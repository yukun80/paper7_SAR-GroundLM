"""Actual pretrained-model acceptance checks, isolated from comparison checkpoints."""
import gc
import time
import numpy as np
import torch
from .common import CLASSES,atomic_json,read_json,seed_all,core_source_identity,sha256
from .training import DATA,RUN,Budget,records,sample,segmentation_prompt,checkpoint,restore,train_step
from .model import SARGroundLM
from .losses import mask_loss


def grad_norm(module):
    return float(sum(p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None).sqrt()) \
        if any(p.grad is not None for p in module.parameters()) else 0.0


def probe():
    report_path=RUN/"verification.json"
    if report_path.exists():
        prior=read_json(report_path)
        if prior.get("passed") and prior.get("source_identity")==core_source_identity() and prior.get("verification_source_sha256")==sha256(__file__):
            return prior
    budget=Budget();segs,langs=records()
    smoke_ids=read_json(DATA/"manifest.json")["smoke_ids"]
    smoke=[next(r for r in segs if r["id"]==i) for i in smoke_ids]
    model=SARGroundLM(variant="B0")
    image,label=sample(smoke[0]);c=smoke[0]["query_classes"][0]
    target=torch.from_numpy((label==c).copy());valid=torch.from_numpy((label!=255).copy())
    report={"environment":"qwen3vl","base":"Qwen3.5-0.8B","trained_checkpoint_used":False,"checks":{},
            "source_identity":core_source_identity(),"verification_source_sha256":sha256(__file__)}
    with budget.charge("language_vs_detached_actual_shared_update"):
        initial={n:t.clone() for n,t in model.trainable_state().items()}
        model.variant="language"
        opt=torch.optim.AdamW(model.parameter_groups(),weight_decay=0.01)
        train_step(model,opt,smoke[0],langs[0],0,smoke=True)
        language_update={n:p.detach().cpu().clone() for n,p in model.base.named_parameters() if p.requires_grad}
        model.load_trainable(initial);model.zero_grad(set_to_none=True);del opt
        model.variant="detached";seed_all(42)
        opt=torch.optim.AdamW(model.parameter_groups(),weight_decay=0.01)
        train_step(model,opt,smoke[0],langs[0],0,smoke=True)
        shared_difference=max(float((p.detach().cpu()-language_update[n]).abs().max()) for n,p in model.base.named_parameters() if p.requires_grad)
        assert shared_difference==0,shared_difference
        report["checks"]["language_vs_detached_shared_update_max_difference"]=shared_difference
        model.load_trainable(initial);model.zero_grad(set_to_none=True);model.variant="B0";seed_all(42)
        del opt,initial,language_update
    with budget.charge("gradient_paths_and_forward"):
        for variant in ["detached","B0","B1","B2","B3"]:
            model.variant=variant;model.zero_grad(set_to_none=True);model.train()
            out=model.forward_task(image,segmentation_prompt(CLASSES[c]),mask=True)
            loss,_=mask_loss(out["mask_logits"],target,valid,structure=variant in ("B2","B3"),water=c==0)
            assert out["mask_logits"].shape==image.shape and torch.isfinite(loss)
            loss.backward()
            norms={"visual":grad_norm(model.base.model.visual),
                   "language":grad_norm(model.base.model.language_model),"mask":grad_norm(model.mask_head),
                   "semantic":grad_norm(model.semantics),"structure":grad_norm(model.structure)}
            if variant=="detached":
                assert norms["visual"]==0 and norms["language"]==0 and norms["mask"]>0,norms
            else:
                assert norms["visual"]>0 and norms["language"]>0 and norms["mask"]>0,norms
            if variant in ("B1","B3"):
                assert norms["semantic"]>0
            if variant in ("B2","B3"):
                assert norms["structure"]>0
            report["checks"][variant+"_mask_gradient"]=norms
        model.variant="detached";model.zero_grad(set_to_none=True)
        out=model.forward_task(image,"What is visible in this SAR image?","A SAR scene.")
        out["lm_loss"].backward()
        assert grad_norm(model.base.model.visual)>0 and grad_norm(model.base.model.language_model)>0
        report["checks"]["detached_keeps_language_learning"]=True
        model.variant="B3";model.zero_grad(set_to_none=True)
        out=model.forward_task(image,"What is visible in this SAR image?","A SAR scene.")
        out["lm_loss"].backward()
        language_shared={"semantic":grad_norm(model.semantics),"structure":grad_norm(model.structure)}
        assert min(language_shared.values())>0,language_shared
        report["checks"]["both_mechanisms_receive_language_gradients"]=language_shared
        model.variant="B0";model.eval()
        with torch.no_grad():
            a=model.forward_task(image,segmentation_prompt(CLASSES[c]),mask=True)["mask_logits"].float()
            b=model.forward_task(image,segmentation_prompt(CLASSES[(c+1)%11]),mask=True)["mask_logits"].float()
            response=float((a-b).abs().mean())
            assert response>1e-7
            # Changing teacher-forced future answers must not affect the query mask.
            a=model.forward_task(image,segmentation_prompt(CLASSES[c]),"Water in the north.",mask=True)["mask_logits"].float()
            b=model.forward_task(image,segmentation_prompt(CLASSES[c]),"Buildings in the south.",mask=True)["mask_logits"].float()
            leak=float((a-b).abs().max())
            assert leak<0.05,leak
        report["checks"]["different_query_mean_logit_change"]=response
        report["checks"]["future_answer_max_logit_change"]=leak
    model.zero_grad(set_to_none=True)
    optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=0.01)

    def smoke_losses():
        model.eval();values=[];ious=[]
        with torch.no_grad():
            for r in smoke:
                im,lab=sample(r);k=r["query_classes"][0]
                pred=model.forward_task(im,segmentation_prompt(CLASSES[k]),mask=True)["mask_logits"]
                value,_=mask_loss(pred,torch.from_numpy((lab==k).copy()),torch.from_numpy((lab!=255).copy()))
                values.append(float(value))
                from .metrics import binary_metrics
                ious.append(binary_metrics(pred.float().cpu().numpy()>=0,lab==k,lab!=255)["iou"])
        return values,ious

    with budget.charge("eight_image_learning_before"):
        before,before_iou=smoke_losses()
    curve=[]
    for i in range(64):
        if budget.remaining<120:
            raise TimeoutError("Insufficient budget for probe; did not start matrix")
        with budget.charge("eight_image_learning_step_"+str(i+1)):
            row=train_step(model,optimizer,smoke[i%8],langs[i%8],i,smoke=True)
        curve.append(row)
        if (i+1)%8==0:
            print("smoke learning",i+1,"mask",row.get("mask"),flush=True)
    with budget.charge("eight_image_learning_after"):
        after,after_iou=smoke_losses()
    report["learning"]={"images":smoke_ids,"updates":len(curve),"before_mask_loss":before,
                        "after_mask_loss":after,"before_query_iou":before_iou,
                        "after_query_iou":after_iou,"curve":curve}
    if not np.mean(after)<np.mean(before):
        atomic_json(report_path,report)
        raise RuntimeError("Eight-image learning check failed; inspect and fix before matrix")
    report["checks"]["eight_image_loss_improved"]=True
    atomic_json(report_path,report)
    with budget.charge("checkpoint_resume_and_inference"):
        ckpt=RUN/"probe"/"resume.pt"
        checkpoint(model,optimizer,ckpt,64,{"isolated_probe":True})
        # One update followed by restore and identical update checks optimizer/RNG recovery.
        train_step(model,optimizer,smoke[0],langs[0],64,smoke=True)
        model.eval()
        with torch.no_grad():
            expected=model.forward_task(image,segmentation_prompt(CLASSES[c]),mask=True)["mask_logits"].float().cpu()
        restore(model,optimizer,ckpt)
        train_step(model,optimizer,smoke[0],langs[0],64,smoke=True)
        model.eval()
        with torch.no_grad():
            actual=model.forward_task(image,segmentation_prompt(CLASSES[c]),mask=True)["mask_logits"].float().cpu()
        difference=float((expected-actual).abs().max())
        assert difference<1e-5,difference
        report["checks"]["resumed_update_max_difference"]=difference
        # Save and load into a newly constructed pretrained model.
        checkpoint(model,optimizer,ckpt,65,{"isolated_probe":True})
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
        model=SARGroundLM(variant="B0");optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=0.01)
        step=restore(model,optimizer,ckpt);model.eval()
        with torch.no_grad():
            reloaded=model.forward_task(image,segmentation_prompt(CLASSES[c]),mask=True)["mask_logits"].float().cpu()
        reload_diff=float((actual-reloaded).abs().max())
        assert reload_diff<1e-5 and step==65
        report["checks"]["fresh_load_max_difference"]=reload_diff
        report["generated_answer"]=model.generate(image,"Describe the observable contents of this SAR image.",64)
        from .evaluation import overlay
        from PIL import Image,ImageDraw
        canvas=Image.new("RGB",(image.shape[1]*3,image.shape[0]+24),"white")
        for i,m in enumerate([np.zeros_like(label,dtype=bool),label==c,reloaded.numpy()>=0]):
            canvas.paste(overlay(image,m),(i*image.shape[1],24))
        ImageDraw.Draw(canvas).text((4,4),"Training example: SAR | reference | learned prediction",fill="black")
        canvas.save(RUN/"probe"/"learning_example.png")
    report["passed"]=True
    report["scientific_scope"]="engineering and training-set learning only"
    report["remaining_training_budget_seconds"]=budget.remaining
    atomic_json(report_path,report)
    return report
