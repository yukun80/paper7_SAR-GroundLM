import io
import tempfile
import unittest
import zipfile
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sar_groundlm.data import SplitArchive,load_intensity,region_facts,region_qa,balance_questions
from sar_groundlm.model import to_spatial,from_spatial,SharedSemantics,SharedStructure,QueryMaskHead
from sar_groundlm.losses import mask_loss
from sar_groundlm.metrics import binary_metrics,score_answer
from sar_groundlm.recovery import recover_logs
from sar_groundlm.common import ROOT


class DataTests(unittest.TestCase):
    def test_interrupted_updates_are_archived_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder=root/"B0";folder.mkdir()
            torch.save({"step":2},folder/"last.pt")
            log=folder/"train.jsonl"
            log.write_text('{"step":1}\n{"step":2}\n{"step":3}\n{"step":')
            changes=recover_logs(root)
            self.assertEqual([json.loads(x)["step"] for x in log.read_text().splitlines()],[1,2])
            self.assertEqual(changes[0]["archived_lines"],2)
            self.assertFalse(changes[0]["budget_reset"])
            self.assertEqual(recover_logs(root),[])

    def test_question_types_and_no_duplicate_answers_in_selection(self):
        rows=[{"qa":[{"question":"q"+str(j),"answer":a} for j,a in enumerate(["yes","yes","no","ship","three"])]} for _ in range(3)]
        balance_questions(rows)
        self.assertEqual([r["qa"][0]["answer"] for r in rows],["yes","no","ship"])
        for r in rows:
            self.assertEqual(len(r["qa"]),4)
            self.assertEqual(len({q["question"] for q in r["qa"]}),4)

    def test_split_archive_cross_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            raw=io.BytesIO()
            with zipfile.ZipFile(raw,"w") as z:
                z.writestr("sample.txt","real archive content")
            data=raw.getvalue();paths=[]
            for i,part in enumerate([data[:17],data[17:63],data[63:]]):
                p=Path(td)/str(i);p.write_bytes(part);paths.append(p)
            with SplitArchive(paths) as source,zipfile.ZipFile(source) as z:
                self.assertEqual(z.read("sample.txt"),b"real archive content")

    def test_db_is_not_quantized(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"db.tif"
            values=np.array([[-20.001,-20.002],[np.nan,-3]],np.float32)
            Image.fromarray(values).save(p)
            x,v=load_intensity(p,"db",[-30,0])
            self.assertNotEqual(x[0,0],x[0,1])
            self.assertFalse(v[1,0]);self.assertEqual(x[1,0],0)
            with self.assertRaises(ValueError):
                load_intensity(p,"uint8")

    def test_composite_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"mixed.png";a=np.zeros((8,8,3),np.uint8);a[...,1]=120
            Image.fromarray(a).save(p)
            with self.assertRaises(ValueError):
                load_intensity(p)

    def test_finite_db_nodata_is_excluded(self):
        import rasterio
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"db.tif"
            with rasterio.open(p,"w",driver="GTiff",height=2,width=2,count=1,dtype="float32",nodata=-9999) as dst:
                dst.write(np.array([[-15,-9999],[-7,-2]],np.float32),1)
            a,valid=load_intensity(p,"db",[-30,0])
            self.assertFalse(valid[0,1]);self.assertEqual(a[0,1],0)

    def test_facts_do_not_invent_instances_or_attributes(self):
        label=np.full((10,10),255,np.uint8);label[:5,:5]=0;label[5:,5:]=1
        facts=region_facts(label,0)
        self.assertEqual(facts["area_fraction"],0.5)
        self.assertEqual(facts["position"],"upper-left")
        self.assertNotIn("instances",facts)
        self.assertEqual(region_facts(label,2)["position"],"absent")
        self.assertEqual(region_qa(facts)[0]["answer"],"yes")


class ModelContractTests(unittest.TestCase):
    def test_real_native_processor_spatial_alignment(self):
        from transformers import AutoImageProcessor
        processor=AutoImageProcessor.from_pretrained(ROOT/"models/Qwen3.5-0.8B",local_files_only=True)
        for h,w in [(16,16),(8,32)]:
            expected=np.arange(h*w,dtype=np.float32).reshape(h,w)/(h*w-1)
            plane=np.repeat(np.repeat(expected,16,axis=0),16,axis=1)
            rgb=np.repeat(plane[...,None],3,axis=2)
            encoded=processor(images=[rgb],do_resize=False,do_rescale=False,do_normalize=False,return_tensors="pt")
            values=encoded["pixel_values"].float().mean(1,keepdim=True)
            recovered=to_spatial(values,encoded["image_grid_thw"][0].tolist(),2)[0,0].numpy()
            np.testing.assert_allclose(recovered,expected,rtol=0,atol=1e-6)

    def test_native_spatial_order_roundtrip(self):
        x=torch.arange(8*6*3).reshape(48,3).float()
        self.assertTrue(torch.equal(x,from_spatial(to_spatial(x,[1,8,6],2),2)))

    def test_ignored_pixels_cannot_change_loss_or_gradients(self):
        logits=torch.zeros((16,16),requires_grad=True);target=torch.zeros(16,16)
        valid=torch.ones(16,16,dtype=torch.bool);valid[:5]=False
        first,_=mask_loss(logits,target,valid,True,True);first.backward()
        self.assertEqual(float(logits.grad[:5].abs().sum()),0)
        changed=target.clone();changed[:5]=1
        second,_=mask_loss(logits,changed,valid,True,True)
        self.assertAlmostEqual(float(first),float(second),places=6)

    def test_empty_valid_mask_zero_gradient(self):
        logits=torch.randn(8,8,requires_grad=True)
        loss,_=mask_loss(logits,torch.ones(8,8),torch.zeros(8,8,dtype=torch.bool),True,True)
        loss.backward();self.assertEqual(float(loss),0);self.assertEqual(float(logits.grad.abs().sum()),0)

    def test_detach_cuts_both_routes(self):
        head=QueryMaskHead(16,20,dim=16)
        features=torch.randn(64,16,requires_grad=True);query=torch.randn(20,requires_grad=True)
        loss=head(features.detach(),query.detach(),[1,8,8],2,(32,32)).square().mean();loss.backward()
        self.assertIsNone(features.grad);self.assertIsNone(query.grad)
        self.assertGreater(float(head.query[0].weight.grad.abs().sum()),0)

    def test_shared_adapters_identity_at_initialization(self):
        torch.manual_seed(42);x=torch.randn(64,16);hint=torch.randn(20)
        semantic=SharedSemantics(16,20,dim=16)
        spatial=SharedStructure(16,dim=8)
        self.assertTrue(torch.equal(x,semantic(x,hint)))
        self.assertTrue(torch.equal(x,spatial([x,x,x],[1,8,8],2)))
        self.assertEqual(semantic.last_assignments.shape,(64,4))


class MetricsTests(unittest.TestCase):
    def test_ignore_not_background_and_hole(self):
        target=np.zeros((32,32),bool);target[5:27,5:27]=True;target[13:18,13:18]=False
        valid=np.ones_like(target);valid[:3]=False
        pred=target.copy();pred[:3]=True
        perfect=binary_metrics(pred,target,valid)
        self.assertEqual(perfect["iou"],1);self.assertEqual(perfect["fp"],0)
        pred[13:18,13:18]=True
        bad=binary_metrics(pred,target,valid)
        self.assertEqual(bad["internal_fp"],25)

    def test_missing_denominators_stay_undefined(self):
        a=np.zeros((8,8),bool)
        m=binary_metrics(a,a,np.ones_like(a))
        self.assertIsNone(m["iou"]);self.assertIsNone(m["boundary_f1"])

    def test_answer_scoring(self):
        self.assertTrue(score_answer("Upper left.","upper-left","position"))
        self.assertFalse(score_answer("lower-right","upper-left","position"))
        self.assertFalse(score_answer("Not upper-left.","upper-left","position"))
        self.assertFalse(score_answer("upper-left or lower-right","upper-left","position"))
        self.assertFalse(score_answer("There might be water.","yes","presence"))
        self.assertTrue(score_answer("42%","40%","area"))


if __name__=="__main__":
    unittest.main()
