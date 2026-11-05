from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .common import atomic_json,sha256,read_json
from .data import load_intensity
from .model import SARGroundLM


def infer(args):
    image,valid=load_intensity(args.image,args.encoding,args.db_limits)
    model=SARGroundLM(variant=args.variant)
    if args.checkpoint:
        state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        if state["variant"]!=args.variant:
            raise ValueError("Checkpoint variant mismatch")
        if state["model_code_sha256"]!=sha256(Path(__file__).with_name("model.py")):
            raise ValueError("Checkpoint model code mismatch")
        if state["model_provenance"]!=read_json(Path(model.model_path)/"LOCAL_PROVENANCE.json"):
            raise ValueError("Checkpoint base model revision mismatch")
        model.load_trainable(state["model"])
    model.eval()
    with torch.no_grad():
        out=model.forward_task(image,args.question,mask=args.variant!="language")
        answer=model.generate(image,args.question)
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    if out["mask_logits"] is not None:
        p=out["mask_logits"].float().sigmoid().cpu().numpy()
        np.save(folder/"probability.npy",p)
        Image.fromarray(((p>=0.5)&valid).astype(np.uint8)*255).save(folder/"mask.png")
    result={"answer":answer,"variant":args.variant,"checkpoint":args.checkpoint,
            "trained":bool(args.checkpoint),"input_encoding":args.encoding,
            "db_limits":args.db_limits,"output_size":list(image.shape)}
    atomic_json(folder/"result.json",result);return result
