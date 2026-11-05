import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["water", "building", "road", "cropland", "forest", "shrubland",
           "grassland", "wetland", "bareland", "snow and ice", "airport"]
VARIANTS = ["language", "detached", "B0", "B1", "B2", "B3"]


def require_environment():
    import sys
    if Path(sys.prefix).name != "qwen3vl":
        raise RuntimeError("Activate conda environment qwen3vl, or use ./run.sh")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(name, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def seed_all(seed=42):
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False


def image_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()


def core_source_identity():
    return {name:sha256(ROOT/"sar_groundlm"/name) for name in
            ["model.py","data.py","losses.py","training.py","common.py"]}
