---
license: apache-2.0
task_categories:
- visual-question-answering
- image-segmentation
language:
- en
tags:
- remote-sensing
- earth-observation
- chain-of-thought
- referring-segmentation
- SAR
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: train-*.parquet
  - split: vqa
    path: vqa-*.parquet
---

# TerraCoT

TerraCoT is a remote-sensing **chain-of-thought VQA** dataset in which every `[SEG]` token in
an answer is **grounded by a pixel-level ground-truth mask**. Each sample pairs an optical
image with a reasoning trace, and (where applicable) a second modality in `image2`
(Sentinel-1 SAR, or a bi-temporal companion image).

Two splits:

- **`train`** — samples that carry an ordered `labels` list aligned to the `[SEG]` tokens.
- **`vqa`** — question/answer samples without the `labels` column.

## Format

Each row has the following columns.

| column | type | description |
|---|---|---|
| `image` | image | primary optical image |
| `image2` | image | second modality: 3-channel SAR PNG (R=VH, G=VV, B=avg) or a temporal companion image; `null` when not applicable |
| `image_id` | string | source patch / tile id |
| `question` | string | the prompt |
| `answer` | string | reasoning + answer; contains one or more `[SEG]` tokens |
| `seg_rle` | string | JSON list of COCO-RLE masks — **one per `[SEG]`, in the order the `[SEG]` tokens appear in `answer`** |
| `labels` | list[str] | *(train split only)* ordered class names, aligned to `[SEG]` |
| `source` | string | subset name |
| `task_type`, `height`, `width`, `num_objects` | — | metadata |

**Key invariant:** for every row, `len(json.loads(seg_rle))` equals the number of `[SEG]`
tokens in `answer`. Mask *i* corresponds to the *i*-th `[SEG]`.

Each mask is a COCO-RLE dict: `{"size": [height, width], "counts": "<rle-string>"}`.

## Usage

### Load

```python
from datasets import load_dataset

ds = load_dataset("sy1998/TerraCoT", split="vqa", streaming=True)
row = next(iter(ds))
print(row["source"], row["answer"])
```

### Decode the `[SEG]` masks

```python
import json
import pycocotools.mask as mask_utils

rles = json.loads(row["seg_rle"])                 # one entry per [SEG]
assert len(rles) == row["answer"].count("[SEG]")

masks = []
for rle in rles:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    m = mask_utils.decode({"size": rle["size"], "counts": counts})  # HxW uint8 {0,1}
    masks.append(m)
```

### Overlay a mask on the image

```python
import numpy as np
from PIL import Image

img = row["image"].convert("RGB")
arr = np.array(img).astype(np.float32)
m = masks[0].astype(bool)
arr[m] = 0.5 * arr[m] + 0.5 * np.array([255, 0, 0])     # red overlay for the 1st [SEG]
Image.fromarray(arr.astype(np.uint8))
```

### Read the second image

```python
img2 = row["image2"]     # PIL image, or None
# For SAR (chatearthnet / bigearthnet): R=VH, G=VV, B=(VH+VV)/2,
# from Sentinel-1 GRD dB clipped to [-30, 0] and scaled to 0-255.
```
