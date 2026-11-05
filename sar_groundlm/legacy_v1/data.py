"""Read-only source access, selected extraction and evidence-derived region facts."""
import collections
import io
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from .common import ROOT, CLASSES, atomic_json, image_hash, read_json, sha256


class SplitArchive(io.RawIOBase):
    """Seek across existing .zip.001 files without concatenating or altering them."""
    def __init__(self, paths):
        self.files = [Path(p).open("rb") for p in paths]
        self.sizes = [Path(p).stat().st_size for p in paths]
        self.length, self.position = sum(self.sizes), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if pos < 0:
            raise ValueError("Negative archive offset")
        self.position = pos
        return pos

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        result, offset = [], 0
        for file, length in zip(self.files, self.sizes):
            if size <= 0:
                break
            if self.position < offset + length:
                local = self.position - offset
                file.seek(local)
                block = file.read(min(size, length - local))
                result.append(block)
                self.position += len(block)
                size -= len(block)
            offset += length
        return b"".join(result)

    def close(self):
        for file in getattr(self, "files", []):
            file.close()
        super().close()


def load_intensity(path, encoding="uint8", db_limits=None):
    """One normalized float channel; never infers physical dB units."""
    nodata=None
    if encoding=="db":
        import rasterio
        with rasterio.open(path) as source:
            if source.count!=1:
                raise ValueError("dB route requires one explicitly selected polarization")
            a=source.read(1);nodata=source.nodata
    else:
        a = np.asarray(Image.open(path))
    if a.ndim == 3:
        if a.shape[2] < 3 or not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
            raise ValueError("Non-gray input needs an explicit verified polarization selection")
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError("Expected single-channel SAR")
    if encoding == "uint8":
        if a.dtype != np.uint8:
            raise ValueError("uint8 route requires uint8 input")
        return a.astype(np.float32) / 255.0, np.ones(a.shape, dtype=bool)
    if encoding != "db" or db_limits is None or not db_limits[0] < db_limits[1]:
        raise ValueError("dB route requires explicit fixed physical limits")
    if not np.issubdtype(a.dtype, np.floating):
        raise ValueError("dB input must be float, not reverse-converted grayscale")
    valid = np.isfinite(a)
    if nodata is not None and np.isfinite(nodata):
        valid &= a!=nodata
    lo, hi = db_limits
    out = np.where(valid, np.clip((a - lo) / (hi - lo), 0, 1), 0).astype(np.float32)
    return out, valid


def region_facts(label, category):
    valid = label != 255
    mask = (label == category) & valid
    n = int(mask.sum())
    facts = {"category": CLASSES[category], "class_id": category, "present": bool(n),
             "pixels": n, "valid_pixels": int(valid.sum()), "area_fraction": float(n / max(1, valid.sum()))}
    if not n:
        facts.update(position="absent", shape="absent", bbox=None)
        return facts
    ys, xs = np.where(mask); h, w = mask.shape
    y, x = float(ys.mean() / h), float(xs.mean() / w)
    vertical = "upper" if y < 1/3 else "lower" if y >= 2/3 else "middle"
    horizontal = "left" if x < 1/3 else "right" if x >= 2/3 else "center"
    components, count = ndi.label(mask); areas = np.bincount(components.ravel())[1:]
    border = mask & ~ndi.binary_erosion(mask); perimeter = int(border.sum())
    slender = perimeter * perimeter / max(1, n)
    holes = ndi.binary_fill_holes(mask) & valid & ~mask
    facts.update(position=vertical + "-" + horizontal,
                 bbox=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                 shape="elongated_or_fragmented" if slender > 100 else "compact_or_broad",
                 semantic_components=int(count), small_region_count=int((areas <= 64).sum()),
                 boundary_pixels=perimeter, internal_non_target_pixels=int(holes.sum()))
    return facts


def region_qa(facts):
    name = facts["category"]
    # Named category is supplied by the question, never scored as inferred recognition.
    return [
        {"kind": "presence", "question": "Is " + name + " present in the valid annotated area? Answer yes or no.",
         "answer": "yes" if facts["present"] else "no"},
        {"kind": "position", "question": "Where is the center of the " + name +
         " region? Answer with one position (upper/middle/lower-left/center/right) or absent.",
         "answer": facts["position"]},
        {"kind": "area", "question": "What percentage of the valid annotated area is " + name +
         "? Give the nearest integer percent.", "answer": str(round(100*facts["area_fraction"])) + "%"},
    ]


def _label_stat(path):
    with Image.open(path) as im:
        a = np.asarray(im)
        if a.ndim != 2 or a.dtype != np.uint8:
            raise ValueError(str(path))
        counts = np.bincount(a.ravel(), minlength=256)
        if set(np.flatnonzero(counts)) - set(range(11)) - {255}:
            raise ValueError("Unknown labels")
        return {"id": path.stem, "counts": counts[:11].tolist(),
                "ignore": int(counts[255]), "size": list(im.size)}


def prepare(root=ROOT / "data/pilot_v1"):
    root = Path(root); manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return validate(root)
    root.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "artifacts/audit/train_label_counts.json"
    source = ROOT / "datasets/SAR-OVSeg_release/train"
    if cache.exists():
        stats = read_json(cache)
    else:
        with ThreadPoolExecutor(max_workers=8) as pool:
            stats = list(pool.map(_label_stat, sorted((source/"annotations").glob("*.tif"))))
        atomic_json(cache, stats)
    rng = np.random.default_rng(42); order = rng.permutation(len(stats)).tolist()
    by_class = [[i for i in order if stats[i]["counts"][c] >= 8] for c in range(11)]
    used, hashes, dhashes, records, quarantined = set(), set(), [], [], []
    for split, size in [("train",128),("dev",32),("heldout",64)]:
        for j in range(size):
            c = j % 11
            candidates = by_class[c][:512] + order
            for i in candidates:
                if i in used:
                    continue
                used.add(i); stem = stats[i]["id"]; image_path = source/"images"/(stem+".tif")
                try:
                    a, _ = load_intensity(image_path)
                except (ValueError, OSError) as error:
                    quarantined.append({"source":stem,"reason":"not_verified_single_uint8","detail":str(error)[:180]})
                    continue
                gray = np.rint(a*255).astype(np.uint8); h = image_hash(gray)
                small = np.asarray(Image.fromarray(gray).resize((9,8)))
                dh = np.packbits(small[:,1:] > small[:,:-1]).tobytes()
                near = any(s != split and sum((x^y).bit_count() for x,y in zip(dh,old)) <= 2 for old,s in dhashes)
                if h in hashes or near:
                    quarantined.append({"source":stem,"reason":"duplicate_or_near_duplicate"}); continue
                label_path = source/"annotations"/(stem+".tif"); label = np.asarray(Image.open(label_path))
                if label.shape != gray.shape:
                    raise ValueError("Image-label size mismatch")
                hashes.add(h); dhashes.append((dh,split))
                dest = root/"images"/"ovseg"/(stem+".tif"); lab = root/"labels"/(stem+".tif")
                dest.parent.mkdir(parents=True,exist_ok=True); lab.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(image_path,dest); shutil.copyfile(label_path,lab)
                facts = [region_facts(label,k) for k in range(11)]
                present = [k for k in range(11) if facts[k]["present"]]
                if not present:
                    raise ValueError("Selected sample has no valid positive class")
                primary = c if c in present else present[j % len(present)]
                queries = [primary] + [k for k in present if k != primary]
                absent = [k for k in range(11) if not facts[k]["present"]]
                if absent:
                    queries.append(absent[j % len(absent)])
                records.append({"id":"ovseg:"+stem,"dataset":"SAR-OVSeg","split":split,
                    "original_split":"train","image":str(dest.relative_to(root)), "label":str(lab.relative_to(root)),
                    "source_image":str(image_path.relative_to(ROOT)), "source_label":str(label_path.relative_to(ROOT)),
                    "image_id":stem,"satellite":None,"source_scene":None,"provenance":"unavailable_user_attested_split",
                    "encoding":"uint8","physical_polarization":"unverified",
                    "sha256":sha256(dest),"pixel_sha256":h,"label_sha256":sha256(lab),
                    "query_classes":queries,"facts":facts})
                break
            else:
                raise RuntimeError("Could not fill class-balanced subset")
    language_path = ROOT/"datasets/SARLANG-1M/extracted/Text/Text/VQA/train/SARVQA1_train.json"
    rows = read_json(language_path)
    names = sorted({Path(r["images"][0]).name for r in rows}); rng.shuffle(names); selected = {}
    archive_paths = sorted((ROOT/"datasets/SARDet-100K/Images").glob("train.zip.*"))
    with SplitArchive(archive_paths) as parts, zipfile.ZipFile(parts) as archive:
        members = {Path(i.filename).name:i for i in archive.infolist() if not i.is_dir()}; cursor = 0
        for split, size in [("train",128),("dev",32),("heldout",64)]:
            n=0
            while n<size:
                name=names[cursor];cursor+=1
                if name not in members:
                    continue
                payload=archive.read(members[name])
                im=Image.open(io.BytesIO(payload)); im.load();a=np.asarray(im)
                if a.ndim==3:
                    if a.shape[2]<3 or not (np.array_equal(a[...,0],a[...,1]) and np.array_equal(a[...,0],a[...,2])):
                        quarantined.append({"source":name,"reason":"non_gray_input"});continue
                    a=a[...,0]
                if a.ndim!=2 or a.dtype!=np.uint8:
                    continue
                h=image_hash(a);small=np.asarray(Image.fromarray(a).resize((9,8)))
                dh=np.packbits(small[:,1:] > small[:,:-1]).tobytes()
                near=any(s!=split and sum((x^y).bit_count() for x,y in zip(dh,old))<=2 for old,s in dhashes)
                if h in hashes or near:
                    quarantined.append({"source":name,"reason":"duplicate_or_near_duplicate"});continue
                hashes.add(h);dhashes.append((dh,split))
                dest=root/"images"/"sarlang"/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
                record={"id":"sarlang:"+name,"dataset":"SARLANG-1M","split":split,"original_split":"train",
                    "image":str(dest.relative_to(root)),"label":None,
                    "source_archive":[str(p.relative_to(ROOT)) for p in archive_paths],
                    "archive_member":members[name].filename,"annotation_source":str(language_path.relative_to(ROOT)),
                    "image_id":name,"satellite":None,"source_scene":None,"encoding":"uint8",
                    "physical_polarization":"unverified","sha256":sha256(dest),"pixel_sha256":h,"qa":[]}
                selected[name]=record;records.append(record);n+=1
    for idx,row in enumerate(rows):
        name=Path(row["images"][0]).name
        if name not in selected:
            continue
        msgs=row["messages"]
        question=next(m["content"] for m in msgs if m["role"]=="user").replace("<image>","").strip()
        answer=next(m["content"] for m in msgs if m["role"]=="assistant").strip()
        if not any(q["question"]==question for q in selected[name]["qa"]):
            selected[name]["qa"].append({"question":question,"answer":answer,"annotation_row":idx,"kind":"native_vqa"})
    balance_questions(list(selected.values()))
    del rows
    if any(not r["qa"] for r in selected.values()):
        raise RuntimeError("Missing native language annotation")
    smoke=[r["id"] for r in records if r["dataset"]=="SAR-OVSeg" and r["split"]=="train"][:8]
    manifest={"schema":"sar_groundlm.pilot.v1","seed":42,"classes":CLASSES,"ignore":255,
              "qa_selection":"answer_type_balanced_v1",
              "split_attestation":"User confirms target tests do not leak; source scenes/sensors not independently established.",
              "native_test_used_for_selection":False,"smoke_ids":smoke,"records":records}
    atomic_json(manifest_path,manifest);atomic_json(root/"quarantine.json",quarantined)
    report=validate(root);contact_sheet(root);return report


def balance_questions(rows):
    """Retain <=4 questions, including different answer types; rotate first type."""
    for index,r in enumerate(rows):
        pool=r["qa"];buckets={"yes":[],"no":[],"other":[]}
        for qa in pool:
            answer=qa["answer"].strip().lower()
            buckets[answer if answer in ("yes","no") else "other"].append(qa)
        chosen=[bucket[0] for bucket in buckets.values() if bucket]
        chosen.extend(q for q in pool if q not in chosen)
        chosen=chosen[:4]
        preferred=("yes","no","other")[index%3]
        first=next((q for q in chosen if (q["answer"].lower() if q["answer"].lower() in ("yes","no") else "other")==preferred),chosen[0])
        r["qa"]=[first]+[q for q in chosen if q is not first]


def rebalance_selected_questions(root=ROOT/"data/pilot_v1"):
    root=Path(root);manifest=read_json(root/"manifest.json")
    selected={r["image_id"]:r for r in manifest["records"] if r["dataset"]=="SARLANG-1M"}
    for r in selected.values():
        r["qa"]=[]
    source=ROOT/"datasets/SARLANG-1M/extracted/Text/Text/VQA/train/SARVQA1_train.json"
    for idx,row in enumerate(read_json(source)):
        name=Path(row["images"][0]).name
        if name not in selected:
            continue
        question=next(m["content"] for m in row["messages"] if m["role"]=="user").replace("<image>","").strip()
        answer=next(m["content"] for m in row["messages"] if m["role"]=="assistant").strip()
        if not any(q["question"]==question for q in selected[name]["qa"]):
            selected[name]["qa"].append({"question":question,"answer":answer,"annotation_row":idx,"kind":"native_vqa"})
    balance_questions(list(selected.values()))
    manifest["qa_selection"]="answer_type_balanced_v1"
    atomic_json(root/"manifest.json",manifest);contact_sheet(root);return validate(root)


def validate(root=ROOT/"data/pilot_v1"):
    root=Path(root);manifest=read_json(root/"manifest.json");seen=set();counts=collections.Counter()
    pixels=collections.defaultdict(lambda:np.zeros(256,dtype=np.int64))
    for r in manifest["records"]:
        path=root/r["image"]
        if sha256(path)!=r["sha256"]:
            raise ValueError("Image identity changed: "+r["id"])
        a,v=load_intensity(path,r["encoding"]);h=image_hash(np.rint(a*255).astype(np.uint8))
        if h in seen:
            raise ValueError("Duplicate selected image")
        seen.add(h);counts[r["dataset"]+"/"+r["split"]]+=1
        if r["label"]:
            label=np.asarray(Image.open(root/r["label"]))
            if label.shape!=a.shape or sha256(root/r["label"])!=r["label_sha256"]:
                raise ValueError("Label mismatch")
            if set(np.unique(label))-set(range(11))-{255}:
                raise ValueError("Invalid labels")
            pixels[r["split"]]+=np.bincount(label.ravel(),minlength=256)
        elif not 1<=len(r["qa"])<=4:
            raise ValueError("Language record must have 1..4 questions")
    for dataset in ["SAR-OVSeg","SARLANG-1M"]:
        for split,n in [("train",128),("dev",32),("heldout",64)]:
            assert counts[dataset+"/"+split]==n
    report={"images":len(seen),"counts":dict(counts),"manifest_sha256":sha256(root/"manifest.json"),
            "all_selected_images_decoded":True,"all_selected_labels_checked":True,
            "pixel_counts":{k:{str(i):int(n) for i,n in enumerate(v) if n} for k,v in pixels.items()},
            "native_test_used":False,"physical_polarization_verified":False,
            "scene_level_independence":"user_attested_not_independently_verified"}
    atomic_json(root/"validation.json",report);return report


def contact_sheet(root):
    import textwrap
    root=Path(root);rows=read_json(root/"manifest.json")["records"]
    selected=[r for r in rows if r["dataset"]=="SAR-OVSeg" and r["split"]=="train"][:8]
    selected += [r for r in rows if r["dataset"]=="SARLANG-1M" and r["split"]=="train"][:4]
    sheet=Image.new("RGB",(768, len(selected)*150),"white");draw=ImageDraw.Draw(sheet)
    colors=np.random.default_rng(42).integers(30,255,(256,3),dtype=np.uint8);colors[255]=[128,128,128]
    for i,r in enumerate(selected):
        im=Image.open(root/r["image"]).convert("RGB");im.thumbnail((145,145));sheet.paste(im,(0,i*150))
        if r["label"]:
            lab=np.asarray(Image.open(root/r["label"]));p=Image.fromarray(colors[lab]);p.thumbnail((145,145))
            sheet.paste(p,(150,i*150));text=", ".join(CLASSES[c] for c in r["query_classes"])
        else:
            text=r["qa"][0]["question"]+" / "+r["qa"][0]["answer"]
        draw.text((305,i*150+8),r["id"],fill="black")
        draw.multiline_text((305,i*150+30),"\n".join(textwrap.wrap(text,62)),fill="black")
    sheet.save(root/"sample_review.png")
