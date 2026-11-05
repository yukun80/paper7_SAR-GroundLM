"""Resource audit and source immutability checks without downloading datasets."""
from collections import Counter
from pathlib import Path
import json
import hashlib
import zipfile
import pyarrow.parquet as pq
from .common import ROOT,CLASSES,atomic_json,read_json,sha256
from .data import SplitArchive


def audit():
    root=ROOT/"datasets";terra=root/"TerraCoT/downloads"
    source_counts={}
    for split in ("train","test"):
        folder=root/"SAR-OVSeg_release"/split
        images={p.stem for p in (folder/"images").glob("*.tif")}
        labels={p.stem for p in (folder/"annotations").glob("*.tif")}
        source_counts[split]={"images":len(images),"labels":len(labels),
                              "pair_mismatches":len(images^labels)}
    t={"files":0,"readable_shards":0,"readable_rows":0,"sources":Counter(),
       "mask_token_mismatches":0,"label_mask_mismatches":0,"invalid_files":[],
       "admitted_training_rows":0}
    for p in sorted(terra.glob("*.parquet")):
        t["files"]+=1
        try:
            pf=pq.ParquetFile(p)
        except Exception:
            t["invalid_files"].append(p.name);continue
        t["readable_shards"]+=1;t["readable_rows"]+=pf.metadata.num_rows
        for batch in pf.iter_batches(columns=["source","labels","answer","seg_rle"],batch_size=1024):
            for r in batch.to_pylist():
                t["sources"][r["source"]]+=1;masks=json.loads(r["seg_rle"])
                t["mask_token_mismatches"]+=len(masks)!=r["answer"].count("[SEG]")
                t["label_mask_mismatches"]+=len(masks)!=len(r.get("labels") or [])
    t["sources"]=dict(t["sources"])
    t["exclusion_reason"]="Optical-conditioned text and dual-polarization composite SAR need independent single-polarization evidence review"
    archives={}
    for p in (root/"SARLANG-1M/downloads").glob("*.zip"):
        try:
            with zipfile.ZipFile(p) as z:
                archives[p.name]={"central_directory_readable":True,"entries":len(z.infolist()),
                                  "full_crc_checked":False}
        except Exception:
            archives[p.name]={"central_directory_readable":False}
    pilot=read_json(ROOT/"data/pilot_v1/validation.json")
    classes={s:[CLASSES[i] for i in range(11) if counts.get(str(i),0)>0]
             for s,counts in pilot["pixel_counts"].items()}
    report={"sar_ovseg":source_counts,"pilot_class_coverage":classes,
            "terra_cot":t,"sarlang_archives":archives,
            "missing":["HJ/HS/LT target imagery and labels","expert-reviewed independent region-language truth",
                       "confirmed single-polarization input for currently excluded multichannel float scenes"],
            "scene_provenance":"user confirms isolation; public scene provenance search intentionally omitted",
            "no_physical_information_loss_claim":True}
    atomic_json(ROOT/"artifacts/audit/data_audit.json",report)
    return report


def source_protection():
    manifest=read_json(ROOT/"data/pilot_v1/manifest.json")
    checked=0
    for r in manifest["records"]:
        if r["dataset"]=="SAR-OVSeg":
            for source,digest in [(r["source_image"],r["sha256"]),(r["source_label"],r["label_sha256"])]:
                if sha256(ROOT/source)!=digest:
                    raise ValueError("Source changed since selected extraction: "+source)
                checked+=1
    language=[r for r in manifest["records"] if r["dataset"]=="SARLANG-1M"]
    archive_members=0;question_records=0
    if language:
        archives=language[0]["source_archive"]
        with SplitArchive([ROOT/p for p in archives]) as parts,zipfile.ZipFile(parts) as archive:
            for record in language:
                if record["source_archive"]!=archives:
                    raise ValueError("Unexpected multiple source archives")
                if hashlib.sha256(archive.read(record["archive_member"])).hexdigest()!=record["sha256"]:
                    raise ValueError("Source archive member differs from the selected image")
                archive_members+=1
        source=read_json(ROOT/language[0]["annotation_source"])
        for record in language:
            for qa in record["qa"]:
                row=source[qa["annotation_row"]]
                question=next(m["content"] for m in row["messages"] if m["role"]=="user").replace("<image>","").strip()
                answer=next(m["content"] for m in row["messages"] if m["role"]=="assistant").strip()
                if Path(row["images"][0]).name!=record["image_id"] or question!=qa["question"] or answer!=qa["answer"]:
                    raise ValueError("Selected question/answer differs from its original row")
                question_records+=1
    report={"selected_source_files_verified":checked,
            "selected_source_archive_members_verified":archive_members,
            "selected_source_question_records_verified":question_records,
            "selected_source_content_matches_extraction":True,
            "public_test_used_for_training":False,
            "scope":"selected source content, not a full checksum audit of all original archives"}
    atomic_json(ROOT/"artifacts/audit/source_protection.json",report)
    return report
