"""Public model preparation, explicitly without using local login credentials."""
import importlib.metadata
import json
import platform
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
from .common import ROOT,atomic_json,sha256,read_json
MODEL_REVISION="2fc06364715b967f1860aea9cf38778875588b17"


def download():
    path=ROOT/"models/Qwen3.5-0.8B";path.mkdir(parents=True,exist_ok=True)
    record=path/"LOCAL_PROVENANCE.json"
    if record.exists():
        prior=read_json(record)
        if (prior.get("revision")==MODEL_REVISION and prior.get("repo_id")=="Qwen/Qwen3.5-0.8B"
            and any(f["name"].endswith(".safetensors") and f.get("official_lfs_verified") for f in prior["files"])
            and all((path/f["name"]).is_file() and sha256(path/f["name"])==f["sha256"] for f in prior["files"])):
            return {"downloaded":str(path),"revision":MODEL_REVISION,"existing_files_verified":True}
    info=HfApi(token=False).model_info("Qwen/Qwen3.5-0.8B",revision=MODEL_REVISION,files_metadata=True)
    print("Downloading public revision",info.sha,flush=True)
    snapshot_download("Qwen/Qwen3.5-0.8B",revision=info.sha,local_dir=path,token=False,
                      allow_patterns=["*.json","*.safetensors","*.jinja","*.txt"],max_workers=4)
    files=[]
    for item in info.siblings:
        file=path/item.rfilename
        if file.is_file():
            digest=sha256(file)
            expected=item.lfs.sha256 if item.lfs else None
            if expected and digest!=expected:
                raise ValueError("Downloaded file identity mismatch: "+item.rfilename)
            files.append({"name":item.rfilename,"sha256":digest,"official_lfs_verified":bool(expected)})
    provenance={"repo_id":"Qwen/Qwen3.5-0.8B","revision":info.sha,"files":files}
    atomic_json(path/"LOCAL_PROVENANCE.json",provenance)
    return {"downloaded":str(path),"revision":info.sha}


def environment():
    import torch,sys
    names=["torch","transformers","numpy","Pillow","scipy","rasterio","pyarrow","accelerate","peft","matplotlib"]
    report={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
            "packages":{n:importlib.metadata.version(n) for n in names},
            "cuda_available":torch.cuda.is_available(),"torch_cuda":torch.version.cuda}
    if torch.cuda.is_available():
        report["gpu"]=torch.cuda.get_device_name(0);report["free_total_bytes"]=list(torch.cuda.mem_get_info())
    atomic_json(ROOT/"artifacts/audit/environment.json",report)
    return report


def download_parallel(prefix_path=None, mirror=False):
    """Resume the known sequential HTTP prefix, then verify the complete official hash."""
    import concurrent.futures,os,requests,shutil,time
    path=ROOT/"models/Qwen3.5-0.8B"
    api=HfApi(token=False);info=api.model_info("Qwen/Qwen3.5-0.8B",revision=MODEL_REVISION,files_metadata=True)
    item=next(i for i in info.siblings if i.rfilename.endswith(".safetensors"))
    dest=path/item.rfilename
    if dest.exists() and sha256(dest)==item.lfs.sha256:
        return download()
    partial=dest.with_suffix(".parallel-part")
    known_prefix=Path(prefix_path) if prefix_path is not None else None
    if known_prefix is not None and item.lfs.sha256 not in known_prefix.name:
        raise ValueError("Prefix file does not identify this public model weight")
    prefix=known_prefix.stat().st_size if known_prefix else 0
    with partial.open("wb") as f:
        if known_prefix:
            with known_prefix.open("rb") as source:
                shutil.copyfileobj(source,f,length=2**20)
        f.truncate(item.size)
    fd=os.open(partial,os.O_WRONLY)
    url=("https://modelscope.cn/models/Qwen/Qwen3.5-0.8B/resolve/master/"+item.rfilename if mirror else
         "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/"+info.sha+"/"+item.rfilename)
    block=64*2**20
    segments=[(start,min(start+block,item.size)-1) for start in range(prefix,item.size,block)]
    def fetch(segment):
        start,end=segment
        for attempt in range(4):
            try:
                with requests.get(url+"?range_start="+str(start),headers={"Range":"bytes="+str(start)+"-"+str(end)},
                                  stream=True,timeout=(30,90)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or not response.headers.get("Content-Range","").startswith("bytes "+str(start)+"-"):
                        raise ValueError("Server did not honor the requested range")
                    offset=start
                    for data in response.iter_content(2**20):
                        if offset+len(data)>end+1:
                            raise ValueError("Oversized range response")
                        os.pwrite(fd,data,offset);offset+=len(data)
                    if offset!=end+1:
                        raise ValueError("Incomplete byte range")
                print("range complete",start,end,flush=True)
                return
            except Exception as error:
                if attempt==3:
                    raise
                print("range retry",start,type(error).__name__,flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch,segments))
    finally:
        os.close(fd)
    if sha256(partial)!=item.lfs.sha256:
        raise ValueError("Full public LFS digest verification failed")
    os.replace(partial,dest)
    # All small files already exist; no second weight download is needed.
    files=[]
    for sibling in info.siblings:
        file=path/sibling.rfilename
        if file.is_file():
            digest=sha256(file);expected=sibling.lfs.sha256 if sibling.lfs else None
            if expected and digest!=expected:
                raise ValueError("Identity mismatch")
            files.append({"name":sibling.rfilename,"sha256":digest,"official_lfs_verified":bool(expected)})
    atomic_json(path/"LOCAL_PROVENANCE.json",{"repo_id":"Qwen/Qwen3.5-0.8B","revision":info.sha,
                "weight_download_source":"Qwen ModelScope mirror + verified HF HTTP prefix" if mirror else "Hugging Face",
                "files":files})
    return {"downloaded":str(dest),"official_weight_digest_verified":True}
