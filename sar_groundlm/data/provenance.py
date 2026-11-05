"""只读核对所选原文件、压缩成员、问答行、旧结果与原始权重。"""
import hashlib
import zipfile
from pathlib import Path
from ..common import ROOT,read_json,sha256,atomic_json
from ..legacy_v1.data import SplitArchive


def verify_sources(cfg):
    m=read_json(cfg.data/'manifest.json');files=members=questions=0
    native=[r for r in m['records'] if r['dataset']=='SARLANG-1M']
    for r in m['records']:
        if r['dataset']!='SAR-OVSeg':continue
        for field,digest in [('source_image','sha256'),('source_label','label_sha256')]:
            if sha256(ROOT/r[field])!=r[digest]:raise ValueError('来源文件与提取内容不一致：'+r[field])
            files+=1
    archives=[ROOT/p for p in native[0]['source_archive']]
    with SplitArchive(archives) as parts,zipfile.ZipFile(parts) as archive:
        for r in native:
            member=r.get('source_member',r.get('archive_member'))
            if hashlib.sha256(archive.read(member)).hexdigest()!=r['sha256']:
                raise ValueError('原始压缩成员与提取图像不同')
            members+=1
    source=read_json(ROOT/native[0].get('qa_source',native[0].get('annotation_source')))
    for r in native:
        for qa in r['qa']:
            row=source[qa.get('source_row',qa.get('annotation_row'))];msg=row['messages']
            if (Path(row['images'][0]).name!=r['image_id'] or
                msg[0]['content'].replace('<image>','').strip()!=qa['question'] or msg[1]['content'].strip()!=qa['answer']):
                raise ValueError('问答与原始行对应失败')
            questions+=1
    old=read_json(ROOT/'data/pilot_v1/manifest.json');old_files=0
    for r in old['records']:
        for key,check in [('image','sha256'),('label','label_sha256')]:
            if r.get(key):
                if sha256(ROOT/'data/pilot_v1'/r[key])!=r[check]:raise ValueError('第一轮数据发生变化')
                old_files+=1
    old_results=0
    for entry in read_json(ROOT/'artifacts/pilot_v1/delivery_manifest.json')['files']:
        name=entry['path']
        if name.startswith(('artifacts/','data/')) and name!='artifacts/pilot_v1/budget.json':
            if sha256(ROOT/name)!=entry['sha256']:raise ValueError('历史结果被修改：'+name)
            old_results+=1
    weights=read_json(cfg.model/'LOCAL_PROVENANCE.json')['files']
    for item in weights:
        if sha256(cfg.model/item['name'])!=item['sha256']:raise ValueError('原始模型文件变化')
    result={'passed':True,'selected_source_files':files,'source_archive_members':members,'source_questions':questions,
            'v1_extracted_files_unchanged':old_files,'historical_manifest_artifacts_unchanged':old_results,
            'original_model_files_unchanged':len(weights),'budget_is_intentionally_cumulative':True,
            'scope':'只读核对已选来源与已有历史清单；未对全部原始压缩包重新计算校验值'}
    atomic_json(cfg.output/'source_protection.json',result);return result
