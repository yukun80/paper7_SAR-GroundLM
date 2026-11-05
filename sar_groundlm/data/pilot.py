"""固定第二轮小数据：只复制所选文件，所有问题按图片分组。"""
import copy
import io
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from ..common import ROOT, CLASSES, atomic_json, read_json, sha256, image_hash
from ..legacy_v1.data import SplitArchive, region_facts, balance_questions
from .readers import read_sar, read_label


def signatures(image):
    """完整数值校验加逐通道近似指纹；两通道不能拆成不同样本。"""
    hashes=[]
    for c in range(2):
        small=np.asarray(Image.fromarray(image.rgb[...,c]).resize((9,8)))
        hashes.append(int.from_bytes(np.packbits(small[:,1:]>small[:,:-1]).tobytes(),'big'))
    return image_hash(image.rgb), hashes


class DuplicateIndex:
    def __init__(self):
        self.rows=[]

    def conflict(self, digest, perceptual, split):
        for old, ph, group in self.rows:
            if digest==old or (group!=split and min((a^b).bit_count() for a in perceptual for b in ph)<=2):
                return True
        return False

    def add(self, digest, perceptual, split):
        self.rows.append((digest,perceptual,split))


def sample(cfg, record):
    image=read_sar(cfg.data/record['image'],cfg['input_profile'],cfg.values.get('db_limits'))
    if record.get('label'):
        label,valid=read_label(cfg.data/record['label'],image)
    else:
        label=None;valid=image.valid
    return image.rgb,label,valid


def records(cfg, split='train'):
    rows=read_json(cfg.data/'manifest.json')['records']
    return ([r for r in rows if r['split']==split and r['dataset']=='SAR-OVSeg'],
            [r for r in rows if r['split']==split and r['dataset']=='SARLANG-1M'])


def _copy(path, dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and sha256(path)!=sha256(dest):
        raise ValueError('目标已有不同内容：'+str(dest))
    if not dest.exists():
        shutil.copyfile(path,dest)


def prepare(cfg):
    manifest=cfg.data/'manifest.json'
    if manifest.exists():
        return validate(cfg)
    cfg.data.mkdir(parents=True,exist_ok=True)
    v1=read_json(ROOT/'data/pilot_v1/manifest.json')
    stats=read_json(ROOT/'artifacts/audit/train_label_counts.json')
    by_id={s['id']:s for s in stats}
    rng=np.random.default_rng(cfg['seed']);order=rng.permutation(len(stats)).tolist()
    source=ROOT/'datasets/SAR-OVSeg_release/train'
    old_ids={r['id'] for r in v1['records']}
    old_heldout={r['id'] for r in v1['records'] if r['split']=='heldout'}
    index=DuplicateIndex();used=set();rows=[];rejected=Counter()
    # 第一轮所有图像对第二轮全新留出图片均构成排除项。
    old_signatures=[]
    for r in v1['records']:
        im=read_sar(ROOT/'data/pilot_v1'/r['image'])
        old_signatures.append(signatures(im))

    def accept_seg(stem,split,channel,primary=None):
        rid='ovseg:'+stem
        if rid in used or rid in old_heldout or (split=='heldout' and rid in old_ids):
            return False
        path=source/'images'/(stem+'.tif')
        image=read_sar(path,cfg['input_profile'],cfg.values.get('db_limits'))
        actual=1 if image.metadata['source_channels']==1 else 2
        if actual!=channel:
            return False
        digest,ph=signatures(image)
        if index.conflict(digest,ph,split):
            rejected['duplicate_or_near_duplicate']+=1;return False
        if split=='heldout' and any(digest==d or min((a^b).bit_count() for a in ph for b in oldph)<=2 for d,oldph in old_signatures):
            rejected['previous_round_overlap']+=1;return False
        labpath=source/'annotations'/(stem+'.tif');label,valid=read_label(labpath,image)
        facts=[region_facts(label,k) for k in range(11)]
        present=[k for k in range(11) if facts[k]['present']]
        if not present:return False
        primary=primary if primary in present else max(present,key=lambda k:facts[k]['pixels'])
        dest=cfg.data/'images/ovseg'/path.name;labdest=cfg.data/'labels'/labpath.name
        _copy(path,dest);_copy(labpath,labdest)
        absent=[k for k in range(11) if k not in present]
        rows.append({'id':rid,'dataset':'SAR-OVSeg','split':split,'original_split':'train',
                     'image_id':stem,'image':str(dest.relative_to(cfg.data)),'label':str(labdest.relative_to(cfg.data)),
                     'source_image':str(path.relative_to(ROOT)),'source_label':str(labpath.relative_to(ROOT)),
                     'sha256':sha256(dest),'label_sha256':sha256(labdest),'pixel_sha256':digest,'perceptual_hashes':ph,
                     'input':image.metadata,'query_classes':[primary]+[c for c in present if c!=primary]+absent[:1],
                     'facts':facts,'qa':[],'satellite':None,'source_scene':None,
                     'split_attestation':v1['split_attestation']})
        index.add(digest,ph,split);used.add(rid);return True

    # 保留原八张练习图；灰度训练/开发优先沿用各自旧划分，避免不必要变化。
    for rid in v1['smoke_ids']:
        old=next(r for r in v1['records'] if r['id']==rid)
        assert accept_seg(old['image_id'],'train',1,old['query_classes'][0]),rid
    for split,size in cfg['split_sizes'].items():
        for channel in (1,2):
            count=lambda:sum(r['split']==split and r['dataset']=='SAR-OVSeg' and r['input']['source_channels']==channel for r in rows)
            if channel==1 and split in ('train','dev'):
                for r in v1['records']:
                    if count()>=size//2:break
                    if r['dataset']=='SAR-OVSeg' and r['split']==split:
                        accept_seg(r['image_id'],split,1,r['query_classes'][0])
            classes=list(range(11)) if channel==2 else [0,1,2,3,4,6,8]
            while count()<size//2:
                c=classes[count()%len(classes)]
                # 优先挑足够大的正例；小区域另外保留，不让每个稀有类只剩几个像素。
                candidates=[stats[i]['id'] for i in order if stats[i]['counts'][c]>=256]
                candidates += [stats[i]['id'] for i in order if stats[i]['counts'][c]>=8]
                for stem in candidates:
                    if accept_seg(stem,split,channel,c):break
                else:raise RuntimeError(f'无法满足 {split} 通道 {channel} 类别 {c} 的抽样')
            print('prepared SAR',split,channel,count(),flush=True)

    # 原生问答保留完整训练和开发部分，不按单个问题跨划分抽样。
    for old in v1['records']:
        if old['dataset']!='SARLANG-1M' or old['split']=='heldout':continue
        r=copy.deepcopy(old);path=ROOT/'data/pilot_v1'/r['image'];im=read_sar(path)
        digest,ph=signatures(im)
        if index.conflict(digest,ph,r['split']):
            raise RuntimeError('保留的 SARLANG 图像与新分割划分重复；需调整分割样本')
        _copy(path,cfg.data/r['image']);r.update(input=im.metadata,pixel_sha256=digest,perceptual_hashes=ph)
        rows.append(r);index.add(digest,ph,r['split']);used.add(r['id'])
    qa_path=ROOT/'datasets/SARLANG-1M/extracted/Text/Text/VQA/train/SARVQA1_train.json'
    qa_rows=read_json(qa_path)
    names=sorted({Path(r['images'][0]).name for r in qa_rows});rng.shuffle(names)
    old_names={r['image_id'] for r in v1['records'] if r['dataset']=='SARLANG-1M'}
    selected={};archives=sorted((ROOT/'datasets/SARDet-100K/Images').glob('train.zip.*'))
    with SplitArchive(archives) as parts,zipfile.ZipFile(parts) as archive:
        members={Path(i.filename).name:i for i in archive.infolist() if not i.is_dir()}
        for name in names:
            if len(selected)==cfg['split_sizes']['heldout']:break
            if name in old_names or name not in members:continue
            payload=archive.read(members[name]);a=np.asarray(Image.open(io.BytesIO(payload)))
            if a.ndim!=2 or a.dtype!=np.uint8:continue
            # 在内存读取原压缩成员，不展开整个档案。
            from .readers import SARImage
            im=SARImage(np.repeat((a.astype(np.float32)/255)[...,None],3,-1),np.ones(a.shape,bool),{})
            digest,ph=signatures(im)
            if index.conflict(digest,ph,'heldout') or any(digest==d or min((x^y).bit_count() for x in ph for y in oldph)<=2 for d,oldph in old_signatures):continue
            dest=cfg.data/'images/sarlang'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
            real=read_sar(dest)
            r={'id':'sarlang:'+name,'dataset':'SARLANG-1M','split':'heldout','original_split':'train',
               'image_id':name,'image':str(dest.relative_to(cfg.data)),'label':None,
               'source_archive':[str(p.relative_to(ROOT)) for p in archives],
               'source_member':members[name].filename,'qa_source':str(qa_path.relative_to(ROOT)),
               'source_image':None,'sha256':sha256(dest),'input':real.metadata,
               'pixel_sha256':digest,'perceptual_hashes':ph,'qa':[], 'satellite':None,'source_scene':None,
               'split_attestation':v1['split_attestation']}
            selected[name]=r;index.add(digest,ph,'heldout')
    if len(selected)!=64:raise RuntimeError('未取得 64 张新的 SARLANG 留出图')
    for row_index,r in enumerate(qa_rows):
        name=Path(r['images'][0]).name
        if name not in selected:continue
        messages=r.get('messages',r.get('conversations',[]))
        if not messages:raise ValueError('未知问答格式')
        question=messages[0].get('content',messages[0].get('value','')).replace('<image>','').strip()
        answer=messages[1].get('content',messages[1].get('value','')).strip()
        qs=selected[name]['qa']
        if question and answer and not any(q['question']==question for q in qs):
            qs.append({'question':question,'answer':answer,'source_row':row_index,
                       'kind':'presence' if answer.lower() in ('yes','no') else 'native'})
    balance_questions(list(selected.values()))
    rows.extend(selected.values());del qa_rows
    smoke=[{'id':rid,'class_id':next(r for r in rows if r['id']==rid)['query_classes'][0]} for rid in v1['smoke_ids']]
    # 补足原八图没有覆盖的类别，优先非中心位置，降低位置固定回答的优势。
    extras=[8,5,7,9,10,0,2,1];chosen=set(v1['smoke_ids'])
    for c in extras:
        pool=[r for r in rows if r['dataset']=='SAR-OVSeg' and r['split']=='train' and r['input']['source_channels']==2 and r['id'] not in chosen and r['facts'][c]['pixels']>=256]
        pool.sort(key=lambda r:(r['facts'][c]['position']=='middle-center',r['image_id']))
        if not pool:raise RuntimeError('练习图类别覆盖不足：'+CLASSES[c])
        r=pool[0];smoke.append({'id':r['id'],'class_id':c});chosen.add(r['id'])
    atomic_json(manifest,{'schema':'sar_pilot_v2','seed':42,'config_sha256':cfg.digest,
                         'classes':CLASSES,'ignore':255,'records':rows,'smoke_cases':smoke,
                         'v1_manifest_sha256':sha256(ROOT/'data/pilot_v1/manifest.json'),
                         'previous_heldout_role':'historical_diagnostics_only','native_test_used':False,
                         'split_attestation':v1['split_attestation'],'rejected':dict(rejected)})
    return validate(cfg)


def validate(cfg):
    m=read_json(cfg.data/'manifest.json');counts=Counter();coverage={};index=DuplicateIndex()
    if m['config_sha256']!=cfg.digest:raise ValueError('数据清单与设置不一致')
    for r in m['records']:
        if sha256(cfg.data/r['image'])!=r['sha256']:raise ValueError('图像内容已改变')
        im=read_sar(cfg.data/r['image'],cfg['input_profile'],cfg.values.get('db_limits'));d,ph=signatures(im)
        if d!=r['pixel_sha256'] or index.conflict(d,ph,r['split']):raise ValueError('图片重复或指纹改变')
        index.add(d,ph,r['split']);counts[(r['dataset'],r['split'])]+=1
        if r.get('label'):
            if sha256(cfg.data/r['label'])!=r['label_sha256']:raise ValueError('标签内容已改变')
            lab,valid=read_label(cfg.data/r['label'],im)
            coverage.setdefault(r['split'],set()).update(int(k) for k in np.unique(lab) if k!=255)
            for c in range(11):
                if region_facts(lab,c)!=r['facts'][c]:raise ValueError('标签与文字事实不一致')
        elif not (1<=len(r['qa'])<=4 and len({q['question'] for q in r['qa']})==len(r['qa'])):
            raise ValueError('问答条数或问题去重错误')
    for dataset in ('SAR-OVSeg','SARLANG-1M'):
        for split,n in cfg['split_sizes'].items():
            if counts[(dataset,split)]!=n:raise ValueError('图片数量错误')
    for split,n in cfg['split_sizes'].items():
        channels=Counter(r['input']['source_channels'] for r in m['records'] if r['dataset']=='SAR-OVSeg' and r['split']==split)
        if channels!={1:n//2,2:n//2}:raise ValueError('通道配额不匹配')
        if coverage[split]!=set(range(11)):raise ValueError('没有覆盖全部 11 类')
    # 已有清单重跑也复核历史隔离，不能只依赖第一次抽样时的判断。
    v1=read_json(ROOT/'data/pilot_v1/manifest.json')
    old_heldout={r['id'] for r in v1['records'] if r['split']=='heldout'}
    if old_heldout & {r['id'] for r in m['records']}:raise ValueError('混入第一轮已分析留出图')
    previous=DuplicateIndex()
    for old in v1['records']:
        d,ph=signatures(read_sar(ROOT/'data/pilot_v1'/old['image']))
        previous.add(d,ph,'history')
    for r in m['records']:
        if r['split']=='heldout' and previous.conflict(r['pixel_sha256'],r['perceptual_hashes'],'heldout'):
            raise ValueError('新留出图与历史图片重复或近似重复')
    old_native={r['id']:r for r in v1['records'] if r['dataset']=='SARLANG-1M' and r['split']!='heldout'}
    native={r['id']:r for r in m['records'] if r['dataset']=='SARLANG-1M' and r['split']!='heldout'}
    if set(native)!=set(old_native):raise ValueError('原生问答训练开发图片发生变化')
    for key,r in native.items():
        if any(r[k]!=old_native[key][k] for k in ('sha256','split','qa')):raise ValueError('原生问答未保持原安排')
    smoke=m['smoke_cases'];selected={r['id']:r for r in m['records']}
    if len(smoke)!=16 or len({s['id'] for s in smoke})!=16:raise ValueError('固定练习图数量错误')
    if not set(v1['smoke_ids'])<={s['id'] for s in smoke}:raise ValueError('缺少原八图')
    if {s['class_id'] for s in smoke}!=set(range(11)):raise ValueError('练习正目标未覆盖全部类别')
    for s in smoke:
        r=selected[s['id']]
        if r['split']!='train' or not r['facts'][s['class_id']]['present'] or all(f['present'] for f in r['facts']):
            raise ValueError('练习正负目标或划分错误')
    report={'passed':True,'images':len(m['records']),'counts':{f'{a}/{b}':n for (a,b),n in counts.items()},
            'coverage':{s:sorted(v) for s,v in coverage.items()},'manifest_sha256':sha256(cfg.data/'manifest.json'),
            'input_profile':cfg['input_profile'],'original_test_used':False,'config_sha256':cfg.digest,
            'fresh_heldout_against_v1_checked':True,'native_train_dev_unchanged':True,'smoke_positive_classes':list(range(11))}
    atomic_json(cfg.output/'data_validation.json',report)
    return report
