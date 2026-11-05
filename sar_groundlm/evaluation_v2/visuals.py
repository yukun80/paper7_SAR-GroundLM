"""查看真实预测与真实标签，所有图注明用途和训练/评价划分。"""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from ..common import CLASSES,read_json
from ..data.pilot import sample


def overlay(rgb,mask):
    out=np.clip(rgb*255,0,255).astype(np.float32)
    out[mask]=out[mask]*.5+np.array([255,65,30])*.5
    return Image.fromarray(out.astype(np.uint8))


def panel(image,target,pred,path,title):
    h,w=image.shape[:2];canvas=Image.new('RGB',(w*3,h+24),'white')
    for i,m in enumerate((np.zeros((h,w),bool),target,pred)):
        canvas.paste(overlay(image,m),(i*w,24))
    ImageDraw.Draw(canvas).text((4,4),title+' | input / reference / prediction',fill='black')
    Path(path).parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def contact_sheet(cfg):
    m=read_json(cfg.data/'manifest.json');rows={r['id']:r for r in m['records']}
    canvas=Image.new('RGB',(256*4,280*8),'white');draw=ImageDraw.Draw(canvas)
    for i,case in enumerate(m['smoke_cases']):
        r=rows[case['id']];im,label,v=sample(cfg,r);c=case['class_id'];x=(i%2)*512;y=(i//2)*280
        canvas.paste(overlay(im,np.zeros_like(v)),(x,y+24));canvas.paste(overlay(im,(label==c)&v),(x+256,y+24))
        draw.text((x+3,y+4),f'TRAIN {i+1}: {CLASSES[c]}, {r["input"]["source_channels"]} bands',fill='black')
    path=cfg.output/'figures/data_contact_sheet.png';path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    return str(path)
