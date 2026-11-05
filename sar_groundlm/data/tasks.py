"""文字问题只依赖任务要求；标准答案由有效标签计算，不提前进入预测。"""
import numpy as np
from ..common import CLASSES
from ..legacy_v1.data import region_facts

POSITIONS='upper-left, upper-center, upper-right, middle-left, middle-center, middle-right, lower-left, lower-center, lower-right, or absent'


def prompt(target):
    return 'Locate all '+target+' regions in this SAR image.'


def facts_for_mask(mask,valid):
    lab=np.where(valid,np.where(mask,0,1),255).astype(np.uint8)
    return region_facts(lab,0)


def region_view(image,mask):
    if image.shape[:2]!=mask.shape:raise ValueError('指定区域与图像尺寸不同')
    return np.where(mask[...,None],image,np.float32(.5)).astype(np.float32)


def question(kind,target=None,selected=False):
    subject='the selected region' if selected else 'all '+target+' regions'
    prefix='Only the selected region retains original SAR values; outside it is uniform gray. ' if selected else ''
    if kind=='position':return prefix+'Where is the center of '+subject+'? Answer with one of: '+POSITIONS+'.'
    if kind=='presence':return prefix+'Is '+('any selected region' if selected else target)+' present? Answer yes or no.'
    if kind=='area':return prefix+'What percentage of the valid image area is occupied by '+subject+'? Answer with an integer percent.'
    return prefix+'Describe the observable location and area of '+subject+'.'


def answer(facts,kind,target):
    if kind=='presence':return 'yes' if facts['present'] else 'no'
    if kind=='position':return facts['position']
    if kind=='area':return str(round(facts['area_fraction']*100))+'%'
    if not facts['present']:return 'No '+target+' region is present in the valid area.'
    return target+' occupies '+str(round(100*facts['area_fraction']))+'% of the valid area; its center is '+facts['position']+'.'


def choose_class(record,cycle):
    present=[k for k in record['query_classes'] if record['facts'][k]['present']]
    present+= [k for k,f in enumerate(record['facts']) if f['present'] and k not in present]
    absent=[k for k in range(11) if k not in present]
    # 每张图片先学习正例，再周期性学习不存在的目标；六组共用这一顺序。
    if cycle%4==3 and absent:return absent[(cycle//4)%len(absent)]
    return present[(cycle//4+cycle%4)%len(present)]
