"""日志与最后一次完整保存对齐；丢失的更新保留原记录，预算不退回。"""
import json
import os


def recover_journal(path, saved_step):
    lines=path.read_text().splitlines() if path.exists() else []
    kept=[];discarded=[]
    for index,line in enumerate(lines):
        try:row=json.loads(line)
        except json.JSONDecodeError:
            if index!=len(lines)-1:raise ValueError('训练日志中间有损坏记录')
            discarded.append(line);continue
        (kept if row['step']<=saved_step else discarded).append(line)
    rows=[json.loads(line) for line in kept]
    if [r['step'] for r in rows]!=list(range(1,saved_step+1)):
        raise ValueError('已保存更新与日志不一致，不能静默跳过训练样本')
    if discarded:
        folder=path.parent/'interrupted_updates';folder.mkdir(exist_ok=True)
        (folder/f'{len(list(folder.iterdir())):03d}.jsonl').write_text('\n'.join(discarded)+'\n')
    temp=path.with_suffix('.tmp');temp.write_text(''.join(line+'\n' for line in kept));os.replace(temp,path)
    return rows
