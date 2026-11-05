"""从已有证据整理现行说明；不运行模型，不改写任何历史实验记录。"""
from pathlib import Path
import json


def _read(root, relative):
    path = Path(root) / relative
    return json.loads(path.read_text()) if path.exists() else {}


def refresh_results(root):
    """每阶段单独列示，缺少记录时留空，不能用后续成绩替代冻结检查。"""
    root = Path(root)
    first = _read(root, 'artifacts/pilot_v1/acceptance.json')
    probe = _read(root, 'artifacts/pilot_v2/probe/verification.json')
    diagnostic = _read(root, 'artifacts/diagnosis_v1/report.json')
    verified = _read(root, 'artifacts/diagnosis_v1/final_verification.json')
    budget = _read(root, 'artifacts/pilot_v1/budget.json')
    experiment = _read(root, 'artifacts/pilot_v2/experiment.json')
    lines = ['# 实际结果', '', '本页由已有实验记录整理；各阶段分别报告，不将训练图成绩当作新图片表现。', '',
             f"基础学习门槛：{'通过' if probe.get('passed') else '未通过或尚无通过记录'}；第二轮六组共同完成 {experiment.get('common_stage', 0)} 次更新。", '',
             '## 阶段结果', '', '| 阶段 | 最差重合 | 平均重合 | 无目标最多误圈 |', '|---|---:|---:|---:|']
    for key, label in [('head_check', '冻结主模型检查'), ('after', '额外共同学习诊断'), ('after_repair', '额外修正诊断')]:
        values = probe.get(key, {})
        cells = [f'{values[k]*100:.2f}%' if k in values else '未记录' for k in ('minimum_iou', 'mean_iou', 'max_absent_fraction')]
        lines.append('| ' + ' | '.join([label] + cells) + ' |')
    lines += ['', '原门槛：每张至少 30%、平均至少 60%、无目标误圈最多 1%；正确区域位置回答至少 12/16 并超过固定回答。', '',
              '**流程偏差：**冻结检查未通过后，初版仍执行了 96 次共同更新和 92 次修正；这些仅是额外诊断。现行入口已禁止失败后继续，不能把修正后的分数写成冻结成绩。', '',
              '[第二轮完整检查记录](../artifacts/pilot_v2/probe/verification.json) · [执行记录](../artifacts/pilot_v2/execution_audit.json)']
    language = probe.get('after_repair', probe.get('after', {}))
    if 'given_region_correct' in language:
        lines += ['', f"额外诊断给定正确区域的位置回答 {language['given_region_correct']}/16，固定回答 {language.get('constant_answer_correct', '未记录')}/16；不代表自由描述正确。[真实回答](../artifacts/pilot_v2/delivery_checks/report.json)"]
    if first:
        lines += ['', '## 首轮保留结论', '',
                  f"六组完成 {first['common_updates']} 次更新；有正目标的分割查询全部为空：{'是' if first.get('all_positive_segmentation_queries_empty') else '否'}。首轮留出已用于失败分析，不再作为新设置的独立留出。", '',
                  '[首轮验收](../artifacts/pilot_v1/acceptance.json) · [逐组完整结果](../artifacts/pilot_v1/evaluation/heldout/)']
    if diagnostic:
        lines += ['', '## 地面尺度复核', '',
                  f"32 个训练要求的历史指标最大差异为 {diagnostic.get('maximum_historical_metric_difference')}；复现完整：{'是' if diagnostic.get('reproduced') else '否'}。没有重新训练。", '',
                  '两种已知尺度没有共同受检正目标类别，不能归因于尺度；单改圈选规则也不足以通过原门槛。[详细诊断](diagnosis.md)']
    if budget:
        preparation = sum(e['seconds'] for e in budget['entries'] if e['label'].startswith('v2/') and not e['label'].startswith('v2/formal/'))
        lines += ['', '## 预算与验证记录', '',
                  f"按已结算账本，原两小时预算剩余 {(budget['limit_seconds']-budget['used_seconds'])/60:.2f} 分钟；第二轮准备累计 {preparation/60:.2f} 分钟。文档整理没有新增模型计算。", '',
                  '[原预算账本](../artifacts/pilot_v1/budget.json)']
        if budget.get('active'):
            lines += ['', '账本仍有未结算操作，实际可用时间须先核实，不能按上述余额继续训练。']
    if verified:
        lines += ['', f"尺度诊断交付时记录 {verified.get('tests_passed')} 项检查通过；此前第二轮报告记录 32 项。两者是不同时间的检查，当前程序检查以实际运行输出为准。", '',
                  '[尺度复核记录](../artifacts/diagnosis_v1/final_verification.json)']
    lines += ['', '保存后加载及继续更新曾核对一致，共享影响隔离也已检查；这些属于运行可靠性证据，不代表模型效果合格。', '',
              '当前不足以确认核心假设。未完成第二轮六组正式比较；缺少独立人工区域语言评价及 HJ、HS、LT 资料，不报告相应能力。', '',
              '[评价边界](evaluation.md) · [失败分析](diagnosis.md) · [返回目录](../README.md)']
    destination = root / 'docs/results.md'
    destination.write_text('\n'.join(lines) + '\n')
    return destination


def update_scale_summary(root, report):
    """仅替换自动尺度小节，人工维护的失败观察和语言分析必须保留。"""
    path = Path(root) / 'docs/diagnosis.md'
    text = path.read_text()
    start, end = '<!-- scale-summary:start -->', '<!-- scale-summary:end -->'
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
        raise ValueError('诊断文档缺少唯一且顺序正确的自动小节标记')
    groups = report.get('scale_groups', {})
    group_text = '；'.join(('尺度未知' if key == 'unknown' else '像素间距 ' + key) + f'：{count} 张' for key, count in groups.items())
    lines = [f"核对 {report['training_images_audited']} 张训练图；{group_text}。未知不填成 1 米，像素间距不等于实际分辨能力。", '',
             f"原预测完整复现：{'是' if report['reproduced'] else '否'}；历史指标最大差异 {report.get('maximum_historical_metric_difference')}。"]
    if report['reproduced']:
        d = report['threshold_diagnosis']
        lines += ['', f"逐图选择最有利圈选规则，平均上限 {d['per_image_oracle_mean_iou']*100:.2f}%，最差上限 {d['per_image_oracle_minimum_iou']*100:.2f}%；仍有 {d['positive_cases_below_minimum_even_with_oracle']}/16 张低于 30%。统一规则可通过全部区域要求：{'是' if d['shared_threshold_exists'] else '否'}。这只是事后诊断上限，不是新的成绩。", '',
                  '4.6875 米组受检正目标为灌木地、机场；10 米组为裸地、湿地、冰雪、水体、道路、建筑。没有同类可比案例，不支持尺度导致失败的判断。', '',
                  '| 规则 | 4.6875 米组 | 10 米组 |', '|---|---:|---:|',
                  '| 64 像素小区域 | 1,406.25 平方米 | 6,400 平方米 |',
                  '| 2 像素边界距离 | 9.375 米 | 20 米 |',
                  '| 3 像素细长部分半宽 | 14.0625 米 | 30 米 |', '',
                  '附加统一规则为 6,400 平方米、20 米、30 米。规则改变会改变结构评价，但当前基础模型未启用额外空间训练约束，不能据此解释全部失败。', '',
                  '已确认缺少尺度记录及类别不匹配；本次未发现新的输入读取、配准或指标公式错误。只补齐记录与诊断，没有增加模型学习分支。']
    else:
        lines += ['', '原结果未完整复现，停止圈选规则与地面尺度解释。' + report.get('reason', '')]
    path.write_text(text.split(start)[0] + start + '\n\n' + '\n'.join(lines) + '\n\n' + end + text.split(end)[1])
    return path
