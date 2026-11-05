"""用真实已有记录在临时目录验证文档，不写历史结果、不加载模型。"""
import json
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import unquote

from sar_groundlm.common import ROOT
from sar_groundlm.documentation import refresh_results, update_scale_summary

TOPICS = {'usage.md', 'data.md', 'model.md', 'evaluation.md', 'results.md',
          'diagnosis.md', 'next-design.md', 'references.md',
          'third-party-notices.md', 'maintenance.md'}


class DocumentationTests(unittest.TestCase):
    def fixture(self, root):
        shutil.copytree(ROOT / 'docs', root / 'docs')
        for name in ('README.md', 'AGENTS.md', 'artifacts/pilot_v1/acceptance.json',
                     'artifacts/pilot_v1/budget.json', 'artifacts/pilot_v2/probe/verification.json',
                     'artifacts/pilot_v2/data_validation.json', 'artifacts/pilot_v2/execution_audit.json',
                     'artifacts/pilot_v2/final_tests.log', 'artifacts/pilot_v2/source_protection.json',
                     'artifacts/diagnosis_v1/report.json', 'artifacts/diagnosis_v1/final_verification.json',
                     'data/pilot_v2/manifest.json'):
            source = ROOT / name
            if source.exists():
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
        return SimpleNamespace(output=root / 'artifacts/pilot_v2', data=root / 'data/pilot_v2',
                               budget=root / 'artifacts/pilot_v1/budget.json')

    def test_all_local_links_and_small_topic_inventory(self):
        self.assertEqual({p.name for p in (ROOT / 'docs').rglob('*.md')}, TOPICS)
        self.assertFalse((ROOT / 'docs/archive').exists())
        for p in [ROOT / 'README.md', ROOT / 'AGENTS.md', *sorted((ROOT / 'docs').glob('*.md'))]:
            text = p.read_text()
            self.assertLessEqual(len(text.splitlines()), 100, str(p))
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                    continue
                self.assertTrue((p.parent / unquote(target.split('#')[0])).exists(), f'{p}: {target}')
        index = (ROOT / 'README.md').read_text()
        for name in TOPICS:
            self.assertIn(f'(docs/{name})', index)

    def test_current_report_uses_real_records_in_temporary_directory(self):
        from sar_groundlm.evaluation_v2 import reporting
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self.fixture(root)
            before = {p: p.read_bytes() for p in (root / 'artifacts').rglob('*') if p.is_file()}
            with patch.object(reporting, 'ROOT', root):
                result = reporting.report(cfg)
                reporting.report(cfg)
            self.assertFalse(result['learning_gate_passed'])
            text = (root / 'docs/results.md').read_text()
            self.assertIn('38.13%', text)
            self.assertIn('43.98%', text)
            self.assertIn('49.85', text)
            self.assertIn('42 项', text)
            self.assertEqual({p.name for p in (root / 'docs').glob('*.md')}, TOPICS)
            manifest = json.loads((cfg.output / 'delivery_manifest.json').read_text())
            paths = {r['path'] for r in manifest['files']}
            self.assertIn('docs/third-party-notices.md', paths)
            self.assertIn('AGENTS.md', paths)
            for path, value in before.items():
                self.assertEqual(path.read_bytes(), value, str(path))

    def test_scale_publish_preserves_manual_analysis_and_is_repeatable(self):
        from sar_groundlm.diagnosis import reporting
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self.fixture(root)
            report = json.loads((root / 'artifacts/diagnosis_v1/report.json').read_text())
            path = root / 'docs/diagnosis.md'
            before = path.read_text()
            # 图像生成逻辑未改动；本检查只验证发布边界，不能为整理文档重算预测。
            with patch.object(reporting, 'ROOT', root), patch.object(reporting, 'figures') as plots:
                reporting.publish(cfg, root / 'artifacts/diagnosis_v1', report)
                first = path.read_text()
                reporting.publish(cfg, root / 'artifacts/diagnosis_v1', report)
                self.assertEqual(plots.call_count, 2)
            self.assertEqual(first, path.read_text())
            self.assertEqual(before.split('<!-- scale-summary:start -->')[0], first.split('<!-- scale-summary:start -->')[0])
            self.assertIn('49.75%', first)
            report['reproduced'] = False
            update_scale_summary(root, report)
            self.assertIn('停止圈选规则', path.read_text())
            self.assertNotIn('49.75%', path.read_text())
            self.assertFalse((root / 'docs/ground-scale-diagnosis.md').exists())

    def test_missing_markers_fail_without_overwriting_manual_document(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'docs').mkdir()
            p = root / 'docs/diagnosis.md'
            p.write_text('人工说明')
            with self.assertRaises(ValueError):
                update_scale_summary(root, {})
            self.assertEqual(p.read_text(), '人工说明')

    def test_legacy_report_command_only_reads_saved_acceptance(self):
        from sar_groundlm import reporting
        from sar_groundlm.legacy_v1 import cli
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch.object(reporting, 'ROOT', root), patch('sys.argv', ['sar_groundlm', 'report']), patch('builtins.print'):
                cli.main()
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob('*') if p.is_file()})
