"""预算、严格学习顺序、重复运行与未知区域的回归检查。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sar_groundlm.engine.budget import Budget
from sar_groundlm.engine.state import require_learning_gate
from sar_groundlm.engine.trainer import common_start,training_cases
from sar_groundlm.data.readers import read_sar,read_label,sovs_reference
from sar_groundlm.evaluation_v2.metrics import binary_metrics
from sar_groundlm.engine.recovery import recover_journal
from sar_groundlm.engine.schedule import epoch_targets
from test_v2 import write_tif


class Config(dict):
    def __init__(self,root):
        super().__init__(preparation_limit_seconds=1200,seed=42,images_per_update=2)
        self.budget=root/'budget.json';self.output=root;self.values=self


class ControlTests(unittest.TestCase):
    def test_interrupted_log_preserved_and_recovered(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'train.jsonl';path.write_text('{"step":1}\n{"step":2}\n{"step":')
            self.assertEqual(recover_journal(path,1),[{'step':1}])
            self.assertIn('"step":2',next((path.parent/'interrupted_updates').iterdir()).read_text())
            self.assertEqual(recover_journal(path,1),[{'step':1}])
            with self.assertRaises(ValueError):recover_journal(path,2)

    def test_live_budget_counts_current_operation(self):
        with tempfile.TemporaryDirectory() as td:
            cfg=Config(Path(td));cfg.budget.write_text(json.dumps({'limit_seconds':7200,'used_seconds':4000,
              'entries':[{'label':'v2/probe/head','seconds':1100}],
              'active':{'label':'v2/probe/check','started_at':100,'pid':999999}}))
            with patch('sar_groundlm.engine.budget.time.time',return_value=190):
                b=Budget(cfg);self.assertEqual(b.remaining,3110);self.assertEqual(b.preparation_used,1190)
                with self.assertRaises(TimeoutError):b.check_active(15)

    def test_interruption_recovery_charges_once(self):
        with tempfile.TemporaryDirectory() as td:
            cfg=Config(Path(td));cfg.budget.write_text(json.dumps({'limit_seconds':7200,'used_seconds':4100,'entries':[],
               'active':{'label':'v2/probe/check','started_at':100,'pid':999999}}))
            with patch('sar_groundlm.engine.budget.time.time',return_value=150),patch('sar_groundlm.engine.budget.os.kill',side_effect=ProcessLookupError):
                b=Budget(cfg);self.assertEqual(b.recover()['charged_seconds'],50)
                self.assertFalse(b.recover()['recovered']);self.assertEqual(b.remaining,3050)

    def test_existing_common_start_cannot_bypass_failed_gate(self):
        with tempfile.TemporaryDirectory() as td:
            cfg=Config(Path(td));(cfg.output/'probe').mkdir();(cfg.output/'common_start.pt').write_text('stale')
            for report in ({'passed':False},{'passed':True,'head_check':{'passed':False}}):
                (cfg.output/'probe/verification.json').write_text(json.dumps(report))
                with self.assertRaises(RuntimeError):common_start(cfg)

    def test_first_epoch_contains_negatives_and_no_repeated_images(self):
        rows=[{'id':str(i),'query_classes':[10,0],
               'facts':[{'present':c in (0,10)} for c in range(11)]} for i in range(16)]
        cfg={'images_per_update':2,'seed':42}
        with patch('sar_groundlm.engine.trainer.records',return_value=(rows,rows)):
            first=[x for step in range(8) for x in training_cases(cfg,step)]
            again=[x for step in range(8) for x in training_cases(cfg,step)]
        self.assertEqual(first,again);self.assertEqual(len({r['id'] for r,c,n in first}),16)
        self.assertEqual(sum(not r['facts'][c]['present'] for r,c,n in first),4)

    def test_rare_positives_cannot_all_become_negative_queries(self):
        rows=[{'facts':[{'present':c==0 or (i==c and c<11)} for c in range(11)]} for i in range(32)]
        targets=epoch_targets(rows,42,0)
        self.assertEqual({c for i,c in targets if rows[i]['facts'][c]['present']},set(range(11)))
        self.assertEqual(len({i for i,c in targets}),32)
        self.assertEqual(sum(not rows[i]['facts'][c]['present'] for i,c in targets),8)

    def test_unknown_connected_hole_is_not_evaluated(self):
        truth=np.ones((32,32),bool);truth[10:22,10:22]=False
        valid=np.ones_like(truth);valid[10:14,10:14]=False
        result=binary_metrics(np.ones_like(truth),truth,valid)
        self.assertEqual(result['internal_negative_count'],0)
        self.assertGreater(binary_metrics(np.ones_like(truth),truth,np.ones_like(truth))['internal_negative_count'],0)

    def test_single_float_gray_db_and_label_nodata(self):
        np.testing.assert_allclose(sovs_reference(np.array([[0,255]],np.uint8))[0,:,0],[0,1])
        np.testing.assert_allclose(sovs_reference(np.array([[-25.,5.]],np.float32))[0,:,0],[0,1])
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'sar.tif';q=Path(td)/'label.tif'
            write_tif(p,np.array([[-20.,-10.],[0,10]],np.float32))
            im=read_sar(p,'explicit_db',(-20,0));np.testing.assert_allclose(im.rgb[...,0],[[0,.5],[1,1]])
            write_tif(q,np.array([[0,1],[1,1]],np.uint8),nodata=0)
            lab,valid=read_label(q,im);self.assertEqual(lab[0,0],255);self.assertFalse(valid[0,0])


if __name__=='__main__':unittest.main()
