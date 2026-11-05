"""Scientific and runtime boundaries of the core A/B protocol."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from sar_groundlm.common import ROOT
from sar_groundlm.core.state import Settings, Budget, require_gate, immutable_json
from sar_groundlm.core.model import symmetric_alignment
from sar_groundlm.core.data import matching_pairs, probe_case, relation
from sar_groundlm.core.evaluation import binary_counts, competitive_map, score_gate, aggregate, select_hard_pairs
from sar_groundlm.core.review import approved_records


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg=Settings(ROOT/'configs/core_hypothesis.json')

    def test_original_thresholds_are_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            values=dict(self.cfg.values);values['gate_min_iou']=.29
            p=Path(td)/'config.json';p.write_text(json.dumps(values))
            with self.assertRaises(ValueError):Settings(p)

    def test_only_evaluation_revision_can_reuse_prediction_weights(self):
        from sar_groundlm.core.state import evaluation_only_revision
        saved={'config':'a','sources':{'sar_groundlm/core/evaluation.py':'old','model.py':'unchanged'}}
        revised={'config':'a','sources':{'sar_groundlm/core/evaluation.py':'new','model.py':'unchanged'}}
        self.assertEqual(evaluation_only_revision(saved,revised),['sar_groundlm/core/evaluation.py'])
        revised['sources']['model.py']='changed';self.assertEqual(evaluation_only_revision(saved,revised),[])
        revised['sources']['model.py']='unchanged';revised['config']='b'
        self.assertEqual(evaluation_only_revision(saved,revised),[])

    def test_formal_branch_compatibility_is_narrow_and_cannot_resume_optimizer(self):
        import hashlib
        from sar_groundlm.core import state
        name='sar_groundlm/core/training.py';ev='sar_groundlm/core/evaluation.py'
        before="expected=pair.get('description') if kind=='description' else answer(facts,kind,'selected region')"
        after="expected=pair.get('description') if kind=='description' and pair.get('description') else answer(facts,kind,'selected region')"
        with tempfile.TemporaryDirectory() as td:
            cfg=SimpleNamespace(output=Path(td));p=cfg.output/'probe/executed_source'/name;p.parent.mkdir(parents=True);p.write_text(before)
            saved={'config':'same','sources':{name:hashlib.sha256(before.encode()).hexdigest(),ev:'old'}}
            current={'config':'same','sources':{name:hashlib.sha256(after.encode()).hexdigest(),ev:'new'}}
            self.assertEqual(set(state.prediction_compatible_revision(cfg,saved,current)),{name,ev})
            with patch.object(state.torch,'load',return_value={'identity':saved}),patch.object(state,'identity',return_value=current):
                with self.assertRaises(ValueError):state.load(cfg,SimpleNamespace(),'unused',optimizer=object())
            current['sources'][name]='arbitrary modification'
            self.assertEqual(state.prediction_compatible_revision(cfg,saved,current),[])

    def test_full_negative_rotation_and_positive_coverage(self):
        pos=[('p',i) for i in range(16)];neg=[('n',i) for i in range(16)]
        selected=[probe_case(i,pos,neg) for i in range(64)]
        self.assertEqual([selected.count(x) for x in pos],[3]*16)
        self.assertEqual([selected.count(x) for x in neg],[1]*16)

    def test_alignment_updates_both_representations(self):
        torch.manual_seed(3);regions=torch.randn(3,8,requires_grad=True);text=torch.randn(3,8,requires_grad=True)
        loss=symmetric_alignment(regions,text);loss.backward()
        self.assertGreater(float(regions.grad.abs().sum()),0)
        self.assertGreater(float(text.grad.abs().sum()),0)
        right=symmetric_alignment(torch.eye(3),torch.eye(3))
        wrong=symmetric_alignment(torch.eye(3),torch.eye(3).roll(1,0))
        self.assertLess(float(right),float(wrong))
        with self.assertRaises(ValueError):symmetric_alignment(torch.ones(1,3),torch.ones(1,3))

    def test_single_region_formal_description_has_a_real_answer(self):
        from sar_groundlm.core.training import learning_step
        from sar_groundlm.data.tasks import facts_for_mask
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.word=torch.nn.Parameter(torch.tensor(.1));self.mask_head=torch.nn.Linear(1,1)
                self.arm='B';self.received=[]
            def mask_task(self,image,target,answer):
                return {'lm_loss':self.word.square(),'mask_logits':self.mask_head.weight[0,0].expand(8,8),'bundle':{}}
            def forward_task(self,image,q,answer=None,**kwargs):
                if answer is None:raise AssertionError('Missing training answer')
                self.received.append(answer);return {'lm_loss':self.word.square()}
            def shared_parameters(self):return [self.word]
        im=np.zeros((8,8,3),np.float32);lab=np.zeros((8,8),np.uint8);valid=np.ones((8,8),bool)
        record={'id':'single','facts':[facts_for_mask(lab==0,valid)]}
        model=TinyModel();optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        with patch('sar_groundlm.core.training.sample',return_value=(im,lab,valid)):
            result=learning_step(model,optimizer,self.cfg,[(record,0,None)],3)
        self.assertEqual(result['tasks'][0]['matching_pairs'],0)
        self.assertIn('100%',model.received[0])

    def test_alignment_ignores_unknown_pixels_and_avoids_false_pairs(self):
        label=np.full((12,12),255,np.uint8);label[:4,:4]=0;label[8:,8:]=1;valid=label!=255
        pairs=matching_pairs(label,valid,0,[(0,1)])
        self.assertEqual([p['class_id'] for p in pairs],[0,1])
        self.assertTrue(all(not p['mask'][~valid].any() for p in pairs))
        self.assertTrue(all('scatter' not in p['description'] for p in pairs))
        self.assertNotEqual(pairs[0]['description'],pairs[1]['description'])
        self.assertEqual(matching_pairs(np.zeros((8,8),np.uint8),np.ones((8,8),bool),0),[])

    def test_competitive_rule_and_ignore_are_explicit(self):
        p=np.zeros((11,2,2));p[2]=.7;p[3]=.7;p[:,0,0]=.49
        result=competitive_map(p);self.assertEqual(result.tolist(),[[11,2],[2,2]])
        with self.assertRaises(ValueError):competitive_map(p[:10])
        counts=binary_counts(np.ones((2,2),bool),np.eye(2,dtype=bool),np.array([[True,False],[False,True]]))
        self.assertEqual(counts['fp'],0);self.assertEqual(counts['tp'],2)

    def test_gate_requires_language_and_complete_fixed_size(self):
        rows=[{'positive':True,'iou':.8,'false_fraction':.2} for _ in range(16)]
        rows += [{'positive':False,'iou':0.,'false_fraction':0.} for _ in range(16)]
        self.assertFalse(score_gate(self.cfg,rows)['passed'])
        answers=[{'correct':True,'expected':str(i%4)} for i in range(16)]
        self.assertTrue(score_gate(self.cfg,rows,answers)['passed'])
        rows[0]['iou']=.29;self.assertFalse(score_gate(self.cfg,rows,answers)['passed'])
        with self.assertRaises(ValueError):score_gate(self.cfg,rows[:-1],answers)

    def test_constant_answer_does_not_pass_language(self):
        rows=[{'positive':True,'iou':1.,'false_fraction':1.} for _ in range(16)]
        rows += [{'positive':False,'iou':None,'false_fraction':0.} for _ in range(16)]
        answers=[{'correct':True,'expected':'center'} for _ in range(16)]
        self.assertFalse(score_gate(self.cfg,rows,answers)['passed'])

    def test_gate_and_review_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as td:
            cfg=SimpleNamespace(output=Path(td));(cfg.output/'probe').mkdir()
            (cfg.output/'probe/verification.json').write_text(json.dumps({'passed':False,'identity':{'x':1}}))
            with patch('sar_groundlm.core.state.identity',return_value={'x':1}):
                with self.assertRaises(RuntimeError):require_gate(cfg)
            with patch('sar_groundlm.core.review.require_protocol',return_value={}):
                with self.assertRaises(RuntimeError):approved_records(cfg)

    def test_frozen_records_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'frozen.json';immutable_json(p,{'value':1});immutable_json(p,{'value':1})
            with self.assertRaises(ValueError):immutable_json(p,{'value':2})
            self.assertEqual(json.loads(p.read_text()),{'value':1})

    def test_new_budget_counts_old_consumption_and_active_time(self):
        with tempfile.TemporaryDirectory() as td:
            cfg=SimpleNamespace(budget=Path(td)/'budget.json');cfg.budget.write_text(json.dumps({'limit_seconds':7200,'used_seconds':7000,'entries':[]}))
            b=Budget(cfg)
            self.assertEqual(b.remaining,200)
            with self.assertRaises(TimeoutError):b.check(reserve=300)
            b.check(reserve=100)

    def test_hard_pairs_need_development_support(self):
        cm=np.zeros((11,12),int);cm[0,1]=25;cm[1,0]=25
        per=[{'positive_images':3,'positive_pixels':300} for _ in range(11)]
        report={'split':'heldout','confusion':cm.tolist(),'metrics':{'per_class':per},'checkpoint_sha256':'x'}
        with self.assertRaises(ValueError):select_hard_pairs(self.cfg,report)
        report['split']='dev'
        with patch('sar_groundlm.core.evaluation.identity',return_value={}):
            result=select_hard_pairs(self.cfg,report)
            self.assertEqual(result['pairs'],[[0,1]]);self.assertFalse(result['supported'])
            per[0]['positive_images']=2;self.assertEqual(select_hard_pairs(self.cfg,report)['pairs'],[])

    def test_empty_class_metrics_are_not_fake_zero_scores(self):
        result=aggregate([])
        self.assertIsNone(result['mean_iou'])
        self.assertTrue(all(x['iou'] is None for x in result['per_class']))

    def test_relation_only_claims_supported_geometry(self):
        a=np.zeros((10,10),bool);b=a.copy();a[1,1]=True;b[8,8]=True
        self.assertEqual(relation(a,b),'left / above')
        self.assertIsNone(relation(a,np.zeros_like(a)))

    def test_legacy_report_does_not_overwrite_current_docs(self):
        from sar_groundlm.evaluation_v2 import reporting
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'configs').mkdir();(root/'configs/core_hypothesis.json').write_text('{}')
            (root/'docs').mkdir();page=root/'docs/results.md';page.write_text('Current research')
            output=root/'artifacts/pilot_v2';output.mkdir(parents=True);(output/'acceptance.json').write_text('{"learning_gate_passed":false}')
            with patch.object(reporting,'ROOT',root):
                result=reporting.report(SimpleNamespace(output=output))
            self.assertTrue(result['historical_read_only']);self.assertEqual(page.read_text(),'Current research')

    def test_results_publication_uses_missing_values_without_claims(self):
        from sar_groundlm.core.reporting import publish_results
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            result={'learning':{},'learning_gate_passed':False,'conclusion':'目前证据不足','formal_stages':[],
                    'reviewed_images':0,'remaining_seconds':1800,'limitations':['基础学习未通过']}
            p=publish_results(root,result);first=p.read_bytes();publish_results(root,result)
            self.assertEqual(first,p.read_bytes());self.assertIn('未取得',p.read_text())
            self.assertIn('目前证据不足',p.read_text());self.assertLessEqual(len(p.read_text().splitlines()),100)

    def test_predicted_region_input_does_not_reveal_annotation_coverage(self):
        from sar_groundlm.core.evaluation import language_rows
        class Recorder:
            def __init__(self):self.images=[]
            def generate(self,image,*args,**kwargs):self.images.append(image.copy());return 'middle-center'
        im=np.full((8,8,3),.2,np.float32);lab=np.zeros((8,8),np.uint8);lab[4:]=1
        probabilities=np.zeros((11,8,8),np.float32);probabilities[0]=.8
        all_valid=np.ones((8,8),bool);partial=all_valid.copy();partial[4:,4:]=False
        record={'id':'test','query_classes':[0]};first=Recorder();second=Recorder()
        language_rows(first,im,lab,all_valid,record,probabilities)
        language_rows(second,im,lab,partial,record,probabilities)
        # 3 closed questions + 1 raw answer in free condition, then predicted view.
        np.testing.assert_array_equal(first.images[4],second.images[4])

    def test_fresh_candidates_exclude_duplicates_and_require_explicit_review(self):
        from sar_groundlm.core import review as module
        from sar_groundlm.data.readers import read_sar
        from sar_groundlm.data.pilot import signatures
        from test_v2 import write_tif
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'datasets/SAR-OVSeg_release/train'
            (source/'images').mkdir(parents=True);(source/'annotations').mkdir()
            rng=np.random.default_rng(6);old=rng.integers(0,255,(16,16),dtype=np.uint8)
            fresh=rng.integers(0,255,(16,16),dtype=np.uint8);label=np.zeros((16,16),np.uint8);label[8:]=1
            for name,image in [('old',old),('duplicate',old),('fresh',fresh)]:
                write_tif(source/'images'/f'{name}.tif',image);write_tif(source/'annotations'/f'{name}.tif',label)
            digest,ph=signatures(read_sar(source/'images/old.tif'))
            for group in ('pilot_v1','pilot_v2'):
                folder=root/'data'/group;folder.mkdir(parents=True)
                (folder/'manifest.json').write_text(json.dumps({'records':[{'image_id':'old','pixel_sha256':digest,'perceptual_hashes':ph}]}))
            class Config(dict):pass
            cfg=Config(seed=42);cfg.output=root/'artifacts/core';cfg.output.mkdir(parents=True)
            (cfg.output/'final_protocol.json').write_text('{"candidate_count":100}')
            with patch.object(module,'ROOT',root),patch.object(module,'require_protocol',return_value={'candidate_count':100}):
                candidates=module.candidates(cfg);self.assertEqual(candidates['count'],1)
                self.assertEqual(candidates['records'][0]['image_id'],'fresh')
                path=cfg.output/'review/human_review.json'
                with self.assertRaises(ValueError):module.import_review(cfg,path)
                reviewed=json.loads(path.read_text());r=reviewed['records'][0]
                r['reviewer']='synthetic test reviewer';r['reviewed_at']='2026-11-06';r['questions'][0]['approved']=True
                path.write_text(json.dumps(reviewed));approved=module.import_review(cfg,path)
                self.assertEqual(len(approved['records'][0]['questions']),1)
                self.assertTrue(module.approved_records(cfg)[0]['expert_reviewed'])
                (source/'images/fresh.tif').write_bytes(b'changed')
                with self.assertRaises(ValueError):module.approved_records(cfg)


if __name__=='__main__':unittest.main()
