"""检验地面几何和分数并列等真实边界，不依赖大型模型。"""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import rasterio
from affine import Affine
from sar_groundlm.data.scale import geometry,read_scale,processed_geometry
from sar_groundlm.diagnosis.statistics import pixel_counts,threshold_curve,threshold_analysis,ground_metrics


class ScaleTests(unittest.TestCase):
    def test_unknown_units_not_one_metre(self):
        for crs,factor in [(None,None),('EPSG:4326',None),('unknown',None)]:
            s=geometry((20,30),Affine.identity(),crs,None,factor)
            self.assertEqual(s['status'],'unknown');self.assertNotIn('spacing_xy_m',s)
        self.assertEqual(geometry((20,30),Affine.scale(10), 'EPSG:32630','metre',1,False)['status'],'unknown')
        self.assertEqual(geometry((20,30),Affine.scale(10), 'EPSG:32630','unknown',1)['status'],'unknown')

    def test_rotated_anisotropic_resize_preserves_footprint(self):
        transform=Affine.translation(100,200)*Affine.rotation(31)*Affine.scale(4,-9)
        s=geometry((20,30),transform,'EPSG:32630','metre',1)
        np.testing.assert_allclose(s['spacing_xy_m'],[4,9]);self.assertTrue(s['orthogonal'])
        self.assertAlmostEqual(s['pixel_area_m2'],36)
        for shape in [(10,10),(40,60)]:
            q=processed_geometry(s,shape,(2,3))
            self.assertAlmostEqual(s['footprint_area_m2'],q['footprint_area_m2'])
            np.testing.assert_allclose(s['footprint_corners_m'],q['footprint_corners_m'])
        np.testing.assert_allclose(processed_geometry(s,(10,10))['spacing_xy_m'],[12,18])

    def test_feet_and_shear_area_conversion(self):
        s=geometry((10,10),Affine(3,1,0,0,4,0),'projected','feet',.3048)
        self.assertAlmostEqual(s['pixel_area_m2'],12*.3048**2);self.assertFalse(s['orthogonal'])
        m=ground_metrics(np.ones((10,10),bool),np.ones((10,10),bool),np.ones((10,10),bool),s)
        self.assertFalse(m['physical_distance_available']);self.assertGreater(m['target_area_m2'],0)

    def test_label_registration_conflict(self):
        with tempfile.TemporaryDirectory() as td:
            paths=[Path(td)/'im.tif',Path(td)/'lab.tif']
            for i,path in enumerate(paths):
                with rasterio.open(path,'w',driver='GTiff',width=8,height=8,count=1,dtype='uint8',
                                   crs='EPSG:32630',transform=Affine(10,0,i*10,0,-10,100)) as f:f.write(np.ones((1,8,8),np.uint8))
            with self.assertRaises(ValueError):read_scale(*paths)

    def test_same_pixel_size_different_ground_area(self):
        t=np.zeros((20,20),bool);t[5:15,5:15]=True;v=np.ones_like(t)
        fine=geometry(t.shape,Affine.scale(4.6875,-4.6875),'metric','metre',1)
        coarse=geometry(t.shape,Affine.scale(10,-10),'metric','metre',1)
        a=ground_metrics(t,t,v,fine);b=ground_metrics(t,t,v,coarse)
        self.assertEqual(a['small_components_64px'],b['small_components_64px'])
        self.assertEqual(a['small_components_6400m2'],1);self.assertEqual(b['small_components_6400m2'],0)
        self.assertEqual(b['boundary_f1_20m'],1)

    def test_inclusive_area_limit_and_unknown_width(self):
        t=np.zeros((20,20),bool);t[6:14,6:14]=True;v=np.ones_like(t)
        s=geometry(t.shape,Affine.rotation(19)*Affine.scale(10,-10),'metric','metre',1)
        self.assertEqual(ground_metrics(t,t,v,s)['small_components_6400m2'],1)
        t[:]=True;v[:,10]=False
        out=ground_metrics(t,t,v,s,water=True)
        self.assertIsNone(out['water_halfwidth_m_median'])

    def test_replay_is_single_attempt(self):
        import json
        from sar_groundlm.diagnosis.replay import replay
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);protocol={'original':'frozen'}
            (folder/'replay_attempt.json').write_text(json.dumps({'protocol':protocol}))
            (folder/'replay.json').write_text(json.dumps({'complete':True,'rows':[]}))
            with patch('sar_groundlm.diagnosis.replay.SARGroundLM',side_effect=AssertionError('不应再次运行模型')):
                self.assertTrue(replay(None,folder,[],{},protocol)['complete'])
                with self.assertRaises(ValueError):replay(None,folder,[],{},{'original':'changed'})
            (folder/'replay.json').unlink()
            self.assertFalse(replay(None,folder,[],{},protocol)['complete'])


class ThresholdTests(unittest.TestCase):
    def test_deadline_interrupts_and_restores_signal_handler(self):
        import signal,time
        from sar_groundlm.diagnosis.replay import deadline
        before=signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            with deadline(.01):time.sleep(.1)
        self.assertEqual(signal.getsignal(signal.SIGALRM),before)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0)

    def test_ties_empty_predictions_and_ignore(self):
        p=np.array([[.5,.5],[.1,.99]]);t=np.array([[1,0],[0,1]],bool);v=np.array([[1,1],[1,0]],bool)
        points=np.array([0,.1,.5,.50001,1]);iou,fp=threshold_curve(p,t,v,points)
        for j,threshold in enumerate(points):
            exact=pixel_counts(p>=threshold,t,v)
            self.assertAlmostEqual(iou[j],exact['iou'] or 0)
            self.assertAlmostEqual(fp[j],exact['false_fraction'])
        self.assertEqual(iou.max(),.5)

    def test_shared_rule_vs_per_case_oracle(self):
        v=np.ones((1,2),bool);t=np.array([[1,0]],bool)
        cases=[{'case':'positive','probability':np.array([[.8,.2]]),'target':t,'valid':v},
               {'case':'absent','probability':np.array([[.1,.1]]),'target':np.zeros_like(t),'valid':v}]
        summary,_=threshold_analysis(cases);self.assertTrue(summary['shared_threshold_exists'])
        cases[1]['probability'][:]=.9
        summary,_=threshold_analysis(cases);self.assertFalse(summary['shared_threshold_exists'])
        self.assertEqual(summary['verdict'],'per_image_thresholds_help_but_no_shared_rule')
        cases[0]['probability'][:]=.5
        summary,_=threshold_analysis(cases);self.assertEqual(summary['verdict'],'threshold_only_explanation_not_supported')


if __name__=='__main__':unittest.main()
