"""覆盖真实迁移风险，而不是只检查函数能否被调用。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import rasterio
import torch
from sar_groundlm.data.readers import sovs_reference,read_sar,read_label
from sar_groundlm.engine.losses import segmentation_loss,structure_targets
from sar_groundlm.modeling.heads import RegionDecoder
from sar_groundlm.modeling.modules.regions import RegionModes
from sar_groundlm.modeling.modules.structure import SpatialStructure


def write_tif(path,a,nodata=None):
    if a.ndim==2:a=a[None]
    with rasterio.open(path,'w',driver='GTiff',height=a.shape[1],width=a.shape[2],count=len(a),dtype=a.dtype,nodata=nodata) as d:d.write(a)


class InputTests(unittest.TestCase):
    def test_sovs_formula_and_lower_bound_are_distinct(self):
        a=np.array([[[-50.,0.]], [[-40.,-7.5]]],dtype=np.float32)
        ref=sovs_reference(a)
        np.testing.assert_allclose(ref[0,0],[-1,-.3,-.65],atol=1e-6)
        np.testing.assert_allclose(ref[0,1],[1,1,1])
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.tif';write_tif(p,a)
            bounded=read_sar(p)
            np.testing.assert_equal(bounded.rgb[0,0],[0,0,0])
            self.assertEqual(bounded.metadata['lower_clipped_fraction_by_band'],[.5,.5])
            np.testing.assert_array_equal(read_sar(p,'sovs_reference').rgb,ref)

    def test_signed_byte_and_extra_channels_rejected(self):
        for a in (np.ones((2,2),np.int8),np.ones((4,2,2),np.float32)):
            with self.assertRaises(ValueError):sovs_reference(a)

    def test_float_precision_validity_and_labels(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.tif';q=Path(td)/'label.tif'
            a=np.array([[[-20.001,-20.002],[-9999,np.nan]], [[-20,-20],[-20,-20]]],np.float32)
            write_tif(p,a,-9999);im=read_sar(p)
            self.assertNotEqual(im.rgb[0,0,0],im.rgb[0,1,0])
            self.assertEqual(im.valid.tolist(),[[True,True],[False,False]])
            write_tif(q,np.zeros((2,2),np.uint8));label,valid=read_label(q,im)
            self.assertTrue((label[1]==255).all());self.assertFalse(valid[1].any())
            write_tif(q,np.zeros((3,2),np.uint8))
            with self.assertRaises(ValueError):read_label(q,im)

    def test_all_invalid_and_unknown_label_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.tif';q=Path(td)/'y.tif'
            write_tif(p,np.full((2,2),np.nan,np.float32))
            with self.assertRaises(ValueError):read_sar(p)
            write_tif(p,np.zeros((2,2),np.uint8));write_tif(q,np.full((2,2),12,np.uint8))
            with self.assertRaises(ValueError):read_label(q,read_sar(p))


class LearningContractTests(unittest.TestCase):
    def test_tiny_foreground_has_strong_correcting_gradient(self):
        t=np.zeros((64,64),bool);t[10,10]=True;v=np.ones_like(t)
        logits=torch.full((64,64),-3.,requires_grad=True)
        loss,_=segmentation_loss(logits,t,v);loss.backward()
        self.assertLess(float(logits.grad[10,10]),-.1)
        self.assertGreater(float(logits.grad[0,0]),0)

    def test_ignore_and_unknown_holes(self):
        t=np.ones((32,32),bool);t[12:20,12:20]=False
        v=np.ones_like(t);v[12:16,12:16]=False
        self.assertFalse(structure_targets(t,v)['holes'].any())
        self.assertTrue(structure_targets(t,np.ones_like(t))['holes'].any())
        p=torch.zeros((32,32),requires_grad=True);loss,_=segmentation_loss(p,t,v,structure=True,water=True);loss.backward()
        self.assertEqual(float(p.grad[~torch.from_numpy(v)].abs().sum()),0)

    def test_all_four_detach_paths(self):
        head=RegionDecoder(16,20,dim=16)
        f=torch.randn(64,16,requires_grad=True);q=torch.randn(20,requires_grad=True)
        modes=torch.randn(4,16,requires_grad=True);detail=torch.randn(1,64,32,32,requires_grad=True)
        result,_=head(f.detach(),q.detach(),modes.detach(),[1,8,8],2,(32,32),detail.detach())
        result.mean().backward()
        for tensor in (f,q,modes,detail):self.assertIsNone(tensor.grad)
        self.assertGreater(float(head.query[0].weight.grad.abs().sum()),0)

    def test_shared_writes_start_as_identity_and_modes_differ(self):
        torch.manual_seed(42);f=torch.randn(64,16);q=torch.randn(20)
        sem=RegionModes(16,20,16);out,modes=sem(f,q,4)
        self.assertTrue(torch.equal(out,f));self.assertGreater(float(modes.std(0).mean()),0)
        self.assertEqual(sem.last['assignments'].shape,(64,5))
        spatial=SpatialStructure(16,16);out,detail=spatial([f,f,f],[1,8,8],2)
        self.assertTrue(torch.equal(out,f));self.assertEqual(detail.shape[-2:],(32,32))


if __name__=='__main__':unittest.main()
