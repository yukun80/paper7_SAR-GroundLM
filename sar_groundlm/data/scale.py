"""只读提取地面像素间距；它不等于雷达实际空间分辨能力。"""
import numpy as np
import rasterio
from affine import Affine


def geometry(shape, transform, crs=None, unit=None, metres_per_unit=None, aligned=True):
    """保留二维像素基向量，支持旋转、非方形像元与面积换算。"""
    h,w=map(int,shape)
    if h<=0 or w<=0:raise ValueError('图像尺寸必须为正')
    t=Affine(*tuple(transform)[:6])
    result={'shape_hw':[h,w],'transform':list(t)[:6],'crs':crs,'source_unit':unit,
            'status':'unknown','reason':None,'sensor_resolution_m':None,
            'source':'GeoTIFF CRS and affine transform; no sensor-resolution inference'}
    if not aligned:result['reason']='image_label_georeferencing_not_confirmed';return result
    if (not crs or unit is None or str(unit).lower() in ('unknown','undefined','degree','degrees')
            or metres_per_unit is None or not np.isfinite(metres_per_unit) or metres_per_unit<=0):
        result['reason']='missing_projected_crs_or_known_linear_unit';return result
    basis=np.array([[t.a,t.b],[t.d,t.e]],float)*metres_per_unit
    det=float(abs(basis[0,0]*basis[1,1]-basis[0,1]*basis[1,0]))
    if not np.isfinite(basis).all() or not np.isfinite([t.c,t.f]).all() or det<=1e-12:
        result['reason']='invalid_or_degenerate_transform';return result
    sx,sy=np.linalg.norm(basis,axis=0)
    corners=[[(t*x)[0]*metres_per_unit,(t*x)[1]*metres_per_unit] for x in [(0,0),(w,0),(w,h),(0,h)]]
    result.update(status='known_pixel_spacing',reason=None,metres_per_unit=metres_per_unit,
                  pixel_basis_m=basis.tolist(),spacing_xy_m=[float(sx),float(sy)],pixel_area_m2=det,
                  footprint_area_m2=det*h*w,footprint_corners_m=corners,
                  orthogonal=bool(abs(np.dot(basis[:,0],basis[:,1]))<=1e-8*sx*sy))
    return result


def read_scale(image_path,label_path=None):
    """CRS 和配准信息缺失时只报告未知，不把默认单位矩阵认作一米。"""
    with rasterio.open(image_path) as im:
        crs=im.crs;factor=unit=None
        if crs is not None and crs.is_projected:
            try:unit,factor=crs.linear_units_factor
            except (ValueError,AttributeError):pass
        alignment='not_applicable';aligned=True
        if label_path is not None:
            with rasterio.open(label_path) as label:
                if im.shape!=label.shape:raise ValueError('图像与标签尺寸不一致')
                if crs is None:
                    alignment='source_pair_and_dimensions_only'
                elif label.crs is None:
                    alignment='label_georeferencing_missing';aligned=False
                elif crs!=label.crs or not np.allclose(tuple(im.transform),tuple(label.transform),rtol=0,atol=1e-8):
                    raise ValueError('图像和标签地理位置冲突')
                else:alignment='matching_crs_and_transform'
        result=geometry(im.shape,im.transform,crs.to_string() if crs else None,unit,factor,aligned)
        result['alignment']=alignment;return result


def processed_geometry(source,input_hw,grid_hw=None,merge=2):
    """保持同一地面范围，计算实际缩放后及视觉网格的尺度；不声称增加细节。"""
    h,w=source['shape_hw'];ih,iw=map(int,input_hw)
    if min(ih,iw)<=0:raise ValueError('处理后尺寸必须为正')
    out={'input_hw':[ih,iw],'source_pixels_per_input_xy':[w/iw,h/ih],
         'status':source['status'],'real_detail_increased':False}
    if source['status']!='known_pixel_spacing':return out
    scaled=Affine(*source['transform'])*Affine.scale(w/iw,h/ih)
    converted=geometry((ih,iw),scaled,source['crs'],source['source_unit'],source['metres_per_unit'])
    out.update(pixel_basis_m=converted['pixel_basis_m'],spacing_xy_m=converted['spacing_xy_m'],
               pixel_area_m2=converted['pixel_area_m2'],footprint_area_m2=converted['footprint_area_m2'],
               footprint_corners_m=converted['footprint_corners_m'])
    if grid_hw is not None:
        gh,gw=map(int,grid_hw)
        if min(gh,gw)<=0:raise ValueError('视觉网格尺寸必须为正')
        cell=np.asarray(source['pixel_basis_m'])@np.diag([w/gw,h/gh])
        out.update(grid_hw=[gh,gw],visual_cell_basis_m=cell.tolist(),
                   merged_cell_basis_m=(cell*merge).tolist())
    return out
