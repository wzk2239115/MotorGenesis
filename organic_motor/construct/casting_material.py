"""Measured composite-material and casting-shrinkage contracts.

No Fe3O4/resin property is inferred from the existing iron material card.
Geometry can be generated with nominal shrinkage; simulation remains gated.
"""
from __future__ import annotations
import numpy as np


def material_template():
    return {
        'name': 'resin_Fe3O4_user_batch', 'batch_id': None, 'resin_product': None,
        'powder_mass_fraction': None, 'cure_schedule': None,
        'measurement_source': None, 'measurement_date': None,
        'density_kg_m3': None, 'thermal_conductivity_W_mK': None,
        'heat_capacity_J_kgK': None, 'electrical_conductivity_S_m': None,
        'max_temperature_C': None,
        'bh_curve': [],  # [{H_A_m, B_T}] measured initial magnetization curve
        'loss_table': [],  # [{frequency_Hz, B_peak_T, temperature_C, loss_W_kg}]
        'shrinkage': {'measured': False, 'linear_xyz': [0.0,0.0,0.0],
                      'coupon_mold_mm': None, 'coupon_cast_mm': None},
    }


def shrinkage_from_coupon(mold_mm, cast_mm):
    mold=np.asarray(mold_mm,dtype=float); cast=np.asarray(cast_mm,dtype=float)
    if mold.shape!=(3,) or cast.shape!=(3,) or not np.all(np.isfinite([mold,cast])) or np.any(mold<=0) or np.any(cast<=0):
        raise ValueError('需要三轴实际模腔尺寸与固化后尺寸，均为正数 mm')
    s=1-cast/mold
    if np.any(s<0) or np.any(s>=.1):
        raise ValueError('收缩结果超出当前模型范围 [0, 10%)；检查测量或改用其他固化模型')
    return s.tolist()


def compensated_dimensions(target_mm, shrink_xyz):
    t=np.asarray(target_mm,dtype=float);s=np.asarray(shrink_xyz,dtype=float)
    if t.shape!=(3,) or s.shape!=(3,) or not np.all(np.isfinite([t,s])) or np.any(t<=0) or np.any(s<0) or np.any(s>=.1):
        raise ValueError('无效目标尺寸或线性收缩率')
    return (t/(1-s)).tolist()


def validate_material(card):
    missing=[]
    for k in ('batch_id','resin_product','cure_schedule','measurement_source','measurement_date'):
        if not card.get(k):missing.append(k)
    for k in ('density_kg_m3','thermal_conductivity_W_mK','heat_capacity_J_kgK','electrical_conductivity_S_m','max_temperature_C'):
        v=card.get(k)
        if not isinstance(v,(int,float)) or not np.isfinite(v) or (v<=0 if k!='electrical_conductivity_S_m' else v<0):missing.append(k)
    f=card.get('powder_mass_fraction')
    if not isinstance(f,(int,float)) or not 0<f<1:missing.append('powder_mass_fraction')
    try:
        bh=np.array([[p['H_A_m'],p['B_T']] for p in card['bh_curve']],float)
        ok=len(bh)>=5 and bh.shape[1]==2 and np.isfinite(bh).all() and (bh>=0).all() and (np.diff(bh[:,0])>0).all() and (np.diff(bh[:,1])>=0).all()
        if not ok:missing.append('bh_curve (>=5 monotonic measured points)')
    except (KeyError,TypeError,ValueError,IndexError):missing.append('bh_curve')
    try:
        losses=np.array([[p[k] for k in ('frequency_Hz','B_peak_T','temperature_C','loss_W_kg')] for p in card['loss_table']],float)
        if len(losses)<1 or not np.isfinite(losses).all() or not (losses[:,[0,1]]>0).all() or not (losses[:,3]>=0).all():missing.append('loss_table')
    except (KeyError,TypeError,ValueError,IndexError):missing.append('loss_table')
    shrink=card.get('shrinkage',{})
    try:
        compensated_dimensions([1,1,1],shrink.get('linear_xyz',[]))
        if not shrink.get('measured'):missing.append('measured shrinkage')
        else:
            measured=shrinkage_from_coupon(shrink['coupon_mold_mm'],shrink['coupon_cast_mm'])
            if not np.allclose(measured,shrink['linear_xyz'],atol=1e-5):missing.append('shrinkage inconsistent with coupon')
    except (ValueError,KeyError,TypeError):missing.append('shrinkage measurements')
    return {'material_data_complete':not missing,'missing':missing,
            'simulation_ready':False,
            'reason':'新分段拓扑尚需重新网格化、接入实测材料并验证；数据完整不等于求解器已校准'}
