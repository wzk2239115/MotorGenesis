"""Bounded manufacturing-parameter search using existing thermal3d physics.

This is a transparent screening model, NOT a calibrated motor simulation.
Copper loss is computed from actual conductor length, area and equal current.
Subgrid heat/conductivity mapping is explicitly approximate.
"""
import json
from dataclasses import replace, asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from organic_motor.construct.wound_prototype import WoundSpec,winding_route,core_field,sleeve_field,grid


def copper_metrics(spec,current_A=3.0):
    from organic_motor.config import MotorConfig
    route=winding_route(spec)
    length=np.linalg.norm(np.diff(route,axis=0),axis=1).sum()*.001+.160
    sigma=MotorConfig().sigma_copper
    area=np.pi*(spec.wire_diameter_mm*.001/2)**2
    resistance=length/(sigma*area)
    return dict(total_wire_length_m=float(length),resistance_ohm=float(resistance),
                copper_loss_W=float(current_A**2*resistance),current_A=current_A,sigma_S_m=sigma)


def thermal_screen(spec,step_mm):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from organic_motor.physics.thermal3d import solve_temperature,thermal_relative_residual
    xyz,origin,spacing=grid(((27,52),(-10,10),(-29,29)),step_mm)
    route=winding_route(spec);d=cKDTree(route).query(np.column_stack([v.ravel() for v in xyz]))[0].reshape(xyz[0].shape)
    # Finite-volume averaged wire surrogate, not binary sub-voxel thresholding.
    width=max(step_mm*.65,.335);weights=np.exp(-.5*(d/width)**2)
    dv=float(np.prod(spacing*.001));coil_length=np.linalg.norm(np.diff(route,axis=0),axis=1).sum()*.001
    copper_volume=coil_length*np.pi*(spec.wire_diameter_mm*.001/2)**2
    fraction=weights*copper_volume/(weights.sum()*dv)
    if fraction.max()>1:raise ValueError('Wire homogenization exceeds cell volume')
    k=np.full(d.shape,.026)
    k[core_field(*xyz,spec)<=0]=1.0 # ASSUMED composite, not measured
    k[sleeve_field(*xyz,spec,route)<=0]=.2 # ASSUMED polymer
    k=k*(1-fraction)+385*fraction
    metric=copper_metrics(spec)
    # Only in-model conductor is heated; 160 mm external leads are outside box.
    power=metric['copper_loss_W']*coil_length/metric['total_wire_length_m']
    q=weights*power/(weights.sum()*dv)
    cfg=SimpleNamespace(Nx=d.shape[0],Ny=d.shape[1],Nz=d.shape[2],spacing=tuple(spacing*.001),ambient_temperature=25.,thermal_maxiter=2500,thermal_tol=1e-9)
    temp=solve_temperature(jnp.array(q),jnp.array(k),cfg)
    residual=thermal_relative_residual(temp,jnp.array(q),jnp.array(k),cfg)
    return dict(step_mm=step_mm,shape=list(d.shape),peak_temperature_C=float(jnp.max(temp)),
                relative_residual=float(residual),heat_input_W=power,integrated_heat_W=float(q.sum()*dv),
                integrated_copper_m3=float(fraction.sum()*dv),conductor_copper_m3=copper_volume)


def run():
    rows=[]
    for radius in [1.2,2.,3.,4.,5.]:
        spec=replace(WoundSpec(),end_radius_mm=radius);spec.validate()
        row=dict(end_radius_mm=radius,**copper_metrics(spec))
        rows.append(row)
    best=min(rows,key=lambda row:row['copper_loss_W'])
    thermal=[]
    for step in [1.2,.8]:
        for radius in [1.2,best['end_radius_mm']]:
            result=thermal_screen(replace(WoundSpec(),end_radius_mm=radius),step)
            thermal.append(dict(end_radius_mm=radius,**result));print(json.dumps(thermal[-1]),flush=True)
    report=dict(search='bounded end-radius sweep, equal turns/wire/current; no aesthetics reward in loss objective',
        candidates=rows,selected=best,selected_spec=asdict(replace(WoundSpec(),end_radius_mm=best['end_radius_mm'])),
        generator_sha256=hashlib.sha256(Path(__import__('organic_motor.construct.wound_prototype',fromlist=['__file__']).__file__).read_bytes()).hexdigest(),
        copper_loss_reduction_percent=100*(1-best['copper_loss_W']/rows[0]['copper_loss_W']),thermal=thermal,
        assumptions=dict(composite_k_W_mK=1.,polymer_k_W_mK=.2,air_k_W_mK=.026,copper_k_W_mK=385.,boundary='all six box faces fixed at 25 C',current_A=3.,wire_mapping='volume-conserving Gaussian subgrid surrogate; isotropic conductivity approximation'),
        performance_advantage_verified=False,
        limitations=['Same-current copper-loss comparison is not same-torque efficiency comparison','No new rotor, calibrated B-H data, iron loss or torque constraint','Thermal model is an uncalibrated isolated-sector screening model; no airflow or contact resistance','Must verify mesh convergence and real material data before accepting thermal superiority'])
    folder=Path('organic_motor/reports/fabrication');folder.mkdir(exist_ok=True,parents=True)
    (folder/'shape_screen.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--publish',type=Path)
    args=parser.parse_args();result=run()
    if args.publish:
        manifest=json.loads((args.publish/'manifest.json').read_text())
        if manifest['generator_sha256']!=result['generator_sha256'] or manifest['spec']!=result['selected_spec']:
            raise ValueError('Screening and geometry versions differ; result not published')
        (args.publish/'shape-screen.json').write_text(json.dumps(result,indent=2))
