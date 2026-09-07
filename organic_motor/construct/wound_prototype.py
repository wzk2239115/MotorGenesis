"""Manufacturing-first radial motor candidate, generated from constraints.

All construction coordinates are millimetres. GLB is converted to metres.
This is a NEW segmented candidate, never passed off as the old solved motor.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh

from organic_motor.construct.casting_material import material_template, validate_material, shrinkage_from_coupon


@dataclass(frozen=True)
class WoundSpec:
    housing_arc_radius_mm: float = 36.0
    housing_mid_radius_mm: float = 62.0
    end_radius_mm: float = 5.0
    segments: int = 12
    turns: int = 8
    wire_diameter_mm: float = .6  # bare conductor
    enamel_radial_mm: float = .035
    turn_pitch_mm: float = 1.2
    clearance_mm: float = .3
    draft_deg: float = 1.0
    geometry_step_mm: float = .45
    gear_module_mm: float = 1.5
    sun_teeth: int = 18
    planet_teeth: int = 18
    planets: int = 3
    gear_backlash_mm: float = .16
    shaft_radius_mm: float = 8.0

    @property
    def ring_teeth(self):return self.sun_teeth+2*self.planet_teeth
    @property
    def ratio(self):return 1+self.ring_teeth/self.sun_teeth

    def validate(self):
        if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in asdict(self).values()):raise ValueError('参数必须为有限数值')
        if self.housing_arc_radius_mm<36 or not 60<=self.housing_mid_radius_mm<=62:raise ValueError('壳体圆弧参数超出当前装配范围')
        if housing_outer_radius(24.5,self)<52:raise ValueError('圆弧端部壁厚不足 2 mm')
        if not 1.2<=self.end_radius_mm<=5:raise ValueError('端部曲率半径须在 1.2—5 mm')
        if self.segments!=12:raise ValueError('当前磁芯接口按 12 段定义；其他段数需重新计算磁轭与法兰')
        if not isinstance(self.turns,int) or not 1<=self.turns<=8:raise ValueError('当前绕线窗口最多支持 8 匝')
        diameter=self.wire_diameter_mm+2*self.enamel_radial_mm
        if not .1<self.wire_diameter_mm<=.9 or self.enamel_radial_mm<0:raise ValueError('线径超出当前骨架范围')
        if self.turn_pitch_mm<diameter+.5 or self.turns*self.turn_pitch_mm>9.6+1e-9:raise ValueError('匝间间距或绕线窗口不足')
        if not .2<=self.clearance_mm<=.6 or not .5<=self.draft_deg<=2:raise ValueError('检查配合间隙及拔模角')
        if not .2<=self.geometry_step_mm<=.6:raise ValueError('几何步长须为 0.2—0.6 mm')
        if self.sun_teeth<18 or self.planet_teeth<18 or self.planets<2:raise ValueError('当前无变位齿形要求至少 18 齿')
        if (self.sun_teeth+self.ring_teeth)%self.planets:raise ValueError('行星轮等间距装配条件不成立')
        if abs(self.gear_module_mm-1.5)>1e-9 or self.sun_teeth!=18 or self.planet_teeth!=18 or self.planets!=3:
            raise ValueError('当前法兰与轴接口验证限定 m1.5 / 18-18-54 / 三行星')
        if not 0<=self.gear_backlash_mm<.4:raise ValueError('齿侧间隙超出允许范围')
        if self.shaft_radius_mm!=8:raise ValueError('当前太阳轮连接口为 16 mm 轴，改变轴径需重设计')


def grid(bounds, step):
    axes=[np.linspace(a,b,int(np.ceil((b-a)/step))+1) for a,b in bounds]
    return np.meshgrid(*axes,indexing='ij'),np.array([a[0] for a in axes]),np.array([a[1]-a[0] for a in axes])


def mesh_field(field, origin, spacing):
    vertices,faces,_,_=marching_cubes(np.asarray(field,np.float32),0,spacing=spacing,allow_degenerate=False)
    m=trimesh.Trimesh(vertices=vertices+origin,faces=faces,process=True)
    if not m.is_watertight:trimesh.repair.fill_holes(m)
    m.fix_normals()
    if not m.is_watertight:raise ValueError('生成的零件网格未闭合，拒绝导出')
    return m


def box(x,y,z,center,half):
    return np.maximum.reduce([np.abs(x-center[0])-half[0],np.abs(y-center[1])-half[1],np.abs(z-center[2])-half[2]])


def rounded_profile(y,z,half_y,half_z,corner):
    qy=np.abs(y)-(half_y-corner);qz=np.abs(z)-(half_z-corner)
    return np.hypot(np.maximum(qy,0),np.maximum(qz,0))+np.minimum(np.maximum(qy,qz),0)-corner


def core_field(x,y,z,spec):
    r=np.hypot(x,y);a=math.pi/spec.segments-.004
    wedge=np.maximum(np.abs(y)*math.cos(a)-x*math.sin(a),np.maximum(44-r,r-49))
    draft=np.abs(z)*math.tan(math.radians(spec.draft_deg))
    wedge=np.maximum(wedge+draft,np.abs(z)-20)
    tooth=np.maximum(np.abs(x-37.5)-7,rounded_profile(y,z,3,20,max(0,spec.end_radius_mm-2)))
    return np.minimum(wedge,tooth+draft)


def rounded_rectangle(theta, half_y=5.0, half_z=22.0, corner=1.2):
    # Radial intersections with a rounded rectangular perimeter, sampled by
    # arc length below so narrow sides are not undersampled.
    pts=[]
    centers=[(half_y-corner,half_z-corner),(-half_y+corner,half_z-corner),(-half_y+corner,-half_z+corner),(half_y-corner,-half_z+corner)]
    for i,(y,z) in enumerate(centers):
        t=np.linspace(i*np.pi/2,(i+1)*np.pi/2,40,endpoint=False)
        pts.extend(np.column_stack([y+corner*np.cos(t),z+corner*np.sin(t)]))
    pts=np.array(pts);pts=np.vstack([pts,pts[0]])
    lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
    q=np.mod(theta,2*np.pi)/(2*np.pi)*lengths[-1]
    return np.column_stack([np.interp(q,lengths,pts[:,0]),np.interp(q,lengths,pts[:,1])])


def winding_route(spec):
    t=np.linspace(0,2*np.pi*spec.turns,spec.turns*640+1)
    yz=rounded_rectangle(t,corner=spec.end_radius_mm)
    main=np.column_stack([32.5+spec.turn_pitch_mm*t/(2*np.pi),yz])
    # Two explicit lead exits reach beyond the flange, so the last turn is
    # not trapped in a blind groove. Extra external lead allowance is separate.
    a=np.linspace(-np.pi/2,0,40,endpoint=False)
    start=np.column_stack([main[0,0]-2+2*np.cos(a),np.full(len(a),main[0,1]),main[0,2]+2*np.sin(a)])
    end=np.linspace(main[-1],main[-1]+[0,0,6.0],41)[1:]
    return np.vstack([start,main,end])


def sleeve_field(x,y,z,spec,route=None):
    rounded=rounded_profile(y,z,5,22,spec.end_radius_mm)
    outer=np.maximum(rounded,np.abs(x-37.05)-5.75)
    flange_profile=rounded_profile(y,z,7.2,24.2,spec.end_radius_mm+2.2)
    flange=np.maximum(flange_profile,np.minimum(np.abs(x-31.7)-.4,np.abs(x-42.4)-.4))
    hole=rounded_profile(y,z,3+spec.clearance_mm,20+spec.clearance_mm,max(0,spec.end_radius_mm-2+spec.clearance_mm))
    body=np.maximum(np.minimum(outer,flange),-hole)
    if route is not None:
        d=cKDTree(route).query(np.column_stack([x.ravel(),y.ravel(),z.ravel()]))[0].reshape(x.shape)
        body=np.maximum(body,-(d-(spec.wire_diameter_mm/2+spec.enamel_radial_mm+.05)))
    return body


def sweep_tube(points, radius, inner_radius=0, sides=16):
    """Continuous printable wire or open-lumen tube, independent of voxel size."""
    pts=np.asarray(points,float); tangent=np.gradient(pts,axis=0)
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    u=np.empty_like(tangent)
    ref=np.eye(3)[np.argmin(np.abs(tangent[0]))]
    u[0]=ref-tangent[0]*np.dot(ref,tangent[0]);u[0]/=np.linalg.norm(u[0])
    for i in range(1,len(pts)):
        u[i]=u[i-1]-tangent[i]*np.dot(u[i-1],tangent[i])
        u[i]/=np.linalg.norm(u[i])
    v=np.cross(tangent,u);a=np.arange(sides)*2*np.pi/sides
    normals=u[:,None,:]*np.cos(a)[None,:,None]+v[:,None,:]*np.sin(a)[None,:,None]
    vertices=[(pts[:,None,:]+radius*normals).reshape(-1,3)];faces=[];n=len(pts)*sides
    for offset in ([0,n] if inner_radius else [0]):
        for i in range(len(pts)-1):
            for j in range(sides):
                a0=offset+i*sides+j;b=offset+i*sides+(j+1)%sides;c=b+sides;d=a0+sides
                faces.extend([[a0,b,c],[a0,c,d]])
    if inner_radius:
        vertices.append((pts[:,None,:]+inner_radius*normals).reshape(-1,3))
        for base in (0,n-sides):
            for j in range(sides):
                a0=base+j;b=base+(j+1)%sides
                faces.extend([[a0,a0+n,b+n],[a0,b+n,b]])
    else:
        vertices.append(pts[[0,-1]])
        for j in range(sides):
            faces.extend([[n,j,(j+1)%sides],[n+1,n-sides+j,n-sides+(j+1)%sides]])
    m=trimesh.Trimesh(vertices=np.vstack(vertices),faces=faces,process=True);m.fix_normals()
    if not m.is_watertight:raise ValueError('扫掠管不闭合')
    return m


def annular_profile(inner,outer,height):
    """Closed annulus with sampled radial boundaries, no triangulator dependency."""
    n=len(outer);t=np.arange(n)*2*np.pi/n
    rings=[np.column_stack([r*np.cos(t),r*np.sin(t),np.full(n,z)]) for z in (-height/2,height/2) for r in (inner,outer)]
    faces=[]
    for i in range(n):
        j=(i+1)%n
        for a,b,c,d in ((i,j,n+j,n+i),(2*n+i,3*n+i,3*n+j,2*n+j),(i,2*n+i,2*n+j,j),(n+i,n+j,3*n+j,3*n+i)):
            faces.extend([[a,b,c],[a,c,d]])
    m=trimesh.Trimesh(vertices=np.vstack(rings),faces=faces,process=True);m.fix_normals()
    assert m.is_watertight
    return m


def extrude_loops(loops, height):
    # Earcut preserves the sampled tooth boundary and bolt-hole loops. Unlike
    # voxelizing a gear, this does not discard the 0.16 mm backlash.
    import mapbox_earcut
    ends=np.cumsum([len(a) for a in loops]).astype(np.uint32)
    xy=np.vstack(loops).astype(np.float64);n=len(xy)
    tris=mapbox_earcut.triangulate_float64(xy,ends).reshape(-1,3)
    vertices=np.vstack([np.column_stack([xy,np.full(n,z)]) for z in [-height/2,height/2]])
    faces=[*tris.tolist(),*(tris+n).tolist()];start=0
    for end in ends:
        for i in range(start,int(end)):
            j=start if i+1==end else i+1
            faces.extend([[i,j,j+n],[i,j+n,i+n]])
        start=int(end)
    m=trimesh.Trimesh(vertices=vertices,faces=faces,process=True);m.fix_normals()
    if not m.is_watertight:raise ValueError('带孔齿圈网格未闭合')
    return m


def gear_profile(teeth,module,backlash,internal=False,n=None):
    n=n or teeth*64;t=np.arange(n)*2*np.pi/n
    rp=module*teeth/2;rb=rp*math.cos(math.radians(20))
    invp=math.tan(math.radians(20))-math.radians(20)
    d=np.abs((t+np.pi/teeth)%(2*np.pi/teeth)-np.pi/teeth)
    if not internal:
        radii=np.linspace(max(rb,rp-1.25*module),rp+module,240)
        alpha=np.arccos(np.clip(rb/radii,-1,1))
        widths=np.pi/(2*teeth)+invp-(np.tan(alpha)-alpha)-backlash/(2*rp)
        return np.interp(d,widths[::-1],radii[::-1],left=rp+module,right=rp-1.25*module)
    radii=np.linspace(max(rb,rp-module),rp+1.25*module,240)
    alpha=np.arccos(np.clip(rb/radii,-1,1))
    widths=np.pi/(2*teeth)-invp+(np.tan(alpha)-alpha)-backlash/(2*rp)
    return np.interp(d,widths,radii,left=radii[0],right=radii[-1])


def d_profile(radius, flat_x, n=192):
    a=np.arange(n)*2*np.pi/n
    return np.minimum(radius,flat_x/np.maximum(np.cos(a),1e-9))


def stepped_shaft():
    n=192;a=np.arange(n)*2*np.pi/n
    radii=[np.full(n,8.),np.full(n,8.),d_profile(8,6.5,n),d_profile(8,6.5,n)]
    levels=[-33,42.9,43,53]
    verts=np.vstack([np.column_stack([r*np.cos(a),r*np.sin(a),np.full(n,z)]) for r,z in zip(radii,levels)]+[np.array([[0,0,levels[0]],[0,0,levels[-1]]])])
    faces=[]
    for k in range(3):
        for i in range(n):
            j=(i+1)%n;faces.extend([[k*n+i,k*n+j,(k+1)*n+j],[k*n+i,(k+1)*n+j,(k+1)*n+i]])
    for i in range(n):
        j=(i+1)%n;faces.extend([[4*n,j,i],[4*n+1,3*n+i,3*n+j]])
    m=trimesh.Trimesh(vertices=verts,faces=faces,process=True);m.fix_normals();return m


def rotation(a):
    t=np.eye(4);t[:2,:2]=[[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]];return t


def kinematics(input_angle, spec):
    carrier=input_angle/spec.ratio
    planet=carrier-(spec.sun_teeth/spec.planet_teeth)*(input_angle-carrier)
    return {'sun':input_angle,'carrier':carrier,'planet_spin':planet,'ring':0.0}


def segment_molds(spec, shrink_xyz):
    shrink=np.asarray(shrink_xyz,float)
    if shrink.shape!=(3,) or not np.isfinite(shrink).all() or np.any(shrink<0) or np.any(shrink>=.1):raise ValueError('无效收缩率')
    if np.any(shrink>.03):raise ValueError('当前模具外框支持至多 3% 收缩补偿；更大收缩需重设计夹紧孔位置')
    xyz,origin,spacing=grid(((23,58),(-21,21),(-28,28)),spec.geometry_step_mm)
    x,y,z=xyz
    cavity=core_field(x*(1-shrink[0]),y*(1-shrink[1]),z*(1-shrink[2]),spec)-.1
    # Feed into the thick yoke; vent at tooth tip. Both reach tool exterior.
    for xx,rad in ((46.5,2.0),(31.5,.65)):
        cavity=np.minimum(cavity,np.maximum(np.hypot(x-xx,y)-rad,18-z))
    for xx in (27,54):
        for yy in (-16,16):cavity=np.minimum(cavity,np.hypot(x-xx,y-yy)-1.8)
    outer=box(x,y,z,(40.5,0,0),(16.5,20,27))
    result={}
    for name,plane in (('segment_mold_lower',z),('segment_mold_upper',-z)):
        field=np.maximum(np.maximum(outer,-cavity),plane)
        result[name]=mesh_field(field,origin,spacing)
    return result


def housing_outer_radius(z,spec):
    """Exact circular arc in the r-z plane; middle bulges out visibly."""
    return spec.housing_mid_radius_mm-spec.housing_arc_radius_mm+np.sqrt(np.maximum(spec.housing_arc_radius_mm**2-np.asarray(z)**2,0))


def build(spec=None,card=None):
    spec=spec or WoundSpec();spec.validate();card=card or material_template()
    assets={};instances=[];route=winding_route(spec)
    xyz,o,h=grid(((28,52),(-16,16),(-25,25)),spec.geometry_step_mm)
    assets['core_segment']=mesh_field(core_field(*xyz,spec),o,h)
    bxyz,bo,bh=grid(((28,52),(-16,16),(-25,25)),.2)
    assets['grooved_bobbin']=mesh_field(sleeve_field(*bxyz,spec,route),bo,bh)
    from organic_motor.construct.winding_harness import make_harness, verify_harness
    from organic_motor.construct.manufacturability import run_manufacturability_checks
    phase_paths,harness,ports=make_harness(route,spec.turns)
    harness['verification']=verify_harness(phase_paths,harness,spec.wire_diameter_mm/2+spec.enamel_radial_mm)
    if not harness['verification']['passed']:raise ValueError(harness['verification'])
    harness['manufacturability']=run_manufacturability_checks(phase_paths,harness,spec)
    for phase,path in phase_paths.items():
        assets['phase_'+phase]=sweep_tube(path,spec.wire_diameter_mm/2+spec.enamel_radial_mm)
    harness['support_status']='后侧分层引线的固定夹具与端子座尚未落实；不能直接投产'
    harness['phase_routes_mm']={k:v.tolist() for k,v in phase_paths.items()}
    def add(id,asset,label,process,position=(0,0,0),angle=0,group='stator',explode=(0,0,0)):
        transform=rotation(angle);transform[:3,3]=position
        instances.append(dict(id=id,asset=asset,label=label,process=process,group=group,
                              transform=transform.tolist(),explode_mm=list(explode)))
    for i in range(spec.segments):
        a=i*2*np.pi/spec.segments;v=(30*math.cos(a),30*math.sin(a),0)
        add(f'core_{i:02}', 'core_segment',f'磁芯段 {i+1:02}', '分模浇注',angle=a,explode=v)
        add(f'bobbin_{i:02}', 'grooved_bobbin',f'带槽骨架 {i+1:02}', 'FDM 配合样件',angle=a,explode=(v[0]*2,v[1]*2,0))
    for phase in 'UVW':
        add('phase_'+phase,'phase_'+phase,phase+' 相 · 四线圈串联至 N','漆包线与绝缘跨接线；按 S/F 接线表',group='wiring')
    # Housing honeycomb is perforated THROUGH its cylindrical side wall.
    xyz,o,h=grid(((-64,64),(-64,64),(-27,27)),max(.6,spec.geometry_step_mm));x,y,z=xyz;r=np.hypot(x,y)
    shell=np.maximum(np.maximum(50-r,r-housing_outer_radius(z,spec)),np.abs(z)-24.5)
    a=np.arctan2(y,x);u=a*52;period=2*np.pi*52/36
    row=np.round(z/7);v=z-row*7;u=(u-(row.astype(int)%2)*period/2+period/2)%period-period/2
    hexagon=np.maximum.reduce([np.abs(u),np.abs(.5*u+.8660254*v),np.abs(-.5*u+.8660254*v)])-2.6
    holes=np.maximum(hexagon,np.abs(z)-18)
    for i in range(6):
        angle=i*np.pi/3;boss_distance=np.hypot(x-54*np.cos(angle),y-54*np.sin(angle))
        holes=np.maximum(holes,4.8-boss_distance)
        shell=np.minimum(shell,np.maximum(boss_distance-4.2,np.abs(z)-24.5))
    shell=np.maximum(shell,-holes)
    for i in range(6):
        a=i*np.pi/3;shell=np.maximum(shell,1.8-np.hypot(x-54*np.cos(a),y-54*np.sin(a)))
    assets['honeycomb_housing']=mesh_field(shell,o,h)
    add('housing','honeycomb_housing','双侧圆弧蜂窝壳','FDM 结构配合样件',explode=(0,0,-65))
    # Gear teeth generated from involute functions; root fillets/contact
    # strength remain unvalidated, so these are fit/kinematic prototypes.
    for asset,n,bore in (('sun',spec.sun_teeth,8.15),('planet',spec.planet_teeth,3.15)):
        outer=gear_profile(n,spec.gear_module_mm,spec.gear_backlash_mm)
        assets[asset]=annular_profile(d_profile(bore,6.65,len(outer)) if asset=='sun' else np.full(len(outer),bore),outer,10)
    inner=gear_profile(spec.ring_teeth,spec.gear_module_mm,spec.gear_backlash_mm,True)
    assets['ring']=annular_profile(inner,np.full(len(inner),60),10)
    # Integral fixed ring with six tie-bolt passages, cut analytically on an
    # extruded sampled profile. The through bolts ground this to both flanges.
    t=np.arange(len(inner))*2*np.pi/len(inner)
    outerloop=np.column_stack([60*np.cos(t),60*np.sin(t)])
    innerloop=np.column_stack([inner*np.cos(t),inner*np.sin(t)])
    loops=[outerloop,innerloop]
    for i in range(6):
        a=i*np.pi/3;q=np.arange(64)*2*np.pi/64
        loops.append(np.column_stack([54*np.cos(a)+1.8*np.cos(q),54*np.sin(a)+1.8*np.sin(q)]))
    assets['ring']=extrude_loops(loops,10)
    assets['tie_spacer']=annular_profile(np.full(96,1.8),np.full(96,4.2),10)
    for i in range(6):
        a=i*np.pi/3
        add(f'spacer_{i}','tie_spacer','法兰隔柱 · 10 mm','FDM / 加工',position=(54*np.cos(a),54*np.sin(a),38),explode=(0,0,25))
    assets['upper_spacer']=annular_profile(np.full(96,1.8),np.full(96,4.2),9)
    for i in range(6):
        a=i*np.pi/3
        add(f'upper_spacer_{i}','upper_spacer','端盖隔柱 · 9 mm','FDM / 加工',position=(54*np.cos(a),54*np.sin(a),57.5),explode=(0,0,55))
    assets['shaft']=stepped_shaft()
    add('input_shaft','shaft','输入轴 · Ø16 / D 扁位','加工：端部 10 mm 扁位，深 1.5 mm',position=(0,0,0),group='sun')
    add('sun','sun','太阳轮 · 18 齿','渐开线配合样件',position=(0,0,48),group='sun',explode=(0,0,35))
    orbit=spec.gear_module_mm*(spec.sun_teeth+spec.planet_teeth)/2
    for i in range(spec.planets):
        a=2*np.pi*i/spec.planets;phase=np.pi-np.pi/spec.planet_teeth+(spec.sun_teeth+spec.planet_teeth)/spec.planet_teeth*a
        add(f'planet_{i}','planet',f'行星轮 {i+1} · 18 齿','渐开线配合样件',position=(orbit*np.cos(a),orbit*np.sin(a),48),angle=phase,group='planet',explode=(0,0,35))
    add('ring','ring','固定内齿圈 · 54 齿','渐开线配合样件',position=(0,0,48),explode=(0,0,35))
    # Carrier bores share the EXACT planet center radius, with running clearance.
    xyz,o,h=grid(((-38,38),(-38,38),(-4,31)),.45);x,y,z=xyz
    carrier=np.minimum(np.maximum(np.hypot(x,y)-35,np.abs(z)-2),np.maximum(np.hypot(x,y)-8,np.maximum(2-z,z-28)))
    for i in range(3):
        a=2*np.pi*i/3;carrier=np.maximum(carrier,-(np.hypot(x-orbit*np.cos(a),y-orbit*np.sin(a))-3.12))
    assets['carrier']=mesh_field(carrier,o,h)
    add('carrier','carrier','输出行星架','加工 / 配合样件',position=(0,0,57),group='carrier',explode=(0,0,65))
    assets['planet_pin']=trimesh.creation.cylinder(radius=3,height=14,sections=48)
    for i in range(3):
        a=2*np.pi*i/3
        add(f'pin_{i}','planet_pin',f'行星轴销 {i+1} · Ø6','采购 / 加工',position=(orbit*np.cos(a),orbit*np.sin(a),50),group='carrier_pin',explode=(0,0,50))
    # Same 6-hole flange pattern on motor adapter and grounded gear housing.
    for asset,inner,outer in (('adapter',8.3,60),('gear_cover',14.15,60)):
        disk=np.maximum(np.maximum(inner-np.hypot(x,y),np.hypot(x,y)-outer),np.abs(z)-2)
        # The local grid for these caps must include the entire OD.
        gx,go,gh=grid(((-63,63),(-63,63),(-4,4)),.6);xx,yy,zz=gx
        disk=np.maximum(np.maximum(inner-np.hypot(xx,yy),np.hypot(xx,yy)-outer),np.abs(zz)-2)
        for i in range(6):
            a=2*np.pi*i/6
            disk=np.maximum(disk,-(np.hypot(xx-54*np.cos(a),yy-54*np.sin(a))-1.8))
        assets[asset]=mesh_field(disk,go,gh)
    add('adapter','adapter','电机 / 减速器共用法兰','FDM 配合 / 加工',position=(0,0,31),explode=(0,0,20))
    assets['output_bearing']=annular_profile(np.full(96,8.15),np.full(96,14),6)
    add('output_bearing','output_bearing','输出轴承包络 · 16/28/6','采购尺寸待确认',position=(0,0,64),explode=(0,0,75))
    add('gear_cover','gear_cover','减速器端盖','FDM 配合 / 加工',position=(0,0,64),explode=(0,0,85))
    # End retainers trap each core axially with 0.3 mm nominal trial clearance;
    # all fixed assemblies share six M3 tie rods, outside the magnetic yoke.
    gx,go,gh=grid(((-63,63),(-63,63),(18,30)),.45);xx,yy,zz=gx
    cap=np.maximum(np.maximum(26-np.hypot(xx,yy),np.hypot(xx,yy)-60),np.abs(zz-26)-1.5)
    for i in range(12):
        a=i*np.pi/6
        boss=np.maximum(np.hypot(xx-46.5*np.cos(a),yy-46.5*np.sin(a))-1.6,np.maximum(20.3-zz,zz-24.6))
        cap=np.minimum(cap,boss)
    for i in range(6):
        a=i*np.pi/3;cap=np.maximum(cap,1.8-np.hypot(xx-54*np.cos(a),yy-54*np.sin(a)))
    for i in range(12):
        point=rotation(i*np.pi/6)[:3,:3]@route[-1]
        cap=np.maximum(cap,1.1-np.hypot(xx-point[0],yy-point[1]))
    # Actual lead clearance slots, cut from the same routes as the conductors.
    from scipy.spatial import cKDTree
    port_distance=cKDTree(ports).query(np.column_stack([xx.ravel(),yy.ravel(),zz.ravel()]))[0].reshape(xx.shape)
    cap=np.maximum(cap,.85-port_distance)
    assets['motor_retainer']=mesh_field(cap,go,gh)
    add('front_retainer','motor_retainer','前磁芯限位盖','FDM 试配 · 轴向间隙 0.3 mm',explode=(0,0,14))
    rear=assets['motor_retainer'].copy();rear.apply_transform(np.diag([1,1,-1,1]));rear.fix_normals()
    assets['rear_retainer']=rear
    add('rear_retainer','rear_retainer','后磁芯限位盖','FDM 试配 · 轴向间隙 0.3 mm',explode=(0,0,-85))
    assets['motor_adapter_spacer']=annular_profile(np.full(64,1.8),np.full(64,4.2),1.5)
    # Tie rods extended from 105 to 130 mm to also retain the rear wiring bracket
    # below the rear retainer nuts. Position z=15 gives span [-50, 80].
    assets['tie_rod']=trimesh.creation.cylinder(radius=1.5,height=130,sections=32)
    q=np.arange(96)*2*np.pi/96
    assets['tie_nut']=annular_profile(np.full(96,1.65),2.75/np.cos((q+np.pi/6)%(np.pi/3)-np.pi/6),2.4)
    assets['tie_washer']=annular_profile(np.full(96,1.7),np.full(96,3.5),.5)
    for i in range(6):
        a=i*np.pi/3;pos=(54*np.cos(a),54*np.sin(a))
        add(f'motor_spacer_{i}','motor_adapter_spacer','法兰垫柱 · 1.5 mm','FDM / 加工',position=(*pos,28.25),explode=(0,0,17))
        add(f'tie_rod_{i}','tie_rod','M3 贯穿拉杆 · 130 mm 包络','采购螺纹杆，模型未画螺纹；延长 25 mm 用于固定接线支架',position=(*pos,15),explode=(20*np.cos(a),20*np.sin(a),0))
        for suffix,zpos in [('rear',-29.2),('front',67.7)]:
            add(f'washer_{suffix}_{i}','tie_washer','M3 垫圈 · 7/3.4/0.5','采购尺寸包络',position=(*pos,-27.75 if suffix=='rear' else 66.25),explode=(0,0,-88 if suffix=='rear' else 92))
            add(f'nut_{suffix}_{i}','tie_nut','M3 螺母尺寸包络','采购，实物尺寸需核对',position=(*pos,zpos),explode=(0,0,-90 if suffix=='rear' else 95))
    # Rear wiring bracket: 3 concentric channels fix the phase bridge arcs,
    # 12 lead clearance slots pass vertical leads at r=65, 6 tie-rod mounting
    # holes secure the bracket below the rear retainer nuts.
    from organic_motor.construct.wiring_bracket import (bracket_base_field,
        bracket_params, build_terminal_assets, build_clamp_assets, fixing_report,
        _manifold, _mesh)
    bxyz,bo,bh=grid(((-84,84),(-84,84),(-49,-43)),.5)
    bracket_mesh=mesh_field(bracket_base_field(*bxyz,spec),bo,bh)
    # Cut clean clearance for wire paths that pass through the bracket
    # (star-point leads and radial connections) using exact Boolean ops.
    bracket_solid=_manifold(bracket_mesh)
    for phase in 'UVW':
        p=phase_paths[phase]
        mask=(p[:,2]>=-50)&(p[:,2]<=-42)
        if not mask.any():continue
        idx=np.where(mask)[0]
        # Split into contiguous segments
        splits=np.where(np.diff(idx)>1)[0]+1
        for seg in np.split(idx,splits):
            if len(seg)<2:continue
            tube=sweep_tube(p[seg],1.0)
            bracket_solid=bracket_solid-_manifold(tube)
    bracket_mesh=_mesh(bracket_solid);bracket_mesh.fix_normals()
    if not bracket_mesh.is_watertight:trimesh.repair.fill_holes(bracket_mesh);bracket_mesh.fix_normals()
    assets['harness_bracket_base']=bracket_mesh
    bp=bracket_params()
    terminal_assets,terminal_contacts=build_terminal_assets(harness)
    assets.update(terminal_assets)
    clamp_assets,clamp_list=build_clamp_assets()
    assets.update(clamp_assets)
    harness['terminal_contacts']=terminal_contacts
    harness['clamp_list']=clamp_list
    harness['fixing']=fixing_report()
    harness['support_status']='后侧接线支架、线缆夹与端子座已生成；FDM 公差、端子采购型号与绝缘耐压待确认'
    add('harness_bracket','harness_bracket_base','后侧接线支架 · 线缆夹安装面','FDM 配合样件',group='wiring',explode=(0,0,-55))
    for clamp_info in clamp_list:
        add(clamp_info['name'],clamp_info['name'],
            f"线缆夹 · {clamp_info['phase']} 相弧线固定",'FDM 配合样件',group='wiring',
            explode=(0,0,-60))
    for phase in 'UVW':
        add('terminal_'+phase,'terminal_'+phase,phase+' 端子座 · M3 接线柱','采购型号待确认',group='wiring',explode=(0,0,-60))
    add('terminal_N','terminal_N','N 星点端子座 · M3 接线柱','采购型号待确认',group='wiring',explode=(0,0,-75))
    for i in range(6):
        a=i*np.pi/3;pos=(54*np.cos(a),54*np.sin(a))
        add(f'washer_bracket_{i}','tie_washer','M3 垫圈 · 支架','采购尺寸包络',position=(*pos,-45.0),explode=(0,0,-92))
        add(f'nut_bracket_{i}','tie_nut','M3 螺母 · 支架紧固','采购，实物尺寸需核对',position=(*pos,-47.0),explode=(0,0,-94))
    shrink=card.get('shrinkage',{}).get('linear_xyz',[0,0,0])
    if card.get('shrinkage',{}).get('measured'):
        measured=shrinkage_from_coupon(card['shrinkage'].get('coupon_mold_mm'),card['shrinkage'].get('coupon_cast_mm'))
        if not np.allclose(measured,shrink,atol=1e-5):raise ValueError('收缩率与实测试样尺寸不一致，拒绝补偿')
    elif np.any(np.asarray(shrink)!=0):raise ValueError('非零收缩补偿须提供同批试样尺寸并标记 measured')
    assets.update(segment_molds(spec,shrink))
    # Deliberately do not merge a material-change candidate into the old FEA run.
    report=dict(schema='wound-prototype-v1',spec=asdict(spec),candidate='新分段径向磁通候选 · 未完成电磁验证',
        material=card,material_status=validate_material(card),instances=instances,winding_harness=harness,
        winding=dict(turns=spec.turns,wire_diameter_mm=spec.wire_diameter_mm,
            insulated_diameter_mm=spec.wire_diameter_mm+2*spec.enamel_radial_mm,
            route_length_mm=float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum()),
            lead_allowance_each_mm=80,start_mm=route[0].tolist(),end_mm=route[-1].tolist(),
            insertion_direction=[-1,0,0],insertion_travel_mm=35,
            instructions=['单独打印带槽骨架并检查穿孔','沿同源螺旋槽绕线，预留两端引线','将磁芯从骨架外径侧沿径向向内插入','将 12 个已绕线磁芯段装入承载壳','安装前后限位盖与六根贯穿拉杆，扭紧前后螺母','将接线支架套入拉杆，置于后螺母下方，加垫圈与螺母紧固','按三相 S/F 接线表串联，每相四段；引线穿过支架 r=65 清理孔，跨接弧线落入对应层线槽','安装 U/V/W/N 端子座，剥漆压接或焊接引线','盖上压线盖，用 6×M2.5 螺钉从下方紧固；三相末端汇入 N；驱动接 U/V/W；检查绝缘后再装转子'],
            phase_connection_status='三相串联路径与接线支架已生成；新拓扑转矩、绝缘耐压与端子采购型号仍待验证'),
        gearbox=dict(ratio=spec.ratio,ring_teeth=spec.ring_teeth,planet_orbit_mm=orbit,
            shaft_diameter_mm=16,planet_pin_diameter_mm=6,bolt_circle_diameter_mm=108,bolt_holes=6,
            tooth_model='sampled involute flanks; no root-fillet/contact-load certification',
            unresolved=['D 形传扭接口已生成，轴向锁紧仍需落实','贯穿拉杆预紧力与 FDM 限位盖强度','输出轴承实物型号与轴向保持（当前仅尺寸包络）','齿根圆角和接触强度','润滑与实测效率']),
        mold=dict(parting_axis='Z',draft_deg=spec.draft_deg,shrink_xyz=shrink,
            shrinkage_measured=card.get('shrinkage',{}).get('measured',False),
            tooling_allowance_mm=.1,feed_diameter_mm=4,vent_diameter_mm=1.3,
            note='两轴向半模；四孔夹紧，分模面另需定位工装。拔模通过截面嵌套检查，收缩按各轴 1/(1-s) 补偿。'),
        housing=dict(profile='circular arc in radial-axial section',arc_radius_mm=spec.housing_arc_radius_mm,middle_outer_diameter_mm=2*spec.housing_mid_radius_mm,end_outer_diameter_mm=2*float(housing_outer_radius(24.5,spec)),helical_cooling_tube=False),
        assembly_checks=dict(core_axial_clearance_mm=.3,shared_tie_rod_diameter_mm=3,shared_tie_rod_length_mm=130,retention='前后限位盖与六根贯穿拉杆；初始配合间隙需试配修正；接线支架由延长拉杆与附加螺母紧固'),
        manufacturing_release=False)
    return assets,report,route


def export(out, spec=None, card=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assets,report,route=build(spec,card)
    scene=trimesh.Scene()
    colors={'core_segment':[95,120,137,255],'grooved_bobbin':[225,220,191,255],'winding':[208,107,42,255],'phase_U':[214,71,51,255],'phase_V':[217,165,31,255],'phase_W':[49,110,218,255],
        'honeycomb_housing':[44,123,115,255],'helical_tube':[61,165,198,255], 'sun':[217,159,66,255],
        'planet':[179,189,199,255],'ring':[76,101,117,255],'carrier':[206,136,76,255]}
    used={i['asset'] for i in report['instances']}
    for name in used:
        m=assets[name].copy();m.apply_scale(.001)
        m.visual.face_colors=colors.get(name,[141,161,165,255]);scene.geometry[name]=m
    for inst in report['instances']:
        transform=np.array(inst['transform']);transform[:3,3]*=.001
        scene.graph.update(frame_to=inst['id'],matrix=transform,geometry=inst['asset'])
    (out/'assembly.glb').write_bytes(scene.export(file_type='glb'))
    moldscene=trimesh.Scene()
    for name in ('segment_mold_lower','segment_mold_upper'):
        m=assets[name].copy();m.apply_translation([-40.5,0,12 if name.endswith('upper') else -12]);m.apply_scale(.001)
        m.visual.face_colors=[109,165,156,255]
        moldscene.add_geometry(m,node_name=name)
    (out/'molds.glb').write_bytes(moldscene.export(file_type='glb'))
    for name,m in assets.items():
        if not m.is_watertight or m.body_count!=1 or m.volume<=0:
            raise ValueError(f'{name}: 零件须为单个闭合、正体积连通体')
    report['assets']={n:dict(size_mm=np.round(m.extents,3).tolist(),watertight=m.is_watertight,components=m.body_count,mesh_sha256=hashlib.sha256(np.asarray(m.vertices,np.float32).tobytes()+np.asarray(m.faces,np.uint32).tobytes()).hexdigest()) for n,m in assets.items()}
    counts=Counter(i['asset'] for i in report['instances'])
    for name in ('segment_mold_lower','segment_mold_upper'):counts[name]=1
    report['bill_of_materials']=[dict(asset=name,quantity=counts[name],process=next((i['process'] for i in report['instances'] if i['asset']==name),'FDM 工装')) for name in assets]
    with (out/'bill_of_materials.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['asset','quantity','process']);writer.writeheader();writer.writerows(report['bill_of_materials'])
    report['generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['wiring_source_sha256']=hashlib.sha256((Path(__file__).parent/'winding_harness.py').read_bytes()+(Path(__file__).parents[1]/'topology'/'winding_assignment.py').read_bytes()+(Path(__file__).parent/'wiring_bracket.py').read_bytes()).hexdigest()
    report['design_hash']=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()[:16]
    (out/'wiring.json').write_text(json.dumps(report['winding_harness'],ensure_ascii=False,indent=2))
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savetxt(out/'winding_route_mm.csv',route,delimiter=',',header='x_mm,y_mm,z_mm',comments='')
    with zipfile.ZipFile(out/'manufacturing-kit.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n,m in assets.items():z.writestr('parts/'+n+'_mm.stl',m.export(file_type='stl'))
        doc=Path(__file__).resolve().parents[2]/'docs'/'WOUND_PROTOTYPE.md'
        if doc.is_file():z.write(doc,'制造与材料实测指南.md')
        z.write(out/'bill_of_materials.csv','bill_of_materials.csv')
        z.write(out/'wiring.json','wiring.json')
        z.write(out/'manifest.json','manifest.json');z.write(out/'winding_route_mm.csv','winding_route_mm.csv')
        z.writestr('material_card.template.json',json.dumps(material_template(),ensure_ascii=False,indent=2))
        z.writestr('装配说明.md', '# 分段绕线原型\n\n这是新制造候选，不能使用旧电机的性能报告。所有 STL 单位 mm。\n\n'+ '\n'.join(report['winding']['instructions'])+'\n\n磁芯、骨架各 12 件；绕组共 12 组、按三相各四组串联，详见 wiring.json；太阳轮 1、行星轮 3、内齿圈 1、行星架 1、轴销 3。模具只需一套，可重复浇注 12 段。\n\n先测材料与收缩，再重建模具。默认零收缩代表未补偿，并非材料不收缩。丝径含漆膜。齿轮目前是配合/运动学样件，不是额定载荷认证。\n')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='organic_motor/out/wound_prototype');ap.add_argument('--material');ap.add_argument('--spec')
    args=ap.parse_args();card=json.loads(Path(args.material).read_text()) if args.material else None
    spec=WoundSpec(**json.loads(Path(args.spec).read_text())) if args.spec else WoundSpec()
    result=export(args.out,spec,card);print(json.dumps({'out':args.out,'hash':result['design_hash'],'parts':len(result['instances'])},ensure_ascii=False))
