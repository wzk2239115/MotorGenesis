"""Intersection audit on exported solids in assembled millimetre coordinates.

AABB is broad phase only; narrow phase uses solid Boolean intersection.
Positive volume is reported, including nominal press fits; never silently
exclude a pair just because it was intended to mate.
"""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import trimesh
import manifold3d


def load_package(folder):
    folder=Path(folder)
    report=json.loads((folder/'manifest.json').read_text())
    with zipfile.ZipFile(folder/'manufacturing-kit.zip') as archive:
        assets={name:trimesh.load(io.BytesIO(archive.read(f'parts/{name}_mm.stl')),file_type='stl') for name in report['assets']}
    return assets,report


def audit(assets,report,tolerance_mm3=.02):
    solids={}
    for name,m in assets.items():
        solid=manifold3d.Manifold(manifold3d.Mesh(np.asarray(m.vertices,np.float32),np.asarray(m.faces,np.uint32)))
        if solid.status()!=manifold3d.Error.NoError:raise ValueError(f'{name}: invalid Boolean solid: {solid.status()}')
        solids[name]=solid
    instances=[]
    for item in report['instances']:
        tr=np.array(item['transform']);m=assets[item['asset']]
        corners=trimesh.bounds.corners(m.bounds)
        xyz=trimesh.transform_points(corners,tr)
        instances.append((item['id'],solids[item['asset']].transform(tr[:3]),np.array([xyz.min(0),xyz.max(0)])))
    collisions=[];contacts=[];candidates=0
    for i,(name,a,box_a) in enumerate(instances):
        for name_b,b,box_b in instances[i+1:]:
            overlap=np.minimum(box_a[1],box_b[1])-np.maximum(box_a[0],box_b[0])
            if np.any(overlap<=1e-5):continue
            candidates+=1
            intersection=a^b
            volume=intersection.volume()
            if volume>tolerance_mm3:
                # Only the explicit common electrical node permits wire overlap.
                # Still test the full intersection outside that bounded region.
                harness=report.get('winding_harness',{})
                if harness.get('connection')=='star' and name in {'phase_U','phase_V','phase_W'} and name_b in {'phase_U','phase_V','phase_W'}:
                    joint=manifold3d.Manifold.sphere(2,32).translate(harness['star_point_mm'])
                    outside=(intersection-joint).volume()
                    if outside<1e-6:
                        contacts.append(dict(a=name,b=name_b,node='N',intersection_mm3=volume,outside_joint_mm3=outside,region_radius_mm=2))
                        continue
                collisions.append(dict(a=name,b=name_b,intersection_mm3=round(volume,6)))
    return dict(method='assembled exported STL solid Boolean intersection',design_hash=report.get('design_hash'),
                instances=len(instances),broad_phase_candidates=candidates,threshold_mm3=tolerance_mm3,
                collisions=collisions,intended_electrical_contacts=contacts,static_interference_pass=not collisions,
                limitations=['Does not certify FDM tolerances or strength','Zero-volume contact is not classified as interference','Tool access and motion require separate checks'])




def fastener_checks(assets,report):
    """Check bolt shank clearance, closed boss walls and axial socket approach."""
    from organic_motor.construct.wound_prototype import annular_profile
    def solid(mesh):
        return manifold3d.Manifold(manifold3d.Mesh(np.asarray(mesh.vertices,np.float32),np.asarray(mesh.faces,np.uint32)))
    world={i['id']:solid(assets[i['asset']]).transform(np.array(i['transform'])[:3]) for i in report['instances']}
    radius=report['gearbox']['bolt_circle_diameter_mm']/2
    bores=[];tools=[];walls=[]
    targets=['housing','front_retainer','rear_retainer','adapter','ring','gear_cover']
    for i in range(6):
        a=i*np.pi/3;xy=np.array([radius*np.cos(a),radius*np.sin(a)])
        # 3.3 mm gauge verifies 0.15 mm radial clearance around M3 nominal shank.
        gauge=trimesh.creation.cylinder(radius=1.65,height=120,sections=96);gauge.apply_translation([*xy,20]);g=solid(gauge)
        for name in targets:
            volume=(g^world[name]).volume()
            bores.append(dict(axis=i,part=name,obstruction_mm3=float(volume),pass_check=volume<.001))
        # A 2.1..3.8 mm annulus must be supported all around each housing hole.
        wall=annular_profile(np.full(96,2.1),np.full(96,3.8),48)
        wall.apply_translation([*xy,0]);void=(solid(wall)-world['housing']).volume()
        walls.append(dict(axis=i,missing_wall_mm3=float(void),pass_check=void<.01))
        for side,lo,hi in [('front',66.5,84),('rear',-45,-28)]:
            tool=annular_profile(np.full(96,3.5),np.full(96,5.5),hi-lo);tool.apply_translation([*xy,(lo+hi)/2]);tool_s=solid(tool)
            collisions=[]
            for name,obj in world.items():
                volume=(tool_s^obj).volume()
                if volume>.01:collisions.append(dict(part=name,mm3=float(volume)))
            tools.append(dict(axis=i,side=side,collisions=collisions,pass_check=not collisions))
    return dict(shank_gauge_diameter_mm=3.3,nominal_radial_clearance_verified_mm=.15,
                socket_envelope=dict(outer_diameter_mm=11,inner_diameter_mm=7,approach='axial; specific tool must be checked'),
                bore_checks=bores,housing_wall_checks=walls,tool_checks=tools,
                pass_check=all(x['pass_check'] for x in bores+walls+tools))


def publish(folder,result):
    """Attach evidence only to the exact geometry version that was audited."""
    import os
    import tempfile
    folder=Path(folder)
    current=json.loads((folder/'manifest.json').read_text())
    if current['design_hash']!=result['design_hash']:raise ValueError('Geometry changed during audit; evidence not published')
    data=json.dumps(result,indent=2)
    (folder/'assembly-audit.json').write_text(data)
    with tempfile.NamedTemporaryFile(dir=folder,suffix='.zip',delete=False) as temp:
        target=Path(temp.name)
    try:
        with zipfile.ZipFile(folder/'manufacturing-kit.zip') as source,zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as dest:
            for entry in source.infolist():
                if entry.filename!='assembly-audit.json':dest.writestr(entry,source.read(entry.filename))
            dest.writestr('assembly-audit.json',data)
        os.replace(target,folder/'manufacturing-kit.zip')
    finally:
        target.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',default='organic_motor/out/wound_prototype');p.add_argument('--output',required=True);p.add_argument('--publish',action='store_true');args=p.parse_args()
    assets,manifest=load_package(args.folder)
    result=audit(assets,manifest);result['fasteners']=fastener_checks(assets,manifest)
    result['limitations']=['No manufacturing tolerance or strength certification','Zero-volume contact excluded','Static solids and specified axial tool envelope only; gear motion and core insertion have separate tests']
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    if args.publish:publish(args.folder,result)
    print(json.dumps({k:v for k,v in result.items() if k!='fasteners'},indent=2));print('Fastener checks:',result['fasteners']['pass_check'])
