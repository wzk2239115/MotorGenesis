import numpy as np
import pytest
import trimesh
pytest.importorskip('manifold3d')
from organic_motor.construct.assembly_audit import audit
from organic_motor.construct.wound_prototype import annular_profile


def instance(name,asset,offset=0):
    t=np.eye(4);t[0,3]=offset
    return dict(id=name,asset=asset,transform=t.tolist())


def test_boolean_distinguishes_a_hole_from_bounding_box_overlap():
    assets={'ring':annular_profile(np.full(96,2),np.full(96,4),5),
            'pin':trimesh.creation.cylinder(radius=1,height=6)}
    report={'instances':[instance('ring','ring'),instance('pin','pin')]}
    assert audit(assets,report)['static_interference_pass']
    report['instances'][1]=instance('pin','pin',2)
    assert not audit(assets,report)['static_interference_pass']


def test_zero_volume_contact_not_a_false_collision():
    assets={'box':trimesh.creation.box(extents=[2,2,2])}
    report={'instances':[instance('a','box'),instance('b','box',2)]}
    assert audit(assets,report)['static_interference_pass']
    report['instances'][1]=instance('b','box',1)
    assert audit(assets,report)['collisions'][0]['intersection_mm3']==pytest.approx(4)


def test_star_contact_is_allowed_only_inside_declared_joint():
    assets={'wire':trimesh.creation.box(extents=[1,1,1])}
    report={'instances':[instance('phase_U','wire'),instance('phase_V','wire',.2)],
            'winding_harness':{'connection':'star','star_point_mm':[0,0,0]}}
    result=audit(assets,report)
    assert result['static_interference_pass']
    assert len(result['intended_electrical_contacts'])==1
    report['winding_harness']['star_point_mm']=[0,0,10]
    assert not audit(assets,report)['static_interference_pass']
