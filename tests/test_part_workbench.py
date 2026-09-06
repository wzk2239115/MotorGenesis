"""Contracts for the fabrication explorer: identities, units, mold topology."""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import trimesh

from organic_motor.web.parts import part_meshes, manifest, package, casting_molds
from organic_motor.web.builder import checkpoint_to_glb


def fixture_artifact(tmp_path):
    n=36
    x,y,z=np.meshgrid(*(np.linspace(-.018,.018,n),)*3,indexing='ij')
    r=np.hypot(x,y)
    core=(r>.007)&(r<.012)&(abs(z)<.008)
    caps=(r>.006)&(r<.015)&(abs(z)>.009)&(abs(z)<.012)
    iron=(core|caps).astype(np.float32)
    path=tmp_path/'step_000000.npz'
    np.savez(path,rho_iron=iron,rho_pm=np.zeros_like(iron),endcap_mask=caps.astype(float),rotor_mask=np.zeros_like(iron),spacing=[.036/(n-1)]*3,origin=[-.018]*3)
    return path


def test_parts_have_stable_identity_and_no_double_export(tmp_path):
    path=fixture_artifact(tmp_path)
    meshes=list(part_meshes(path))
    assert {n for n,m in meshes}=={'front_endcap_iron','rear_endcap_iron','stator_iron'}
    report=manifest(path,meshes)
    assert all(p['watertight'] for p in report['parts'])
    assert not report['required_morphology']['honeycomb']
    scene=trimesh.load(io.BytesIO(checkpoint_to_glb(path)),file_type='glb')
    assert len(scene.geometry)==3


def test_export_stl_is_mm_and_voids_are_not_print_parts(tmp_path):
    path=fixture_artifact(tmp_path)
    originals=dict(part_meshes(path))
    with zipfile.ZipFile(io.BytesIO(package(path))) as archive:
        m=trimesh.load(io.BytesIO(archive.read('fit_check_parts/stator_iron_mm.stl')),file_type='stl')
        assert np.allclose(m.extents,originals['stator_iron'].extents*1000,atol=.001)
        assert json.loads(archive.read('manifest.json'))['stl_units']=='mm'
        assert '制造说明.md' in archive.namelist()


def test_split_molds_closed_and_contain_no_core_material(tmp_path):
    path=fixture_artifact(tmp_path)
    tools=casting_molds(path)
    assert len(tools)==2
    for _,m in tools:
        assert m.is_watertight
        assert m.volume>0
    assert tools[0][1].bounds[1,2] <= tools[1][1].bounds[0,2]+1e-8


def test_material_coupon_molds_are_closed(tmp_path):
    for name,m in casting_molds(fixture_artifact(tmp_path), sample=True):
        assert m.is_watertight
        assert m.extents.max() < .08


def test_parts_api_and_missing_field(tmp_path):
    from fastapi.testclient import TestClient
    from organic_motor.web.server import create_app
    checkpoint=tmp_path/'assembly'/'checkpoints';checkpoint.mkdir(parents=True)
    fixture_artifact(checkpoint)
    client=TestClient(create_app(tmp_path))
    assert client.get('/api/runs/assembly/checkpoint/0/parts').status_code==200
    assert client.get('/api/runs/assembly/checkpoint/0/parts.zip').content[:2]==b'PK'
    assert client.get('/api/runs/assembly/checkpoint/0/slice?field=temperature').status_code==404


def test_organic_constraints_reject_empty_or_underresolved():
    from types import SimpleNamespace
    from organic_motor.construct.prototype_spec import OrganicPrototypeSpec
    import pytest
    spec=OrganicPrototypeSpec()
    cfg=SimpleNamespace(spacing=(.001,.001,.001))
    empty=SimpleNamespace(sdf=np.ones((3,3,3)))
    filled=SimpleNamespace(sdf=-np.ones((3,3,3)))
    with pytest.raises(ValueError): spec.validate(cfg,empty,filled)
    with pytest.raises(ValueError): spec.validate(SimpleNamespace(spacing=(.003,)*3),filled,filled)
    assert spec.validate(cfg,filled,filled)['required']==['honeycomb','helix']
