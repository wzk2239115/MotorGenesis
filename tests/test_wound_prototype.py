import numpy as np
import pytest
from organic_motor.construct.wound_prototype import (WoundSpec, core_field, sleeve_field, winding_route, sweep_tube, kinematics, gear_profile)
from organic_motor.construct.casting_material import material_template, validate_material, shrinkage_from_coupon, compensated_dimensions


def test_material_never_invents_properties():
    card=material_template()
    assert not validate_material(card)['material_data_complete']
    assert card['density_kg_m3'] is None
    assert not validate_material(card)['simulation_ready']
    assert shrinkage_from_coupon([100,50,20],[98,49,19.6]) == pytest.approx([.02]*3)
    assert compensated_dimensions([98,49,19.6],[.02]*3)==pytest.approx([100,50,20])
    with pytest.raises(ValueError):compensated_dimensions([1,1,1],[np.nan,0,0])


def test_radial_insertion_sweeps_without_bobbin_collision():
    spec=WoundSpec();route=winding_route(spec)
    x,y,z=np.meshgrid(np.linspace(29,44,40),np.linspace(-7,7,40),np.linspace(-24,24,60),indexing='ij')
    sleeve=sleeve_field(x,y,z,spec,route)<-.05
    for travel in np.linspace(35,0,24):
        assert not np.any(sleeve & (core_field(x-travel,y,z,spec)<-.05))
    # All wire centerline points sit in the same generated groove and outside core.
    assert np.min(sleeve_field(*route.T,spec,route))>0
    assert np.min(core_field(*route.T,spec))>0


def test_draft_sections_nested_and_wire_continuous():
    spec=WoundSpec();x,y=np.meshgrid(np.linspace(28,51,70),np.linspace(-15,15,70))
    previous=core_field(x,y,0,spec)<=0
    for height in np.linspace(1,20,20):
        current=core_field(x,y,height,spec)<=0
        assert not np.any(current & ~previous)
        previous=current
    m=sweep_tube(winding_route(spec),.335)
    assert m.is_watertight and m.body_count==1 and m.volume>0


def test_gear_kinematics_and_root_clearance():
    s=WoundSpec();s.validate()
    for theta in np.linspace(-10,10,21):
        k=kinematics(theta,s)
        assert s.sun_teeth*(k['sun']-k['carrier'])+s.ring_teeth*(k['ring']-k['carrier'])==pytest.approx(0)
        assert s.sun_teeth*(k['sun']-k['carrier'])+s.planet_teeth*(k['planet_spin']-k['carrier'])==pytest.approx(0)
    assert gear_profile(18,1.5,.16).min()>8.15
    with pytest.raises(ValueError):WoundSpec(turn_pitch_mm=.5).validate()


def test_manufacturing_api_is_separate_from_simulator(tmp_path):
    from fastapi.testclient import TestClient
    from organic_motor.web.server import create_app
    client=TestClient(create_app(tmp_path))
    assert client.get('/prototype').status_code==200
    assert client.get('/api/prototype/manifest.json').status_code==404
    assert client.get('/api/prototype/secrets').status_code==404
    response=client.post('/api/prototype/material-check',json=material_template())
    assert response.status_code==200
    assert not response.json()['simulation_ready']


def test_exact_ring_holes_and_mesh_volume():
    pytest.importorskip('mapbox_earcut')
    from organic_motor.construct.wound_prototype import extrude_loops
    t=np.arange(256)*2*np.pi/256
    loop=lambda r:np.column_stack([r*np.cos(t),r*np.sin(t)])
    m=extrude_loops([loop(10),loop(5),loop(1)+[7.5,0]],4)
    assert m.is_watertight and m.body_count==1
    assert m.volume==pytest.approx(np.pi*(100-25-1)*4,rel=.001)


def test_gear_profiles_do_not_overlap_through_rotation():
    # Independent occupancy check across an entire tooth period, outside
    # numerical boundary uncertainty. Includes internal ring engagement.
    s=WoundSpec();x,y=np.meshgrid(np.linspace(-45,45,451),np.linspace(-45,45,451))
    def solid(xx,yy,profile,angle=0,internal=False):
        q=np.mod(np.arctan2(yy,xx)-angle,2*np.pi)
        r=np.interp(q,np.r_[np.arange(len(profile))*2*np.pi/len(profile),2*np.pi],np.r_[profile,profile[0]])
        return np.hypot(xx,yy)>r+.015 if internal else np.hypot(xx,yy)<r-.015
    outer=gear_profile(18,1.5,.16);inner=gear_profile(54,1.5,.16,True)
    for a in np.r_[np.linspace(0,2*np.pi/18,13),np.random.default_rng(71).uniform(0,8*np.pi,16)]:
        k=kinematics(a,s);c=k['carrier'];px=27*np.cos(c);py=27*np.sin(c)
        planet=solid(x-px,y-py,outer,np.pi-np.pi/18+k['planet_spin'])
        assert not np.any(planet & solid(x,y,outer,a))
        assert not np.any(planet & solid(x,y,inner,internal=True))


def test_d_shaft_is_continuous_and_matches_sun_bore():
    from organic_motor.construct.wound_prototype import d_profile, stepped_shaft
    shaft=stepped_shaft()
    assert shaft.is_watertight and shaft.body_count==1 and shaft.volume>0
    assert np.all(d_profile(8.15,6.65)-d_profile(8,6.5)>.14)
    assert shaft.bounds[1,2]==53  # stays below carrier bottom at z=55


def test_clamp_holes_do_not_detach_mold_corners():
    from organic_motor.construct.wound_prototype import segment_molds
    tools=segment_molds(WoundSpec(),[0,0,0])
    assert all(m.body_count==1 and m.is_watertight and m.volume>0 for m in tools.values())
    assert tools['segment_mold_lower'].bounds[1,2]<=.0001
    assert tools['segment_mold_upper'].bounds[0,2]>=-.0001


def test_curved_ends_match_core_bobbin_and_reduce_equal_current_loss():
    from dataclasses import replace
    from organic_motor.construct.wound_prototype import rounded_rectangle
    from organic_motor.reports.fabrication_screen import copper_metrics
    spec=WoundSpec()
    profile=rounded_rectangle(np.linspace(0,2*np.pi,1000),corner=spec.end_radius_mm)
    caps=np.abs(profile[:,1])>17
    # A true R5 semicircle, rather than applying a shading trick to a rectangle.
    assert np.max(np.abs(np.hypot(profile[caps,0],np.abs(profile[caps,1])-17)-5))<.005
    assert copper_metrics(spec)['copper_loss_W']<copper_metrics(replace(spec,end_radius_mm=1.2))['copper_loss_W']


def test_housing_sides_are_circular_and_end_wall_is_preserved():
    from organic_motor.construct.wound_prototype import housing_outer_radius
    spec=WoundSpec();z=np.linspace(-24.5,24.5,101);r=housing_outer_radius(z,spec)
    assert np.allclose((r-(spec.housing_mid_radius_mm-spec.housing_arc_radius_mm))**2+z**2,spec.housing_arc_radius_mm**2)
    assert r[50]-r[0]>9
    assert r.min()-50>=2


def test_wiring_bracket_is_printable_and_has_clearance():
    from organic_motor.construct.wiring_bracket import (bracket_base_field,
        bracket_params, cable_clamp, terminal_post, build_terminal_assets,
        build_clamp_assets, fixing_report)
    from organic_motor.construct.wound_prototype import grid, mesh_field, winding_route, WoundSpec, build
    spec=WoundSpec()
    # Bracket base is a watertight solid ring
    bxyz,bo,bh=grid(((-84,84),(-84,84),(-49,-43)),.5)
    m=mesh_field(bracket_base_field(*bxyz,spec),bo,bh)
    assert m.is_watertight and m.body_count==1 and m.volume>0
    # Clamps are watertight, one per phase level
    p=bracket_params()
    for z,r,ph,_,_ in p['clamp_config']:
        clamp=cable_clamp(z,r,0)
        assert clamp.is_watertight and clamp.volume>0
    # Terminal posts are watertight
    tp=terminal_post([0,0,0],p['terminal_post_radius'],p['terminal_post_height'])
    assert tp.is_watertight and tp.volume>0
    # Fixing report documents every wire segment
    fr=fixing_report()
    assert len(fr['assembly_sequence'])>=8
    assert len(fr['tool_access'])>=3
    assert '待确认' in fr['status']


def test_bracket_mounts_on_extended_tie_rods_and_passes_audit():
    from organic_motor.construct.wound_prototype import build, WoundSpec
    from organic_motor.construct.assembly_audit import audit
    assets,report,route=build(WoundSpec())
    # Tie rods are 130 mm (extended for bracket)
    tr=assets['tie_rod']
    assert abs(tr.extents[2]-130)<0.1
    # Bracket is in the assembly
    assert any(i['asset']=='harness_bracket_base' for i in report['instances'])
    # Terminals are in the assembly
    assert any(i['asset']=='terminal_U' for i in report['instances'])
    assert any(i['asset']=='terminal_N' for i in report['instances'])
    # Clamps are in the assembly
    assert sum(1 for i in report['instances'] if i['asset'].startswith('clamp_'))==9
    # Terminal contacts are defined by geometry AND netlist
    contacts=report['winding_harness']['terminal_contacts']
    assert len(contacts)==4  # U,V,W,N
    for c in contacts:
        assert 'position_mm' in c and 'phase' in c and 'contact_radius_mm' in c
    # Audit passes
    result=audit(assets,report)
    assert result['static_interference_pass']
    assert len(result['intended_electrical_contacts'])>=9  # 3 N-node + 3 terminal + 3 clamp


def test_atomic_publish_and_version_consistency(tmp_path):
    """Export writes to staging dir, then atomically replaces live dir."""
    import json, os
    from organic_motor.construct.wound_prototype import export, WoundSpec
    out=tmp_path/'wound_prototype'
    report=export(str(out),WoundSpec())
    # All required files exist
    required={'assembly.glb','molds.glb','manifest.json','wiring.json',
             'winding_route_mm.csv','manufacturing-kit.zip','bill_of_materials.csv'}
    assert required<={p.name for p in out.iterdir()}
    # No staging directory left behind
    assert not (out.parent/'wound_prototype.staging').exists()
    # Manifest design_hash matches report
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['design_hash']==report['design_hash']
    # Manifest does NOT contain high-density phase_routes_mm
    assert 'phase_routes_mm' not in manifest.get('winding_harness',{})
    # Wiring.json DOES contain phase_routes_mm
    wiring=json.loads((out/'wiring.json').read_text())
    assert 'phase_routes_mm' in wiring
    # EM verification present in manifest
    assert 'em_verification' in manifest['winding_harness']
    assert not manifest['winding_harness']['em_verification']['simulation_ready']
    assert not manifest['winding_harness']['em_verification']['old_model_fallback']
