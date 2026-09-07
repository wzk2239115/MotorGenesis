import numpy as np
from organic_motor.topology.winding_assignment import tooth_phase_polarity
from organic_motor.construct.winding_harness import make_harness, verify_harness
from organic_motor.construct.wound_prototype import winding_route, WoundSpec


def test_balanced_phase_assignment():
    sums=np.zeros(3,complex)
    for tooth in range(12):
        phase,sign=tooth_phase_polarity(tooth)
        sums[phase]+=sign*np.exp(1j*5*tooth*np.pi/6)
    assert np.allclose(abs(sums),3.863703305)
    assert abs(sums.sum())<1e-12
    assert np.allclose(np.angle(sums[1:]/sums[:-1]),2*np.pi/3)


def test_continuous_three_phase_routes():
    route=winding_route(WoundSpec())
    paths,report,ports=make_harness(route)
    assert verify_harness(paths,report)['passed']
    for chain in report['phase_chains']:
        path=paths[chain['phase']]
        assert np.isfinite(path).all()
        assert np.max(np.linalg.norm(np.diff(path,axis=0),axis=1))<1
        # Four complete coils retained with the correct traversal direction.
        cursor=0
        for entry in chain['coils']:
            start=np.array(entry['S_mm'] if entry['polarity']>0 else entry['F_mm'])
            end=np.array(entry['F_mm'] if entry['polarity']>0 else entry['S_mm'])
            a=cursor+np.argmin(np.linalg.norm(path[cursor:]-start,axis=1))
            b=a+np.argmin(np.linalg.norm(path[a:]-end,axis=1))
            assert b>a
            assert np.linalg.norm(path[a]-start)<1e-8
            assert np.linalg.norm(path[b]-end)<1e-8
            cursor=b
    assert len(report['series_links'])==9


def test_config_and_netlist_share_manufacturing_convention():
    from organic_motor.config3d import MOTOR_SPEC
    from organic_motor.construct.winding_netlist import PrintedCoilNetlist
    netlist=PrintedCoilNetlist()
    assert netlist.pole_pairs==5
    for tooth,phase,sign in netlist.coil_table():
        assert (phase,sign)==tooth_phase_polarity(tooth)
        assert (MOTOR_SPEC.phase_of_slot(tooth),MOTOR_SPEC.polarity_of_slot(tooth))==(phase,sign)
