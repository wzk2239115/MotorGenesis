"""Tests for per-phase resistance and EM symbol verification."""
import numpy as np
import pytest
from organic_motor.construct.phase_resistance import (
    copper_cross_section_area_mm2, phase_resistance,
    compute_phase_resistances, verify_phase_symbols, em_verification_chain,
)
from organic_motor.topology.winding_assignment import tooth_phase_polarity


def test_copper_area_uses_bare_diameter():
    A = copper_cross_section_area_mm2(0.6)
    assert abs(A - np.pi * 0.09) < 1e-10
    # Insulated diameter (0.67) must NOT be used
    A_ins = copper_cross_section_area_mm2(0.67)
    assert A < A_ins


def test_phase_resistance_uses_bare_copper():
    R = phase_resistance(path_length_mm=4372.0, bare_diameter_mm=0.6,
                         sigma_copper_S_m=5.96e7)
    assert R['resistance_ohm'] > 0
    assert R['copper_cross_section_mm2'] < 0.30  # bare, not insulated
    assert R['joint_resistance_status'] == '待确认'


def test_phase_resistance_temperature_correction():
    R20 = phase_resistance(1000.0, 0.6, 5.96e7, temperature_C=20.0)
    R80 = phase_resistance(1000.0, 0.6, 5.96e7, temperature_C=80.0)
    assert R80['resistance_ohm'] > R20['resistance_ohm']


def test_per_phase_resistances_differ():
    """The three phase paths have different lengths → different R."""
    chains = [
        dict(phase='U', route_length_mm=4372.0),
        dict(phase='V', route_length_mm=4435.3),
        dict(phase='W', route_length_mm=4485.7),
    ]
    result = compute_phase_resistances(chains, 0.6, 5.96e7)
    assert result['per_phase']['U']['resistance_ohm'] < result['per_phase']['V']['resistance_ohm']
    assert result['per_phase']['V']['resistance_ohm'] < result['per_phase']['W']['resistance_ohm']
    assert result['imbalance_percent'] > 0.5  # >0.5% difference


def test_verify_phase_symbols_balanced():
    """12-slot 10-pole winding: balanced, 120° apart."""
    coils = []
    for tooth in range(12):
        phase, polarity = tooth_phase_polarity(tooth)
        coils.append(dict(
            tooth=tooth + 1, phase='UVW'[phase], polarity=polarity,
            input_terminal=f'C{tooth+1:02d}.S'))
    result = verify_phase_symbols(coils)
    assert result['independent_coil_reference']
    assert result['three_phase_balanced']
    assert result['phase_sequence_120deg']
    assert result['rotation_reversal_verified']
    assert not result['magnet_polarity_verified']  # 待确认
    assert result['passed']


def test_verify_phase_symbols_rejects_wrong_polarity():
    """Flipping one coil's polarity breaks balance."""
    coils = []
    for tooth in range(12):
        phase, polarity = tooth_phase_polarity(tooth)
        if tooth == 0:
            polarity = -polarity  # flip
        coils.append(dict(
            tooth=tooth + 1, phase='UVW'[phase], polarity=polarity,
            input_terminal=f'C{tooth+1:02d}.S'))
    result = verify_phase_symbols(coils)
    assert not result['three_phase_balanced']


def test_em_verification_chain_documents_unconnected_links():
    """The new candidate must NOT claim solver/torque is connected."""
    from organic_motor.construct.wound_prototype import winding_route, WoundSpec
    from organic_motor.construct.winding_harness import make_harness
    from organic_motor.config import MotorConfig
    route = winding_route(WoundSpec())
    paths, report, ports = make_harness(route)
    result = em_verification_chain(report, WoundSpec(), MotorConfig().sigma_copper)
    chain = result['verification_chain']
    assert 'verified' in chain['manufacturing_geometry']
    assert 'verified' in chain['current_path']
    assert 'NOT connected' in chain['solver_mesh']
    assert 'NOT computed' in chain['flux_linkage']
    assert 'NOT computed' in chain['torque']
    assert not result['simulation_ready']
    assert not result['old_model_fallback']
    # Per-phase resistances present
    assert 'U' in result['resistance']['per_phase']
    assert 'V' in result['resistance']['per_phase']
    assert 'W' in result['resistance']['per_phase']


def test_resistance_no_hidden_correction():
    """R = ρ*L/A only; no correction constant to force balance."""
    from organic_motor.construct.wound_prototype import winding_route, WoundSpec
    from organic_motor.construct.winding_harness import make_harness
    from organic_motor.config import MotorConfig
    route = winding_route(WoundSpec())
    paths, report, ports = make_harness(route)
    result = compute_phase_resistances(
        report['phase_chains'], WoundSpec().wire_diameter_mm,
        MotorConfig().sigma_copper)
    # Verify R = ρ*L/A exactly (no hidden constant)
    rho = 1.0 / MotorConfig().sigma_copper
    for chain in report['phase_chains']:
        ph = chain['phase']
        L = chain['route_length_mm'] * 1e-3
        A = np.pi * (WoundSpec().wire_diameter_mm / 2) ** 2 * 1e-6
        R_expected = rho * L / A
        R_actual = result['per_phase'][ph]['resistance_ohm']
        assert abs(R_actual - R_expected) / R_expected < 1e-6
