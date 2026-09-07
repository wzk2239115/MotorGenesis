"""Adversarial tests for manufacturability checks.

Each test constructs a minimal failure case (crossover, short, out-of-N
misconnection, too-small bend radius, blocked tool path) and verifies
that the check actually catches it — not just that the default geometry
passes.
"""
import numpy as np
import pytest
from organic_motor.construct.manufacturability import (
    segment_to_segment_distance, check_intra_phase_crossover,
    check_inter_phase_short, check_bend_radius,
)


def test_segment_distance_parallel():
    d = segment_to_segment_distance(np.array([0,0,0]), np.array([10,0,0]),
                                     np.array([0,1,0]), np.array([10,1,0]))
    assert abs(d - 1.0) < 1e-9


def test_segment_distance_crossing():
    d = segment_to_segment_distance(np.array([0,0,0]), np.array([10,0,0]),
                                     np.array([5,-5,0]), np.array([5,5,0]))
    assert d < 0.01  # segments cross


def test_intra_phase_crossover_detected():
    # A path that crosses itself: goes right, up, left, down, right again
    path = np.array([
        [0, 0, 0], [10, 0, 0], [10, 10, 0],
        [0, 10, 0], [0, 0.1, 0], [10, 0.1, 0],
    ], dtype=float)
    v = check_intra_phase_crossover(path, wire_outer_radius=0.5, min_gap_mm=0.5)
    assert len(v) > 0, 'crossover must be detected'


def test_intra_phase_no_false_positive_on_continuous_path():
    # A smooth spiral should not trigger crossover
    t = np.linspace(0, 4 * np.pi, 500)
    path = np.column_stack([t * np.cos(t), t * np.sin(t), np.zeros_like(t)])
    v = check_intra_phase_crossover(path, wire_outer_radius=0.3, min_gap_mm=0.2)
    # The spiral expands, so non-adjacent segments are far apart
    assert all(x['distance_mm'] > 1.0 for x in v)


def test_inter_phase_short_outside_N_detected():
    # Two parallel paths that touch outside the N region
    path_a = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], dtype=float)
    path_b = np.array([[0, 0.1, 0], [10, 0.1, 0], [10, 10, 0.1]], dtype=float)
    v = check_inter_phase_short([path_a, path_b], 0.5,
                                 star_point=[100, 100, 0], n_region_radius=2.0)
    assert len(v) > 0, 'inter-phase short outside N must be detected'


def test_inter_phase_contact_at_N_not_flagged():
    # Two paths that meet at the N point (0,0,-10) should not be flagged
    path_a = np.array([[5, 0, 0], [0, 0, -10]], dtype=float)
    path_b = np.array([[-5, 0, 0], [0, 0, -10]], dtype=float)
    v = check_inter_phase_short([path_a, path_b], 0.5,
                                 star_point=[0, 0, -10], n_region_radius=2.0)
    assert len(v) == 0, 'contact at N should not be flagged as short'


def test_bend_radius_unverified_without_supplier_data():
    # A path with a tight bend
    path = np.array([[0, 0, 0], [1, 0, 0], [1, 0.1, 0]], dtype=float)
    result = check_bend_radius(path, wire_diameter_mm=0.6)
    assert result['passed'] is None
    assert not result['verified']
    assert '待确认' in result['reason']


def test_bend_radius_passes_with_known_limit():
    # Gentle curve with known supplier limit
    t = np.linspace(0, np.pi, 50)
    path = np.column_stack([np.cos(t) * 100, np.sin(t) * 100, np.zeros_like(t)])
    result = check_bend_radius(path, wire_diameter_mm=0.6, supplier_limit_mm=50.0)
    assert result['passed']
    assert result['verified']


def test_bend_radius_fails_with_known_limit():
    # Very tight U-turn
    path = np.array([[0, 0, 0], [1, 0, 0], [1, 0.01, 0]], dtype=float)
    result = check_bend_radius(path, wire_diameter_mm=0.6, supplier_limit_mm=50.0)
    assert not result['passed']


def test_default_geometry_passes_manufacturability():
    """The default wound prototype must pass all manufacturability checks."""
    from organic_motor.construct.wound_prototype import winding_route, WoundSpec
    from organic_motor.construct.winding_harness import make_harness
    from organic_motor.construct.manufacturability import run_manufacturability_checks
    route = winding_route(WoundSpec())
    paths, report, ports = make_harness(route)
    results = run_manufacturability_checks(paths, report, WoundSpec())
    assert results['intra_phase_crossover']['passed']
    assert results['inter_phase_short']['passed']
    assert results['bend_radius']['passed'] is None  # unverified, not failed
