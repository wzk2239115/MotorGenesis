"""Tests for the balanced rotating stator-current model."""

import numpy as np
import pytest

from organic_motor.config import MotorConfig
from organic_motor.geometry.sdf import domain_masks
from organic_motor.physics.excitation import (
    synchronous_electrical_angle,
    three_phase_current_density,
)


def test_current_is_confined_to_winding_band():
    cfg = MotorConfig(N=64)
    jz = np.asarray(three_phase_current_density(0.3, cfg))
    winding = np.asarray(domain_masks(cfg)["winding"])
    assert np.allclose(jz[~winding], 0.0)
    assert np.max(np.abs(jz[winding])) <= cfg.current_density_peak * 1.001


def test_synchronous_phase_uses_pole_pairs():
    cfg = MotorConfig(pole_pairs=3)
    assert synchronous_electrical_angle(0.4, cfg) == pytest.approx(1.2)


def test_balanced_sheet_has_nearly_zero_net_current():
    cfg = MotorConfig(N=96)
    jz = np.asarray(three_phase_current_density(0.7, cfg))
    winding = np.asarray(domain_masks(cfg)["winding"])
    assert abs(jz[winding].mean()) < 0.01 * cfg.current_density_peak
