"""Electromechanical and thermal transient primitive tests."""

import numpy as np

from organic_motor.physics.transient3d import (
    RotorState,
    ThreePhaseState,
    Transient3DState,
    advance_transient_step,
    advance_voxel_temperature,
    transient_joule_loss,
)


def test_transient_joule_loss_preserves_xyz_volume_and_scales_quadratically():
    current = np.zeros((3, 4, 5, 3))
    current[..., 0] = 2.0
    current[..., 2] = -1.0
    loss = transient_joule_loss(current, np.full((3, 4, 5), 5.0))

    assert loss.shape == (3, 4, 5)
    assert np.allclose(np.asarray(loss), 1.0, atol=2e-6)
    doubled = transient_joule_loss(2.0 * current, np.full((3, 4, 5), 5.0))
    assert np.allclose(np.asarray(doubled), 4.0, atol=1e-5)


def test_voxel_temperature_diffuses_along_z_and_conserves_insulated_heat():
    temperature = np.zeros((3, 3, 5))
    temperature[:, :, 2] = 1.0
    advanced = advance_voxel_temperature(
        temperature,
        heat_density=0.0,
        conductivity=1.0,
        volumetric_heat_capacity=1.0,
        spacing=(1.0, 1.0, 1.0),
        dt=0.1,
    )
    values = np.asarray(advanced)

    assert values.shape == temperature.shape
    assert values[:, :, 1].mean() > 0.0
    assert values[:, :, 3].mean() > 0.0
    assert values[:, :, 2].mean() < 1.0
    np.testing.assert_allclose(
        values.sum(), temperature.sum(), rtol=2e-5, atol=2e-5
    )


def test_composed_transient_step_updates_rotor_circuit_and_temperature():
    state = Transient3DState(
        rotor=RotorState(angle=np.asarray(0.0), angular_velocity=np.asarray(0.0)),
        circuit=ThreePhaseState(currents=np.zeros(3)),
        temperature=np.asarray(25.0),
    )
    next_state = advance_transient_step(
        state,
        phase_voltage=np.array([1.0, -0.5, -0.5]),
        electromagnetic_torque=np.asarray(2.0),
        heat_power=np.asarray(10.0),
        dt=0.1,
        pole_pairs=2,
        flux_linkage=0.1,
        phase_resistance=1.0,
        phase_inductance=0.5,
        rotor_inertia=1.0,
        ambient_temperature=25.0,
        thermal_capacity=10.0,
        thermal_conductance=1.0,
    )

    assert float(next_state.rotor.angular_velocity) > 0.0
    assert float(next_state.rotor.angle) > 0.0
    assert np.isclose(np.asarray(next_state.circuit.currents).sum(), 0.0, atol=2e-6)
    assert float(next_state.temperature) > 25.0
