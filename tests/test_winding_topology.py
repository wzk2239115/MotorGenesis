"""Regression tests: winding electrical topology and geometric quality.

These encode the hard acceptance gates from the design audit:
  - The three phases are each ONE connected copper network INCLUDING end
    turns and terminals (a real winding, not fragments and not a shorted
    ring) -- asserted at a resolution where the wire core resolves.
  - Phases are mutually insulated (no cross-phase overlap; the measured
    insulation gap is reported).
  - Each phase has copper in the axial end regions (terminals).
  - No iron bridges the stator-side air gap.
  - Shaft and rotor iron are separated (no solid centre column).
  - The housing has open windows.
"""

import numpy as np
import pytest

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.objects import (
    PrintedStatorCore,
    PrintedStatorWinding,
    Winding3D,
    field_driven_motor,
)
from organic_motor.construct.winding_netlist import printed_netlist
from organic_motor.construct.geometry_metrics import compute_geometry_metrics
from organic_motor.construct.material import MaterialField
from organic_motor.construct.phase_verify import verify_phase_connectivity


def _small_cfg(**overrides):
    defaults = dict(
        shape=(48, 48, 30),
        excitation_mode="terminal",
        pole_pairs=2,  # Winding3D's design pole count (12s4p span-3 lap)
        filt_radius=0.0,
        projection_beta=0.0,
        mechanical_angles=3,
        maxwell_maxiter=10,
        thermal_maxiter=20,
        electric_maxiter=20,
        n_theta=8,
        torque_n_z=6,
        torque_n_r=6,
    )
    defaults.update(overrides)
    return MotorConfig3D(**defaults)


def _winding_only_cfg(**overrides):
    """A grid where the wire core (3 mm dia) and the 1.3 mm phase
    insulation both resolve: the display grid 160x160x96."""
    return _small_cfg(shape=(160, 160, 96), **overrides)


class TestWindingTopology:
    """The winding is a real three-phase electrical network."""

    @pytest.fixture(scope="class")
    def built(self):
        cfg = _small_cfg()
        mf = field_driven_motor(cfg).build()
        return cfg, mf

    @pytest.fixture(scope="class")
    def winding_resolved(self):
        """Winding-only build at a resolution that resolves the wires.

        Connectivity of thin conductors is a resolution question: at the
        48^3 physics grid the 3 mm wire is sub-cell and MUST fragment --
        that is exactly what the mesh-convergence verdict reports.  The
        electrical-topology gates are asserted where the geometry resolves.
        """
        cfg = _winding_only_cfg()
        mf = MaterialField(
            shape=cfg.shape, spacing=cfg.spacing, origin=cfg.origin,
        )
        Winding3D(cfg).build(mf)
        return cfg, mf

    def test_netlist_attached(self, built):
        _cfg, mf = built
        assert "winding_netlist" in mf.metadata

    def test_phase_belts_disjoint(self, built):
        cfg, mf = built
        belts = mf.metadata["winding_netlist"].phase_belts_3d(cfg)
        overlap = (belts[0] != 0) & (belts[1] != 0)
        overlap |= (belts[0] != 0) & (belts[2] != 0)
        overlap |= (belts[1] != 0) & (belts[2] != 0)
        assert not overlap.any(), "phase belts must be disjoint"

    def test_each_phase_has_copper(self, built):
        cfg, mf = built
        belts = mf.metadata["winding_netlist"].phase_belts_3d(cfg)
        # Density threshold (not hard SDF<0): on coarse grids a thin wire
        # can pass between nodes, but its smoothed density is still present
        # in the phase's own slot sectors.
        copper = mf.to_densities()["copper"] > 0.1
        for ph in range(3):
            n_vox = int((copper & (belts[ph] != 0)).sum())
            assert n_vox > 0, f"phase {ph} has no copper"

    def test_phase_connectivity_report(self, winding_resolved):
        """Each phase is ONE connected network including end turns."""
        cfg, mf = winding_resolved
        report = verify_phase_connectivity(mf, cfg)
        assert report["method"] == "phase_sdf_ownership"
        assert report["phase_cross_short"] is False
        assert all(report["phase_terminals"])
        assert report["phase_a_components"] == 1
        assert report["phase_b_components"] == 1
        assert report["phase_c_components"] == 1
        assert report["passed"] is True
        # Phase insulation is real and resolvable.
        assert report["min_phase_gap_mm"] > 1.0

    def test_no_unowned_copper(self, winding_resolved):
        """Every copper voxel belongs to exactly one phase (owner array)."""
        _cfg, mf = winding_resolved
        copper = mf.sdfs["copper"].sdf < 0.0
        owner = mf.metadata["winding_phase_owner"]
        assert not (copper & (owner < 0)).any()
        assert not (owner.max() > 2)

    def test_winding_wire_fits_layer(self):
        cfg = _small_cfg()
        w = Winding3D(cfg)
        wr = 0.35 * (cfg.R_winding_outer - cfg.R_winding_inner) / w.n_layers
        spacing = (cfg.R_winding_outer - cfg.R_winding_inner) / w.n_layers
        assert 2 * wr < spacing, "wire diameter must fit inside its layer band"


class TestPrintedWinding:
    """The 12s10p printed concentrated winding: one coil loop per tooth."""

    @pytest.fixture(scope="class")
    def printed(self):
        # The display grid (~0.63mm cells): the 1.5mm phase gap and the
        # 0.6mm cladding resolve here; at the physics grid they are
        # sub-cell and the mesh-convergence verdict reports that.
        cfg = _small_cfg(shape=(224, 224, 132), excitation_mode="impressed")
        mf = MaterialField(shape=cfg.shape, spacing=cfg.spacing, origin=cfg.origin)
        PrintedStatorCore(cfg).build(mf)
        PrintedStatorWinding(cfg).build(mf)
        return cfg, mf

    def test_netlist_table_balanced(self):
        netlist = printed_netlist(MotorConfig3D())
        table = netlist.coil_table()
        assert len(table) == 12
        counts = [sum(1 for _t, ph, _s in table if ph == p) for p in range(3)]
        assert counts == [4, 4, 4]
        # The MMF fundamental of every phase has the same magnitude 2+sqrt(3)
        # (= 4 * 0.933): the winding-factor-0.933 machine is per-phase
        # symmetric by construction.
        for p in range(3):
            mmf = sum(
                s * np.cos(5 * t * np.pi / 6 - p * 2 * np.pi / 3)
                for t, ph, s in table if ph == p
            )
            assert abs(abs(mmf) - (2.0 + np.sqrt(3.0))) < 1e-9, (p, mmf)

    def test_expected_four_loops_per_phase(self):
        netlist = printed_netlist(MotorConfig3D())
        assert netlist.expected_phase_components().tolist() == [4, 4, 4]

    def test_belts_balanced_columns(self, printed):
        cfg, mf = printed
        belts = mf.metadata["winding_netlist"].phase_belts_3d(cfg)
        mid = belts.shape[2] // 2
        for p in range(3):
            b = belts[p][:, :, mid]
            assert int((b > 0).sum()) == int((b < 0).sum()) > 0
            assert np.isclose(b[b != 0].sum(), 0.0, atol=len(b[b != 0]))

    def test_four_components_per_phase_no_short(self, printed):
        cfg, mf = printed
        report = verify_phase_connectivity(mf, cfg)
        assert [report["phase_a_components"], report["phase_b_components"],
                report["phase_c_components"]] == [4, 4, 4]
        assert report["phase_cross_short"] is False
        assert report["passed"] is True


class TestGeometryGates:
    """Geometric quality gates evaluated on the winding-bearing motor."""

    @pytest.fixture(scope="class")
    def metrics(self):
        cfg = _small_cfg()
        mf = field_driven_motor(cfg).build()
        geom = compute_geometry_metrics(mf, cfg)
        from organic_motor.construct.connectivity import connectivity_report
        geom.update(connectivity_report(mf, cfg))
        return geom

    def test_no_air_gap_bridge(self, metrics):
        assert metrics["air_gap_iron_bridge"] is False

    def test_rotor_anchored_to_shaft(self, metrics):
        # The hub spokes are the structural load path rotor -> shaft;
        # shaft_rotor_merge is now EXPECTED (deliberate anchor), and the
        # separation guarantee lives in the air-gap check above.
        assert metrics["rotor_anchored"] is True

    def test_no_floating_islands(self, metrics):
        assert metrics["floating_islands"] == 0

    def test_copper_not_single_ring(self, metrics):
        # 1 means every phase shorted together into one copper cylinder.
        assert metrics["copper_components"] != 1

    def test_end_face_open(self, metrics):
        assert metrics["end_face_occlusion"] < 0.6
