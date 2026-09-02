"""Phase connectivity verification for constructed motor windings.

Verifies the electrical topology of a constructed winding:
  - Each phase (A, B, C) is internally connected (one component)
  - Phases are mutually insulated (no short between A/B/C)
  - Each phase has at least one terminal

This is the first validation gate before claiming a motor can rotate:
"rho_copper exists" does not prove "current can flow along a closed path."
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.material import MaterialField
from organic_motor.construct.winding_netlist import CoilNetlist
from organic_motor.optimization.objective3d import phase_terminal_masks3d


def verify_phase_connectivity(
    mf: MaterialField, cfg: MotorConfig3D, threshold: float = 0.3,
) -> dict:
    """Verify A/B/C phase connectivity and mutual insulation.

    Returns a dict with:
      - phase_a_components, phase_b_components, phase_c_components
      - phase_cross_short (True if any voxels are shared between phases)
      - phase_terminals (3, bool: each phase has at least one terminal voxel)
      - passed (all phases connected, no shorts)
    """
    netlist = mf.metadata.get("winding_netlist") if hasattr(mf, "metadata") else None
    if not isinstance(netlist, CoilNetlist):
        return {
            "phase_a_components": -1, "phase_b_components": -1, "phase_c_components": -1,
            "phase_cross_short": False, "phase_terminals": [False] * 3,
            "passed": False, "reason": "no winding netlist in metadata",
        }

    densities = mf.to_densities()
    copper_sdf = mf.sdfs.get("copper")
    belts = netlist.phase_belts_3d(cfg)
    structure = ndimage.generate_binary_structure(3, 1)

    phase_names = ["a", "b", "c"]
    result = {}
    phase_masks = []
    for i, name in enumerate(phase_names):
        belt = belts[i] != 0
        if copper_sdf is not None:
            phase_copper = (copper_sdf.sdf < 0.0) & belt
        else:
            phase_copper = (densities["copper"] > threshold) & belt
        phase_masks.append(phase_copper)
        if not phase_copper.any():
            result[f"phase_{name}_components"] = 0
            result[f"phase_{name}_voxels"] = 0
            continue
        labels, n_comp = ndimage.label(phase_copper, structure=structure)
        sizes = ndimage.sum(phase_copper, labels, range(1, n_comp + 1))
        result[f"phase_{name}_components"] = n_comp
        result[f"phase_{name}_voxels"] = int(phase_copper.sum())
        result[f"phase_{name}_largest_fraction"] = float(max(sizes)) / float(phase_copper.sum())

    cross_short = False
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = phase_masks[i] & phase_masks[j]
            if overlap.any():
                cross_short = True
                result[f"overlap_{phase_names[i]}_{phase_names[j]}"] = int(overlap.sum())
    result["phase_cross_short"] = cross_short

    import jax.numpy as jnp
    belts_jnp = jnp.asarray(belts)
    terminals = np.asarray(phase_terminal_masks3d(cfg, belts_jnp))
    for i, name in enumerate(phase_names):
        result[f"phase_{name}_has_terminal"] = bool(terminals[i].any())
    result["phase_terminals"] = [result[f"phase_{p}_has_terminal"] for p in phase_names]

    # Each phase legitimately splits into up to n_layers parallel layer
    # paths (layers are radially insulated by design); more than that means
    # fragmentation, fewer than one means disconnection.
    max_components = max(1, netlist.n_layers)
    all_connected = all(
        1 <= result.get(f"phase_{n}_components", 0) <= max_components
        for n in phase_names
    )
    has_terminals = all(result.get(f"phase_{n}_has_terminal", False) for n in phase_names)
    result["passed"] = all_connected and not cross_short and has_terminals
    return result
