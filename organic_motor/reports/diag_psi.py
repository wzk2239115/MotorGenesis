"""Diagnose the psi inconsistency: print the full unit chain of both
the FEA flux extraction and the map nominal-current normalization."""

import numpy as np
import jax.numpy as jnp

from organic_motor.construct.model_artifact import ModelArtifact
from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.startup_validation import _logits_from_densities
from organic_motor.construct.transient_bridge import extract_fea_flux_linkage
from organic_motor.optimization.objective3d import forward3d_fields, _phase_belts
from organic_motor.experiments.motor3d_powered import _single_phase_current, compute_powered_maps, Powered3DSettings

ART = "organic_motor/out/assembly"

artifact = ModelArtifact.load(ART)
import inspect
valid = set(inspect.signature(MotorConfig3D.__init__).parameters) - {"self"}
kw = {"shape": tuple(artifact.shape)}
for k, v in artifact.config_dict.items():
    if k in valid:
        kw[k] = tuple(v) if isinstance(v, list) and len(v) == 3 else v
cfg = MotorConfig3D(**kw)
fields, mag = artifact.solver_fields(cfg)
reg = artifact.centerline_registry

print("=== CONFIG ===")
print(f"M_sat = {cfg.M_sat:.0f} A/m  (Br = {cfg.mu0*cfg.M_sat:.3f} T)")
print(f"current_per_turn = {cfg.current_density_peak*1e-6:.1f} A")
print(f"pole_pairs = {cfg.pole_pairs}")
print(f"_n_turns_per_cell = {getattr(cfg, '_n_turns_per_cell', 'NOT SET -> 1')}")

print(f"\n=== CENTERLINE REGISTRY ({len(reg)} entries) ===")
for i, entry in enumerate(reg):
    pts = entry["points"]
    print(f"  entry {i}: tooth={entry['tooth']} phase={entry['phase']} "
          f"pol={entry['polarity']} n_turns={entry['n_turns']} "
          f"n_pts={len(pts)} area={entry['cross_section_area']:.2e}")

# --- FEA flux at angle 0 ---
print("\n=== FEA FLUX AT ANGLE 0 (PM only) ===")
r0 = forward3d_fields(cfg, fields, mag, [0.0], None,
                      phase_amplitudes=jnp.zeros(3),
                      centerline_registry=reg)
A_vec = np.asarray(r0.vector_potential)
print(f"A field: max|A| = {np.max(np.abs(A_vec)):.4f} Wb/m")

for i, entry in enumerate(reg):
    pts = entry["points"]
    phase = entry["phase"]
    polarity = entry["polarity"]
    flux = 0.0
    for seg in range(len(pts)):
        p1 = pts[seg]
        p2 = pts[(seg + 1) % len(pts)]
        dl = p2 - p1
        mid = 0.5 * (p1 + p2)
        idx = ((mid - cfg.origin) / cfg.spacing).astype(int)
        ix = int(np.clip(idx[0], 0, cfg.shape[0]-1))
        iy = int(np.clip(idx[1], 0, cfg.shape[1]-1))
        iz = int(np.clip(idx[2], 0, cfg.shape[2]-1))
        A_mid = A_vec[ix, iy, iz, :]
        flux += float(np.dot(A_mid, dl))
    print(f"  tooth {entry['tooth']} (phase {phase}, pol {polarity}): "
          f"flux = {polarity*flux:.6f} Wb  ({flux:.6f} raw)")

# --- Full flux extraction ---
print("\n=== FULL FEA FLUX LINKAGE ===")
flux_fea = extract_fea_flux_linkage(None, cfg, artifact.magnetization,
                                    fields=fields, registry=reg)
print(f"psi_FEA = {flux_fea:.6f} Wb")

# --- Map nominal current ---
print("\n=== MAP NOMINAL CURRENT ===")
full = _phase_belts(cfg)
zero = jnp.zeros_like(full[0])
singles = [jnp.stack([full[p] if q == p else zero for q in range(3)])
           for p in range(3)]
one = jnp.asarray([1.0, 0.0, 0.0])
r_plus = forward3d_fields(cfg, fields, mag, [0.0], singles[0],
                          phase_amplitudes=jnp.roll(one, 0),
                          centerline_registry=reg)
raw_current = float(_single_phase_current(r_plus, 0, cfg))
print(f"raw _single_phase_current (phase 0) = {raw_current:.2f} A")
n_series = 12 // 3  # = 4
n_turns_per_cell = getattr(cfg, "_n_turns_per_cell", 1)
n_series_total = n_series * max(1, n_turns_per_cell)
print(f"n_series = {n_series} x n_turns_per_cell={n_turns_per_cell} = {n_series_total}")
print(f"nominal = {raw_current}/{n_series_total} = {raw_current/n_series_total:.2f} A")

# --- Cross-check: psi from map ---
print("\n=== PSI FROM MAP ===")
# T1 at unit excitation ~ 0.046 Nm (from verify)
t1_amp = 0.046  # approximate
I_nom = raw_current / n_series_total
psi_map = t1_amp / (1.5 * cfg.pole_pairs * I_nom)
print(f"T1_amp ~ {t1_amp:.4f} Nm at i_norm=1 (terminal={I_nom:.2f} A)")
print(f"psi_map = T1/(1.5*p*I_nom) = {t1_amp}/{1.5*cfg.pole_pairs*I_nom:.1f} = {psi_map:.6f} Wb")
print(f"\n=== RATIO ===")
print(f"psi_FEA / psi_map = {flux_fea/psi_map:.1f}")
print(f"psi_FEA / psi_physical(~0.009) = {flux_fea/0.009:.1f}")
print(f"psi_map / psi_physical(~0.009) = {psi_map/0.009:.1f}")

# --- What if _n_turns_per_cell = 7? ---
print("\n=== IF _n_turns_per_cell = 7 ===")
n_series_7 = n_series * 7
nominal_7 = raw_current / n_series_7
psi_map_7 = t1_amp / (1.5 * cfg.pole_pairs * nominal_7)
print(f"nominal = {raw_current}/{n_series_7} = {nominal_7:.2f} A")
print(f"psi_map = {psi_map_7:.6f} Wb")
print(f"psi_FEA / psi_map(7) = {flux_fea/psi_map_7:.1f}")
