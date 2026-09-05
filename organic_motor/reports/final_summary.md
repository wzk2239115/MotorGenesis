# Final Summary Report — Direction Convention Fix

**Date**: 2026-09-06
**Commits**: `9919d70` (fix) → `90f48be` (validation + morphology)

---

## Root Cause

`objects.py:1593` centerline rotation matrix was transposed:
```python
# WRONG (was): rotates by -θ
pts[:, 0:2] = pts[:, 0:2] @ np.array([[ca, -sa], [sa, ca]])
# CORRECT (now): rotates by +θ
pts[:, 0:2] = pts[:, 0:2] @ np.array([[ca, sa], [-sa, ca]])
```

This placed all 12 copper centerlines at -n×pitch instead of +n×pitch.
Iron teeth were at +n×pitch (build_iron correct). Copper was on the
wrong side of each tooth → stator field rotated opposite to rotor →
zero average torque.

## Evidence Chain

### 1. Phase sweep (pre-fix)
| sign | δ | T_odd mean |
|------|---|------------|
| +1 | any | ≈ 0 |
| -1 | 90° | +0.076 Nm |

### 2. Direction audit
- Rotor: +mech_angle → +θ rotation ✓
- Iron teeth: at +n×pitch ✓
- Copper centerlines: at -n×pitch ✗ (rotation matrix transposed)

### 3. Post-fix ablation
| Test | Before fix | After fix |
|------|-----------|-----------|
| θ=0 torque (mi=240) | -0.0194 Nm | **+0.0213 Nm** |
| 6-angle mean | +0.0007 | **+0.0348** |
| 12-angle mean | +0.0010 | **+0.0206** |
| 24-angle mean | -0.0009 | **+0.0189** |
| 96³→128³ torque | sign flip | **both positive** |

### 4. Reference coil validation
| Check | Result | Expected |
|-------|--------|----------|
| B_z magnitude (z=10mm) | -1.2% error | ✓ |
| Current reversal | ratio=-1.000 | ✓ |
| Current scaling | ratio=2.000 | ✓ |

## What Was NOT Done (and Why)

- **No hardcoded sign=-1 or δ=90°** — fix is in the geometry rotation
- **No 224³ as default physics grid** — only used as display/audit grid
- **No thermal-flow models added** — physics validation first
- **No surrogate model trained** — needs verified data first
- **No morphology optimization** — generators provided but not yet integrated

## Current Model Capabilities

### Can support (verified):
- Winding topology (224³ display): 4+4+4 components ✓
- Structural connectivity: rotor/stator anchored ✓
- Manufacturing: min wall 1.26mm, no powder pockets ✓
- Copper loss: 3.56W (analytical, grid-independent) ✓
- DDA current conservation: <1e-8 ✓
- Phase resistance: 0.071 Ω (analytical) ✓
- **Average torque: +0.019 Nm (default excitation)** ✓
- Current deposit direction: validated via reference coil ✓

### Still limited:
- Torque magnitude not grid-converged (0.021→0.024)
- Solver residual grid-dependent (2.9e-4→9.5e-4 at mi=240)
- Thermal residual not converged at 96³
- No cooling channels (build_coolant disabled)
- No electromechanical startup transient in scoring loop

## New Capabilities

### Morphology generators (morphology.py):
- `HoneycombGenerator`: hexagonal strut lattice, parametric cell_size/wall_thickness
- `HelicalChannelGenerator`: helical cooling, parametric pitch/turns/handedness
- `BranchingManifold`: Y-junction coolant distribution
- 6 tests validating parameter sensitivity

### Integration tests:
- 125 total (was 105 at start of session)
- Agent loop: propose→build→score→verdict→select proven end-to-end
- Winding direction regression: teeth at +θ, torque > 0.01 Nm
- Reference coil: Biot-Savart analytical comparison

## Next Steps (Expert's 6-step plan)

1. ✅ Real closed-loop + integration tests
2. ✅ Frozen baseline diagnostic
3. ✅ Resolution study (holes are artifacts)
4. ✅ EM model credibility (root cause found and fixed)
5. ⬜ Thermal-flow + rotation (simplified models)
6. ⬜ Morphology optimization (generators ready, need integration)
