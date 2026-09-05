# Step 4: EM Model Credibility — Sensitivity and Reference Checks

**Commit**: `8200a21`
**Date**: 2026-09-05

---

## 1. Grid Sensitivity (same geometry, different physics grids)

| Grid | Spacing | Torque (Nm) | cu_loss (W) | iron_loss (W) | max_res | th_res | div_res |
|------|---------|-------------|------------|---------------|---------|--------|---------|
| 96³ | 1.47mm | **+0.00488** | 3.558 | 0.734 | 0.011 | 3.16 | 1.5e-8 |
| 128³ | 1.10mm | **-0.00633** | 3.558 | 0.854 | 0.016 | 0.57 | 1.1e-8 |
| 160³ | 0.88mm | **-0.00415** | 3.558 | 1.096 | 0.015 | 0.57 | 7.9e-9 |

### Findings:
- **Torque SIGN FLIPS** between 96³ (+0.005) and 128³/160³ (-0.006/-0.004). **Not grid-convergent.**
- **Copper loss is identical** (3.558W) — analytical from centerline, grid-independent. ✓
- **Iron loss increases** with resolution (0.73→0.85→1.10) — not converged.
- **Maxwell residual** doesn't improve (0.011→0.016→0.015) — solver not converging better at higher res.
- **Thermal residual** improves from 96³ (3.16) to 128³+ (0.57) — 96³ needs more iterations.
- **DDA divergence** excellent at all grids (1e-8). ✓

**Conclusion**: Torque is NOT reliable at any tested resolution. Copper loss and DDA conservation ARE reliable.

---

## 2. Angle Sampling Sensitivity (96³)

| m_angles | n_theta | Torque (Nm) | Ripple |
|----------|---------|-------------|--------|
| 1 | 8 | -0.0175 | 0.0001 |
| 1 | 16 | -0.0184 | 0.0001 |
| 1 | 32 | -0.0194 | 0.0001 |
| 3 | 8 | +0.0022 | 32.9 |
| 3 | 16 | +0.0049 | 9.0 |
| 3 | 32 | +0.0118 | 5.2 |
| 6 | 8 | +0.0012 | 46.1 |
| 6 | 16 | -0.0028 | 20.1 |
| 6 | 32 | +0.0128 | 4.1 |

### Findings:
- **Single angle**: torque stable at ~-0.018 Nm across n_theta. This is cogging at one position.
- **Multi-angle**: torque varies wildly (+0.002 to +0.013). Average is not stable.
- **Torque ripple** decreases with more n_theta but mean torque is unstable.

**Conclusion**: Torque averaging over 3-6 angles at 96³ is unreliable. The torque is a small difference of large stress fields, dominated by discretization error at this resolution.

---

## 3. Current Source vs Visible Winding

| Item | Value | Assessment |
|------|-------|------------|
| Centerline entries | 12 (4 per phase) | ✓ correct |
| Turns per phase | 28 (4×7) | ✓ correct |
| Cross-section area | 1.13e-6 m² (π×0.6²) | ✓ correct |
| Phase R (analytical) | 0.0711 Ω | ✓ grid-independent |
| Physical path length | 4785.6 mm per phase | ✓ |
| Virtual closure | 5 extra points per entry | solver-only |
| solver_closure flag | True | DDA conservation |
| Current on copper voxels | 14.7% | by design (line source) |
| Current off copper | 85.3% | centerline passes through air voxels at 96³ |

### Findings:
- The line current is a **mathematical source** deposited along the centerline, not a volume current in copper voxels.
- At 96³, copper is fragmented (440 components), so most centerline points fall in air voxels.
- The current source is **correct by design** (DDA conservation, correct ampere-turns), but the solver can't properly model the electromagnetic interaction because copper isn't resolved.
- The virtual closure (5 extra points) serves only DDA conservation — it's not in the physical copper geometry.

---

## 4. Reference Cross-Check

### Analytical checks (grid-independent):
- **Per-phase turns**: N = 28 ✓ (4 coils × 7 turns)
- **Winding factor**: kw = 0.933 ✓ (frozen MotorSpec)
- **Phase resistance**: R = 0.0711 Ω ✓ (ρ×L/A from centerline)
- **Cross-section**: A = 1.13e-6 m² ✓ (π×r²)
- **DDA divergence**: < 1e-8 ✓ (face-flux conservation)
- **Phase balance**: < 1e-9 ✓

### NOT verified:
- **Torque**: no reliable reference (sign flips with grid)
- **Iron loss**: not converged (0.73→1.10)
- **Air-gap flux density**: not extracted
- **Inductance**: analytical gap formula only, not FEA-extracted

---

## 5. What the Model CAN and CANNOT Support

### CAN support (at 96³ physics + 224³ display):
- ✓ Winding topology (on display grid)
- ✓ Structural connectivity (on display grid)
- ✓ Manufacturing constraints (wall, powder, overhang)
- ✓ Copper loss (analytical from centerline, grid-independent)
- ✓ DDA current conservation (1e-8)
- ✓ Phase balance (1e-9)
- ✓ Phase resistance (0.071 Ω, analytical)
- ✓ Material distribution (qualitative)

### CANNOT support (at current resolution):
- ✗ Quantitative torque (sign flips, unstable with angle sampling)
- ✗ Iron loss (not converged)
- ✗ Mesh convergence (physics grid fragments copper)
- ✗ Thermal convergence at 96³ (residual 3.16)
- ✗ Air-gap flux density (not extracted)
- ✗ FEA inductance (analytical only)

### Root cause:
The 0.6mm copper bands are sub-voxel at 96³ (spacing 1.47mm). The line current source is correct, but the solver cannot properly model the electromagnetic interaction because the copper material is fragmented into 440 pieces. The torque, which depends on the interaction between current and iron/PM fields, is dominated by discretization error.

### Fix direction:
1. **Short-term**: Use 224³ physics grid (15s build + ~60s solve = manageable)
2. **Medium-term**: Implement homogenized material properties for sub-voxel copper
3. **Long-term**: Adaptive mesh refinement near copper bands
