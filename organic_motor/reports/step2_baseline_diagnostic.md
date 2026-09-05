# Step 2: Frozen Baseline Diagnostic Report

**Commit**: `981d296` (corrected `f36935e`)
**Date**: 2026-09-05
**Config**: MotorConfig3D(shape=(96,96,58), excitation_mode="impressed", filt_radius=0.0, projection_beta=0.0)
  - Physics grid: spacing=(1.474, 1.474, 1.754)mm, origin=(-70,-70,-50)mm, domain=(141.5,141.5,101.8)mm
  - Display grid: shape=(224,224,136), spacing=(0.628, 0.628, 0.741)mm
  - Solver: maxwell_maxiter=60, thermal_maxiter=60, electric_maxiter=60, mechanical_angles=3, n_theta=16
  - Code: BASELINE_CODE (field_driven_motor, n_bands=7, arch_slope=1.0, channel_wall=0.0003)

---

## 1. WINDING — PASS

| Item | Value | Status |
|------|-------|--------|
| Phase A components | 4 | ✓ (expected 4) |
| Phase B components | 4 | ✓ (expected 4) |
| Phase C components | 4 | ✓ (expected 4) |
| Phase A has terminal | True | ✓ |
| Phase B has terminal | True | ✓ |
| Phase C has terminal | True | ✓ |
| Cross-phase short | False | ✓ |
| Min phase gap | 2.59 mm | ✓ |
| Phase A voxels | 20436 | — |
| Phase B voxels | 20446 | — |
| Phase C voxels | 20014 | — |

**结论**: 每相4个连通分量（4线圈串联），均有端子，无相间短路。绕组拓扑正确。

---

## 2. STRUCTURE — PASS

| Item | Value | Status |
|------|-------|--------|
| Structural components | 2 | ✓ (rotor + stator) |
| Floating islands | 0 | ✓ |
| Rotor anchored | True | ✓ |
| Stator anchored | True | ✓ |
| Rotor-stator cross bridge | False | ✓ |
| Air gap solid bridge | False | ✓ |
| Min neck | 1.26 mm | ✓ (> 0.4mm) |
| Anchored fraction | 1.0 | ✓ |

**结论**: 转子→轴、定子→机壳承载路径完整，无悬浮结构。

---

## 3. COOLING — NOT EVALUATED (None)

| Item | Value | Status |
|------|-------|--------|
| Dedicated coolant | False | ✗ no coolant material |
| Coolant components | 1 | (air void, not real coolant) |
| Through flow networks | 1 | (void network, not coolant) |
| Trapped voids | 0 | — |

**根因**: `build_coolant` 在 `objects.py:1715` 硬编码 `return mf`，冷却通道被显式禁用。`channel_wall=0.3mm < band_radius=0.6mm`，条件检查通过但代码直接返回。
**修复方向**: 实现 `build_coolant` 中的通道构建逻辑，或在外部结构中实现共形冷却。

---

## 4. GEOMETRY

| Item | Value | Status |
|------|-------|--------|
| Copper components (224³) | 12 | ✓ (4×3) |
| Copper min gap | 2.59 mm | ✓ |
| Air gap iron bridge | False | ✓ |
| Shaft-rotor merge | True | ✓ |
| End face occlusion | 0.0 | ✓ |
| Min neck | 1.26 mm | ✓ |
| Insulator voxels | 5296 | present (treated as air in solver) |
| Sleeve | in iron SDF | present (non-magnetic via metadata) |

**材料清单**:
- Solver-facing: iron, copper, pm
- Aux: insulator (5296 vox), air (8352 vox)
- Sleeve: structural iron, non-magnetic (realize() subtracts from rho_iron)

**问题**: 224³ display 能解析铜带，但 96³ physics 碎片化严重（440 vs 12 components）。

---

## 5. SOLVER

| Metric | Value | Assessment |
|--------|-------|------------|
| excitation_mode | impressed | line-current deposit |
| maxwell_residual | 0.0113 | partially converged |
| thermal_residual | 3.157 | **not converged** |
| electric_residual | 0.0 | N/A (impressed, no solve) |
| source_divergence | 1.5e-8 | excellent (DDA conservation) |
| phase_balance | 5.5e-10 | excellent |
| mechanical_angles | 3 | — |
| n_theta | 16 | — |

| Output | Value | Assessment |
|--------|-------|------------|
| torque | 0.00488 Nm | **very low** — copper fragmented at 96³ |
| torque_ripple | 9.05 | meaningless (torque ≈ 0) |
| copper_loss | 3.56 W | reasonable |
| iron_loss | 0.73 W | reasonable |
| loss_total | 4.29 W | — |
| temperature_max | 28.4°C | low (small losses, no cooling needed) |
| vol_iron | 0.70 | 70% domain |
| vol_copper | 0.07 | 7% |
| vol_pm | 0.05 | 5% |
| obj | 2.137 | — |

**问题**:
1. 热残差 3.157 未收敛 — 需要 more iterations
2. 转矩 0.00488 Nm 极低 — 96³ 物理网格铜碎片化导致有效电流密度错误
3. 未执行的验收项：electromechanical (startup transient)、cooling (no coolant)、mesh convergence (physics grid fragments)

---

## 6. FAILURE ITEMS SUMMARY

| # | Item | Root Cause | Fix Direction |
|---|------|-----------|---------------|
| F1 | mesh_convergence FAIL | Hypothesis: 96³ fragments copper (440 vs 12) —待验证 | Sub-voxel features need local refinement or higher physics grid |
| F2 | cooling None | build_coolant hardcoded return (objects.py:1715) | Implement channel construction or external conformal cooling |
| F3 | thermal residual 3.16 | Not converged — may need more iterations or better preconditioner | Increase thermal_maxiter, investigate operator/boundary |
| F4 | torque 0.005 Nm | Low value — 待验证是否由铜碎片化或电流沉积/求解问题导致 | Ablation experiment (Step 4 corrected) |
| F5 | electromechanical None | Startup transient not run in scoring | Separate validation step |

---

## 7. WHAT THE MODEL CAN/CANNOT SUPPORT

**Can support** (at 96³ physics + 224³ display):
- Winding topology verification (on display grid)
- Structural connectivity (on display grid)
- Manufacturing constraints (wall, powder, overhang)
- DDA current conservation (source_divergence 1.5e-8)
- Phase balance (5.5e-10)
- Approximate torque/loss (order of magnitude)

**Cannot support** (at current resolution):
- Quantitative torque (copper fragmented at physics grid)
- Mesh convergence (physics vs display topology mismatch)
- Thermal convergence (residual 3.16)
- Cooling network (no coolant material)
- Electromechanical startup (not run during scoring)

**Next**: Step 3 — single stator cell fine geometry at 3 resolution levels to determine whether visual holes are real structure or sampling artifacts.
