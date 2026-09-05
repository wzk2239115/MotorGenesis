# EM Error Separation Report (Corrected)

**Date**: 2026-09-05
**Commit**: to be tagged after commit
**Script**: `organic_motor/reports/ablation/run_ablation.py`
**Raw data**: `ablation_e1_e3.json`, `ablation_e4_e5.json`, `ablation_e6.json`

**Config** (auto-generated from MotorConfig3D):
- shape=(96,96,58), spacing=(1.474, 1.474, 1.754)mm
- origin=(-70, -70, -50)mm, node_extent=(140.0, 140.0, 99.8)mm
- pole_pairs=5, excitation_mode="impressed"
- proper 3-phase: [cos(ea), cos(ea-2π/3), cos(ea-4π/3)]

---

## E1: Baseline Torque Discrepancy — EXPLAINED

| Phase amplitudes | Torque (Nm) | Note |
|-----------------|-------------|------|
| default cos at ea=0 = [1, -0.5, -0.5] | -0.019489 | proper 3-phase |
| explicit [1, -0.5, -0.5] | -0.019489 | matches default ✓ |
| explicit [1, 1, 1] | -0.006322 | NOT proper 3-phase |

**Root cause**: Previous Ablation 3 used `phase_amplitudes=[1,1,1]` which applies equal current to all phases — this is not a physical 3-phase excitation. The correct excitation uses `phase_amplitudes=None` (default cos) which gives [1,-0.5,-0.5] at ea=0.

---

## E2: Solver Convergence (fixed 96×96×58, θ=0, [1,-0.5,-0.5])

| maxiter | Torque (Nm) | residual | B_max | Δ from prev |
|---------|-------------|----------|-------|------------|
| 5 | -0.004567 | 3.32e-01 | 1.62 | — |
| 10 | -0.009687 | 1.21e-01 | 1.65 | 112% |
| 20 | -0.017267 | 4.64e-02 | 1.96 | 78% |
| 40 | -0.021893 | 1.82e-02 | 1.74 | 27% |
| 60 | -0.022237 | 9.02e-03 | 1.75 | 1.6% |
| 120 | -0.019489 | 2.35e-03 | 1.40 | 12.4% |
| 240 | -0.019375 | 2.83e-04 | 1.38 | 0.6% |

**Findings**:
- Torque converges to ~-0.0194 at maxiter≥240
- At maxiter=60 (our default): 14% error
- B_max also converges: 1.62→1.38 (settles at ~1.38)
- **"maxiter=240" is not a universal convergence standard** — residual still 2.8e-4, and different grids reach different residuals (see E6)

---

## E3: Integration Sampling (maxiter=240)

| n_z × n_r | Torque (Nm) |
|-----------|-------------|
| 4×4 | -0.021371 |
| 8×8 | -0.021470 |
| 16×16 | -0.019375 |
| 32×32 | -0.020339 |

**Range**: -0.0194 to -0.0215, **10.5% variation**

Previous report claimed <7% — corrected. This is larger than typical design improvements (few %), so integration sampling error is significant. Need to also check circumferential sampling and integration surface position.

---

## E4: Per-Angle Decomposition (maxiter=240, 6 angles)

| θ (deg) | T(+I) | T(-I) | T(0) | T_odd | T_even |
|---------|-------|-------|------|-------|--------|
| 0 | -0.0194 | +0.0218 | +0.0012 | -0.0206 | +0.0012 |
| 12 | +0.0748 | -0.0787 | -0.0019 | +0.0767 | -0.0019 |
| 24 | -0.0436 | +0.0659 | +0.0112 | -0.0548 | +0.0112 |
| 36 | -0.0173 | +0.0218 | +0.0022 | -0.0196 | +0.0023 |
| 48 | +0.1061 | -0.0484 | +0.0289 | +0.0773 | +0.0289 |
| 60 | -0.0079 | +0.1016 | +0.0469 | -0.0548 | +0.0468 |

| Component | Mean | Std | Range |
|-----------|------|-----|-------|
| T_odd (current) | **+0.0007** | 0.0558 | [-0.055, +0.077] |
| T_zero (cogging) | +0.0147 | 0.0176 | [-0.002, +0.047] |

**Critical finding**: T_odd mean ≈ **zero**. The current-dependent torque averages to approximately zero over the electrical cycle. This means the winding current is NOT producing useful average torque.

T_even = T_zero at all angles — confirms torque = T_cogging + T_odd(I).

---

## E5: Angle Refinement (maxiter=240)

| n_angles | Mean torque | Std | Range |
|----------|-------------|-----|-------|
| 3 | +0.0144 | 0.066 | [-0.044, +0.106] |
| 6 | +0.0155 | 0.055 | [-0.044, +0.106] |
| 12 | +0.0010 | 0.061 | [-0.119, +0.106] |
| 24 | -0.0009 | 0.058 | [-0.119, +0.106] |

**Finding**: Mean torque → 0 as n_angles increases. With 12-24 angles, mean ≈ 0. The earlier positive mean (+0.014) was from insufficient angle sampling (3-6 angles can't average out the 150% variation).

---

## E6: Grid Refinement (maxiter=240, θ=0, [1,-0.5,-0.5])

| Grid | Spacing (mm) | Torque (Nm) | residual | B_max |
|------|-------------|-------------|----------|-------|
| 96×96×58 | (1.47, 1.47, 1.75) | -0.0194 | 2.8e-4 | 1.38 |
| 128×128×78 | (1.10, 1.10, 1.30) | -0.0229 | 9.4e-4 | 2.03 |
| 160×160×96 | (0.88, 0.88, 1.05) | -0.0277 | 1.9e-3 | 2.96 |

**Findings**:
- Torque magnitude increases with grid (-0.019 → -0.023 → -0.028)
- **B_max increases dramatically** (1.38 → 2.03 → 2.96) — field not converging
- **Residual INCREASES** at finer grids (2.8e-4 → 9.4e-4 → 1.9e-3) — same maxiter gives worse convergence at higher resolution
- "Fixed 240 iterations" does NOT give same precision across grids

---

## Summary: Error Source Hierarchy (Corrected)

| # | Factor | Effect | Status |
|---|--------|--------|--------|
| 1 | 铜碎片化 | 0.0% (E1 ablation) | **否定** |
| 2 | 求解器收敛 | 14% at mi=60, <1% at mi=240 | 受控 but grid-dependent |
| 3 | 积分采样 | 10.5% variation | **需要更多检查** |
| 4 | 角度采样 | mean ±0.015 with 3-6, →0 with 12+ | **主要问题** |
| 5 | 网格收敛 | magnitude -0.019→-0.028, B_max 1.4→3.0 | **未收敛** |
| 6 | **T_odd mean ≈ 0** | **电流不产生平均转矩** | **待调查根因** |

## Key Open Question

**T_odd mean ≈ 0**: 电流相关转矩在完整电周期内平均为零。这可能意味着：
1. 绕组配置不正确（极性/相序错误）
2. 电流相位与转子位置关系错误
3. 中心线沉积的电流方向有误

这不是铜碎片化问题——需要检查绕组网表、极性分配和电流沉积方向。

## Next Steps

1. 检查 T_odd ≈ 0 的根因：绕组极性、相序、电流方向
2. 用独立参考线圈算例验证电流沉积（E6 尚未完成）
3. 固定积分采样误差在可接受范围内
4. 224³ 只作为对照实验——不预设为可信基准
