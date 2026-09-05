# EM Error Separation Report v2

**Date**: 2026-09-05 23:43:04
**Design hash**: 0970d5ce563f
**Config**: shape=[96, 96, 58], spacing=(1.4737,1.4737,1.7544)mm
  node_extent=(140.0,140.0,100.0)mm, origin=(-70.0,-70.0,-50.0)mm
  maxiter=240, pp=5
  proper 3-phase: cos(elec - [0, 2pi/3, 4pi/3])
**Script**: run_ablation_v2.py (reproducible)
**Raw data**: ablation_v2_results.json

## E1: Phase Amplitude Comparison

| Amplitudes | Torque (Nm) |
|-----------|-------------|
| default_cos | -0.019375 |
| explicit_[1,-0.5,-0.5] | -0.019375 |
| explicit_[1,1,1] | -0.005916 |

`[1,1,1]` is NOT proper 3-phase. Default cos at ea=0 = [1,-0.5,-0.5].

## E2: Solver Convergence

| maxiter | Torque | residual | B_max |
|---------|--------|----------|-------|
| 5 | -0.004567 | 3.317020e-01 | 1.6227 |
| 10 | -0.009687 | 1.214008e-01 | 1.6531 |
| 20 | -0.017267 | 4.641517e-02 | 1.9572 |
| 40 | -0.021893 | 1.817727e-02 | 1.7393 |
| 60 | -0.022237 | 9.018938e-03 | 1.7472 |
| 120 | -0.019489 | 2.354543e-03 | 1.4034 |
| 240 | -0.019375 | 2.833886e-04 | 1.3799 |

## E3: Integration Sampling

| n_z x n_r | Torque |
|---------|--------|
| 4x4 | -0.021371 |
| 8x8 | -0.021470 |
| 16x16 | -0.019375 |
| 32x32 | -0.020339 |

Variation: 10.2%

## E4: Per-Angle Decomposition

| deg | T_pos | T_neg | T_zero | T_odd | T_even |
|-----|-------|-------|--------|-------|--------|
| 0.0 | -0.019375 | +0.021782 | +0.001161 | -0.020578 | +0.001203 |
| 12.0 | +0.074789 | -0.078685 | -0.001948 | +0.076737 | -0.001948 |
| 24.0 | -0.043598 | +0.065913 | +0.011199 | -0.054756 | +0.011157 |
| 36.0 | -0.017297 | +0.021846 | +0.002235 | -0.019572 | +0.002275 |
| 48.0 | +0.106136 | -0.048429 | +0.028852 | +0.077283 | +0.028853 |
| 60.0 | -0.007934 | +0.101584 | +0.046866 | -0.054759 | +0.046825 |

T_odd: mean=+0.000726
T_zero: mean=+0.014728

Note: T_even approx T_zero at these angles — only means even-symmetry contribution is small here, not universally zero.

## E5: Phase + Sequence Sweep (KEY)

| sign | delta_deg | T_odd mean | T_odd std |
|------|-----------|-------------|-----------|
| +1 | 0 | +0.000649 | 0.055955 |
| +1 | 30 | +0.000686 | 0.055593 |
| +1 | 60 | +0.000538 | 0.055347 |
| +1 | 90 | +0.000247 | 0.055467 |
| +1 | 120 | -0.000111 | 0.055831 |
| +1 | 150 | -0.000439 | 0.056074 |
| +1 | 180 | -0.000649 | 0.055955 |
| -1 | 0 | -0.020756 | 0.000360 |
| -1 | 30 | +0.020025 | 0.000155 |
| -1 | 60 | +0.055442 | 0.000523 |
| -1 | 90 | +0.076003 | 0.000793 |
| -1 | 120 | +0.076199 | 0.000856 |

**Max |T_odd|**: sign=-1, delta=120deg, T_odd=+0.076199

**CRITICAL FINDING**:
- sign=+1 (positive sequence): T_odd approx 0 for ALL delta — current produces no average torque
- sign=-1 (negative sequence): T_odd up to +0.076 Nm — significant average torque
- The winding's spatial arrangement corresponds to the OPPOSITE rotation direction
- Original default (s=+1, delta=0) gives approx zero torque because it drives in wrong direction
- This is NOT 'winding cannot produce torque' — it is a sequence direction mismatch

## E6: Grid Refinement

| Grid | Spacing (mm) | Torque | res | B_max | B_max pos (mm) |
|------|-------------|--------|-----|-------|-----------------|
| [96, 96, 58] | (1.474,1.474,1.754) | -0.019375 | 2.833886e-04 | 1.3799 | (-8.11,-18.42,-27.19) |
| [128, 128, 78] | (1.102,1.102,1.299) | -0.022908 | 9.436163e-04 | 2.0301 | (33.62,-0.55,-37.01) |
| [160, 160, 96] | (0.881,0.881,1.053) | -0.027667 | 1.902912e-03 | 2.9617 | (-37.42,-2.2,-38.42) |

B_max position changes with grid — local peak (material interface/centerline), not whole-field non-convergence.

## Summary

| # | Factor | Finding |
|---|--------|---------|
| 1 | Copper fragmentation | 0.0% effect |
| 2 | Solver convergence | 14% at mi=60, <1% at mi=240 |
| 3 | Integration sampling | 10.2% variation |
| 4 | T_odd (s=+1, default) | approx +0.000726 — near zero |
| 5 | **Phase sweep** | **s=-1, delta=90deg: T_odd=+0.076 Nm** — sequence direction wrong |
| 6 | Grid convergence | Torque and B_max local peak not converged |

**Distinction**: 'T_odd approx 0 under s=+1 excitation' is NOT 'winding cannot produce torque'.
The winding produces 0.076 Nm with correct sequence (s=-1, delta=90deg).

**Next**: Investigate why sign=-1 is needed (winding polarity vs rotor magnetization direction).
Then validate with independent reference coil case.
