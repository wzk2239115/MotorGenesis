# Energy Audit (git 00ce79b)

## Synthetic Model

| 场景 | 电路误差 | 转子误差 | E_elec | E_conv | W_load | 末速 |
|---|---|---|---|---|---|---|
| zero_voltage | n/a (无能量流动) | 0.00% | 0.00 mJ | 0.00 mJ | -0.00 mJ | -0.0 rad/s |
| no_load_start | 1.60% | 0.00% | 1725.42 mJ | 675.99 mJ | 0.00 mJ | 116.3 rad/s |
| load_step_high | 0.89% | 0.00% | 1392.19 mJ | 357.88 mJ | 247.82 mJ | 46.9 rad/s |
| reversal | 1.19% | 0.00% | 1538.03 mJ | 497.74 mJ | 227.00 mJ | -73.6 rad/s |
| power_off_coast | 7.87% | 0.00% | 681.83 mJ | 143.51 mJ | 105.05 mJ | 27.7 rad/s |
| dt_half | 1.26% | 0.00% | 1718.88 mJ | 675.69 mJ | 0.00 mJ | 116.2 rad/s |
| windage_on | 1.60% | 0.00% | 1725.33 mJ | 675.91 mJ | 0.22 mJ | 116.2 rad/s |

## Real Assembly Artifact (hash 6056e6ccfa5f37e9)
- ψ_FEA = 0.00230 Wb
- ψ_map = 0.00162 Wb
- n_turns = 7

| 场景 | 电路误差 | 转子误差 | E_elec | E_conv | W_load | 末速 |
|---|---|---|---|---|---|---|
| no_load | 3.19% | 0.00% | 590.42 mJ | 4.30 mJ | 0.10 mJ | 6.5 rad/s |
| load_step | 1.53% | 0.00% | 577.61 mJ | 1.53 mJ | 0.86 mJ | 2.6 rad/s |
| power_off | 3.43% | 0.00% | 364.02 mJ | 0.91 mJ | 0.05 mJ | 2.9 rad/s |