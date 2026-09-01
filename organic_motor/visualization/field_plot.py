"""Field visualisation: flux-density heatmap and magnetic field lines."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from organic_motor.config import MotorConfig


def _domain_circles(ax, cfg: MotorConfig) -> None:
    for r, ls, label in [
        (cfg.R_shaft, "k-", "shaft"),
        (cfg.R_gap, "k--", "air gap"),
        (cfg.R_design, "k--", "design"),
        (cfg.R_split, "k:", "rotor/stator"),
    ]:
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(r * np.cos(th), r * np.sin(th), ls, lw=0.8)


def plot_field_panel(cfg: MotorConfig, Bx, By, out_path: str,
                     title: str = "") -> None:
    X = np.linspace(-cfg.L, cfg.L, cfg.N)
    Y = np.linspace(-cfg.L, cfg.L, cfg.N)
    Bmag = np.sqrt(np.asarray(Bx) ** 2 + np.asarray(By) ** 2)
    bx = np.asarray(Bx)
    by = np.asarray(By)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    im = axes[0].pcolormesh(X, Y, Bmag.T, shading="auto", cmap="inferno")
    fig.colorbar(im, ax=axes[0], label="|B| [T]")
    axes[0].set_title(f"{title}  flux density |B|")
    axes[0].set_aspect("equal")
    _domain_circles(axes[0], cfg)

    s = max(2, cfg.N // 32)
    axes[1].streamplot(X, Y, bx.T, by.T, color=Bmag.T, cmap="viridis",
                       density=1.4, linewidth=0.8, arrowsize=0.6)
    axes[1].set_title("magnetic field lines (B)")
    axes[1].set_aspect("equal")
    _domain_circles(axes[1], cfg)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
