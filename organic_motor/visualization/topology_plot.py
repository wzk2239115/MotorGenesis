"""Topology visualisation: material maps and optimisation histories."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from organic_motor.config import MotorConfig


def material_rgb(rho_iron, rho_pm) -> np.ndarray:
    """Composite RGB image: white=air, steel-grey=iron, red=PM."""
    ri = np.asarray(rho_iron)
    rp = np.asarray(rho_pm)
    ra = np.clip(1.0 - ri - rp, 0.0, 1.0)
    iron = np.array([0.35, 0.35, 0.40])
    pm = np.array([0.90, 0.20, 0.15])
    air = np.array([1.0, 1.0, 1.0])
    img = (ri[..., None] * iron + rp[..., None] * pm + ra[..., None] * air)
    return np.clip(img, 0, 1)


def plot_material(cfg: MotorConfig, rho_iron, rho_pm, out_path: str,
                  title: str = "") -> None:
    X = np.linspace(-cfg.L, cfg.L, cfg.N)
    Y = np.linspace(-cfg.L, cfg.L, cfg.N)
    img = material_rgb(rho_iron, rho_pm)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img.transpose(1, 0, 2), origin="lower",
              extent=[-cfg.L, cfg.L, -cfg.L, cfg.L])
    th = np.linspace(0, 2 * np.pi, 400)
    for r, ls in [(cfg.R_shaft, "k-"), (cfg.R_gap, "k--"),
                  (cfg.R_design, "k--"), (cfg.R_split, "k:")]:
        ax.plot(r * np.cos(th), r * np.sin(th), ls, lw=0.8)
    ax.set_title(title or "material distribution (white=air, grey=iron, red=PM)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_magnetization(cfg: MotorConfig, rho_pm, Mx, My, out_path: str,
                       title: str = "") -> None:
    """PM regions with magnetisation direction arrows overlaid."""
    rp = np.asarray(rho_pm)
    mx = np.asarray(Mx)
    my = np.asarray(My)
    X = np.linspace(-cfg.L, cfg.L, cfg.N)
    Y = np.linspace(-cfg.L, cfg.L, cfg.N)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.ones_like(rp), cmap="Greys", vmin=0, vmax=1,
              extent=[-cfg.L, cfg.L, -cfg.L, cfg.L], origin="lower")
    step = max(2, cfg.N // 40)
    yy, xx = np.mgrid[0:cfg.N:step, 0:cfg.N:step]
    u = mx[xx, yy]
    v = my[xx, yy]
    m = rp[xx, yy] > 0.25
    ax.quiver(X[xx], Y[yy], u * m, v * m, angles="xy", scale_units="xy",
              scale=4e5, color="red", width=0.004)
    ax.contour(X, Y, rp.T, levels=[0.5], colors="red", linewidths=1.0)
    th = np.linspace(0, 2 * np.pi, 400)
    for r, ls in [(cfg.R_shaft, "k-"), (cfg.R_gap, "k--"), (cfg.R_design, "k--")]:
        ax.plot(r * np.cos(th), r * np.sin(th), ls, lw=0.8)
    ax.set_title(title or "magnetisation direction (PM)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_history(history: dict, out_path: str) -> None:
    keys = [k for k in history if isinstance(history[k], list)]
    if not keys:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(history.get("torque", []))
    ax.set_title("torque [N.m/m]")
    ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(history.get("obj", []))
    ax.set_title("objective")
    ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.plot(history.get("vol_pm", []), label="PM")
    ax.plot(history.get("vol_iron", []), label="iron")
    ax.set_title("volume fractions")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    ax.plot(history.get("torque/mass", []))
    ax.set_title("torque / mass")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
