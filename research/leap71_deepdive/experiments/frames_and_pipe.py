"""Experiment 2: skeleton (spine) + Frames + field-modulated cross-section.

Reproduces the LEAP71 ShapeKernel `BasePipe` recipe:
  surface_point(s, u, v) = center(s) + frame(s) * cross_section(s, u, v)

where the cross-section is a 2D profile placed in the local frame, scaled/morphed
by a field (1D "line" modulation along s, and 2D "surface" modulation around the
profile). This is Frisken & Perry 2006 section 5.5 "fleshing out" made explicit.

We build a helical spine, an orthonormal frame at each point (parallel-transport
style, the same idea as MIN_ROTATION), and a radius field r(s,theta) = base * line(s)
* surface(theta). Then we verify the frame is orthonormal and the sampled tube
surface has the expected radius, and dump a point cloud to CSV.
"""
import numpy as np

# ---- 1. spine: helix ----
t = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
radius_helix = 10.0
pitch = 2.0
center = np.stack([
    radius_helix * np.cos(t),
    radius_helix * np.sin(t),
    pitch * t,
], axis=1)                      # (N,3)

# ---- 2. frame at each spine point (parallel transport / Gram-Schmidt) ----
tangent = np.gradient(center, axis=0)          # dC/ds (unnormalized)
tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

up = np.array([0.0, 0.0, 1.0])
normal = np.cross(tangent, up)
normal /= np.linalg.norm(normal, axis=1, keepdims=True)
binormal = np.cross(tangent, normal)
# frame = (normal, binormal, tangent)  -> 3x3 per point

# orthonormality check
dot_nb = np.abs(np.einsum("ij,ij->i", normal, binormal))
dot_nt = np.abs(np.einsum("ij,ij->i", normal, tangent))
dot_bt = np.abs(np.einsum("ij,ij->i", binormal, tangent))
print(f"frame orthonormality  max|n.b|={dot_nb.max():.2e} max|n.t|={dot_nt.max():.2e} max|b.t|={dot_bt.max():.2e}")

# ---- 3. field-modulated cross-section ----
base_r = 2.5


def line_field(s):
    # 1D modulation along the spine: radius pinches/expands twice per turn
    return 1.0 + 0.3 * np.sin(2.0 * s / pitch)


def surface_field(theta):
    # 2D modulation around the profile: make a 5-lobed "flower" cross-section
    return 1.0 + 0.35 * np.cos(5.0 * theta)


n_theta = 64
theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
points = []
for i in range(len(t)):
    r = base_r * line_field(t[i])
    n = normal[i]
    b = binormal[i]
    c = center[i]
    for th in theta:
        rr = r * surface_field(th)
        local = rr * np.cos(th) * n + rr * np.sin(th) * b
        points.append(c + local)

pts = np.asarray(points)
np.savetxt("helical_pipe_surface.csv", pts, delimiter=",", header="x,y,z", comments="")
print(f"generated {pts.shape[0]} surface points -> helical_pipe_surface.csv")

# ---- 4. verify: reconstructed radius vs expected at a mid-spine point ----
i = len(t) // 2
sub = pts[i * n_theta:(i + 1) * n_theta]
radii = np.linalg.norm(sub - center[i], axis=1)
expected = base_r * line_field(t[i]) * surface_field(theta)
err = np.abs(radii - expected).max()
print(f"radius reconstruction max error = {err:.2e}  (expected lobe range "
      f"{expected.min():.2f}..{expected.max():.2f})")

# ---- 5. verify cross-section plane is perpendicular to tangent ----
vecs = sub - center[i]
dot = np.abs(vecs @ tangent[i])
print(f"cross-section perpendicular to tangent: max|dot|={dot.max():.2e}")
