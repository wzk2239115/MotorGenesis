"""Experiment 3: differential growth (space-filling curve) in 2D.

Reproduces the core of the Pedersen & Singh 2006 "Organic Labyrinths" /
"inconvergent" differential-line-growth algorithm that Seidler et al. 2023
used to synthesize heat-transferring walls (cf. LEAP71 HelixHeatX / lattice layer).

A closed loop of nodes grows inside a rectangle through three forces:
  - repulsion  (all nearby nodes push apart  -> self-avoidance)
  - cohesion   (neighboring nodes spring to a rest length)
  - curl       (slight tangent bias to encourage folding)
Edges longer than max_len split (growth); nodes too close merge (pruning).

Output: ASCII raster of the final space-filling curve + node CSV.
"""
import numpy as np

rng = np.random.default_rng(7)

# domain: rectangle (mirrors Seidler's 40x20 mm cross-section aspect ratio)
W, H = 2.0, 1.0
halfw, halfh = W / 2.0, H / 2.0

# start with a small circle
N0 = 24
theta = np.linspace(0, 2 * np.pi, N0, endpoint=False)
pos = np.stack([0.35 * np.cos(theta), 0.2 * np.sin(theta)], axis=1)

R = 0.09          # repulsion radius
rest = 0.075       # rest edge length
max_len = 0.11     # split threshold
min_len = 0.032    # merge threshold
curl = 0.04        # curl strength
repel = 0.55       # repulsion strength
cohe = 0.30        # cohesion strength
boundary_k = 0.4   # boundary push strength

iters = 2400


def forces(p):
    n = len(p)
    f = np.zeros_like(p)
    # repulsion: all-pairs within R (O(n^2) but n stays ~1-2k)
    dx = p[:, None, :] - p[None, :, :]
    d = np.linalg.norm(dx, axis=2)
    d[d < 1e-9] = 1e-9
    mask = (d < R) & (d > 0)
    f += repel * np.sum((dx / d[..., None]) * (R - d)[..., None] * mask[..., None], axis=1)
    # cohesion: neighbors (prev, next)
    prev = np.roll(p, 1, axis=0)
    nxt = np.roll(p, -1, axis=0)
    for q in (prev, nxt):
        v = q - p
        d = np.linalg.norm(v, axis=1, keepdims=True)
        d[d < 1e-9] = 1e-9
        f += cohe * v / d * (d - rest)
    # curl: tangential bias (perpendicular to local direction)
    tang = nxt - prev
    tn = np.linalg.norm(tang, axis=1, keepdims=True)
    tn[tn < 1e-9] = 1e-9
    tang /= tn
    perp = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    f += curl * perp
    return f


for it in range(iters):
    f = forces(pos)
    pos = pos + 0.01 * f
    # boundary: push back inside rectangle
    pos[:, 0] = np.clip(pos[:, 0], -halfw + 0.03, halfw - 0.03)
    pos[:, 1] = np.clip(pos[:, 1], -halfh + 0.03, halfh - 0.03)
    # growth: split long edges
    newpos = [pos[0]]
    for i in range(len(pos) - 1):
        a, b = pos[i], pos[i + 1]
        newpos.append(a)
        if np.linalg.norm(b - a) > max_len:
            newpos.append((a + b) / 2)
    a, b = pos[-1], pos[0]
    newpos.append(b)
    if np.linalg.norm(b - a) > max_len:
        newpos.append((a + b) / 2)
    pos = np.asarray(newpos)
    # pruning: merge nodes too close to predecessor
    if len(pos) > 3:
        keep = [True] * len(pos)
        for i in range(1, len(pos)):
            if np.linalg.norm(pos[i] - pos[i - 1]) < min_len:
                keep[i] = False
        pos = pos[keep]

print(f"final node count = {len(pos)}")

np.savetxt("differential_growth_nodes.csv", pos, delimiter=",", header="x,y", comments="")
print("saved differential_growth_nodes.csv")


# ---- ASCII raster ----
def raster(p, gw=140, gh=70):
    img = [[" " for _ in range(gw)] for _ in range(gh)]
    for (x, y) in p:
        cx = int((x + halfw) / W * (gw - 1))
        cy = int((y + halfh) / H * (gh - 1))
        cy = gh - 1 - cy
        if 0 <= cx < gw and 0 <= cy < gh:
            img[cy][cx] = "#"
    return "\n".join("".join(r) for r in img)


print(raster(pos))
