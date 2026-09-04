"""Experiment 1: Signed distance field boolean algebra (Frisken & Perry 2006).

Verifies the core CSG formulas that PicoGK implements, using the SAME sign
convention as PicoGK (negative = inside, surface at f=0):

  union         f = min(fA, fB)          (both inside -> more negative wins)
  intersection  f = max(fA, fB)
  difference    f = max(fA, -fB)         (A minus B)
  smooth union  f = -smoothmin(-fA,-fB)  -> "real clay" filleted blend

Note: Frisken & Perry use positive=inside, so their formulas are min/max flipped.
Both are equivalent; here we follow PicoGK's negative-inside convention.

Output: ASCII visualization of each field + numeric spot checks.
"""
import numpy as np

W = 81
H = 41
x = np.linspace(-2.0, 2.0, W)
y = np.linspace(-1.0, 1.0, H)
X, Y = np.meshgrid(x, y)

# two overlapping circles (signed distance, negative inside)
cA = (-0.7, 0.0, 0.9)
cB = (0.7, 0.0, 0.9)


def circle_sdf(cx, cy, r):
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r


fA = circle_sdf(*cA)
fB = circle_sdf(*cB)

uni = np.minimum(fA, fB)
inter = np.maximum(fA, fB)
diff = np.maximum(fA, -fB)


def smoothmin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


k = 0.35
smooth = smoothmin(fA, fB, k)


def ascii_field(f):
    chars = []
    for row in f:
        line = "".join("#" if v < -0.02 else "." if v > 0.02 else "o" for v in row)
        chars.append(line)
    return "\n".join(chars)


print("=== UNION (min) ===")
print(ascii_field(uni))
print("\n=== INTERSECTION (max) ===")
print(ascii_field(inter))
print("\n=== DIFFERENCE A-B (max(fA,-fB)) ===")
print(ascii_field(diff))
print("\n=== SMOOTH UNION (fillet k=0.35) ===")
print(ascii_field(smooth))

# numeric spot checks
inside = uni[H // 2, int(W * 0.25)] < 0
outside = uni[H // 2, 0] > 0
print(f"\nunion inside-left={inside} outside-left-edge={outside}")

# save numeric grids for the report
np.savez("sdf_fields.npz", X=X, Y=Y, fA=fA, fB=fB, uni=uni, inter=inter, diff=diff, smooth=smooth)
print("saved sdf_fields.npz")
