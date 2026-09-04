# LEAP71 Deep-Dive — Final Synthesis Report

## Goal
Trace the technical foundation of LEAP71's engine, verify it against the academic
literature AND the vendored source code, and reproduce the core math with minimal
experiments.

## What LEAP71's stack actually is

One sentence: **a computational-geometry engine that models every shape as an
implicit signed-distance field (SDF), then layers three abstractions on top —
a sparse SDF kernel, a skeleton/frame skinning layer, and a field-driven lattice
layer — all inherited directly from ~25 years of computer-graphics research.**

```
LEAP71_HelixHeatX / applications      heat exchangers, heat sinks (differential growth)
LEAP71_LatticeLibrary                 biomimetic lattices, microstructures, TPMS
LEAP71_ShapeKernel                    skeleton ("spine") + Frames + modulated cross-section
PicoGK                                SDF voxel field = OpenVDB (Museth 2013)
```

## The four papers and what each one contributes

1. **Frisken/Perry/Rockwood/Jones 2000 (ASDF, MERL TR2000-15)** — the ancestor:
   distance fields as a *unifying* shape representation (geometry, sculpting, offsetting,
   booleans, LOD). Adaptive = hierarchical.
2. **Frisken & Perry 2006 ("Designing With Distance Fields")** — the conceptual blueprint.
   §5.5 "Concept Modeling" literally describes LEAP71's workflow: sketch **skeleton curves**
   → "**fleshing out** the geometry" by lofting 2D cross-sections along the skeleton → edit.
   Also defines the boolean algebra over distance values.
3. **Museth 2013 ("VDB", DreamWorks/OpenVDB)** — the kernel. Sparse, hierarchical,
   B+-tree-like grid for **narrow-band level sets**; native CSG, flood-fill, adaptive
   resolution, virtually infinite index space. This is exactly what PicoGK vendors.
4. **Seidler et al. 2023 (differential-growth heat-exchanger walls, TU Dresden)** — the
   *application* motivation. Differential-growth walls beat gyroid/TPMS on area (+37.76 %),
   pressure drop (−10 %), and efficiency (73.1 % vs 62.1 %), and the paper explicitly calls
   for "**field functions for functionally targeted control of the growth behavior**" —
   which is precisely what LEAP71's `BeamThickness`/modulation fields provide.
   (Pasko 1995/2011 F-rep + function-based microstructures is the theoretical backbone of
   the lattice layer.)

## Verified against source code

- **PicoGK = OpenVDB.** `vendor/leap71/PicoGK/Base/Voxels.cs` exposes `BoolAdd`,
  `BoolSubtract`, `BoolIntersect`, `Smoothen`, `OverOffset`, `UnderCut`; the native layer is
  behind `Internals/Interop.cs`. Negative = inside (confirmed by `fValue <= 0` → solid in
  the raster code, `Voxels.cs:1140`).
- **ShapeKernel = Frisken & Perry §5.5.** `BasePipe.cs:302` `vecGetSurfacePoint()` does
  exactly the skeleton+frame+modulation recipe:
  `surfacePoint = spine(s) + r·cosφ·localX + r·sinφ·localY`, where inner/outer radius come
  from `fGetOuterRadius/fGetInnerRadius` **modulation** objects — the "field-modulated
  cross-section" (flesh) along the spine (skeleton). Frames are built along the spine
  (`Frames/Frames.cs`, `LocalFrame.cs`).
- **LatticeLibrary = Pasko microstructures + Seidler's field control.**
  `BeamThickness/` has `Boundary`, `GlobalFunc`, `CellBased` — thickness as a *field* over
  the lattice (functional grading). `ImplicitLibrary/TPMSPresets/ImplicitSchwarzPrimitive.cs`
  is the implicit TPMS class benchmarked in the heat-exchanger literature; `RandomDeformationField`
  and `RandomSplineLattice` are the procedural-noise "tissue".

## Experiments (all reproducible in `experiments/`)

| # | File | What it proves | Result |
|---|------|----------------|--------|
| 1 | `sdf_booleans.py` | union=min, intersection=max, difference, **smooth min = filleted "clay" blend** | smooth union merges two circles with a rounded neck (exactly the Frisken & Perry "real clay" claim) |
| 2 | `frames_and_pipe.py` | skeleton + orthonormal frame + 5-lobed field-modulated cross-section | frame orthonormal to ~1e-17, radius reconstruction error ~1e-15, section ⊥ tangent to ~1e-16 |
| 3 | `differential_growth.py` | Pedersen/Singh space-filling loop (repel + cohesion + curl + split/prune) | closed loop grows into a bounded, self-avoiding folded curve (292 nodes) → `differential_growth_nodes.csv` |
| 4 | `tpms_beam_thickness.py` | gyroid implicit `|f|−t`, **thickness as a field** | solid fraction 0.257 (uniform) → 0.323 (graded); x=0 slice 10.9 % vs x=2π slice 46.9 % solid — grading verified |

## Second-pass findings (application layer — see `deepdive_addendum.md`)

- **`HelixHeatX` is a real, buildable two-fluid heat exchanger**, ~200 lines of SDF ops.
  Its 0.8 mm separating wall is a literal `voxOffset` + subtract (`HelixHeatX.cs:152`):
  `voxHotFluidVoid -= voxCoolFluidVoid.voxOffset(wall)`. Fins, splitters, fillets, threads
  all assembled as voxel booleans, then exported to STL.
- **Modulation is a function algebra**: `LineModulation`/`SurfaceModulation` are first-class
  `f(ratio)->float` / `f(phi,length)->float` with `+ - *` overloads, plus **image-driven**
  modulation (grayscale bitmap → physical radius). Design = composing functions.
- **Frames = 4 orientation modes** (Z / CYLINDRICAL / SPHERICAL / MIN_ROTATION) with a
  brute-force 0.01°-scan alignment to target X.
- **Beam thickness = the field-grading primitive** answering Seidler 2023's call: Constant /
  CellBased (radial) / Boundary (distance-to-skin via `fTransSmooth` tanh) / GlobalFunc.
  `ImplicitModular.cs` confirms TPMS = `abs(rawSD) − thicknessField` (my experiment 4 formula).
- **Lattice = native OpenVDB tapered beams/spheres** (`Lattice.cs` header: "thin layer on top
  of OpenVDB"); `PicoGK_SimulationExample` exports geometry **and** physics fields
  (density/viscosity/velocity) as VDB — geometry & simulation share one field format (CEM).

## The bottom line

LEAP71's "technical root" is **the F-rep/SDF tradition** (Pasko → Frisken → Museth/OpenVDB),
repackaged as a code-first CAD kernel for additive manufacturing. Its three pillars map 1:1
onto the module layout: **(1) OpenVDB SDF kernel (PicoGK), (2) skeleton+frames+modulated
cross-sections (ShapeKernel), (3) field-graded biomimetic lattices/TPMS (LatticeLibrary)**,
with differential-growth heat exchangers (`HelixHeatX`) as the flagship application that
ties it all together. The "magic" is not any single algorithm — it is the deliberate
collapse of all geometry into *fields*, which makes booleans, blends, offsets, topology
changes, and functional grading near-free, and therefore makes *code* (not clicking) the
authoring interface.
