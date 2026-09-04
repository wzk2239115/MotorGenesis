# LEAP71 Deep-Dive Addendum — Application layer & field/modulation internals

Second-pass findings from reading the flagship application, the frame/modulation
math, and the lattice beam primitives. Fills in the "how it actually works" below
the earlier stack summary.

---

## 1. The flagship app is a REAL, buildable heat exchanger (`HelixHeatX`)

`LEAP71_HelixHeatX/src/HelixHeatX/HelixHeatX.cs` constructs a complete two-fluid
helical heat exchanger purely in code, then exports STL. The build order is the
proof of the whole thesis — everything is a boolean-on-SDF operation:

1. Inlet/outlet cylinders + bounding box (`BaseCylinder`, `BaseBox`).
2. Corner/straight **fins** as separate voxel sets (`InternalFins.cs`).
3. **Helical fluid voids** built as *lattices of beams* (`HelicalVoids.cs`):
   - two interleaved helices (COOL at `phi=0`, HOT at `phi=pi`).
   - each sampled at ~0.005 mm spacing, each sample adds a `Lattice.AddBeam`
     tapered beam (inner/outer radius from `fGetInnerRadius/fGetOuterRadius`,
     which come from a `LineModulation` contour — the *flesh* on the helix *skeleton*).
4. **The key line** — the 0.8 mm separating wall between the two fluids is created by
   *offsetting* one fluid void and subtracting it from the other:
   ```csharp
   voxHotFluidVoid  -= voxCoolFluidVoid.voxOffset(m_fWallThickness);
   voxCoolFluidVoid -= voxHotFluidVoid.voxOffset(m_fWallThickness);
   ```
   A uniform-thickness wall, produced by a distance-field offset + boolean subtract.
   This is *exactly* the Frisken & Perry "offsetting / shelling" primitive, applied
   to a real product.
5. Assemble: `result = (outer - innerVoid) + fins + splitters`, intersected with the
   bounding box, then `Fillet(5)`, `Smoothen(0.5)`, subtract screw holes and a
   print web, add IO threads. `ProjectZSlice` cuts the bottom flat.

Conclusion: the "magic" of a heat exchanger (thin uniform walls, interleaved
channels, fillets) reduces to ~200 lines of SDF boolean/offset calls.

---

## 2. Modulation = a first-class *function algebra* (not just "a scalar")

`ShapeKernel/Modulations/LineModulation(1D).cs` and `SurfaceModulation(2D).cs`
are the heart of the "field" abstraction:

- `LineModulation` is `f(float ratio) -> float`. Constructed from:
  a constant, a delegate function, or a **discrete point list** (linear-interpolated,
  endpoints auto-clamped to 0..1).
- `SurfaceModulation` is `f(float phi, float lengthRatio) -> float`. Constructed from:
  a constant, a 2D delegate, a `LineModulation` (projected along phi OR length), or
  an **image** (grayscale value → physical radius via a user mapping function).
- Both overload `+`, `-`, and `* scalar` → **modulations compose arithmetically**.
  You literally write `radius = 2 * A + 0.5 * (B - C)`.

Two consequences:
1. A cross-section profile is a *function*, so "design" = writing/composing functions
   (the "code-first CAD" claim is literal, not marketing).
2. **Image-driven geometry**: a bitmap can modulate a surface — gray values become
   physical features. This is the direct bridge from 2D image/sensor data to 3D.

---

## 3. Frames: 4 orientation modes + a brute-force alignment

`ShapeKernel/Frames/Frames.cs` (`EFrameType`):

| mode | `vecGetTargetX` | meaning |
|------|-----------------|---------|
| `Z` | `(0,0,1)` | frame X aims at global +Z |
| `CYLINDRICAL` | `normalize(x,y,0)` | radial in the XY plane (tubes/helix) |
| `SPHERICAL` | `normalize(x,y,z)` | radial from origin (balls) |
| `MIN_ROTATION` | carry forward previous X | rotation-minimizing (parallel transport) |

`vecAlignWithTargetX` scans 0–180° in 0.01° steps, maximizing `|dot(X, target)|`,
then flips for alignment. Simple, robust, and *discretized* — deliberately trading a
little CPU for zero singularities. The tangent is a forward finite difference
(`pt[i] - pt[i-1]`), localY = `cross(localZ, localX)` (right-handed, see
`LocalFrame.vecGetLocalY`).

---

## 4. Beam thickness = the field-grading primitive (Seidler 2023's ask, answered)

`IBeamThickness.fGetBeamThickness(Vector3) -> float`, four impls:

- `ConstantBeamThickness` — uniform.
- `CellBasedBeamThickness` — radius gradient within each unit cell
  (`fDist / cellHalfDiagonal`, `fTransFixed(min,max,ratio)`).
- `BoundaryBeamThickness` — **distance to a bounding surface** field via
  `bClosestPointOnSurface`, `fTransSmooth(max,min,dist, 15, 5)` → beams thicken
  toward the skin, thin in the core (functionally-graded infill).
- `GlobalFuncBeamThickness` — arbitrary global function (here `0.02*X`).

The transition helpers in `UsefulFormulas.cs`:
- `fTransFixed` — an open **B-spline** ramp (control points 0→0.5→0.5→1).
- `fTransSmooth` — **tanh** smoothstep `0.5+0.5*tanh((s-t)/k)` (doc cites the
  classic smooth-transition trick).

A beam is then a **truncated cone** between two points with two radii (plus optional
hemispherical cap) — `Lattice.AddBeam(vecA, rA, vecB, rB, roundCap)`. The radii are
`samples` from the thickness field, so thickness varies *continuously along* the beam.

TPMS ties it together in `ImplicitModular.cs`:
```
fRawSD        = rawTPMSPattern(x,y,z)         // gyroid / Schwarz primitive
fWallThickness= beamThicknessField(point)
final         = splittingLogic(rawSD, thickness)   // = abs(rawSD) - thickness/2
```
i.e. **TPMS surface thickened by a field** — the exact formula I reproduced in
`experiments/tpms_beam_thickness.py`.

---

## 5. Lattice & simulation handoff confirm OpenVDB

- `PicoGK/Base/Lattice.cs` header states verbatim: *"The foundation of PicoGK is a
  thin layer on top of the powerful open-source OpenVDB project"* — the VDB paper is
  the kernel, confirmed at the source level.
- `PicoGK_SimulationExample` shows the **closed loop**: geometry is exported as a
  **VDB file** carrying not only the solid/fluid domains but also `ScalarField`
  (density, viscosity) and `VectorField` (velocity) — so a CFD/FEA solver consumes
  the *same* representation the geometry engine produced. This is the "Computational
  Engineering Model (CEM)" concept: geometry and physics share one field format.

---

## 6. Updated one-line summary

LEAP71 = a thin OpenVDB wrapper (PicoGK) + a **functional modulation algebra**
(ShapeKernel: `Line/SurfaceModulation` + `Frames` + `BasePipe`) + a **field-graded
beam/TPMS lattice** layer (LatticeLibrary), all driven by code and exportable, with
physics fields, into the VDB format solvers natively read. The flagship `HelixHeatX`
is a working two-fluid heat exchanger whose 0.8 mm walls are a `voxOffset`+subtract.
