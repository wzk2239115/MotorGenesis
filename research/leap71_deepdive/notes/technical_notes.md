# LEAP71 — Technical Foundation Deep-Dive Notes

Status: research notes from paper digests + source-code reading. Working theory of the
company's "technical DNA", traced through the academic lineage and verified against the
vendored source in `vendor/leap71/`.

---

## 1. The four-layer conceptual stack

LEAP71's entire CAD engine is a **computational geometry stack built on one primitive**:
the signed distance field (SDF). Everything else (frames, pipes, lattices, microstructures,
heat exchangers) is constructed *on top of* that single representation.

```
Layer 4  LatticeLibrary   biomimetic lattices, microstructures, randomness, TPMS
Layer 3  ShapeKernel      skeleton ("spine") + Frames + modulated cross-sections
Layer 2  PicoGK            SDF voxel field (OpenVDB), booleans, offsets, meshing
Layer 1  Papers           math foundation (see section 2)
```

The key architectural insight: **LEAP71 never works with B-rep/precise CAD surfaces.**
It works with fields. A shape is a function `f : R3 -> R` (distance to surface), and the
surface is the level set `f = 0`. This is the *same* representation behind VFX sparse
volumes (OpenVDB), which is why the stack can do things classic parametric CAD cannot
(implicit blending, topology changes, fillets, offsets) cheaply and robustly.

---

## 2. Academic lineage (the papers, in dependency order)

### 2.1 Frisken, Perry, Rockwood, Jones 2000 — Adaptively Sampled Distance Fields (ASDF)
- MERL TR2000-15, Siggraph 2000.
- Unifying representation: distance fields for geometry **and** volume data; rendering,
  sculpting, LOD, offsetting, collision, boolean ops, gradient-based reconstruction.
- ADF = *adaptive* (hierarchical) distance field, vs. a uniform voxel grid. Precursor to VDB.

### 2.2 Frisken & Perry 2006 — Designing With Distance Fields (SIGGRAPH course)
- The single most important paper for LEAP71's ShapeKernel (`papers/FriskenPerry2006.txt`).
- Signed distance field definition, and the boolean algebra over distance values:
  - **Union / add**: `min(dA, dB)`  (equivalently `max(fA,fB)` for the "inside < 0" convention)
  - **Intersection**: `max(dA, dB)`  (`min(fA,fB)`)
  - **Difference / subtract**: `min(dA, -dB)` (`max(fA, -fB)`)
  - **Smooth/rounded variants**: smooth min/max produce *filleted* blends (the "real clay" look).
- **Section 5.5 "Concept Modeling"** is the smoking gun for LEAP71's design:
  - Stage 1: freehand **skeleton curves** roughing out the shape.
  - Stage 2: "**fleshing out** the geometry" — 2D cross-sectional profiles **lofted along the
    skeleton** via implicit blend.
  - Stage 3: brush-based carving edits.
- This is *exactly* LEAP71's "spine = skeleton, cross-section = muscle". The paper even uses
  the word "fleshing out". `BasePipe` + `Frames` is a literal implementation of this.

### 2.3 Museth 2013 — VDB: High-Resolution Sparse Volumes with Dynamic Topology (OpenVDB)
- DreamWorks. `papers/Museth2013_VDB.txt`.
- **VDB = Volumetric, Dynamic grid**, sharing characteristics with B+ trees. Virtually
  infinite 3D index space, memory scales with meaningful voxels not the dense bounding volume.
- Hierarchical tree with configurable, variable branching factors (powers of two), cache-coherent.
- Purpose-built for **narrow-band signed-distance level sets**: the field is only stored near the
  `f=0` surface (a thin band), not everywhere.
- Native support for **CSG (booleans), flood-filling, topology dilation, adaptive resolution**.
- Used for high-res animated clouds in *Puss in Boots* / *Rise of the Guardians* (15,000×900×500 voxels).
- **This is the exact engine PicoGK vendors** (`PicoGK/Base/Voxels.cs`, `Internals/Interop.cs`).

### 2.4 Pasko et al. 1995 / 2011 — Function Representation (F-Rep) & microstructures
- F-rep 1995: "Function representation in geometric modeling" — the theoretical basis of using
  real-valued functions of point coordinates + R-functions (min/max) as a *complete* modeling
  language. PicoGK's ScalarField/IImplicit is F-rep.
- Microstructures 2011 (Graphical Models): *procedural, function-based* representation of
  heterogeneous objects with internal volumetric structures — size of details orders of
  magnitude smaller than the object. "Compact, precise, arbitrarily parametrized models of
  coherent microstructures." This is the lineage of `LatticeLibrary`'s `CellBasedBeamThickness`,
  periodic TPMS (`ImplicitSchwarzPrimitive`), and the "tissue"-like dense lattices.

### 2.5 Seidler et al. 2023 — Differential-Growth walls for heat exchangers
- TU Dresden, ICED23. `papers/Seidler2023_DifferentialGrowth.txt`.
- Uses the **differential-growth method** (Pedersen & Singh 2006 "Organic Labyrinths and Mazes")
  to synthesize complex heat-transferring walls, CFD-validated (Ansys Fluent).
- vs. gyroid (TPMS) heat exchanger [Peng 2019]:
  - +37.76 % larger heat-transferring wall area, ~10 % lower max pressure drop,
    efficiency 73.1 % vs. 62.1 % for gyroid. Laminar flow (Re ≈ 149 / 44).
  - Heat transfer best in the "furrow" regions; furrows let you *steer* heat transfer locally.
- Conclusion names the open problem LEAP71's HelixHeatX/lattices attack: **"additional
  parameters or field functions for a functionally targeted control of the growth behavior
  require further definitions."**
- This is why the LEAP71 lattice layer is field-driven, not purely geometric.

---

## 3. Source-code mapping (vendor/leap71)

### 3.1 PicoGK — the kernel (OpenVDB wrapper + SDF API)
- `Base/Voxels.cs` (~1255 lines): `Voxels` class; `IImplicit` interface (`fValue(x,y,z)`),
  booleans, offset/`OverOffset`, `SmoothenLattice`/`UnderCut`, sampling. Convention:
  **negative = inside** (surface at `f=0`).
- `Base/ScalarField.cs` / `Base/VectorField.cs`: field abstractions feeding Implicits
  (the F-rep "function representation").
- `Internals/Interop.cs`: P/Invoke boundary to the native OpenVDB C++ layer — confirms VDB
  (Museth 2013) is the underlying data structure.
- Sources are vendored + fully checked out (see `git submodule status`).

### 3.2 ShapeKernel — skeleton + frames + modulation
- `Frames/LocalFrame.cs`, `Frames/Frames.cs`: coordinate frame along a spine with kine modes
  (CYLINDRICAL, SPHERICAL, Z, MIN_ROTATION). Implementation of the "skeleton curve →
  locally-oriented cross-section" from Frisken & Perry §5.5.
- `BaseShapes/BaseShape.cs`, `BaseShapes/BasePipe.cs`: `fGetSurfacePoint()` liquidates the
  spine+frame into an Implicit — directly the "flesh out along the skeleton" idea.
- `Modulations/SurfaceModulation(2D).cs`, `Modulations/LineModulation(1D).cs`: field-valued
  modifiers over (u,v) / (s) that scale/morph the cross-section — the "functional control"
  anticipated by Seidler 2023.

### 3.3 LatticeLibrary — biomimetic structures
- `LatticeTypes/RandomSplineLattice.cs`: randomly perturbed splines → organic, sponge-like tissue.
- `BeamThickness/*.cs` (`Boundary`, `GlobalFunc`, `CellBased`): thickness as a *field* over the
  lattice — supports **functional grading** (thick near boundary, thin in core, etc.).
- `ImplicitLibrary/RandomDeformationField.cs`: procedural noise-driven deformation (cf. VDB
  clouds pipeline where noise makes the puffy look).
- `ImplicitLibrary/TPMSPresets/ImplicitSchwarzPrimitive.cs`: implicit TPMS evaluation — the
  same P-minimal-surface class benchmarked in the heat-exchanger literature.

### 3.4 Applications
- `LEAP71_HelixHeatX`: the differential-growth / bio-inspired heat-exchanger demo — ties
  Seidler 2023 → LatticeLibrary → PicoGK together.
- `PicoGK_SimulationExample`: a minimal FEA simulation holder (attaches a solver to the voxel
  field), demonstrating the geometry → simulation handoff.

---

## 4. Synthesis — "what is LEAP71's technical root?"

The company is the first to **productize implicit (F-rep / SDF / VDB) modeling for mechanical
engineering and AM**, collapsing 25+ years of VFX/computer-graphics math (ASDF → VDB →
Frisken's "clay" metaphor) into a code-first pipeline where *shapes are fields and code*.

The three technical pillars:
1. **SDF kernel (PicoGK = OpenVDB)** — robust booleans, offsets, blends, meshing on a sparse
   narrow-band grid. Correctness/extensibility over raw speed.
2. **Skeleton + frames + field-modulated cross-sections (ShapeKernel)** — the Frisken & Perry
   "flesh out along a skeleton" concept, made code-first/parametric.
3. **Field-driven biomimetic lattices & microstructures (LatticeLibrary)** — Pasko's function-based
   microstructures + random splines + local thickness fields, aimed at the differentiable /
   CFD-validated heat-exchanger / heat-sink problem space (Seidler 2023).

These three map 1:1 onto the module layout the codex identified, and each has a direct,
verifiable academic ancestor. Experiments in `../experiments/` reproduce the math of each pillar.
```