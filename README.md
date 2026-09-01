# MotorGenesis

MotorGenesis is a differentiable 2-D topology-optimization benchmark for an
inner-rotor permanent-magnet motor.  Air, iron, permanent-magnet density and
magnetization direction are optimized through a JAX magnetostatic solve.

The current model includes a physical rotor/stator air gap, rigid rotation of
both material and magnetization vectors, and a balanced distributed three-phase
stator excitation.  Torque is evaluated across synchronized mechanical and
electrical positions; the optimizer rewards cycle-average directional torque
and penalizes signed torque ripple.

## Install and test

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

## Run a small rotating-field experiment

```bash
python -m organic_motor.experiments.motor2d_ripple --N 64 --K 8 --steps 100
```

The distributed current sheet is an initial winding model.  Explicit copper
topology is represented as a continuous fourth material phase and modulates
the winding current density. The second model tier includes differentiable
copper loss, a replaceable first-order iron-loss proxy, a saturation-risk
penalty and steady-state heat conduction. These are optimization-grade reduced
models, not yet a substitute for measured material curves. Structural
constraints and 3-D end effects are deliberately not claimed by this version.

Every optimization checkpoint contains all four continuous material fields,
magnetization, vector potential, flux density, winding current and metric
history. Launch the research viewer with:

```bash
pip install -e '.[morphogenesis]'
python -m organic_motor.visualization.morphogenesis_viewer path/to/run
```

Each completed experiment also writes a static visual acceptance package under
`growth_report/`: one contact sheet spanning the optimization timeline and one
detailed four-panel PNG per selected checkpoint. Regenerate it independently:

```bash
python -m organic_motor.visualization.growth_report path/to/run/checkpoints
```

MotorGenesis defaults to float64 for CPU verification. Set
`MOTORGENESIS_X64=0` for float32 accelerator growth runs. Generate an explicit
precision and grid-convergence report with:

```bash
python -m organic_motor.experiments.precision_study --grids 32 48 64
```

## Native 3-D organic motor

The three-dimensional path uses independent `(Nx, Ny, Nz)` design and field
arrays.  It is not a z-extrusion of the 2-D benchmark.  The powered forward
model solves terminal-driven three-phase conduction, vector magnetostatics,
closed-surface Maxwell torque, Joule/iron losses, and steady heat conduction.

Validate the default 3-D geometry and memory estimate:

```bash
python -m organic_motor.experiments.motor3d_organic validate
```

Run a small CPU smoke simulation:

```bash
python -m organic_motor.experiments.motor3d_organic simulate \
  --shape 14,14,7 --angles 1 --maxwell-iters 30 --thermal-iters 60
```

Grow a design through native three-dimensional resolution stages:

```bash
python -m organic_motor.experiments.motor3d_organic grow \
  --shape 48,48,32 --levels 24,24,16:32,32,24:48,48,32 \
  --steps-per-level 20 --angles 3
```

Each run writes full 3-D NPZ fields, watertight material meshes, engineering
field sections, an organic isosurface view, and a checkpoint growth report.
The default coarse air gap is intentionally wider than the converged 2-D gap
so it spans roughly two cells; final engineering claims still require a grid
and air-padding convergence study.

Generate the native 3-D grid-convergence report:

```bash
python -m organic_motor.experiments.precision_study3d \
  --shapes 24,24,16 32,32,24 48,48,32
```

Run the subsequent thermal-mechanical and powered transient stage from a grown
design:

```bash
python -m organic_motor.experiments.motor3d_powered \
  --design organic_motor/out/motor3d_organic/final_design3d.npz \
  --angles 6 --steps 500 --out organic_motor/out/motor3d_powered
```

The powered stage is explicitly a quasi-static 3-D field-map transient with
three-phase RL/back-EMF dynamics.  It includes thermal expansion, centrifugal
loading, stress and air-gap collision diagnostics, but does not claim a full
eddy-current time-domain solution.
