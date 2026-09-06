"""E3: startup-parameter provenance — where every number comes from.

The audit requires each dynamic-model parameter to carry its source
(FEA / analytical / centerline / config default) and verification level.
This table is the machine-readable record; `format_provenance_table`
renders it for reports and the web UI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterProvenance:
    symbol: str
    description: str
    unit: str
    source: str          # e.g. "centerline analytical", "FEA line integral"
    method: str          # how it is computed
    verified_against: str  # what it has been cross-checked with
    confidence: str      # "verified" | "estimate" | "unverified"


PROVENANCE: tuple[ParameterProvenance, ...] = (
    ParameterProvenance(
        "R_phase", "per-phase winding resistance", "ohm",
        "centerline analytical",
        "rho*L/A summed over registry polylines in series "
        "(line_current.centerline_resistance)",
        "voxel |J|^2/sigma dissipation (path B5 unification pending)",
        "estimate",
    ),
    ParameterProvenance(
        "L_phase", "synchronous inductance", "H",
        "air-gap reluctance analytical",
        "N^2*mu0*A_pole/(2*g_eff), gap dominated (mu_r iron excluded "
        "by design)",
        "no independent check yet (needs locked-rotor AC test or FEA "
        "energy perturbation)",
        "unverified",
    ),
    ParameterProvenance(
        "psi_pm", "PM flux linkage", "Wb",
        "FEA line integral",
        "closed-loop integral of A_PM along winding centerlines over "
        "one electrical cycle, fundamental fit "
        "(transient_bridge.extract_fea_flux_linkage)",
        "reversal/scaling linearity of the reference coil battery; "
        "absolute value not yet vs external solver",
        "estimate",
    ),
    ParameterProvenance(
        "J_rotor", "rotor inertia", "kg m^2",
        "config default",
        "fixed 2e-4 unless overridden — NOT computed from the realized "
        "rotor density field",
        "none (should be integral of rho*r^2 over rotor voxels)",
        "unverified",
    ),
    ParameterProvenance(
        "T_load", "load torque constant", "N m",
        "user input",
        "scenario parameter for startup tests",
        "n/a (boundary condition, not a property)",
        "verified",
    ),
    ParameterProvenance(
        "T1/T0/T2 maps", "torque map decomposition", "N m",
        "FEA Maxwell stress",
        "zero/plus/minus current solves per angle on the realized fields "
        "(compute_powered_maps)",
        "reference-coil Biot-Savart battery for the underlying solver; "
        "map phase convention cross-checked vs back-EMF (cos axis); "
        "QUALITY GATE active: at 96^3 the real-motor maps exceed the "
        "physical bound (T1 up to 4935 Nm vs 1.5*p*psi*I_nom*x2.5 = 147) "
        "at specific (phase, angle) alignments — phase-1 current at map "
        "angles 2/3/6-of-6 — magnetostatic solve failures; transient "
        "REFUSED until maps are fixed (finer grid / more iterations)",
        "unverified (gate active)",
    ),
    ParameterProvenance(
        "back-EMF", "sinusoidal phase EMF", "V",
        "analytical from psi_pm",
        "p*omega*psi*cos(p*theta+s_p); cos convention consistent with "
        "the torque maps (energy identity tested <5%)",
        "RL energy balance test (test_powered_control)",
        "verified",
    ),
    ParameterProvenance(
        "k_hyst, k_eddy", "iron loss coefficients", "-",
        "config defaults",
        "cfg.iron_loss_coeff / eddy_loss_coefficient with B_ref scaling",
        "no datasheet curve fitted yet",
        "unverified",
    ),
    ParameterProvenance(
        "alpha_R(T)", "copper resistivity temp coefficient", "1/K",
        "literature",
        "0.00393 /K linear model about 20 degC",
        "standard copper value (handbook)",
        "verified",
    ),
    ParameterProvenance(
        "k_pm(T)", "NdFeB remanence temp coefficient", "1/K",
        "literature",
        "-0.0012 /K linear model about 20 degC (grade-typical)",
        "typical N35-N45 range; exact grade not pinned",
        "estimate",
    ),
    ParameterProvenance(
        "c_f windage", "air friction coefficients", "-",
        "derived + engineering estimate",
        "laminar side exact (2/Re_delta, tested); disks Daily-Nece "
        "(3.87/sqrt(Re), 0.146/Re^0.2); turbulent side 0.08/Re^0.25 "
        "flagged estimate",
        "laminar branch unit-tested against analytic value",
        "estimate",
    ),
    ParameterProvenance(
        "h_gap", "air-gap convection", "W/m2K",
        "correlation",
        "laminar exact k/delta; Taylor Nu=max(1,0.2*Ta^0.25) with "
        "prefactor flagged (sources 0.1-0.4)",
        "continuity + laminar limit tested",
        "estimate",
    ),
)


def format_provenance_table(markdown: bool = True) -> str:
    """Render the provenance table (Markdown or TSV)."""
    header = ["symbol", "description", "unit", "source", "confidence"]
    if markdown:
        lines = [
            "| 参数 | 含义 | 单位 | 来源 | 置信度 |",
            "|---|---|---|---|---|",
        ]
        for p in PROVENANCE:
            lines.append(
                f"| `{p.symbol}` | {p.description} | {p.unit} | "
                f"{p.source} | {p.confidence} |"
            )
        return "\n".join(lines)
    lines = ["\t".join(header)]
    for p in PROVENANCE:
        lines.append("\t".join([
            p.symbol, p.description, p.unit, p.source, p.confidence,
        ]))
    return "\n".join(lines)


def confidence_counts() -> dict[str, int]:
    out = {"verified": 0, "estimate": 0, "unverified": 0}
    for p in PROVENANCE:
        out[p.confidence] = out.get(p.confidence, 0) + 1
    return out
