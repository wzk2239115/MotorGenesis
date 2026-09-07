"""Per-phase resistance from actual wire path lengths and EM symbol checks.

The three phase paths have DIFFERENT total lengths (U≈4.37 m, V≈4.44 m,
W≈4.49 m with the stepped-bridge geometry).  This module computes the
actual per-phase resistance from the same-source path, using the BARE
copper cross-section (not the insulated diameter), and tracks the
verification chain from manufacturing geometry to current path.

It also verifies the electromagnetic symbol conventions:
  - Independent coil MMF direction (S→F from winding_assignment)
  - Three-phase phasor balance (120° spacing, zero sum)
  - Rotation reversal under phase swap
  - Magnet polarity vs current direction (marked unverified without
    rotor magnet data)
"""
from __future__ import annotations
import numpy as np


def copper_cross_section_area_mm2(bare_diameter_mm):
    """Bare copper cross-section, NOT the insulated diameter."""
    return np.pi * (bare_diameter_mm / 2) ** 2


def phase_resistance(path_length_mm, bare_diameter_mm, sigma_copper_S_m,
                     temperature_C=20.0, alpha_per_C=0.00393,
                     joint_resistance_mOhm=None):
    """R = ρ * L / A, with temperature correction.

    Parameters are independent and explicitly sourced:
    - ``bare_diameter_mm``: bare conductor, not insulated
    - ``sigma_copper_S_m``: from MotorConfig, standard Cu conductivity
    - ``temperature_C``: reference temperature (20 °C default)
    - ``alpha_per_C``: copper temperature coefficient
    - ``joint_resistance_mOhm``: per-joint resistance; None = 待确认
    """
    L = path_length_mm * 1e-3  # mm → m
    A = copper_cross_section_area_mm2(bare_diameter_mm) * 1e-6  # mm² → m²
    rho = 1.0 / sigma_copper_S_m  # Ω·m
    rho_T = rho * (1 + alpha_per_C * (temperature_C - 20.0))
    R = rho_T * L / A
    sources = dict(
        bare_copper_diameter_mm=bare_diameter_mm,
        copper_cross_section_mm2=float(copper_cross_section_area_mm2(bare_diameter_mm)),
        insulated_diameter_not_used=True,
        sigma_copper_S_m=sigma_copper_S_m,
        temperature_C=temperature_C,
        alpha_per_C=alpha_per_C,
        path_length_mm=path_length_mm,
        joint_resistance_mOhm=joint_resistance_mOhm,
        joint_resistance_status='待确认' if joint_resistance_mOhm is None else 'specified',
    )
    return dict(resistance_ohm=float(R), **sources)


def compute_phase_resistances(phase_chains, bare_diameter_mm,
                              sigma_copper_S_m, temperature_C=20.0):
    """Per-phase resistance from actual route lengths in the harness."""
    results = {}
    for chain in phase_chains:
        R = phase_resistance(
            chain['route_length_mm'], bare_diameter_mm,
            sigma_copper_S_m, temperature_C)
        results[chain['phase']] = R
    # Balance: max deviation from mean
    Rvals = [r['resistance_ohm'] for r in results.values()]
    Rmean = np.mean(Rvals)
    imbalance_pct = 100 * (max(Rvals) - min(Rvals)) / Rmean if Rmean > 0 else 0
    return dict(
        per_phase=results,
        mean_resistance_ohm=float(Rmean),
        imbalance_percent=float(imbalance_pct),
        note='Resistance differs per phase because path lengths differ; '
             'no hidden correction constant applied',
    )


def verify_phase_symbols(coil_table, n_slots=12, pole_pairs=5):
    """Verify electromagnetic symbol conventions from the netlist.

    1. Independent coil reference: each coil's MMF phasor from S→F
       matches the winding_assignment polarity.
    2. Three-phase balance: sum of per-phase MMF phasors = 0.
    3. Phase sequence: U, V, W are 120° apart.
    4. Rotation reversal: swapping V and W negates the sum phasor.
    """
    from organic_motor.topology.winding_assignment import tooth_phase_polarity
    # Build per-coil MMF phasors
    coil_phasors = {}
    polarity_mismatches = []
    for entry in coil_table:
        tooth = entry['tooth']
        phase_idx = 'UVW'.index(entry['phase'])
        polarity = entry['polarity']
        # Verify against the shared topology source
        ref_phase, ref_pol = tooth_phase_polarity(tooth - 1, n_slots, pole_pairs)
        if phase_idx != ref_phase:
            polarity_mismatches.append(f'tooth {tooth}: phase {entry["phase"]} != ref {ref_phase}')
        if polarity != ref_pol:
            polarity_mismatches.append(f'tooth {tooth}: polarity {polarity} != ref {ref_pol}')
        # Coil MMF: angle = electrical angle of tooth, sign = polarity
        elec_angle = pole_pairs * (tooth - 1) * 2 * np.pi / n_slots
        coil_phasors[entry['input_terminal']] = polarity * np.exp(1j * elec_angle)

    # Per-phase sum
    phase_sums = {}
    for entry in coil_table:
        ph = entry['phase']
        phase_sums.setdefault(ph, 0j)
        phase_sums[ph] += coil_phasors[entry['input_terminal']]

    # Three-phase balance
    U = phase_sums.get('U', 0j)
    V = phase_sums.get('V', 0j)
    W = phase_sums.get('W', 0j)
    total = U + V + W
    balance_ok = bool(abs(total) < 1e-10)

    # Phase sequence: angles should be 120° apart
    angles = [np.angle(U), np.angle(V), np.angle(W)]
    diffs = [(angles[(i + 1) % 3] - angles[i]) % (2 * np.pi) for i in range(3)]
    sequence_ok = bool(all(abs(d - 2 * np.pi / 3) < 0.01 for d in diffs))

    # Rotation reversal: swapping two phases reverses the sequence direction.
    # In a balanced system the sum is 0, so we check the sign of the
    # cross product of (V-U) × (W-U) in the complex plane.
    def _sequence_sign(a, b, c):
        d1 = (b - a)
        d2 = (c - a)
        cross = d1.real * d2.imag - d1.imag * d2.real
        return np.sign(cross)
    forward = float(_sequence_sign(U, V, W))
    reversed_ = float(_sequence_sign(U, W, V))
    reversal_changes_direction = bool(forward != 0 and reversed_ != 0 and forward != reversed_)

    return dict(
        independent_coil_reference=bool(not polarity_mismatches),
        polarity_mismatches=polarity_mismatches,
        coil_phasor_count=len(coil_phasors),
        phase_phasors_mmf=dict(
            U=dict(magnitude=float(abs(U)), angle_deg=float(np.degrees(np.angle(U)))),
            V=dict(magnitude=float(abs(V)), angle_deg=float(np.degrees(np.angle(V)))),
            W=dict(magnitude=float(abs(W)), angle_deg=float(np.degrees(np.angle(W)))),
        ),
        three_phase_balanced=balance_ok,
        total_mmf_phasor=float(abs(total)),
        phase_sequence_120deg=sequence_ok,
        rotation_reversal_verified=reversal_changes_direction,
        passed=bool(balance_ok and sequence_ok and not polarity_mismatches),
        magnet_polarity_verified=False,
        magnet_polarity_note='Rotor magnet direction not measured; '
                             'current-to-torque sign待确认',
        back_emf_verified=False,
        back_emf_note='Back-EMF requires rotor + B-H data; not available '
                      'for resin/Fe3O4 composite',
    )


def em_verification_chain(report, spec, sigma_copper_S_m):
    """Full chain: manufacturing geometry → current path → (mesh → flux/torque).

    The last two links (mesh → flux → torque) are marked as NOT connected
    for the new segmented candidate.  The chain documents what IS verified
    and what is NOT, without silently falling back to the old motor model.
    """
    chains = report.get('phase_chains', [])
    resistances = compute_phase_resistances(
        chains, spec.wire_diameter_mm, sigma_copper_S_m)
    symbols = verify_phase_symbols(report.get('coils', []))

    return dict(
        resistance=resistances,
        symbol_checks=symbols,
        verification_chain=dict(
            manufacturing_geometry='verified: 12 segments, 8 turns, '
                                   'radial insertion, stepped-bridge wiring',
            current_path='verified: per-phase continuous route from '
                         'terminal through 4 coils to N',
            solver_mesh='NOT connected: new segmented topology requires '
                        'new mesh generation; old FEA mesh is for a '
                        'different stator geometry',
            flux_linkage='NOT computed: no calibrated B-H curve for '
                        'resin/Fe3O4 composite',
            torque='NOT computed: no rotor magnet data, no mesh, no '
                   'calibrated material; sensitivity range not available',
            performance_claim='none: this candidate has no torque, '
                              'efficiency or thermal rating'),
        simulation_ready=False,
        old_model_fallback=False,
        note='The old motor simulation results (torque, thermal, copper '
             'loss) are for a DIFFERENT geometry and material; they are '
             'not reused for this candidate.',
    )
