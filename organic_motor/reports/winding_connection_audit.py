"""Read-only diagnostic: coil solids are not the same as connected phases."""
import json
from pathlib import Path
import numpy as np
from organic_motor.config3d import MOTOR_SPEC
from organic_motor.construct.winding_netlist import PrintedCoilNetlist


def summarize(table,pole_pairs):
    phasors=[sum(sign*np.exp(1j*pole_pairs*2*np.pi*tooth/12) for tooth,phase,sign in table if phase==p) for p in range(3)]
    return {'table_zero_based':table,'fundamental_axis_phasor_magnitudes':np.abs(phasors).tolist(),
            'fundamental_axis_phasor_angles_deg':np.angle(phasors,deg=True).tolist(),
            'note':'Equal-turn radial tooth-axis phasor check only; not a field-solved torque or wiring release'}


def run():
    printed=PrintedCoilNetlist(n_slots=12,pole_pairs=5,n_phases=3,turns_per_coil=8)
    canonical=[(i,MOTOR_SPEC.phase_of_slot(i),MOTOR_SPEC.polarity_of_slot(i)) for i in range(12)]
    result={'geometry_findings':['Manufacturing generator realizes three continuous phase routes, four coils each','S/F table, nine series links and U/V/W/N coordinates are exported in wiring.json'],
            'legacy_checker_expected_components_per_phase':printed.expected_phase_components().tolist(),
            'canonical_MOTOR_SPEC':summarize(canonical,5),
            'explicit_12s10p_printed_netlist':summarize(printed.coil_table(),5),
            'winding_ready_for_power':False,
            'required':['Design rear harness retention and terminal hardware','Qualify insulation, bend radius and physical joints','Validate candidate current/flux polarity and torque with new material measurements']}
    out=Path('organic_motor/reports/fabrication/winding_connection_audit.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':run()
