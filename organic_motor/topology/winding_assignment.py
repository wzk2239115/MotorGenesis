"""Shared integer phase-belt convention: +Z view, increasing tooth angle.

Phase indices 0/1/2 correspond to U/V/W. Polarity applies to current from
local winding start S to finish F; it must not also reverse the geometry.
"""
def tooth_phase_polarity(tooth, n_slots=12, pole_pairs=5):
    if not 0<=tooth<n_slots or n_slots%3 or pole_pairs<1:
        raise ValueError('Invalid three-phase winding specification')
    belt=((6*pole_pairs*tooth+n_slots//2)//n_slots)%6
    return (0,2,1,0,2,1)[belt], (1 if belt in (0,2,4) else -1)
