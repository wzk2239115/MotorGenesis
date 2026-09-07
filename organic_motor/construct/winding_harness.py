"""Realized star winding: three continuous routes, four coils per phase.

Coordinates mm. Local coil geometry is traversed in reverse for negative
polarity. External interconnects occupy separate rear axial routing levels.
"""
import numpy as np
from scipy.spatial import cKDTree
from organic_motor.topology.winding_assignment import tooth_phase_polarity


def resample_polyline(points,step=.15):
    points=np.asarray(points,float)
    result=[]
    for a,b in zip(points[:-1],points[1:]):
        n=max(2,int(np.ceil(np.linalg.norm(b-a)/step))+1)
        result.extend(np.linspace(a,b,n,endpoint=False))
    return np.vstack([result,points[-1]])


def rounded_path(points,radius=1.0,step=.15):
    """Quadratic corner transitions with exact retained terminal endpoints."""
    p=np.asarray(points,float);out=[p[0]]
    for i in range(1,len(p)-1):
        incoming=p[i-1]-p[i];outgoing=p[i+1]-p[i]
        a=np.linalg.norm(incoming);b=np.linalg.norm(outgoing)
        if min(a,b)<1e-8:continue
        cut=min(radius,.35*a,.35*b)
        before=p[i]+incoming/a*cut;after=p[i]+outgoing/b*cut
        out.extend(resample_polyline([out[-1],before],step)[1:])
        for t in np.linspace(0,1,max(3,int(2*cut/step)+1))[1:]:out.append((1-t)**2*before+2*(1-t)*t*p[i]+t*t*after)
    out.extend(resample_polyline([out[-1],p[-1]],step)[1:])
    return np.array(out)


def lead_path(endpoint,level):
    a=np.arctan2(endpoint[1],endpoint[0]);xy=65*np.array([np.cos(a),np.sin(a)])
    return rounded_path([endpoint,[endpoint[0],endpoint[1],28.25],[*xy,28.25],[*xy,level]])


def series_bridge(a,b,phase,bridge_idx=0):
    # Each bridge within a phase is offset radially by 1.0 mm so that
    # adjacent arcs (whose S/F endpoints differ by ~0.9°) do not coincide.
    # Transitions step up 2 mm in z to avoid crossing other bridges' arcs.
    level=-34-4*phase;radius=70+3*phase+bridge_idx*1.0
    la=lead_path(a,level);lb=lead_path(b,level)
    aa=np.arctan2(a[1],a[0]);ab=np.arctan2(b[1],b[0]);delta=(ab-aa)%(2*np.pi)
    if delta>np.pi:delta-=2*np.pi
    angles=np.linspace(aa,aa+delta,max(12,int(abs(delta)*radius/.15)))
    arc=np.column_stack([radius*np.cos(angles),radius*np.sin(angles),np.full(len(angles),level)])
    dz=2.0
    mid=rounded_path([la[-1],[la[-1][0],la[-1][1],level+dz],
        [arc[0][0],arc[0][1],level+dz],arc[0],*arc[1:-1],arc[-1],
        [arc[-1][0],arc[-1][1],level+dz],[lb[-1][0],lb[-1][1],level+dz],lb[-1]],radius=.8)
    return np.vstack([la,mid[1:],lb[-2::-1]])


def make_harness(local_route,turns=8):
    from organic_motor.construct.wound_prototype import rotation
    coils={};table=[]
    for i in range(12):
        phase,polarity=tooth_phase_polarity(i,12,5)
        coils[i]=np.einsum('ij,kj->ik',local_route,rotation(i*np.pi/6)[:3,:3])
        table.append(dict(tooth=i+1,phase='UVW'[phase],polarity=polarity,turns=turns,
                          input_terminal=f'C{i+1:02d}.'+('S' if polarity>0 else 'F'),
                          output_terminal=f'C{i+1:02d}.'+('F' if polarity>0 else 'S'),
                          S_mm=coils[i][0].tolist(),F_mm=coils[i][-1].tolist()))
    paths={};chains=[];port_paths=[]
    star=np.array([0.,0.,-47.]);terminals={};links=[]
    for phase in range(3):
        entries=[e for e in table if e['phase']=='UVW'[phase]]
        oriented=[coils[e['tooth']-1][::e['polarity']] for e in entries]
        start_lead=lead_path(oriented[0][0],-34-4*phase)
        terminals['UVW'[phase]]=start_lead[-1].tolist()
        segments=[start_lead[::-1],oriented[0][1:]]
        for idx,(left,right,e1,e2) in enumerate(zip(oriented[:-1],oriented[1:],entries[:-1],entries[1:])):
            bridge=series_bridge(left[-1],right[0],phase,idx)
            segments.extend([bridge[1:],right[1:]])
            links.append(dict(phase='UVW'[phase],source=e1['output_terminal'],target=e2['input_terminal']))
        last_lead=lead_path(oriented[-1][-1],-47)
        star_link=rounded_path([last_lead[-1],star],radius=1)
        segments.extend([last_lead[1:],star_link[1:]])
        path=np.vstack(segments);distance=np.linalg.norm(np.diff(path,axis=0),axis=1)
        path=path[np.r_[True,distance>1e-8]]
        paths['UVW'[phase]]=path
        chains.append(dict(phase='UVW'[phase],coils=entries,start='UVW'[phase],finish='N',
                           route_length_mm=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())))
    for route in coils.values():
        for point in route[[0,-1]]:port_paths.append(lead_path(point,-34))
    report=dict(connection='star',n_slots=12,pole_pairs=5,turns_per_coil=turns,
        convention='View from +Z; tooth 1 at +X; numbering increases counterclockwise; positive coil current S to F',
        coils=table,phase_chains=chains,series_links=links,terminals_mm=terminals,star_point_mm=star.tolist(),
        electrical_topology_complete=True,performance_validated=False,
        assembly_sequence=['Wind four bobbins per phase following S/F order','Insert the twelve core segments','Route leads through the front retainer slots before closing the adapter','Lay the three phase chains in separate rear levels','Join the three finish ends at N and connect external drive to U/V/W'],
        caution='Insulated conductor routing geometry; star joint and external terminals require actual solder/crimp qualification. No calibrated torque or dielectric rating.')
    return paths,report,np.vstack(port_paths)


def verify_harness(paths,report,wire_outer_radius=.335):
    # Each entire phase is one ordered route, with terminal->4 coils->N topology.
    assert len(report['series_links'])==9
    assert sorted(e['tooth'] for e in report['coils'])==list(range(1,13))
    for name,path in paths.items():
        assert np.allclose(path[0],report['terminals_mm'][name])
        assert np.allclose(path[-1],report['star_point_mm'])
        assert len([e for e in report['coils'] if e['phase']==name])==4
    gaps=[]
    for i,a in enumerate('UVW'):
        for b in 'UVW'[i+1:]:
            # The common N pad occupies radius 2 mm around the star position.
            star=np.array(report['star_point_mm'])
            pa=paths[a][np.linalg.norm(paths[a]-star,axis=1)>2.5]
            pb=paths[b][np.linalg.norm(paths[b]-star,axis=1)>2.5]
            gap=float(cKDTree(pa).query(pb)[0].min()-2*wire_outer_radius)
            gaps.append(dict(phases=a+b,envelope_gap_mm=gap))
    return dict(phase_routes=3,coils_per_phase=4,series_links=9,common_node='N',
                phase_gaps=gaps,passed=all(g['envelope_gap_mm']>.2 for g in gaps),
                method='Ordered full phase paths plus sampled interphase distance, excluding N joint region',
                limitations='Nominal geometry only; does not certify insulation voltage or joint resistance')
