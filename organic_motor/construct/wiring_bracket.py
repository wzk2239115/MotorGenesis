"""FDM-printable rear wiring bracket, cable clamps and terminal posts.

The bracket is a thin annular mounting ring below all wire routing levels.
Tall cable clamps reach up from the bracket to each phase's bridge arc.
Terminal posts define the U/V/W drive and N star-point connection envelopes
with parametric采购尺寸.  No terminal part number is assumed without
verification; fastener sizes are marked待确认.

Coordinates are millimetres, consistent with wound_prototype.
"""
from __future__ import annotations
import numpy as np
import trimesh
import manifold3d


def bracket_params():
    return dict(
        inner_radius=58.0,
        outer_radius=82.0,
        z_top=-44.0,
        z_bottom=-48.0,
        mount_radius=54.0,
        mount_hole_radius=1.8,
        # Clamp positions: 3 per phase, angularly offset so they don't overlap
        # One clamp per bridge: (z, r, phase, bridge_idx, angle)
        # Radii match series_bridge offset: 70+3*phase+bridge_idx*1.0
        # Angles offset so clamps at different radii don't collide
        clamp_config=[
            (-34.0, 70.0, 'U', 0, 0),
            (-34.0, 71.0, 'U', 1, 2*np.pi/3),
            (-34.0, 72.0, 'U', 2, 4*np.pi/3),
            (-38.0, 73.0, 'V', 0, np.pi/3),
            (-38.0, 74.0, 'V', 1, np.pi),
            (-38.0, 75.0, 'V', 2, 5*np.pi/3),
            (-42.0, 76.0, 'W', 0, np.pi/6),
            (-42.0, 77.0, 'W', 1, 5*np.pi/6),
            (-42.0, 78.0, 'W', 2, 3*np.pi/2),
        ],
        clamp_width=6.0,        # tangential
        clamp_depth=3.0,        # radial (separates phase arcs at 3 mm spacing)
        clamp_groove_radius=0.8,
        clamp_screw_radius=1.65,
        terminal_post_radius=4.0,
        terminal_post_height=10.0,
        terminal_screw_radius=1.65,
        n_terminal_radius=6.0,
        n_terminal_height=10.0,
        contact_region_radius=3.0,
    )


def _manifold(mesh):
    return manifold3d.Manifold(manifold3d.Mesh(
        np.asarray(mesh.vertices, np.float32),
        np.asarray(mesh.faces, np.uint32)))


def _mesh(solid):
    m = solid.to_mesh()
    verts = np.asarray(m.vert_properties, dtype=np.float32)
    faces = np.asarray(m.tri_verts, dtype=np.uint32)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def _difference(a, b):
    return _mesh(_manifold(a) - _manifold(b))


def bracket_base_field(x, y, z, spec):
    """SDF for the rear harness bracket base: thin solid ring with holes.

    Convention: field <= 0 is solid.  Holes use (radius - dist) > 0 inside.
    """
    p = bracket_params()
    r = np.hypot(x, y)
    body = np.maximum(np.maximum(p['inner_radius'] - r, r - p['outer_radius']),
                      np.maximum(z - p['z_top'], p['z_bottom'] - z))
    for i in range(6):
        a = i * np.pi / 3
        hole_dist = np.hypot(x - p['mount_radius'] * np.cos(a),
                             y - p['mount_radius'] * np.sin(a))
        body = np.maximum(body, p['mount_hole_radius'] - hole_dist)
    for z_level, r_bridge, _, _, angle in p['clamp_config']:
        cx, cy = r_bridge * np.cos(angle), r_bridge * np.sin(angle)
        clamp_dist = np.hypot(x - cx, y - cy)
        body = np.maximum(body, p['clamp_screw_radius'] - clamp_dist)
    return body


def cable_clamp(z_level, r_bridge, angle):
    """Tall clamp block reaching from bracket top up to the wire level."""
    p = bracket_params()
    W = p['clamp_width']
    D = p['clamp_depth']
    height = z_level - p['z_top'] + 2  # from bracket top up to 2mm above wire
    # Build in local coords: X=radial, Y=tangential, Z=axial
    # Bottom of clamp at z=0, top at z=height
    block = trimesh.creation.box(extents=[D, W, height])
    block.apply_translation([0, 0, height / 2])
    # Wire groove: semi-cylinder along Y at the top, radius groove_r
    gr = p['clamp_groove_radius']
    groove = trimesh.creation.cylinder(radius=gr, height=W + 2, sections=24)
    groove.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    groove.apply_translation([0, 0, height - 2])
    # Screw hole: vertical cylinder along Z through the base
    screw = trimesh.creation.cylinder(radius=p['clamp_screw_radius'], height=height + 2, sections=24)
    m = _difference(block, groove)
    m = _difference(m, screw)
    m.fix_normals()
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
        m.fix_normals()
    # Position: center at (r_bridge*cos, r_bridge*sin, z_top) so base sits on bracket
    m.apply_translation([r_bridge, 0, p['z_top']])
    m.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
    return m


def terminal_post(position, radius, height, screw_axis='y'):
    """Terminal post with a transverse set-screw hole."""
    p = bracket_params()
    post = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    post.apply_translation([0, 0, height / 2])
    screw = trimesh.creation.cylinder(
        radius=p['terminal_screw_radius'], height=radius * 3, sections=24)
    if screw_axis == 'x':
        screw.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [0, 1, 0]))
    else:
        screw.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0]))
    screw.apply_translation([0, 0, height * 0.6])
    m = _difference(post, screw)
    m.apply_translation(position)
    m.fix_normals()
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
        m.fix_normals()
    return m


def build_terminal_assets(report):
    """Generate terminal post meshes and contact-region definitions."""
    p = bracket_params()
    terminals = {}
    contacts = []
    for phase in 'UVW':
        pos = np.array(report['terminals_mm'][phase])
        terminals['terminal_' + phase] = terminal_post(
            pos, p['terminal_post_radius'], p['terminal_post_height'],
            screw_axis='y' if abs(pos[0]) > abs(pos[1]) else 'x')
        contacts.append(dict(terminal='terminal_' + phase, phase='phase_' + phase,
                              position_mm=pos.tolist(),
                              contact_radius_mm=p['contact_region_radius']))
    star = np.array(report['star_point_mm'])
    terminals['terminal_N'] = terminal_post(
        star, p['n_terminal_radius'], p['n_terminal_height'], screw_axis='x')
    contacts.append(dict(terminal='terminal_N', phase='phase_U,phase_V,phase_W',
                         position_mm=star.tolist(),
                         contact_radius_mm=p['contact_region_radius']))
    return terminals, contacts


def build_clamp_assets():
    """Generate cable clamp meshes, one per bridge."""
    p = bracket_params()
    clamps = {}
    clamp_list = []
    for z_level, r_bridge, phase_label, bridge_idx, angle in p['clamp_config']:
        name = f'clamp_{phase_label}_{bridge_idx}'
        clamps[name] = cable_clamp(z_level, r_bridge, angle)
        clamp_list.append(dict(
            name=name, phase=phase_label, bridge_idx=bridge_idx,
            r_mm=float(r_bridge), z_mm=float(z_level), angle_rad=float(angle),
            position_mm=[float(r_bridge * np.cos(angle)),
                         float(r_bridge * np.sin(angle)), float(z_level)]))
    return clamps, clamp_list


def fixing_report():
    """Per-segment wire fixing method summary for the harness report."""
    p = bracket_params()
    n_clamps = len(p['clamp_config'])
    return dict(
        bracket=f'FDM thin mounting ring r={p["inner_radius"]}-{p["outer_radius"]} mm, z={p["z_top"]} to {p["z_bottom"]} mm',
        mounting=f'6 tie-rod holes at r={p["mount_radius"]} mm; bracket secured by extended 130 mm tie rods + M3 nuts below rear retainer nuts',
        cable_clamps=f'{n_clamps} clamps (one per bridge, per-bridge radial offset 1.0 mm)',
        clamp_details=[dict(phase=ph, bridge_idx=bi, r_mm=r, z_mm=z, angle_deg=round(float(a)*180/np.pi,1))
                       for z, r, ph, bi, a in p['clamp_config']],
        terminals='Cylindrical terminal posts with M3 set-screw;采购型号待确认',
        terminal_positions='U/V/W at phase start endpoints; N at star point',
        assembly_sequence=[
            '1. Assemble motor with rear retainer, tie rods and rear nuts torqued',
            '2. Slide harness bracket over tie rods below rear nuts',
            '3. Add bracket washers and nuts; torque bracket nuts',
            '4. Route phase leads down to terminal positions at r=65',
            '5. Lay phase bridge arcs at routing levels (z=-34/-38/-42, r=70/73/76)',
            '6. Install cable clamps (3 per phase) with M3 screws from below bracket',
            '7. Install terminal posts at U/V/W and N positions',
            '8. Strip enamel and crimp/solder leads to terminal posts',
            '9. Verify insulation; install rotor after electrical checks',
        ],
        tool_access=[
            dict(step='Rear nut torque', tool='M3 socket', access='axial from -Z before bracket install', note='Must remove bracket to re-torque rear nuts'),
            dict(step='Bracket nut torque', tool='M3 socket', access='axial from -Z', note='Accessible after bracket install, before clamps'),
            dict(step='Cable clamp screws', tool='M3 hex', access='from -Z below bracket', note='Remove individual clamps to re-route wires'),
            dict(step='Terminal set-screws', tool='M3 hex', access='radial from outer side', note='Accessible with clamps installed'),
        ],
        status='Bracket and terminal geometry defined; FDM tolerance, terminal采购型号 and dielectric rating待确认',
    )
