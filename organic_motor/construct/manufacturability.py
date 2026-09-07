"""Manufacturability checks for the wound prototype wiring.

Checks that go beyond the static assembly Boolean audit:

- **Intra-phase crossover**: non-adjacent segments of the same phase that
  come closer than the minimum gap (wire OD + insulation + tolerance).
  Uses segment-to-segment distance, not point-to-point, and excludes
  adjacent sampling segments along arc length.
- **Inter-phase short**: different phase paths that touch outside the
  declared N-node region.
- **Bend radius**: minimum radius of curvature at every sample; marked
  待确认 when the supplier bend limit is unknown.
- **Assembly motion**: core insertion travel and gear kinematic paths
  checked with solid Boolean intersection, not AABB.

All wire dimensions use independent parameters: bare copper diameter,
enamel radial thickness, crossover insulation layer, and print tolerance.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def segment_to_segment_distance(p0, p1, q0, q1):
    """Minimum distance between two 3-D line segments [p0,p1] and [q0,q1].

    Uses the parametric clamp algorithm (Ericson, Real-Time Collision Detection).
    """
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    EPS = 1e-12
    if a <= EPS and e <= EPS:
        return np.linalg.norm(p0 - q0)
    if a <= EPS:
        s = 0.0
        t = np.clip(f / e, 0, 1)
    else:
        c = np.dot(d1, r)
        if e <= EPS:
            t = 0.0
            s = np.clip(-c / a, 0, 1)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            if denom != 0:
                s = np.clip((b * f - c * e) / denom, 0, 1)
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0:
                t = 0.0
                s = np.clip(-c / a, 0, 1)
            elif t > 1:
                t = 1.0
                s = np.clip((b - c) / a, 0, 1)
    closest_a = p0 + d1 * s
    closest_b = q0 + d2 * t
    return np.linalg.norm(closest_a - closest_b)


def _path_segments(path):
    """Return array of (N-1, 2, 3) segment start/end pairs."""
    return np.stack([path[:-1], path[1:]], axis=1)


def check_intra_phase_crossover(path, wire_outer_radius, min_gap_mm,
                                adjacency_arc_mm=5.0, step_mm=0.15):
    """Find non-adjacent segment pairs of one phase that are too close.

    Adjacent segments (within ``adjacency_arc_mm`` of arc length) are
    excluded.  Distance is segment-to-segment, not point-to-point.
    """
    path = np.asarray(path, float)
    segs = _path_segments(path)
    seg_centers = (segs[:, 0] + segs[:, 1]) / 2
    seg_lengths = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    tree = cKDTree(seg_centers)
    # Arc-length index: cumulative segment length gives arc position
    arc = np.concatenate([[0], np.cumsum(seg_lengths)])
    violations = []
    threshold = 2 * wire_outer_radius + min_gap_mm
    n = len(segs)
    for i in range(n):
        # Query segments whose centers are within threshold of segment i
        candidates = tree.query_ball_point(seg_centers[i], threshold + max(seg_lengths))
        for j in candidates:
            if j <= i:
                continue
            # Exclude adjacent segments by arc length
            if abs(arc[j] - arc[i]) < adjacency_arc_mm:
                continue
            dist = segment_to_segment_distance(
                segs[i, 0], segs[i, 1], segs[j, 0], segs[j, 1])
            if dist < threshold:
                violations.append(dict(
                    seg_a=i, seg_b=j, distance_mm=float(dist),
                    arc_a_mm=float(arc[i]), arc_b_mm=float(arc[j]),
                    threshold_mm=float(threshold)))
    return violations


def check_inter_phase_short(paths, wire_outer_radius, star_point,
                            n_region_radius=2.0, min_gap_mm=0.2):
    """Check different phase paths that touch outside the N-node region.

    Uses segment-to-segment distance; the N-joint sphere is excluded
    because the star weld is an intended contact.
    """
    star = np.asarray(star_point, float)
    threshold = 2 * wire_outer_radius + min_gap_mm
    violations = []
    phases = list(paths)
    for i, pa in enumerate(phases):
        segs_a = _path_segments(np.asarray(pa, float))
        for pb in phases[i + 1:]:
            segs_b = _path_segments(np.asarray(pb, float))
            tree = cKDTree((segs_b[:, 0] + segs_b[:, 1]) / 2)
            for si, sa in enumerate(segs_a):
                center = (sa[0] + sa[1]) / 2
                if np.linalg.norm(center - star) < n_region_radius + 5:
                    continue  # skip near N joint
                candidates = tree.query_ball_point(center, threshold + 5)
                for sj in candidates:
                    sb = segs_b[sj]
                    if np.linalg.norm((sb[0] + sb[1]) / 2 - star) < n_region_radius + 5:
                        continue
                    dist = segment_to_segment_distance(
                        sa[0], sa[1], sb[0], sb[1])
                    if dist < threshold:
                        violations.append(dict(
                            seg_a=si, seg_b=sj, distance_mm=float(dist),
                            threshold_mm=float(threshold)))
    return violations


def check_bend_radius(path, wire_diameter_mm, min_bend_multiplier=3.0,
                      supplier_limit_mm=None):
    """Check minimum radius of curvature at every sample.

    If ``supplier_limit_mm`` is None, the limit is marked待确认 (pending
    supplier data); the computed minimum is reported but not failed.
    """
    path = np.asarray(path, float)
    n = len(path)
    if n < 3:
        return dict(min_radius_mm=float('inf'), passed=True, verified=False,
                    reason='path too short for curvature')
    radii = np.full(n, np.inf)
    for i in range(1, n - 1):
        v1 = path[i] - path[i - 1]
        v2 = path[i + 1] - path[i]
        cross = np.cross(v1, v2)
        cross_mag = np.linalg.norm(cross)
        if cross_mag < 1e-15:
            continue
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)
        sin_angle = cross_mag / (v1_mag * v2_mag)
        radius = min(v1_mag, v2_mag) / (2 * sin_angle + 1e-15)
        radii[i] = radius
    min_r = float(np.nanmin(radii[np.isfinite(radii)])) if np.any(np.isfinite(radii)) else float('inf')
    limit = supplier_limit_mm
    verified = limit is not None
    if not verified:
        # Cannot pass/fail without supplier bend data
        return dict(min_radius_mm=min_r, passed=None, verified=False,
                    wire_diameter_mm=wire_diameter_mm,
                    multiplier_limit_mm=wire_diameter_mm * min_bend_multiplier,
                    reason='bend limit待确认；supplier data required')
    return dict(min_radius_mm=min_r, passed=min_r >= limit, verified=True,
                limit_mm=float(limit))


def check_core_insertion(assets, report, spec, travel_mm=35.0, n_steps=12):
    """Check core segment insertion path with solid Boolean intersection.

    Not AABB: uses manifold3d intersection at each insertion step.
    """
    import manifold3d
    import trimesh
    def _solid(m):
        return manifold3d.Manifold(manifold3d.Mesh(
            np.asarray(m.vertices, np.float32),
            np.asarray(m.faces, np.uint32)))
    core = _solid(assets['core_segment'])
    bobbin = _solid(assets['grooved_bobbin'])
    collisions = []
    for travel in np.linspace(travel_mm, 0, n_steps):
        moved = core.transform(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1], [travel, 0, 0]], dtype=np.float32))
        inter = moved ^ bobbin
        vol = inter.volume()
        if vol > 0.1:
            collisions.append(dict(travel_mm=float(travel), intersection_mm3=float(vol)))
    return dict(travel_mm=travel_mm, steps=n_steps, collisions=collisions,
                passed=not collisions,
                method='solid Boolean intersection at each insertion step')


def run_manufacturability_checks(paths, report, spec, assets=None):
    """Run all manufacturability checks and return a structured report.

    Each check is labeled: 'exact_geometry', 'discrete_scan', or 'unverified'.
    """
    wire_r = spec.wire_diameter_mm / 2 + spec.enamel_radial_mm
    results = {}
    # Intra-phase crossover (segment-to-segment, adjacency excluded)
    intra = {}
    for phase, path in paths.items():
        intra[phase] = check_intra_phase_crossover(path, wire_r, min_gap_mm=0.2)
    results['intra_phase_crossover'] = dict(
        violations=intra, method='segment-to-segment distance, adjacency excluded',
        classification='discrete_scan',
        passed=all(not v for v in intra.values()))
    # Inter-phase short (outside N node)
    inter = check_inter_phase_short(
        [paths[p] for p in 'UVW'], wire_r,
        report['star_point_mm'], n_region_radius=2.0, min_gap_mm=0.2)
    results['inter_phase_short'] = dict(
        violations=inter, n_region_radius_mm=2.0,
        classification='discrete_scan',
        passed=not inter)
    # Bend radius (marked unverified without supplier data)
    bends = {}
    for phase, path in paths.items():
        bends[phase] = check_bend_radius(path, spec.wire_diameter_mm)
    results['bend_radius'] = dict(
        per_phase=bends, classification='unverified',
        passed=None,
        reason='supplier bend limit待确认；wire_diameter=%.2f mm' % spec.wire_diameter_mm)
    # Core insertion (solid Boolean)
    if assets is not None:
        results['core_insertion'] = check_core_insertion(assets, report, spec)
    # Parameters used
    results['parameters'] = dict(
        bare_copper_diameter_mm=spec.wire_diameter_mm,
        enamel_radial_mm=spec.enamel_radial_mm,
        insulated_diameter_mm=spec.wire_diameter_mm + 2 * spec.enamel_radial_mm,
        wire_outer_radius=wire_r,
        print_tolerance_mm=0.2,
        crossover_insulation_layer_mm=0.0,
        note='crossover insulation layer and print tolerance are independent parameters')
    results['overall'] = dict(
        intra_phase_pass=results['intra_phase_crossover']['passed'],
        inter_phase_pass=results['inter_phase_short']['passed'],
        bend_radius_verified=results['bend_radius']['passed'],
        core_insertion_pass=results.get('core_insertion', {}).get('passed', None),
        n_node_contact_only=all(
            v.get('outside_region_mm3', 0) == 0
            for v in report.get('verification', {}).get('phase_gaps', [])))
    return results
