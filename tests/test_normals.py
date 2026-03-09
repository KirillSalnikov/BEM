"""
Check triangle normal orientation and RWG basis function consistency.
For a sphere centered at origin, all normals should point outward (dot with centroid > 0).
"""
import numpy as np
from bem_core import icosphere, build_rwg

for refine in [1, 2]:
    verts, tris = icosphere(1.0, refinements=refine)
    rwg = build_rwg(verts, tris)
    N = rwg['N']
    print(f"\nRefine={refine}: {len(tris)} tris, {N} RWG")

    # Check triangle normals
    v0 = verts[tris[:,0]]; v1 = verts[tris[:,1]]; v2 = verts[tris[:,2]]
    normals = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(normals, axis=1)
    normals_hat = normals / (2 * areas[:,None])
    centroids = (v0 + v1 + v2) / 3

    # For outward normals, centroid · normal > 0
    dots = np.sum(centroids * normals_hat, axis=1)
    n_outward = np.sum(dots > 0)
    n_inward = np.sum(dots < 0)
    print(f"  Triangle normals: {n_outward} outward, {n_inward} inward")
    if n_inward > 0:
        print(f"  WARNING: Some normals point inward!")
        print(f"  Inward dot products: {dots[dots < 0][:5]}")

    # Check RWG edge structure
    # For each RWG edge, tri_p and tri_m should be two different triangles sharing the edge
    # The RWG function should be f_n = l/(2A+)(r-r_free+) on T+ and f_n = -l/(2A-)(r-r_free-) on T-
    # The edge normals (within the triangle plane) should point from free vertex toward the edge

    # Check: free vertex should be OPPOSITE to the edge
    for n in range(min(5, N)):
        tp = rwg['tri_p'][n]; tm = rwg['tri_m'][n]
        fp = rwg['free_p'][n]; fm = rwg['free_m'][n]
        # Edge vertices
        edge_verts_p = set(tris[tp].tolist()) - {np.argmin(np.linalg.norm(verts - fp, axis=1))}
        edge_verts_m = set(tris[tm].tolist()) - {np.argmin(np.linalg.norm(verts - fm, axis=1))}
        # These should be the same two vertices (the shared edge)
        shared = edge_verts_p & edge_verts_m

        # Check that the RWG div has correct sign
        # On T+: div(f) = +l/(2A+) (positive divergence)
        # On T-: div(f) = -l/(2A-) (negative divergence)
        div_p = rwg['length'][n] / (2 * rwg['area_p'][n])
        div_m = -rwg['length'][n] / (2 * rwg['area_m'][n])
        print(f"  RWG {n}: tri_p={tp}, tri_m={tm}, l={rwg['length'][n]:.4f}")
        print(f"    div_p={div_p:.4f}, div_m={div_m:.4f}")
        print(f"    area_p={rwg['area_p'][n]:.4f}, area_m={rwg['area_m'][n]:.4f}")

        # Check that free_p is a vertex of tri_p
        verts_p = verts[tris[tp]]
        dists_p = np.linalg.norm(verts_p - fp, axis=1)
        print(f"    free_p dists to tri_p verts: {dists_p}")

    # Check RWG basis function values at centroid of T+
    print(f"\n  Sample RWG basis function values:")
    for n in range(min(3, N)):
        tp = rwg['tri_p'][n]
        centroid_p = (verts[tris[tp][0]] + verts[tris[tp][1]] + verts[tris[tp][2]]) / 3
        fp = rwg['free_p'][n]
        f_val = rwg['length'][n] / (2 * rwg['area_p'][n]) * (centroid_p - fp)
        print(f"    RWG {n} at T+ centroid: f = {f_val}, |f| = {np.linalg.norm(f_val):.4f}")
