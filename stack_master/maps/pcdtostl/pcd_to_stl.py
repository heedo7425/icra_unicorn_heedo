#!/usr/bin/env python3
"""PCD → STL converter for Isaac Sim track ingestion.

Reads a point-cloud (.pcd) file, reconstructs a surface mesh, and writes a
binary .stl suitable for `add_reference_to_stage` in Isaac Sim. Two
reconstruction backends:
  - poisson     : Screened Poisson, smooth closed surface (default)
  - ball_pivot  : Ball-pivoting, preserves raw topology better for open tracks

Usage (ROS 1 launch wrapper):
    roslaunch stack_master pcd_to_stl.launch \
        pcd:=/path/to/input.pcd  stl:=/path/to/output.stl  method:=poisson

Direct CLI:
    rosrun stack_master pcd_to_stl.py --pcd in.pcd --stl out.stl
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import open3d as o3d


def load_pcd(path: str) -> o3d.geometry.PointCloud:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PCD not found: {path}")
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"PCD has 0 points: {path}")
    print(f"[pcd_to_stl] loaded {len(pcd.points)} points from {path}")
    return pcd


def preprocess(
    pcd: o3d.geometry.PointCloud,
    voxel: float,
    normal_radius: float,
) -> o3d.geometry.PointCloud:
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
        print(f"[pcd_to_stl] voxel downsample → {len(pcd.points)} points (voxel={voxel})")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)
    return pcd


def reconstruct_poisson(
    pcd: o3d.geometry.PointCloud,
    depth: int,
    density_quantile: float,
) -> o3d.geometry.TriangleMesh:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    densities = np.asarray(densities)
    thr = np.quantile(densities, density_quantile)
    keep = densities > thr
    mesh.remove_vertices_by_mask(~keep)
    print(f"[pcd_to_stl] poisson: {len(mesh.triangles)} tris (depth={depth}, "
          f"density>q{density_quantile:.2f})")
    return mesh


def reconstruct_ball_pivot(
    pcd: o3d.geometry.PointCloud,
    radii: list[float],
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    print(f"[pcd_to_stl] ball_pivot: {len(mesh.triangles)} tris (radii={radii})")
    return mesh


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd",  required=True, help="input .pcd")
    ap.add_argument("--stl",  required=True, help="output .stl")
    ap.add_argument("--method", choices=("poisson", "ball_pivot"), default="poisson")
    ap.add_argument("--voxel",         type=float, default=0.05,
                    help="voxel-downsample size [m]; 0 = skip")
    ap.add_argument("--normal-radius", type=float, default=0.15,
                    help="normal-estimation radius [m]")
    ap.add_argument("--poisson-depth", type=int,   default=9,
                    help="poisson octree depth (larger=more detail, more memory)")
    ap.add_argument("--density-quantile", type=float, default=0.02,
                    help="drop low-density vertices below this quantile")
    ap.add_argument("--ball-radii", type=str, default="0.1,0.2,0.4",
                    help="comma-separated ball radii for ball_pivot [m]")
    args = ap.parse_args()

    pcd = load_pcd(args.pcd)
    pcd = preprocess(pcd, args.voxel, args.normal_radius)

    if args.method == "poisson":
        mesh = reconstruct_poisson(pcd, args.poisson_depth, args.density_quantile)
    else:
        radii = [float(x) for x in args.ball_radii.split(",")]
        mesh = reconstruct_ball_pivot(pcd, radii)

    mesh = clean_mesh(mesh)

    out_dir = os.path.dirname(os.path.abspath(args.stl))
    os.makedirs(out_dir, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(args.stl, mesh, write_ascii=False)
    if not ok:
        print(f"[pcd_to_stl] ERROR: failed to write {args.stl}", file=sys.stderr)
        sys.exit(1)
    print(f"[pcd_to_stl] wrote {args.stl}  ({len(mesh.vertices)} verts, "
          f"{len(mesh.triangles)} tris)")


if __name__ == "__main__":
    main()
