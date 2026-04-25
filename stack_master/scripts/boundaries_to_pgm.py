#!/usr/bin/env python3
"""Rasterize boundary_left.csv / boundary_right.csv → <map>.pgm for map_server.

Intended for maps created from 3D track data where a real occupancy grid was
never generated. Produces a grayscale PGM where the track interior (between
left and right bounds) is free space (254) and everything else is occupied (0).

Usage:
    python3 boundaries_to_pgm.py /path/to/maps/<map_name>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml


def main(map_dir: str) -> None:
    map_dir = Path(map_dir)
    map_name = map_dir.name
    yaml_path = map_dir / f"{map_name}.yaml"
    pgm_path  = map_dir / f"{map_name}.pgm"
    bl_path   = map_dir / "boundary_left.csv"
    br_path   = map_dir / "boundary_right.csv"

    if not yaml_path.exists():
        print(f"[error] {yaml_path} not found", file=sys.stderr); sys.exit(1)
    for p in (bl_path, br_path):
        if not p.exists():
            print(f"[error] {p} not found", file=sys.stderr); sys.exit(1)

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    res    = float(meta["resolution"])

    # Load boundary points (x, y) — skip header row
    def _load(p):
        a = np.genfromtxt(p, delimiter=",", skip_header=1)
        # assume columns: x, y, ... (take first two numeric cols)
        return a[:, :2]
    L = _load(bl_path)
    R = _load(br_path)
    pts_xy = np.vstack([L, R])

    # Auto-size pgm from boundary bbox with 2m margin.
    margin = 2.0
    x_min, x_max = pts_xy[:, 0].min() - margin, pts_xy[:, 0].max() + margin
    y_min, y_max = pts_xy[:, 1].min() - margin, pts_xy[:, 1].max() + margin
    origin = [float(x_min), float(y_min)]
    w = int(np.ceil((x_max - x_min) / res))
    h = int(np.ceil((y_max - y_min) / res))
    print(f"[info] bbox ({x_min:.2f},{y_min:.2f})→({x_max:.2f},{y_max:.2f}) "
          f"→ pgm {w}x{h}, res {res}, origin {origin}")

    # Update yaml origin so map_server places the pgm correctly in world.
    meta["origin"] = [origin[0], origin[1], 0]
    with open(yaml_path, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=None)
    print(f"[info] updated {yaml_path} origin")

    # World → pixel: px = (x - origin_x) / res, py_img = h - 1 - (y - origin_y)/res
    def w2p(xy):
        px = ((xy[:, 0] - origin[0]) / res).astype(int)
        py = (h - 1 - (xy[:, 1] - origin[1]) / res).astype(int)
        return px, py

    img = np.full((h, w), 0, dtype=np.uint8)   # 0 = occupied
    lpx, lpy = w2p(L)
    rpx, rpy = w2p(R)

    # Fill polygon defined by L + reversed R as track free region.
    # Use simple scanline: for each row intersect L and R polylines.
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[error] pip install Pillow", file=sys.stderr); sys.exit(2)
    poly = [(int(x), int(y)) for x, y in zip(lpx, lpy)]
    poly += [(int(x), int(y)) for x, y in zip(rpx[::-1], rpy[::-1])]
    pil = Image.fromarray(img, mode="L")
    ImageDraw.Draw(pil).polygon(poly, fill=254)   # interior free

    # Then redraw boundary lines as occupied (0) so Map display shows walls
    ImageDraw.Draw(pil).line(list(zip(lpx, lpy)), fill=0, width=1)
    ImageDraw.Draw(pil).line(list(zip(rpx, rpy)), fill=0, width=1)

    pil.save(pgm_path)
    print(f"[info] wrote {pgm_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/maps/<map_name>", file=sys.stderr); sys.exit(1)
    main(sys.argv[1])
