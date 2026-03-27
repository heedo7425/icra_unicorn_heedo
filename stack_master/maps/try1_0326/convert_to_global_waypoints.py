#!/usr/bin/env python3
"""
Convert raceline JSON + centerline CSV to global_waypoints.json format
that global_trajectory_publisher.py expects.
"""

import json
import csv
import math
import numpy as np

MAP_DIR = "/home/hmcl/catkin_ws/src/race_stack/stack_master/maps/try1_0326"
RACELINE_JSON = f"{MAP_DIR}/localization_bridge_2_3d_rc_car_timeoptimal_safety_0.1.json"
CENTERLINE_CSV = f"{MAP_DIR}/localization_bridge_2_3d.csv"
OUTPUT_JSON = f"{MAP_DIR}/global_waypoints.json"


def make_marker(id, x, y, z=0.0, marker_type=2, scale=0.05, r=0.0, g=0.0, b=1.0, a=1.0):
    return {
        "header": {"seq": 0, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": "map"},
        "ns": "", "id": id, "type": marker_type, "action": 0,
        "pose": {
            "position": {"x": x, "y": y, "z": z},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1}
        },
        "scale": {"x": scale, "y": scale, "z": scale},
        "color": {"r": r, "g": g, "b": b, "a": a},
        "lifetime": {"secs": 0, "nsecs": 0},
        "frame_locked": False, "points": [], "colors": [],
        "text": "", "mesh_resource": "", "mesh_use_embedded_materials": False
    }


def make_wpnt(id, s_m, d_m, x_m, y_m, z_m, d_right, d_left, psi_rad, kappa_radpm, vx_mps, ax_mps2):
    return {
        "id": id, "s_m": s_m, "d_m": d_m,
        "x_m": x_m, "y_m": y_m, "z_m": z_m,
        "d_right": d_right, "d_left": d_left,
        "psi_rad": psi_rad, "kappa_radpm": kappa_radpm,
        "vx_mps": vx_mps, "ax_mps2": ax_mps2
    }


def load_raceline():
    with open(RACELINE_JSON, "r") as f:
        data = json.load(f)
    wpnts = []
    for wp in data["waypoints"]:
        wpnts.append(make_wpnt(
            id=wp["id"], s_m=wp["s_m"], d_m=wp["d_m"],
            x_m=wp["x_m"], y_m=wp["y_m"], z_m=wp["z_m"],
            d_right=wp["d_right"], d_left=wp["d_left"],
            psi_rad=wp["psi_rad"], kappa_radpm=wp["kappa_radpm"],
            vx_mps=wp["vx_mps"], ax_mps2=wp["ax_mps2"]
        ))
    return wpnts, data["laptime"]


def load_centerline():
    wpnts = []
    with open(CENTERLINE_CSV, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            wpnts.append(make_wpnt(
                id=i,
                s_m=float(row["s_m"]),
                d_m=0.0,
                x_m=float(row["x_m"]),
                y_m=float(row["y_m"]),
                z_m=float(row["z_m"]),
                d_right=float(row["w_tr_right_m"]),
                d_left=float(row["w_tr_left_m"]),
                psi_rad=float(row["theta_rad"]),
                kappa_radpm=float(row["dtheta_radpm"]),
                vx_mps=0.0,
                ax_mps2=0.0
            ))
    return wpnts


def make_wpnt_markers(wpnts, marker_type=2, scale=0.05, r=0.0, g=0.0, b=1.0):
    markers = []
    for i, wp in enumerate(wpnts):
        markers.append(make_marker(i, wp["x_m"], wp["y_m"], z=wp["z_m"],
                                   marker_type=marker_type, scale=scale,
                                   r=r, g=g, b=b))
    return markers


#### IY : raceline marker z → 실제 지형 높이(z_m), 색상 → 속도(파랑=느림, 빨강=빠름) ####
def speed_to_color(t):
    """red=slow(0.0) → yellow=mid(0.5) → green=fast(1.0)"""
    r = max(0.0, min(1.0, 1.0 - 2.0 * (t - 0.5)))
    g = max(0.0, min(1.0, 2.0 * t))
    b = 0.0
    return r, g, b


def make_raceline_markers(wpnts):
    """Raceline markers: z = actual terrain height (z_m), color = speed (blue→red)"""
    # 기존 구현: z=vx_mps/2.0 (속도를 높이로 표현), 단색 빨강, cylinder
    # markers = []
    # for i, wp in enumerate(wpnts):
    #     m = make_marker(i, wp["x_m"], wp["y_m"],
    #                     z=wp["vx_mps"] / 2.0,
    #                     marker_type=3, scale=0.1,
    #                     r=1.0, g=0.0, b=0.0)
    #     m["scale"]["z"] = wp["vx_mps"]
    #     markers.append(m)
    v_min = min(wp["vx_mps"] for wp in wpnts)
    v_max = max(wp["vx_mps"] for wp in wpnts)
    markers = []
    for i, wp in enumerate(wpnts):
        t = (wp["vx_mps"] - v_min) / (v_max - v_min) if v_max > v_min else 0.5
        r, g, b = speed_to_color(t)
        m = make_marker(i, wp["x_m"], wp["y_m"],
                        z=wp["z_m"],
                        marker_type=2, scale=0.08,
                        r=r, g=g, b=b)
        markers.append(m)
    return markers
#### IY : raceline marker z → 실제 지형 높이(z_m), 색상 → 속도(파랑=느림, 빨강=빠름) ####


def make_trackbound_markers(centerline_wpnts):
    markers = []
    mid = 0
    for wp in centerline_wpnts:
        theta = wp["psi_rad"]
        x, y = wp["x_m"], wp["y_m"]
        w_right = wp["d_right"]
        w_left = wp["d_left"]

        z = wp["z_m"]
        # Right boundary point (perpendicular to heading)
        rx = x + w_right * math.sin(theta)
        ry = y - w_right * math.cos(theta)
        markers.append(make_marker(mid, rx, ry, z=z, scale=0.05, r=0.5, g=0.0, b=0.5))
        mid += 1

        # Left boundary point
        lx = x - w_left * math.sin(theta)
        ly = y + w_left * math.cos(theta)
        markers.append(make_marker(mid, lx, ly, z=z, scale=0.05, r=0.5, g=0.0, b=0.5))
        mid += 1

    return markers


def main():
    print("Loading raceline...")
    raceline_wpnts, laptime = load_raceline()
    print(f"  -> {len(raceline_wpnts)} waypoints, laptime={laptime:.3f}s")

    print("Loading centerline...")
    centerline_wpnts = load_centerline()
    print(f"  -> {len(centerline_wpnts)} waypoints")

    max_speed = max(wp["vx_mps"] for wp in raceline_wpnts)

    print("Generating markers...")
    result = {
        "map_info_str": {
            "data": f"IQP estimated lap time: {laptime:.4f}s; IQP maximum speed: {max_speed:.4f}m/s; "
                    f"SP estimated lap time: {laptime:.4f}s; SP maximum speed: {max_speed:.4f}m/s; "
        },
        "est_lap_time": {"data": laptime},
        "centerline_markers": {"markers": make_wpnt_markers(centerline_wpnts, scale=0.05, r=0.0, g=0.0, b=1.0)},
        "centerline_waypoints": {"header": {"seq": 0, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": ""}, "wpnts": centerline_wpnts},
        "global_traj_markers_iqp": {"markers": make_raceline_markers(raceline_wpnts)},
        "global_traj_wpnts_iqp": {"header": {"seq": 0, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": ""}, "wpnts": raceline_wpnts},
        "global_traj_markers_sp": {"markers": make_raceline_markers(raceline_wpnts)},
        "global_traj_wpnts_sp": {"header": {"seq": 0, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": ""}, "wpnts": raceline_wpnts},
        "trackbounds_markers": {"markers": make_trackbound_markers(centerline_wpnts)},
    }

    print(f"Writing {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f)

    print(f"Done! File size: {len(json.dumps(result)) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
