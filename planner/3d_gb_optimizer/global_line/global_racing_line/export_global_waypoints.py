"""
3D racing line CSV → UNICORN 호환 global waypoints JSON 변환.

입력: data/global_racing_lines/{raceline_name} (time-optimal 또는 sector-tuned CSV)
출력: data/global_racing_lines/{output_name} (UNICORN WpntArray 포맷 JSON)

변환 과정:
  1. curvilinear (s_opt, n_opt) → Cartesian (x, y, z) via sn2cartesian
  2. 레이싱라인 실제 arc length 계산
  3. 실제 arc length 기준 등간격(0.1m) 재보간
  4. heading, kappa는 재보간된 x,y 좌표에서 직접 계산
  5. s_m = 레이싱라인 실제 arc length, d_m = 0
  6. RViz markers 생성 (centerline, raceline, trackbounds)

Usage:
  python global_racing_line/export_global_waypoints.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

params = {
    'track_name': 'experiment_3d_2_3d_smoothed_waypoint1.csv',
    'raceline_name': 'experiment_3d_2_3d_smoothed_waypoint1_timeoptimal.csv',
    'output_name': 'experiment_3d_2_3d_smoothed_waypoint1_timeoptimal.json',
    'waypoint_spacing': 0.1,  # 재보간 간격 [m]
}

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
track_path = os.path.join(data_path, 'smoothed_track_data', params['track_name'])
raceline_path = os.path.join(data_path, 'global_racing_lines', params['raceline_name'])
output_path = os.path.join(data_path, 'global_racing_lines', params['output_name'])
sys.path.append(os.path.join(dir_path, '..', 'src'))

from track3D import Track3D


# ── RViz Marker 생성 헬퍼 ──

def _make_marker_template():
    return {
        'header': {'seq': 0, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': 'map'},
        'ns': '', 'id': 0, 'type': 2, 'action': 0,
        'pose': {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1},
        },
        'scale': {'x': 0.05, 'y': 0.05, 'z': 0.05},
        'color': {'r': 0.0, 'g': 0.0, 'b': 0.0, 'a': 1.0},
        'lifetime': {'secs': 0, 'nsecs': 0},
        'frame_locked': False,
        'points': [], 'colors': [], 'text': '',
        'mesh_resource': '', 'mesh_use_embedded_materials': False,
    }


def _build_sphere_markers(xs, ys, zs, r, g, b, scale=0.05):
    """type=2 (SPHERE) markers."""
    markers = []
    for i in range(len(xs)):
        m = _make_marker_template()
        m['id'] = i
        m['type'] = 2  # SPHERE
        m['pose']['position'] = {'x': float(xs[i]), 'y': float(ys[i]), 'z': float(zs[i])}
        m['scale'] = {'x': scale, 'y': scale, 'z': scale}
        m['color'] = {'r': r, 'g': g, 'b': b, 'a': 1.0}
        markers.append(m)
    return markers


### HJ : sphere markers with speed-based color (red=slow, green=fast)
def _build_speed_sphere_markers(xs, ys, zs, vs, scale=0.05):
    """type=2 (SPHERE) markers with color mapped to speed (red-yellow-green)."""
    v_min, v_max = float(min(vs)), float(max(vs))
    v_range = v_max - v_min if v_max > v_min else 1.0
    markers = []
    for i in range(len(xs)):
        m = _make_marker_template()
        m['id'] = i
        m['type'] = 2  # SPHERE
        m['pose']['position'] = {'x': float(xs[i]), 'y': float(ys[i]), 'z': float(zs[i])}
        m['scale'] = {'x': scale, 'y': scale, 'z': scale}
        t = (float(vs[i]) - v_min) / v_range  # 0=slow, 1=fast
        m['color'] = {'r': 1.0 - t, 'g': t, 'b': 0.0, 'a': 1.0}
        markers.append(m)
    return markers
### HJ : end


def _build_cylinder_markers(xs, ys, zs, vs, r, g, b):
    """type=3 (CYLINDER) markers. 높이 = 속도 비례 (scale_factor=0.1317)."""
    scale_factor = 0.1317
    markers = []
    for i in range(len(xs)):
        m = _make_marker_template()
        m['id'] = i
        m['type'] = 3  # CYLINDER
        height = float(vs[i]) * scale_factor
        m['pose']['position'] = {'x': float(xs[i]), 'y': float(ys[i]), 'z': height / 2.0}
        m['scale'] = {'x': 0.1, 'y': 0.1, 'z': height}
        m['color'] = {'r': r, 'g': g, 'b': b, 'a': 1.0}
        markers.append(m)
    return markers


def _build_trackbounds_markers(track):
    """좌/우 트랙 경계 sphere markers. 좌=보라(0.5,0,0.5), 우=노랑(0.5,1,0)."""
    markers = []
    marker_id = 0

    for k in range(len(track.s)):
        s = track.s[k]
        theta = track.theta[k]
        x_c, y_c, z_c = track.x[k], track.y[k], track.z[k]

        # 왼쪽 경계
        w_l = track.w_tr_left[k]
        x_l = x_c - w_l * np.sin(theta)
        y_l = y_c + w_l * np.cos(theta)

        m = _make_marker_template()
        m['id'] = marker_id
        m['pose']['position'] = {'x': float(x_l), 'y': float(y_l), 'z': float(z_c)}
        m['color'] = {'r': 0.5, 'g': 0.0, 'b': 0.5, 'a': 1.0}
        markers.append(m)
        marker_id += 1

        # 오른쪽 경계
        w_r = track.w_tr_right[k]  # 음수
        x_r = x_c - w_r * np.sin(theta)
        y_r = y_c + w_r * np.cos(theta)

        m = _make_marker_template()
        m['id'] = marker_id
        m['pose']['position'] = {'x': float(x_r), 'y': float(y_r), 'z': float(z_c)}
        m['color'] = {'r': 0.5, 'g': 1.0, 'b': 0.0, 'a': 1.0}
        markers.append(m)
        marker_id += 1

    return markers


# ── 메인 export ──

def export_waypoints():
    # 트랙 로드
    track = Track3D(path=track_path)

    # racing line 로드
    rl = pd.read_csv(raceline_path)
    s_opt = rl['s_opt'].values
    v_opt = rl['v_opt'].values
    n_opt = rl['n_opt'].values
    chi_opt = rl['chi_opt'].values
    ax_opt = rl['ax_opt'].values
    laptime = rl['laptime'].iloc[0]

    # ── Step 1: curvilinear → Cartesian 변환 ──
    n_points = len(s_opt)
    x_raw = np.zeros(n_points)
    y_raw = np.zeros(n_points)
    z_raw = np.zeros(n_points)

    for k in range(n_points):
        cart = track.sn2cartesian(s_opt[k], n_opt[k])
        x_raw[k] = float(cart[0])
        y_raw[k] = float(cart[1])
        z_raw[k] = float(cart[2])

    ### HJ : CubicSpline 보간 (C² 연속, 기존 np.interp 선형보간 대체)
    ### IY : periodic CubicSpline으로 폐곡선 연결부 oscillation 제거
    # ── Step 3: periodic CubicSpline + 등간격 0.1m 재보간 ──
    spacing = params['waypoint_spacing']

    # 중복 끝점 제거 (raw[-1] == raw[0])
    x_r, y_r, z_r = x_raw[:-1], y_raw[:-1], z_raw[:-1]
    v_r = v_opt[:-1]
    ax_r = ax_opt[:-1]
    s_opt_r = s_opt[:-1]
    n_opt_r = n_opt[:-1]

    # arc length 계산 (마지막→첫 점 closure gap 포함)
    ds_r = np.sqrt(np.diff(x_r)**2 + np.diff(y_r)**2 + np.diff(z_r)**2)
    ds_close = np.sqrt((x_r[0] - x_r[-1])**2 + (y_r[0] - y_r[-1])**2 + (z_r[0] - z_r[-1])**2)
    arc_r = np.zeros(len(x_r) + 1)
    arc_r[1:-1] = np.cumsum(ds_r)
    arc_r[-1] = arc_r[-2] + ds_close  # 한 바퀴 전체
    total_loop = arc_r[-1]

    # periodic CubicSpline (첫=끝 값 동일, C² 연속 보장)
    cs_x = CubicSpline(arc_r, np.append(x_r, x_r[0]), bc_type='periodic')
    cs_y = CubicSpline(arc_r, np.append(y_r, y_r[0]), bc_type='periodic')
    cs_z = CubicSpline(arc_r, np.append(z_r, z_r[0]), bc_type='periodic')

    # 등간격 리샘플링 (endpoint=False: 폐곡선이므로 마지막 점 ≠ 시작점)
    n_new = round(total_loop / spacing)
    arc_new = np.linspace(0, total_loop, n_new, endpoint=False)

    x_new = cs_x(arc_new)
    y_new = cs_y(arc_new)
    z_new = cs_z(arc_new)

    # v, ax, s_opt, n_opt 보간 (arc length 기준, 선형보간)
    arc_r_inner = arc_r[:-1]  # 중복 끝점 제외한 arc values
    v_new = np.interp(arc_new, arc_r_inner, v_r, period=total_loop)
    ax_new = np.interp(arc_new, arc_r_inner, ax_r, period=total_loop)
    s_opt_new = np.interp(arc_new, arc_r_inner, s_opt_r, period=total_loop)
    n_opt_new = np.interp(arc_new, arc_r_inner, n_opt_r, period=total_loop)

    # ── Step 4: heading, kappa, pitch 계산 (스플라인 미분) ──
    dx_dt = cs_x(arc_new, 1)
    dy_dt = cs_y(arc_new, 1)
    psi = np.arctan2(dy_dt, dx_dt)

    d2x_dt2 = cs_x(arc_new, 2)
    d2y_dt2 = cs_y(arc_new, 2)
    kappa = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5

    dz_dt = cs_z(arc_new, 1)
    mu = -np.arcsin(np.clip(dz_dt / np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2), -1.0, 1.0))
    ### IY : end

    # ── Step 5: 트랙 경계 거리 계산 + waypoints 생성 ──
    waypoints = []
    for k in range(n_new):
        s = s_opt_new[k]
        n = n_opt_new[k]

        w_tr_left = float(track.w_tr_left_interpolator(s))
        w_tr_right = float(track.w_tr_right_interpolator(s))
        d_left = w_tr_left - n
        d_right = -w_tr_right + n

        wpnt = {
            'id': k,
            's_m': float(arc_new[k]),
            'd_m': 0.0,
            'x_m': float(x_new[k]),
            'y_m': float(y_new[k]),
            'z_m': float(z_new[k]),
            'd_right': float(abs(d_right)),
            'd_left': float(abs(d_left)),
            'psi_rad': float(psi[k]),
            'kappa_radpm': float(kappa[k]),
            'vx_mps': float(v_new[k]),
            'ax_mps2': float(ax_new[k]),
            'mu_rad': float(mu[k]),
        }
        waypoints.append(wpnt)

    # ── Step 6: RViz Markers 생성 ──
    centerline_markers = _build_sphere_markers(
        track.x, track.y, track.z,
        r=0.0, g=0.0, b=1.0, scale=0.05,  # 파란색
    )

    ### HJ : two raceline marker types — sphere (3D pos + speed color) and cylinder (speed height)
    raceline_markers = _build_speed_sphere_markers(
        x_new, y_new, z_new, v_new, scale=0.05,
    )
    raceline_vel_markers = _build_cylinder_markers(
        x_new, y_new, z_new, v_new,
        r=1.0, g=0.0, b=0.0,  # red
    )
    ### HJ : end

    trackbounds_markers = _build_trackbounds_markers(track)

    # ── JSON 출력 (참고 포맷 호환) ──
    output = {
        'map_info_str': {
            'data': f'estimated lap time: {laptime:.4f}s; maximum speed: {v_new.max():.4f}m/s; '
        },
        'est_lap_time': {'data': float(laptime)},
        'centerline_markers': {'markers': centerline_markers},
        'centerline_waypoints': _build_centerline_waypoints(track),
        ### HJ : IQP = SP (3D pipeline has single raceline, no separate IQP/SP)
        'global_traj_markers_iqp': {'markers': raceline_markers},
        'global_traj_wpnts_iqp': {
            'header': {'seq': 0, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
            'wpnts': waypoints,
        },
        ### HJ : end
        'global_traj_markers_sp': {'markers': raceline_markers},
        'global_traj_vel_markers_sp': {'markers': raceline_vel_markers},
        'global_traj_wpnts_sp': {
            'header': {'seq': 1, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
            'wpnts': waypoints,
        },
        'trackbounds_markers': {'markers': trackbounds_markers},
        ### IY : centerline 기준 s, n (local racing line solver 초기 guess용)
        ### Wpnt msg에 없는 필드이므로 별도 섹션으로 저장
        'centerline_ref': {
            's_center_m': [float(s_opt_new[k]) for k in range(n_new)],
            'n_center_m': [float(n_opt_new[k]) for k in range(n_new)],
        },
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # 검증 출력
    ds_check = np.sqrt(np.diff(x_new)**2 + np.diff(y_new)**2 + np.diff(z_new)**2)
    print(f'입력: {raceline_path}')
    print(f'출력: {output_path}')
    print(f'  waypoints: {n_new}개 (spacing={spacing}m)')
    print(f'  실제 총 경로 길이: {total_loop:.3f}m')
    print(f'  laptime: {laptime:.3f}s')
    print(f'  v: [{v_new.min():.2f}, {v_new.max():.2f}] m/s')
    print(f'  z: [{z_new.min():.3f}, {z_new.max():.3f}] m')
    print(f'  s_m 간격: {np.diff(arc_new).mean():.6f} (등간격)')
    print(f'  실제 거리 간격: min={ds_check.min():.6f}, max={ds_check.max():.6f}, std={ds_check.std():.6f}')
    print(f'  markers: centerline={len(centerline_markers)}, raceline={len(raceline_markers)}, trackbounds={len(trackbounds_markers)}')
    print(f'\nSample waypoint [0]:')
    for key, val in waypoints[0].items():
        print(f'  {key}: {val}')


def _build_centerline_waypoints(track):
    """smoothed track 센터라인을 waypoint 형식으로 변환."""
    n_pts = len(track.s)
    wpnts = []
    for k in range(n_pts):
        wpnts.append({
            'id': k,
            's_m': float(track.s[k]),
            'd_m': 0.0,
            'x_m': float(track.x[k]),
            'y_m': float(track.y[k]),
            'z_m': float(track.z[k]),
            'd_right': float(abs(track.w_tr_right[k])),
            'd_left': float(abs(track.w_tr_left[k])),
            'psi_rad': float(track.theta[k]),
            'kappa_radpm': float(track.Omega_z[k]),
            'vx_mps': 0.0,
            'ax_mps2': 0.0,
        })
    return {
        'header': {'seq': 0, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
        'wpnts': wpnts,
    }


if __name__ == '__main__':
    export_waypoints()
