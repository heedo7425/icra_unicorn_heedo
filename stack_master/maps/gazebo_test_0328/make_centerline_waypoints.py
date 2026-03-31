"""센터라인 기반 global_waypoints.json 생성. 0.1m 재보간 + 속도는 최적화 속도 그대로 매핑."""
### HJ : centerline test waypoints — raceline 좌표 대신 센터라인 사용, 0.1m 재보간, 속도는 time-optimal 유지
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import os

SPACING = 0.1  # 재보간 간격 [m]

map_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(map_dir, 'global_waypoints_backup.json')) as f:
    data = json.load(f)

cwpnts = data['centerline_waypoints']['wpnts']
rwpnts = data['global_traj_wpnts_sp']['wpnts']

rl = pd.read_csv(os.path.join(map_dir, 'gazebo_wall_3d_rc_car_10th_timeoptimal.csv'))
s_opt = rl['s_opt'].values
v_opt = rl['v_opt'].values
ax_opt = rl['ax_opt'].values
laptime = rl['laptime'].iloc[0]

# ── Step 1: 센터라인 원본 데이터 추출 ──
c_s = np.array([w['s_m'] for w in cwpnts])
c_x = np.array([w['x_m'] for w in cwpnts])
c_y = np.array([w['y_m'] for w in cwpnts])
c_z = np.array([w['z_m'] for w in cwpnts])
c_d_left = np.array([w['d_left'] for w in cwpnts])
c_d_right = np.array([w['d_right'] for w in cwpnts])

# ── Step 2: CubicSpline 0.1m 재보간 (폐곡선 periodic BC) ──
total_length = c_s[-1] + (c_s[1] - c_s[0])  # 마지막→처음 간격 포함한 전체 길이
# periodic BC용: 끝점 = 시작점
s_cl = np.append(c_s, total_length)
x_cl = np.append(c_x, c_x[0])
y_cl = np.append(c_y, c_y[0])
z_cl = np.append(c_z, c_z[0])

cs_x = CubicSpline(s_cl, x_cl, bc_type='periodic')
cs_y = CubicSpline(s_cl, y_cl, bc_type='periodic')
cs_z = CubicSpline(s_cl, z_cl, bc_type='periodic')

n_new = int(total_length / SPACING) + 1
s_new = np.linspace(0, total_length, n_new, endpoint=False)

x_new = cs_x(s_new)
y_new = cs_y(s_new)
z_new = cs_z(s_new)

# d_left, d_right는 선형보간
d_left_new = np.interp(s_new, c_s, c_d_left, period=total_length)
d_right_new = np.interp(s_new, c_s, c_d_right, period=total_length)

# ── Step 3: heading, kappa 계산 (스플라인 해석적 미분) ──
dx_ds = cs_x(s_new, 1)
dy_ds = cs_y(s_new, 1)
dz_ds = cs_z(s_new, 1)
psi = np.arctan2(dy_ds, dx_ds)

d2x_ds2 = cs_x(s_new, 2)
d2y_ds2 = cs_y(s_new, 2)
kappa = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (dx_ds**2 + dy_ds**2)**1.5

# pitch (mu_rad)
mu = -np.arcsin(np.clip(dz_ds / np.sqrt(dx_ds**2 + dy_ds**2 + dz_ds**2), -1.0, 1.0))

# ── Step 4: 속도/가속도 보간 (s_opt 기준) ──
v_new = np.interp(s_new, s_opt, v_opt, period=total_length)
ax_new = np.interp(s_new, s_opt, ax_opt, period=total_length)

# ── Step 5: 실제 arc length 기준 s_m 재계산 ──
ds_real = np.sqrt(np.diff(x_new)**2 + np.diff(y_new)**2 + np.diff(z_new)**2)
arc_new = np.zeros(n_new)
arc_new[1:] = np.cumsum(ds_real)

print(f'센터라인 원본: {len(cwpnts)}개, spacing~{np.mean(np.diff(c_s)):.3f}m')
print(f'재보간 결과: {n_new}개, spacing={SPACING}m')
print(f's_opt 범위: [{s_opt.min():.3f}, {s_opt.max():.3f}]')
print(f'보간 속도 범위: [{v_new.min():.3f}, {v_new.max():.3f}]')

# ── Step 6: waypoints 생성 ──
new_wpnts = []
for k in range(n_new):
    wpnt = {
        'id': k,
        's_m': float(arc_new[k]),
        'd_m': 0.0,
        'x_m': float(x_new[k]),
        'y_m': float(y_new[k]),
        'z_m': float(z_new[k]),
        'd_right': float(d_right_new[k]),
        'd_left': float(d_left_new[k]),
        'psi_rad': float(psi[k]),
        'kappa_radpm': float(kappa[k]),
        'vx_mps': float(v_new[k]),
        'ax_mps2': float(ax_new[k]),
        'mu_rad': float(mu[k]),
    }
    new_wpnts.append(wpnt)

xs = [w['x_m'] for w in new_wpnts]
ys = [w['y_m'] for w in new_wpnts]
zs = [w['z_m'] for w in new_wpnts]
vs = [w['vx_mps'] for w in new_wpnts]


# ── Marker 헬퍼 ──

def _make_marker_template():
    return {
        'header': {'seq': 0, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': 'map'},
        'ns': '', 'id': 0, 'type': 2, 'action': 0,
        'pose': {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1}},
        'scale': {'x': 0.05, 'y': 0.05, 'z': 0.05},
        'color': {'r': 0.0, 'g': 0.0, 'b': 0.0, 'a': 1.0},
        'lifetime': {'secs': 0, 'nsecs': 0}, 'frame_locked': False,
        'points': [], 'colors': [], 'text': '',
        'mesh_resource': '', 'mesh_use_embedded_materials': False,
    }


def _build_speed_sphere_markers(xs, ys, zs, vs, scale=0.05):
    v_min, v_max = min(vs), max(vs)
    v_range = v_max - v_min if v_max > v_min else 1.0
    markers = []
    for i in range(len(xs)):
        m = _make_marker_template()
        m['id'] = i
        m['type'] = 2
        m['pose']['position'] = {'x': float(xs[i]), 'y': float(ys[i]), 'z': float(zs[i])}
        m['scale'] = {'x': scale, 'y': scale, 'z': scale}
        t = (float(vs[i]) - v_min) / v_range
        m['color'] = {'r': 1.0 - t, 'g': t, 'b': 0.0, 'a': 1.0}
        markers.append(m)
    return markers


def _build_cylinder_markers(xs, ys, zs, vs, r, g, b):
    scale_factor = 0.1317
    markers = []
    for i in range(len(xs)):
        m = _make_marker_template()
        m['id'] = i
        m['type'] = 3
        height = float(vs[i]) * scale_factor
        m['pose']['position'] = {'x': float(xs[i]), 'y': float(ys[i]), 'z': height / 2.0}
        m['scale'] = {'x': 0.1, 'y': 0.1, 'z': height}
        m['color'] = {'r': r, 'g': g, 'b': b, 'a': 1.0}
        markers.append(m)
    return markers


raceline_markers = _build_speed_sphere_markers(xs, ys, zs, vs)
raceline_vel_markers = _build_cylinder_markers(xs, ys, zs, vs, 1.0, 0.0, 0.0)

# 새 JSON 구성 — centerline_markers, centerline_waypoints, trackbounds는 기존 유지
output = dict(data)
output['global_traj_markers_iqp'] = {'markers': raceline_markers}
output['global_traj_wpnts_iqp'] = {
    'header': {'seq': 0, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
    'wpnts': new_wpnts,
}
output['global_traj_markers_sp'] = {'markers': raceline_markers}
output['global_traj_vel_markers_sp'] = {'markers': raceline_vel_markers}
output['global_traj_wpnts_sp'] = {
    'header': {'seq': 1, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
    'wpnts': new_wpnts,
}

out_path = os.path.join(map_dir, 'global_waypoints.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f'\n=== 새 global_waypoints.json 생성 완료 ===')
print(f'  출력: {out_path}')
print(f'  waypoints: {len(new_wpnts)}개 (센터라인 기반)')
print(f'  속도: [{min(vs):.3f}, {max(vs):.3f}] m/s')
print(f'  raceline markers: {len(raceline_markers)}개')

orig_xs = np.array([w['x_m'] for w in rwpnts])
orig_ys = np.array([w['y_m'] for w in rwpnts])
dists = [float(np.min(np.sqrt((orig_xs - x)**2 + (orig_ys - y)**2))) for x, y in zip(xs, ys)]
print(f'  센터라인↔레이싱라인 거리: mean={np.mean(dists):.3f}m, max={np.max(dists):.3f}m')
