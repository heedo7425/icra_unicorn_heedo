"""
Post-process: 기존 time-optimal racing line에 섹터별 velocity scaling 적용.
NLP 재계산 없이 velocity_scale만 곱하는 방식 (UNICORN vel_scaler_node.py 방식).

입력:
  - data/global_racing_lines/{raceline_name}  (기존 time-optimal CSV)
  - data/sector_config/sector_velocity.yaml   (velocity_tuner.py에서 생성)
  - data/smoothed_track_data/{track_name}     (s 매핑용)

출력:
  - data/global_racing_lines/{output_name}    (velocity 스케일 적용된 새 CSV)

Usage:
  python global_racing_line/apply_sector_velocity.py
"""
import os
import yaml
import numpy as np
import pandas as pd

params = {
    'track_name': 'experiment_3d_2_3d_smoothed.csv',
    'raceline_name': 'experiment_3d_2_3d_rc_car_timeoptimal.csv',
    'output_name': 'experiment_3d_2_3d_rc_car_sector_tuned.csv',
}

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
track_path = os.path.join(data_path, 'smoothed_track_data', params['track_name'])
raceline_path = os.path.join(data_path, 'global_racing_lines', params['raceline_name'])
output_path = os.path.join(data_path, 'global_racing_lines', params['output_name'])
sector_config_path = os.path.join(data_path, 'sector_config')


def build_velocity_scale_array(sector_yaml_path, s_track):
    """
    sector_velocity.yaml을 읽고, 트랙 포인트별 velocity_scale 배열 생성.
    섹터 경계에서 linear interpolation으로 smooth transition 적용.
    """
    n_points = len(s_track)
    scale_arr = np.ones(n_points)

    if not os.path.exists(sector_yaml_path):
        print(f'sector_velocity.yaml 없음 → velocity_scale=1.0 적용')
        return scale_arr

    with open(sector_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    n_sectors = cfg.get('n_sectors', 0)
    global_limit = cfg.get('global_limit', 1.0)

    # 섹터별 기본 스케일 적용
    for i in range(n_sectors):
        sec = cfg[f'Sector{i}']
        start, end = sec['start'], sec['end']
        scale = min(sec.get('velocity_scale', 1.0), global_limit)
        scale_arr[start:end + 1] = scale

    # 섹터 경계 smooth transition (UNICORN hl_change 방식)
    ds = np.mean(np.diff(s_track)) if len(s_track) > 1 else 1.0
    hl_change = max(1, int(2.0 / ds))  # 2m transition

    for i in range(n_sectors):
        sec = cfg[f'Sector{i}']
        start = sec['start']
        sec_scale = min(sec.get('velocity_scale', 1.0), global_limit)

        if i > 0:
            prev_sec = cfg[f'Sector{i-1}']
            prev_scale = min(prev_sec.get('velocity_scale', 1.0), global_limit)
        else:
            last_sec = cfg[f'Sector{n_sectors-1}']
            prev_scale = min(last_sec.get('velocity_scale', 1.0), global_limit)

        t_start = max(0, start - hl_change)
        t_end = min(n_points - 1, start + hl_change)
        if t_end > t_start:
            indices = np.arange(t_start, t_end + 1)
            scale_arr[t_start:t_end + 1] = np.interp(
                indices, [t_start, t_end], [prev_scale, sec_scale]
            )

    print(f'Loaded {n_sectors} sectors (global_limit={global_limit})')
    return scale_arr


def apply_velocity_scaling():
    # 트랙 데이터 (s 좌표)
    track = pd.read_csv(track_path)
    s_track = track['s_m'].values

    # 기존 racing line
    rl = pd.read_csv(raceline_path)
    print(f'입력: {raceline_path}')
    print(f'  원본 laptime: {rl["laptime"].iloc[0]:.3f}s')
    print(f'  원본 v_opt: [{rl["v_opt"].min():.2f}, {rl["v_opt"].max():.2f}] m/s')

    # velocity scale 배열 생성 (트랙 포인트 기준)
    vel_yaml = os.path.join(sector_config_path, 'sector_velocity.yaml')
    scale_track = build_velocity_scale_array(vel_yaml, s_track)

    # 트랙 s → raceline s 매핑 (interpolation)
    scale_rl = np.interp(rl['s_opt'].values, s_track, scale_track)

    # velocity scaling 적용
    rl_out = rl.copy()
    rl_out['v_opt'] = rl['v_opt'] * scale_rl

    # acceleration도 스케일에 맞게 재계산 (v가 줄어들면 ax, ay도 줄어듦)
    # a ∝ v^2 이므로 scale^2 적용
    rl_out['ax_opt'] = rl['ax_opt'] * scale_rl ** 2
    rl_out['ay_opt'] = rl['ay_opt'] * scale_rl ** 2

    # laptime 재계산: dt = ds / v
    ds = np.diff(rl_out['s_opt'].values)
    v_avg = (rl_out['v_opt'].values[:-1] + rl_out['v_opt'].values[1:]) / 2.0
    v_avg = np.maximum(v_avg, 0.01)  # 0 나눗셈 방지
    new_laptime = np.sum(ds / v_avg)
    rl_out['laptime'] = new_laptime

    # 저장
    rl_out.to_csv(output_path, sep=',', index=True, float_format='%.6f')

    print(f'\n출력: {output_path}')
    print(f'  스케일 적용 v_opt: [{rl_out["v_opt"].min():.2f}, {rl_out["v_opt"].max():.2f}] m/s')
    print(f'  새 laptime: {new_laptime:.3f}s (원본: {rl["laptime"].iloc[0]:.3f}s)')

    # 섹터별 요약
    with open(vel_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    n_sectors = cfg.get('n_sectors', 0)
    print(f'\n섹터별 요약:')
    for i in range(n_sectors):
        sec = cfg[f'Sector{i}']
        start, end = sec['start'], sec['end']
        vs = sec.get('velocity_scale', 1.0)
        v_sec = rl_out['v_opt'].values[start:end+1]
        print(f'  Sector{i} [{start}-{end}]: scale={vs:.2f}, v=[{v_sec.min():.2f}, {v_sec.max():.2f}] m/s')


if __name__ == '__main__':
    apply_velocity_scaling()
