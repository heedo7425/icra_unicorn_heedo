"""
Sector-tuned global racing line optimization.

Reads sector_friction.yaml and sector_velocity.yaml from data/sector_config/
and applies per-sector friction and velocity scaling to the GG diagram constraints.

- friction_scale: scales GG diagram limits (ax_min, ax_max, ay_max) per sector
- velocity_scale: caps V_max per sector (V_max * velocity_scale)

Output is saved as a separate CSV so the original racing line is untouched.
"""
import os
import sys
import casadi as ca
import numpy as np
import yaml
import pandas as pd

params = {
    'track_name': 'experiment_3d_2_3d_smoothed.csv',
    'raceline_name': 'experiment_3d_2_3d_rc_car_sector_tuned.csv',
    'warmstart_name': 'experiment_3d_2_3d_rc_car_timeoptimal.csv',  # 기존 racing line (warm start용)
    'vehicle_name': 'rc_car_10th',
    'gg_vehicle_name': 'rc_car_10th',
    'safety_distance': 0.05,
    'gg_mode': 'diamond',  # polar, diamond
    'gg_margin': 0.0,
    'neglect_w_omega_y': True,
    'neglect_w_omega_x': True,
    'neglect_euler': True,
    'neglect_centrifugal': True,
    'neglect_w_dot': False,
    'neglect_V_omega': False,
    'V_guess': 3.0,
    'w_jx': 1e-2,
    'w_jy': 1e-2,
    'w_dOmega_z': 0.0,
    'w_T': 1e0,
    'RK4_steps': 1,
    'sol_opts': {
        "ipopt.max_iter": 5000,
        "ipopt.hessian_approximation": 'limited-memory',
        "ipopt.line_search_method": 'cg-penalty',
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.acceptable_dual_inf_tol": 1e-4,
        "ipopt.constr_viol_tol": 1e-4,
    }
}

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
vehicle_params_path = os.path.join(data_path, 'vehicle_params', 'params_' + params['vehicle_name'] + '.yml')
gg_diagram_path = os.path.join(data_path, 'gg_diagrams', params.get('gg_vehicle_name', params['vehicle_name']), 'velocity_frame')
track_path = os.path.join(data_path, 'smoothed_track_data')
raceline_out_path = os.path.join(data_path, 'global_racing_lines')
sector_config_path = os.path.join(data_path, 'sector_config')
sys.path.append(os.path.join(dir_path, '..', 'src'))

# load vehicle and tire parameters
with open(vehicle_params_path, 'r') as stream:
    params.update(yaml.safe_load(stream))

from track3D import Track3D
from ggManager import GGManager


def _smooth_sector_array(arr, ds, transition_length_m=2.0):
    """
    Smooth step-function sector array with Gaussian-like filter to avoid
    abrupt constraint changes that cause IPOPT convergence issues.

    Parameters
    ----------
    arr : np.ndarray
        Per-point sector scaling array (step function).
    ds : float
        Distance between track discretization points [m].
    transition_length_m : float
        Smoothing half-width in meters (default 2.0m).
    """
    from scipy.ndimage import uniform_filter1d
    # Convert transition length to number of points
    kernel_size = max(3, int(2 * transition_length_m / ds) | 1)  # ensure odd
    return uniform_filter1d(arr, size=kernel_size, mode='wrap')


def load_sector_arrays(sector_config_path, n_points, ds):
    """
    Load sector YAML files and build per-point friction_scale and velocity_scale arrays.
    If YAML files don't exist, returns arrays of 1.0 (no scaling).
    Arrays are smoothed at sector boundaries to help IPOPT convergence.
    """
    friction_arr = np.ones(n_points)
    velocity_arr = np.ones(n_points)

    # --- friction_scale ---
    friction_yaml = os.path.join(sector_config_path, 'sector_friction.yaml')
    if os.path.exists(friction_yaml):
        with open(friction_yaml, 'r') as f:
            friction_cfg = yaml.safe_load(f)
        n_sectors = friction_cfg.get('n_sectors', 0)
        for i in range(n_sectors):
            sec = friction_cfg.get(f'Sector{i}', {})
            start = sec.get('start', 0)
            end = sec.get('end', n_points - 1)
            scale = sec.get('friction_scale', 1.0)
            friction_arr[start:end + 1] = scale
        print(f'Loaded sector friction config: {n_sectors} sectors from {friction_yaml}')
    else:
        print(f'No sector_friction.yaml found at {friction_yaml}, using friction_scale=1.0 everywhere.')

    # --- velocity_scale ---
    velocity_yaml = os.path.join(sector_config_path, 'sector_velocity.yaml')
    if os.path.exists(velocity_yaml):
        with open(velocity_yaml, 'r') as f:
            velocity_cfg = yaml.safe_load(f)
        n_sectors = velocity_cfg.get('n_sectors', 0)
        for i in range(n_sectors):
            sec = velocity_cfg.get(f'Sector{i}', {})
            start = sec.get('start', 0)
            end = sec.get('end', n_points - 1)
            scale = sec.get('velocity_scale', 1.0)
            velocity_arr[start:end + 1] = scale
        print(f'Loaded sector velocity config: {n_sectors} sectors from {velocity_yaml}')
    else:
        print(f'No sector_velocity.yaml found at {velocity_yaml}, using velocity_scale=1.0 everywhere.')

    # Smooth transitions at sector boundaries
    friction_arr = _smooth_sector_array(friction_arr, ds)
    velocity_arr = _smooth_sector_array(velocity_arr, ds)

    return friction_arr, velocity_arr


def calc_global_raceline_sector_tuned(
        track_name: str,
        vehicle_params: dict,
        gg_mode: str,
        gg_margin: float,
        safety_distance: float,
        w_T: float,
        w_jx: float,
        w_jy: float,
        w_dOmega_z: float,
        RK4_steps: int,
        V_guess: float,
        neglect_w_omega_x: bool,
        neglect_w_omega_y: bool,
        neglect_euler: bool,
        neglect_centrifugal: bool,
        neglect_w_dot: bool,
        neglect_V_omega: bool,
        sol_opt: dict,
        out_path: str,
):
    track_handler = Track3D(
        path=os.path.join(track_path, track_name)
    )

    gg_handler = GGManager(
        gg_path=gg_diagram_path,
        gg_margin=gg_margin
    )

    # 섹터별 스케일링 배열 로드 (경계에서 smooth transition 적용)
    friction_scale, velocity_scale = load_sector_arrays(sector_config_path, track_handler.s.size, track_handler.ds)

    # 섹터 스케일링 요약 출력
    print('\n--- Sector scaling summary (smoothed) ---')
    for k in range(0, track_handler.s.size, max(1, track_handler.s.size // 20)):
        print(f'  point {k:4d} (s={track_handler.s[k]:7.2f}m): friction={friction_scale[k]:.3f}, velocity={velocity_scale[k]:.3f}')
    print('---\n')

    # 기존 racing line 결과를 warm start로 사용 (수렴 속도/안정성 향상)
    warmstart_path = os.path.join(raceline_out_path, params.get('warmstart_name', ''))
    warmstart_v = None
    warmstart_n = None
    if os.path.exists(warmstart_path):
        ws_df = pd.read_csv(warmstart_path)
        warmstart_v = ws_df['v_opt'].values
        warmstart_n = ws_df['n_opt'].values
        print(f'Warm start 사용: {warmstart_path}')
    else:
        print(f'Warm start 파일 없음, 상수 초기값 사용 (V_guess={V_guess})')

    # Define state variables.
    V = ca.MX.sym('V')
    n = ca.MX.sym('n')
    chi = ca.MX.sym('chi')
    ax = ca.MX.sym('ax')
    ay = ca.MX.sym('ay')

    x = ca.vertcat(V, n, chi, ax, ay)
    nx = x.shape[0]

    # Define control variables.
    jx = ca.MX.sym('jx')
    jy = ca.MX.sym('jy')

    u = ca.vertcat(jx, jy)
    nu = u.shape[0]

    s = ca.MX.sym('s')

    # Time-distance scaling factor (dt/ds).
    s_dot = (V * ca.cos(chi)) / (1.0 - n * track_handler.Omega_z_interpolator(s))
    # vertical velocity
    w = n * track_handler.Omega_x_interpolator(s) * s_dot

    # Differential equations for scaled point mass model.
    dV = 1.0 / s_dot * ax
    if not neglect_w_omega_y:
        dV += w * (track_handler.Omega_x_interpolator(s) * ca.sin(chi) - track_handler.Omega_y_interpolator(s) * ca.cos(chi))

    dn = 1.0 / s_dot * V * ca.sin(chi)

    dchi = 1.0 / s_dot * ay / V - track_handler.Omega_z_interpolator(s)
    if not neglect_w_omega_x:
        dchi += w * (track_handler.Omega_x_interpolator(s) * ca.cos(chi) + track_handler.Omega_y_interpolator(s) * ca.sin(chi)) / V

    dax = jx / s_dot
    day = jy / s_dot

    dOmega_z = track_handler.Omega_z_interpolator(s) + dchi

    dx = ca.vertcat(dV, dn, dchi, dax, day)

    # Objective function.
    L_t = w_T * 1.0 / s_dot
    L_reg = w_jx * (jx / s_dot) ** 2 + w_jy * (jy / s_dot) ** 2 + w_dOmega_z * dOmega_z ** 2

    # Discrete time dynamics using fixed step Runge-Kutta 4 integrator.
    M = RK4_steps
    ds_rk = track_handler.ds / M
    f = ca.Function('f', [x, u, s], [dx, L_t, L_reg])
    X0 = ca.MX.sym('X0', nx)
    U = ca.MX.sym('U', nu)
    S0 = ca.MX.sym('S0')
    X = X0
    S = S0
    Q_t = 0
    Q_reg = 0
    for j in range(M):
        k1, k1_q_t, k1_q_reg = f(X, U, S)
        k2, k2_q_t, k2_q_reg = f(X + ds_rk / 2 * k1, U, S + ds_rk / 2)
        k3, k3_q_t, k3_q_reg = f(X + ds_rk / 2 * k2, U, S + ds_rk / 2)
        k4, k4_q_t, k4_q_reg = f(X + ds_rk * k3, U, S + ds_rk)
        X = X + ds_rk / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q_t = Q_t + ds_rk / 6 * (k1_q_t + 2 * k2_q_t + 2 * k3_q_t + k4_q_t)
        Q_reg = Q_reg + ds_rk / 6 * (k1_q_reg + 2 * k2_q_reg + 2 * k3_q_reg + k4_q_reg)
        S = S + ds_rk
    F = ca.Function('F', [X0, U, S0], [X, Q_t, Q_reg], ['x0', 'u', 's0'], ['xf', 'q_t', 'q_reg'])

    # Empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J_t = 0.0
    J_reg = 0.0
    g = []
    lbg = []
    ubg = []

    # Per-point V_max with velocity_scale applied
    V_max_arr = gg_handler.V_max * velocity_scale

    # 초기 조건 (warm start가 있으면 기존 해를 초기값으로 사용)
    Xk = ca.MX.sym('X0', nx)
    w += [Xk]
    lbw += [0.0, track_handler.w_tr_right[0] + vehicle_params['total_width'] / 2.0 + safety_distance, -np.pi / 2.0, -np.inf, -np.inf]
    ubw += [V_max_arr[0], track_handler.w_tr_left[0] - vehicle_params['total_width'] / 2.0 - safety_distance, np.pi / 2.0, np.inf, np.inf]
    v0 = warmstart_v[0] if warmstart_v is not None else V_guess
    n0 = warmstart_n[0] if warmstart_n is not None else (track_handler.w_tr_left[0] + track_handler.w_tr_right[0]) / 2.0
    w0 += [v0, n0, 0.0, 1e-6, 1e-6]

    # Formulate the NLP
    for k in range(0, track_handler.s.size):
        # Apparent accelerations
        ax_tilde, ay_tilde, g_tilde = track_handler.calc_apparent_accelerations(
            V=Xk[0],
            n=Xk[1],
            chi=Xk[2],
            ax=Xk[3],
            ay=Xk[4],
            s=k * track_handler.ds,
            h=vehicle_params['h'],
            neglect_w_omega_y=neglect_w_omega_y,
            neglect_w_omega_x=neglect_w_omega_x,
            neglect_euler=neglect_euler,
            neglect_centrifugal=neglect_centrifugal,
            neglect_w_dot=neglect_w_dot,
            neglect_V_omega=neglect_V_omega
        )

        # friction scaling: GG 한계는 그대로 두고, apparent acceleration을 스케일
        # friction_scale=0.7이면 |ay_tilde|/0.7 <= ay_max (수학적 동치, 수치적으로 안정)
        fric = friction_scale[k]
        ax_tilde_s = ax_tilde / fric
        ay_tilde_s = ay_tilde / fric

        # Constraints of gggv-Diagram
        if gg_mode == 'polar':
            alpha = ca.arctan2(ax_tilde_s, ay_tilde_s)
            rho = ca.sqrt(ax_tilde_s ** 2 + ay_tilde_s ** 2)
            adherence_radius = gg_handler.gggv_interpolator(ca.vertcat(Xk[0], g_tilde, alpha))
            g += [adherence_radius - rho]
            lbg += [0.0]
            ubg += [np.inf]
        elif gg_mode == 'diamond':
            # V와 g_tilde를 interpolator 유효 범위로 clamp (IPOPT iteration 중 범위 벗어남 방지)
            V_clamped = ca.fmax(Xk[0], 0.01)
            g_tilde_clamped = ca.fmax(g_tilde, 0.01)
            gg_exponent, ax_min, ax_max, ay_max = ca.vertsplit(gg_handler.acc_interpolator(ca.vertcat(V_clamped, g_tilde_clamped)))
            # GG 출력 안전 clamp
            ay_max = ca.fmax(ay_max, 1e-2)
            ax_max = ca.fmax(ax_max, 1e-2)
            ax_min = ca.fmin(ax_min, -1e-2)
            gg_exponent = ca.fmax(gg_exponent, 0.5)

            g += [ay_max - ca.fabs(ay_tilde_s)]
            lbg += [0.0]
            ubg += [np.inf]

            g += [ca.fabs(ax_min) * ca.power(
                    ca.fmax(
                        (1.0 - ca.power(
                            ca.fmin(ca.fabs(ay_tilde_s) / ay_max, 1.0),
                            gg_exponent
                        ))
                        , 1e-3
                    ),
                    1.0 / gg_exponent
                ) - ca.fabs(ax_tilde_s)
            ]
            lbg += [0.0]
            ubg += [np.inf]

            g += [ax_max - ax_tilde_s]
            lbg += [0.0]
            ubg += [np.inf]
        else:
            raise RuntimeError('Unknown gg_mode.')

        # If last iteration, don't add new control or states.
        if k == track_handler.s.size - 1:
            break

        # New NLP variable for the control.
        Uk = ca.MX.sym('U_' + str(k), nu)
        w += [Uk]
        lbw += [-np.inf] * nu
        ubw += [np.inf] * nu
        w0 += [0.0] * nu

        # Integrate till the end of the interval.
        Fk = F(x0=Xk, u=Uk, s0=k * track_handler.ds)
        Xk_end = Fk['xf']
        J_t = J_t + Fk['q_t']
        J_reg = J_reg + Fk['q_reg']

        # New NLP variable for state at end of interval.
        Xk = ca.MX.sym('X_' + str(k + 1), nx)
        w += [Xk]
        lbw += [0.0, track_handler.w_tr_right[k+1] + vehicle_params['total_width'] / 2.0 + safety_distance, -np.pi / 2.0, -np.inf, -np.inf]
        ubw += [V_max_arr[k+1], track_handler.w_tr_left[k+1] - vehicle_params['total_width'] / 2.0 - safety_distance, np.pi / 2.0, np.inf, np.inf]
        vk = warmstart_v[k+1] if warmstart_v is not None else V_guess
        nk = warmstart_n[k+1] if warmstart_n is not None else (track_handler.w_tr_left[k+1] + track_handler.w_tr_right[k+1]) / 2.0
        w0 += [vk, nk, 0.0, 1e-6, 1e-6]

        # Add equality constraint for continuity.
        g += [Xk_end - Xk]
        lbg += [0.0] * nx
        ubg += [0.0] * nx

    # Boundary constraint: start states = final states.
    g += [w[0] - Xk]
    lbg += [0.0] * nx
    ubg += [0.0] * nx

    # Concatenate NLP vectors.
    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    w0 = ca.vertcat(w0)
    lbw = ca.vertcat(lbw)
    ubw = ca.vertcat(ubw)
    lbg = ca.vertcat(lbg)
    ubg = ca.vertcat(ubg)

    # Create an NLP solver.
    nlp = {'f': J_t + J_reg, 'x': w, 'g': g}
    solver = ca.nlpsol('solver', 'ipopt', nlp, sol_opt)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    # Extract lap time.
    laptime = ca.Function('f_laptime', [w], [J_t])
    print(f'Laptime: {float(laptime(sol["x"]))}')
    regularization = ca.Function('f_regularization', [w], [J_reg])

    # Extract solution.
    w_opt = sol['x'].full().flatten()
    s_opt = track_handler.s
    v_opt = np.array(w_opt[0::nx + nu])
    n_opt = np.array(w_opt[1::nx + nu])
    chi_opt = np.array(w_opt[2::nx + nu])
    ax_opt = np.array(w_opt[3::nx + nu])
    ay_opt = np.array(w_opt[4::nx + nu])
    jx_opt = np.array(w_opt[5::nx + nu])
    jy_opt = np.array(w_opt[6::nx + nu])

    trajectory_data_frame = pd.DataFrame()
    trajectory_data_frame['s_opt'] = s_opt
    trajectory_data_frame['v_opt'] = v_opt
    trajectory_data_frame['n_opt'] = n_opt
    trajectory_data_frame['chi_opt'] = chi_opt
    trajectory_data_frame['ax_opt'] = ax_opt
    trajectory_data_frame['ay_opt'] = ay_opt
    trajectory_data_frame['jx_opt'] = np.concatenate([jx_opt, [0]])
    trajectory_data_frame['jy_opt'] = np.concatenate([jy_opt, [0]])
    trajectory_data_frame['laptime'] = float(laptime(sol["x"]))

    # Save solution.
    if out_path:
        trajectory_data_frame.to_csv(path_or_buf=out_path, sep=',', index=True, float_format='%.6f')
        print(f'Saved sector-tuned racing line: {out_path}')

    return trajectory_data_frame


if __name__ == '__main__':
    os.makedirs(raceline_out_path, exist_ok=True)
    raceline = calc_global_raceline_sector_tuned(
            track_name=params['track_name'],
            vehicle_params=params['vehicle_params'],
            gg_mode=params['gg_mode'],
            gg_margin=params['gg_margin'],
            safety_distance=params['safety_distance'],
            w_T=params['w_T'],
            w_jx=params['w_jx'],
            w_jy=params['w_jy'],
            w_dOmega_z=params['w_dOmega_z'],
            RK4_steps=params['RK4_steps'],
            V_guess=params['V_guess'],
            neglect_w_omega_x=params['neglect_w_omega_x'],
            neglect_w_omega_y=params['neglect_w_omega_y'],
            neglect_euler=params['neglect_euler'],
            neglect_centrifugal=params['neglect_centrifugal'],
            neglect_w_dot=params['neglect_w_dot'],
            neglect_V_omega=params['neglect_V_omega'],
            sol_opt=params['sol_opts'],
            out_path=os.path.join(raceline_out_path, params['raceline_name']),
    )
