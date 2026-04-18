"""
MPCC Solver - CasADi + IPOPT based Model Predictive Contouring Control.

Ported verbatim from icra_unicorn_heedo/mpc/scripts/mpcc_solver.py.
Based on EVO-MPCC (github.com/zhouhengli/EVO-MPCC) architecture,
using CasADi B-spline LUTs instead of discrete waypoint arrays.

States:   [x, y, psi, s]      - position, heading, arc-length progress
Controls: [v, delta, p]       - velocity, steering, progress velocity

Cost (EVO-MPCC):
  - contouring error (e_c / 0.5)^2
  - lag error (e_l / 0.5)^2
  - progress reward: -gamma * (p / p_max) * dT
  - velocity tracking: q_v * ((v - ref_v) / Vbias_max)^2
  - control smoothness: q_dv, q_d_delta, q_dvp

Constraints:
  - Dynamics: kinematic bicycle (Euler)
  - Track boundary: half-plane inequality
  - Box: state/control bounds
"""

import numpy as np
from casadi import *


class MPCCSolver:

    def __init__(self, params):
        # Horizon
        self.N = params['N']
        self.dT = params['dT']
        self.L = params['vehicle_L']

        # Limits
        self.v_max = params['max_speed']
        self.v_min = params.get('min_speed', 0.0)
        self.theta_max = params['max_steering']
        self.p_max = params['max_progress']

        # Cost weights
        self.w_contour = params['w_contour']
        self.w_lag = params['w_lag']
        self.w_progress = params['w_progress']
        self.w_velocity = params['w_velocity']
        self.v_bias_max = params.get('v_bias_max', 10.0)
        self.w_dv = params['w_dv']
        self.w_dsteering = params['w_dsteering']
        self.w_dprogress = params['w_dprogress']

        # Boundary
        self.inflation = params.get('boundary_inflation', 0.1)

        # IPOPT
        self.ipopt_max_iter = params.get('ipopt_max_iter', 200)
        self.ipopt_print_level = params.get('ipopt_print_level', 0)

        # Internal
        self.n_states = 4
        self.n_controls = 3
        self.track_length = None
        self.solver = None
        self.ready = False

        # Warm start
        self.X0 = None
        self.u0 = None
        self.warm = False

        # Global waypoint data (for nearest-s lookup)
        self.global_s = None
        self.global_x = None
        self.global_y = None

    def build_luts(self, s_vals, cx, cy, dx, dy, psi_vals, d_left, d_right):
        self.track_length = float(s_vals[-1])

        self.global_s = np.array(s_vals, dtype=float)
        self.global_x = np.array(cx, dtype=float)
        self.global_y = np.array(cy, dtype=float)

        # 3x wrap-around extension (track loop handling)
        tl = self.track_length
        s_ext = np.concatenate([s_vals, s_vals[1:] + tl, s_vals[1:] + 2 * tl])
        cx_ext = np.concatenate([cx, cx[1:], cx[1:]])
        cy_ext = np.concatenate([cy, cy[1:], cy[1:]])
        dx_ext = np.concatenate([dx, dx[1:], dx[1:]])
        dy_ext = np.concatenate([dy, dy[1:], dy[1:]])
        psi_ext = np.concatenate([psi_vals, psi_vals[1:], psi_vals[1:]])
        dl_ext = np.concatenate([d_left, d_left[1:], d_left[1:]])
        dr_ext = np.concatenate([d_right, d_right[1:], d_right[1:]])

        self.lut_cx = interpolant('lut_cx', 'bspline', [s_ext], cx_ext)
        self.lut_cy = interpolant('lut_cy', 'bspline', [s_ext], cy_ext)
        self.lut_dx = interpolant('lut_dx', 'bspline', [s_ext], dx_ext)
        self.lut_dy = interpolant('lut_dy', 'bspline', [s_ext], dy_ext)
        self.lut_psi = interpolant('lut_psi', 'bspline', [s_ext], psi_ext)
        self.lut_dl = interpolant('lut_dl', 'bspline', [s_ext], dl_ext)
        self.lut_dr = interpolant('lut_dr', 'bspline', [s_ext], dr_ext)

        self._build_nlp()
        self.ready = True

    def _build_nlp(self):
        N = self.N
        ns = self.n_states
        nc = self.n_controls

        X = MX.sym('X', ns, N + 1)
        U = MX.sym('U', nc, N)
        # Parameters: initial_state(4) + per_step(ref_cx, ref_cy, ref_dx, ref_dy, ref_v, bound_a, bound_b) * N
        self.REF_BLOCK = 7
        n_params = ns + self.REF_BLOCK * N
        P = MX.sym('P', n_params)

        obj = 0
        g = []

        g.append(X[:, 0] - P[0:ns])

        for k in range(N):
            st = X[:, k]
            st_next = X[:, k + 1]
            con = U[:, k]

            ref_idx = ns + self.REF_BLOCK * k
            ref_x = P[ref_idx]
            ref_y = P[ref_idx + 1]
            t_dx = P[ref_idx + 2]
            t_dy = P[ref_idx + 3]
            ref_v = P[ref_idx + 4]
            bound_a = P[ref_idx + 5]
            bound_b = P[ref_idx + 6]

            t_angle = atan2(t_dy, t_dx)

            # Contouring/lag error (normalized by half-width 0.5)
            e_c = (sin(t_angle) * (st_next[0] - ref_x) - cos(t_angle) * (st_next[1] - ref_y)) / 0.5
            e_l = (-cos(t_angle) * (st_next[0] - ref_x) - sin(t_angle) * (st_next[1] - ref_y)) / 0.5

            obj += self.w_contour * (e_c ** 2)
            obj += self.w_lag * (e_l ** 2)

            # Progress reward
            obj -= self.w_progress * (con[2] / self.p_max) * self.dT

            # Velocity tracking
            obj += self.w_velocity * ((con[0] - ref_v) / self.v_bias_max) ** 2

            # Control smoothness
            if k < N - 1:
                con_next = U[:, k + 1]
                obj += self.w_dv * (con_next[0] - con[0]) ** 2
                obj += self.w_dsteering * (con_next[1] - con[1]) ** 2
                obj += self.w_dprogress * (con_next[2] - con[2]) ** 2

            # Dynamics (Euler kinematic bicycle)
            st_next_euler = st + self.dT * vertcat(
                con[0] * cos(st[2]),
                con[0] * sin(st[2]),
                (con[0] / self.L) * tan(con[1]),
                con[2]
            )
            g.append(st_next - st_next_euler)

            # Boundary half-plane: bound_a * x - bound_b * y in [lo, hi]
            g.append(bound_a * st_next[0] - bound_b * st_next[1])

        g = vertcat(*g)

        OPT_variables = vertcat(
            reshape(X, ns * (N + 1), 1),
            reshape(U, nc * N, 1)
        )

        n_per_step = ns + 1
        n_constraints = ns + N * n_per_step

        self.lbg = np.zeros((n_constraints, 1))
        self.ubg = np.zeros((n_constraints, 1))

        n_vars = ns * (N + 1) + nc * N
        self.lbx = np.zeros((n_vars, 1))
        self.ubx = np.zeros((n_vars, 1))

        for k in range(N + 1):
            self.lbx[ns * k:ns * (k + 1), 0] = [-200, -200, -1000, 0]
            self.ubx[ns * k:ns * (k + 1), 0] = [200, 200, 1000, 3 * self.track_length]

        state_count = ns * (N + 1)
        for k in range(N):
            self.lbx[state_count:state_count + nc, 0] = [self.v_min, -self.theta_max, 0]
            self.ubx[state_count:state_count + nc, 0] = [self.v_max, self.theta_max, self.p_max]
            state_count += nc

        nlp = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
        opts = {
            'ipopt': {
                'max_iter': self.ipopt_max_iter,
                'print_level': self.ipopt_print_level,
                'acceptable_tol': 1e-4,
                'acceptable_obj_change_tol': 1e-3,
                'fixed_variable_treatment': 'make_parameter',
            },
            'print_time': 0,
        }
        self.solver_nlp = nlpsol('solver', 'ipopt', nlp, opts)
        self.n_params = n_params

    def find_nearest_s(self, x, y):
        dists = (self.global_x - x) ** 2 + (self.global_y - y) ** 2
        idx = int(np.argmin(dists))
        return float(self.global_s[idx]), idx

    def filter_s(self, raw_s):
        """Wrap-around handling using previous solution continuity."""
        if not self.warm or self.X0 is None:
            return raw_s
        prev_s = self.X0[0, 3]
        tl = self.track_length
        candidates = [raw_s, raw_s + tl, raw_s - tl]
        best = min(candidates, key=lambda c: abs(c - prev_s))
        return best

    def solve(self, initial_state, ref_data):
        """
        Returns: speed, steering, trajectory (N+1 x 4), success.
        """
        if not self.ready:
            return 0.0, 0.0, None, False

        N = self.N
        ns = self.n_states

        # Yaw wrap-around (EVO-MPCC style)
        if self.warm:
            delta_yaw = self.X0[1, 2] - initial_state[2]
            if abs(delta_yaw) >= np.pi:
                ceil_val = initial_state[2] + np.ceil(delta_yaw / (2 * np.pi)) * (2 * np.pi)
                floor_val = initial_state[2] + np.floor(delta_yaw / (2 * np.pi)) * (2 * np.pi)
                if abs(ceil_val - self.X0[1, 2]) < abs(floor_val - self.X0[1, 2]):
                    initial_state[2] = ceil_val
                else:
                    initial_state[2] = floor_val

        s_cur = initial_state[3]
        s_lo = max(0, s_cur - 2.0)
        s_hi = s_cur + N * self.dT * self.v_max + 5.0
        for k in range(N + 1):
            self.lbx[ns * k + 3, 0] = s_lo
            self.ubx[ns * k + 3, 0] = s_hi

        p = np.zeros(self.n_params)
        p[0:ns] = initial_state

        center = ref_data['center_points']
        left = ref_data['left_points']
        right = ref_data['right_points']
        ref_v_arr = ref_data['ref_v']
        ref_dx = ref_data['ref_dx']
        ref_dy = ref_data['ref_dy']

        for k in range(N):
            ref_idx = ns + self.REF_BLOCK * k
            idx = min(k, len(center) - 1)

            p[ref_idx] = center[idx, 0]
            p[ref_idx + 1] = center[idx, 1]
            p[ref_idx + 2] = ref_dx[idx]
            p[ref_idx + 3] = ref_dy[idx]
            p[ref_idx + 4] = ref_v_arr[idx]

            delta_bx = right[idx, 0] - left[idx, 0]
            delta_by = right[idx, 1] - left[idx, 1]
            p[ref_idx + 5] = -delta_bx
            p[ref_idx + 6] = delta_by

            val_r = -delta_bx * right[idx, 0] - delta_by * right[idx, 1]
            val_l = -delta_bx * left[idx, 0] - delta_by * left[idx, 1]
            lo = min(val_r, val_l)
            hi = max(val_r, val_l)

            constraint_idx = ns + k * (ns + 1) + ns
            self.lbg[constraint_idx, 0] = lo
            self.ubg[constraint_idx, 0] = hi

        ref_v_for_warm = ref_v_arr[:N] if len(ref_v_arr) >= N else np.pad(ref_v_arr, (0, N - len(ref_v_arr)), mode='edge')
        if not self.warm:
            self._construct_warm_start(initial_state, ref_v_for_warm)

        x_init = vertcat(
            reshape(self.X0.T, ns * (N + 1), 1),
            reshape(self.u0.T, self.n_controls * N, 1),
        )

        sol = self.solver_nlp(x0=x_init, lbx=self.lbx, ubx=self.ubx,
                              lbg=self.lbg, ubg=self.ubg, p=p)

        stats = self.solver_nlp.stats()
        success = stats['success']

        x_sol = reshape(sol['x'][:ns * (N + 1)], ns, N + 1).T
        u_sol = reshape(sol['x'][ns * (N + 1):], self.n_controls, N).T

        speed = float(u_sol[0, 0])
        steering = float(u_sol[0, 1])
        trajectory = np.array(x_sol.full())

        self.X0 = np.array(vertcat(x_sol[1:, :], x_sol[-1, :]).full())
        self.u0 = np.array(vertcat(u_sol[1:, :], u_sol[-1, :]).full())

        if not self.warm:
            self.warm = True

        return speed, steering, trajectory, success

    def reset_warm_start(self):
        self.warm = False

    def _construct_warm_start(self, initial_state, ref_v):
        """EVO-MPCC style: forward-propagate along centerline."""
        N = self.N
        self.X0 = np.zeros((N + 1, self.n_states))
        self.u0 = np.zeros((N, self.n_controls))

        s0 = initial_state[3]
        psi0 = float(np.arctan2(
            float(self.lut_dy(s0)),
            float(self.lut_dx(s0))
        ))
        self.X0[0, :] = [initial_state[0], initial_state[1], initial_state[2], s0]

        for k in range(N):
            init_speed = max(float(ref_v[k]) * 0.3, 1.0)
            s_next = self.X0[k, 3] + init_speed * self.dT
            psi_next = float(np.arctan2(
                float(self.lut_dy(s_next)),
                float(self.lut_dx(s_next))
            ))
            x_next = float(self.lut_cx(s_next))
            y_next = float(self.lut_cy(s_next))

            dpsi = np.arctan2(
                np.sin(psi_next - self.X0[k, 2]),
                np.cos(psi_next - self.X0[k, 2])
            )
            steer = np.clip(
                np.arctan(dpsi * self.L / max(init_speed * self.dT, 0.01)),
                -self.theta_max * 0.3, self.theta_max * 0.3
            )
            self.X0[k + 1, :] = [x_next, y_next, psi_next, s_next]
            self.u0[k, :] = [init_speed, steer, init_speed]
