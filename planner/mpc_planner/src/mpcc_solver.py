"""
Kinematic MPC Solver for planner use.

Simplified from the original EVO-MPCC port (see mpcc_solver_original.py).
This version removes s (arc-length) and p (progress) from the NLP since
they are dead variables when the reference is pre-sliced by the caller
(the solver's job is to track waypoint[k] at step k, not to decide
progress).  LUTs are removed for the same reason: with pre-sliced
reference, no symbolic interpolation in CasADi is needed.

States:   [x, y, psi]   - position and heading
Controls: [v, delta]    - forward speed and steering angle

Cost:
  - contouring error (e_c / 0.5)^2
  - lag error (e_l / 0.5)^2
  - velocity tracking: w_velocity * ((v - ref_v) / v_bias_max)^2
  - control smoothness: w_dv, w_dsteering

Constraints:
  - Dynamics: kinematic bicycle (Euler)
  - Track boundary: single half-plane per step, from (left, right) pair
  - Box: state/control bounds
"""

import numpy as np
from casadi import (MX, atan2, cos, sin, tan, vertcat, reshape, nlpsol)


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

        # Cost weights
        self.w_contour = params['w_contour']
        self.w_lag = params['w_lag']
        self.w_velocity = params['w_velocity']
        self.v_bias_max = params.get('v_bias_max', 10.0)
        self.w_dv = params['w_dv']
        self.w_dsteering = params['w_dsteering']

        # Boundary
        self.inflation = params.get('boundary_inflation', 0.1)
        # Soft-boundary slack penalty (heavier = harder boundary)
        self.w_slack = params.get('w_slack', 1000.0)

        # IPOPT
        self.ipopt_max_iter = params.get('ipopt_max_iter', 200)
        self.ipopt_print_level = params.get('ipopt_print_level', 0)

        # Dimensions
        self.n_states = 3    # x, y, psi
        self.n_controls = 2  # v, delta

        # State
        self.solver_nlp = None
        self.ready = False

        # Warm start
        self.X0 = None
        self.u0 = None
        self.warm = False

    def setup(self):
        """Build the NLP. Call once after construction."""
        self._build_nlp()
        self.ready = True

    def _build_nlp(self):
        N = self.N
        ns = self.n_states
        nc = self.n_controls

        X = MX.sym('X', ns, N + 1)
        U = MX.sym('U', nc, N)
        # Slack variables for soft boundary (one per step, non-negative).
        # Keeps the NLP always feasible: the solver may overshoot the track
        # half-plane at the cost of a heavy quadratic penalty.
        SL = MX.sym('SL', N, 1)
        # Per-step reference block:
        #   [ref_x, ref_y, ref_dx, ref_dy, ref_v, bound_a, bound_b]
        self.REF_BLOCK = 7
        n_params = ns + self.REF_BLOCK * N
        P = MX.sym('P', n_params)

        obj = 0
        g = []

        # Initial-state constraint
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

            # Contouring/lag (normalized by half-width 0.5)
            e_c = (sin(t_angle) * (st_next[0] - ref_x) - cos(t_angle) * (st_next[1] - ref_y)) / 0.5
            e_l = (-cos(t_angle) * (st_next[0] - ref_x) - sin(t_angle) * (st_next[1] - ref_y)) / 0.5

            obj += self.w_contour * (e_c ** 2)
            obj += self.w_lag * (e_l ** 2)

            # Velocity tracking
            obj += self.w_velocity * ((con[0] - ref_v) / self.v_bias_max) ** 2

            # Control smoothness
            if k < N - 1:
                con_next = U[:, k + 1]
                obj += self.w_dv * (con_next[0] - con[0]) ** 2
                obj += self.w_dsteering * (con_next[1] - con[1]) ** 2

            # Slack penalty (quadratic, large weight keeps boundary near-hard)
            obj += self.w_slack * SL[k] ** 2

            # Dynamics (Euler kinematic bicycle)
            st_next_euler = st + self.dT * vertcat(
                con[0] * cos(st[2]),
                con[0] * sin(st[2]),
                (con[0] / self.L) * tan(con[1]),
            )
            g.append(st_next - st_next_euler)

            # Soft boundary: lo - slack <= bound_a*x - bound_b*y <= hi + slack
            # Implemented as two inequalities with slack_k >= 0:
            #   c - slack_k <= hi    (upper)
            #   c + slack_k >= lo    (lower)
            c_expr = bound_a * st_next[0] - bound_b * st_next[1]
            g.append(c_expr - SL[k])   # upper side (ubg = hi)
            g.append(c_expr + SL[k])   # lower side (lbg = lo)

        g = vertcat(*g)

        OPT_variables = vertcat(
            reshape(X, ns * (N + 1), 1),
            reshape(U, nc * N, 1),
            SL,
        )

        # Constraint vector layout:
        #   [initial(ns)] + N * [dynamics(ns) + boundary_upper(1) + boundary_lower(1)]
        n_per_step = ns + 2
        n_constraints = ns + N * n_per_step
        self.lbg = np.zeros((n_constraints, 1))
        self.ubg = np.zeros((n_constraints, 1))

        # Box constraints
        n_slack = N
        n_vars = ns * (N + 1) + nc * N + n_slack
        self.lbx = np.zeros((n_vars, 1))
        self.ubx = np.zeros((n_vars, 1))

        for k in range(N + 1):
            self.lbx[ns * k:ns * (k + 1), 0] = [-200, -200, -1000]
            self.ubx[ns * k:ns * (k + 1), 0] = [200, 200, 1000]

        state_count = ns * (N + 1)
        for k in range(N):
            self.lbx[state_count:state_count + nc, 0] = [self.v_min, -self.theta_max]
            self.ubx[state_count:state_count + nc, 0] = [self.v_max, self.theta_max]
            state_count += nc

        # Slack bounds: slack_k >= 0, upper = +inf (no cap, penalty keeps it small)
        for k in range(N):
            self.lbx[state_count, 0] = 0.0
            self.ubx[state_count, 0] = 1e3
            state_count += 1

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

    def solve(self, initial_state, ref_data):
        """
        Args:
            initial_state: [x, y, psi]
            ref_data: dict with
                center_points: (>=N, 2)   reference centers
                left_points:   (>=N, 2)   left boundary points
                right_points:  (>=N, 2)   right boundary points
                ref_v:         (>=N,)     reference speeds
                ref_dx, ref_dy: (>=N,)    reference tangent unit vector

        Returns:
            speed (u_0[0]), steering (u_0[1]), trajectory (N+1, 3), success
        """
        if not self.ready:
            return 0.0, 0.0, None, False

        N = self.N
        ns = self.n_states

        # Yaw wrap-around (keep continuity with previous warm-start solution)
        if self.warm:
            delta_yaw = self.X0[1, 2] - initial_state[2]
            if abs(delta_yaw) >= np.pi:
                ceil_val = initial_state[2] + np.ceil(delta_yaw / (2 * np.pi)) * (2 * np.pi)
                floor_val = initial_state[2] + np.floor(delta_yaw / (2 * np.pi)) * (2 * np.pi)
                if abs(ceil_val - self.X0[1, 2]) < abs(floor_val - self.X0[1, 2]):
                    initial_state[2] = ceil_val
                else:
                    initial_state[2] = floor_val

        # Parameter vector
        p = np.zeros(self.n_params)
        p[0:ns] = initial_state

        center = ref_data['center_points']
        left = ref_data['left_points']
        right = ref_data['right_points']
        ref_v_arr = ref_data['ref_v']
        ref_dx = ref_data['ref_dx']
        ref_dy = ref_data['ref_dy']

        # Very large finite bounds (used in place of +/- inf so IPOPT's
        # default bound-treatment stays numerically well-behaved).
        BIG = 1e6

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

            # Layout per step: ns dynamics (lbg=ubg=0) + upper bnd + lower bnd
            base = ns + k * (ns + 2)
            # Dynamics rows already zero-initialized.
            # Upper side: (c - slack_k) <= hi, i.e. free below
            self.lbg[base + ns, 0] = -BIG
            self.ubg[base + ns, 0] = hi
            # Lower side: (c + slack_k) >= lo, i.e. free above
            self.lbg[base + ns + 1, 0] = lo
            self.ubg[base + ns + 1, 0] = BIG

        # Warm start
        if not self.warm:
            self._construct_warm_start(initial_state, ref_data)

        x_init = vertcat(
            reshape(self.X0.T, ns * (N + 1), 1),
            reshape(self.u0.T, self.n_controls * N, 1),
            np.zeros((N, 1)),  # slack warm start: zero (boundary satisfied)
        )

        sol = self.solver_nlp(x0=x_init, lbx=self.lbx, ubx=self.ubx,
                              lbg=self.lbg, ubg=self.ubg, p=p)

        stats = self.solver_nlp.stats()
        success = stats['success']
        self.last_return_status = stats.get('return_status', 'unknown')
        self.last_iter_count = stats.get('iter_count', -1)

        state_end = ns * (N + 1)
        ctrl_end = state_end + self.n_controls * N
        x_sol = reshape(sol['x'][:state_end], ns, N + 1).T
        u_sol = reshape(sol['x'][state_end:ctrl_end], self.n_controls, N).T
        slack_sol = np.array(sol['x'][ctrl_end:ctrl_end + N].full()).flatten()
        self.last_slack_max = float(np.max(slack_sol)) if slack_sol.size else 0.0

        speed = float(u_sol[0, 0])
        steering = float(u_sol[0, 1])
        trajectory = np.array(x_sol.full())

        self.X0 = np.array(vertcat(x_sol[1:, :], x_sol[-1, :]).full())
        self.u0 = np.array(vertcat(u_sol[1:, :], u_sol[-1, :]).full())

        if not self.warm:
            self.warm = True

        return speed, steering, trajectory, success

    def reset_warm_start(self):
        """Drop stored warm-start so next solve uses a fresh propagation."""
        self.warm = False

    def _construct_warm_start(self, initial_state, ref_data):
        """
        Seed X0, u0 using the sliced reference directly — each step anchors
        to the k-th center point with heading from the ref tangent.
        """
        N = self.N
        self.X0 = np.zeros((N + 1, self.n_states))
        self.u0 = np.zeros((N, self.n_controls))

        center = ref_data['center_points']
        ref_dx = ref_data['ref_dx']
        ref_dy = ref_data['ref_dy']
        ref_v_arr = ref_data['ref_v']
        n_ref = len(center)

        self.X0[0, :] = initial_state

        for k in range(N):
            idx = min(k + 1, n_ref - 1)
            x_next = float(center[idx, 0])
            y_next = float(center[idx, 1])
            psi_next = float(np.arctan2(ref_dy[idx], ref_dx[idx]))

            # Gentle initial speed guess (half of local ref, floor 1 m/s)
            init_speed = max(float(ref_v_arr[idx]) * 0.5, 1.0)

            dpsi = np.arctan2(
                np.sin(psi_next - self.X0[k, 2]),
                np.cos(psi_next - self.X0[k, 2])
            )
            steer = np.clip(
                np.arctan(dpsi * self.L / max(init_speed * self.dT, 0.01)),
                -self.theta_max * 0.5, self.theta_max * 0.5
            )

            self.X0[k + 1, :] = [x_next, y_next, psi_next]
            self.u0[k, :] = [init_speed, steer]
