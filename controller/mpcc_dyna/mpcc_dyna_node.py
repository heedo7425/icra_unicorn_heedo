#!/usr/bin/env python3
"""
mpcc_dyna_node — Dynamic single-track + Pacejka MPCC controller (Liniger-style).

State (8):  [X, Y, psi, vx, vy, omega, delta, theta]
Input (3):  [u_ddelta, u_a, vs]
NX/NU/NP : see mpcc_dyna.vehicle_model

Topics:
    /car_state/odom        Odometry      input  (vx/vy/omega 측정)
    /car_state/pose        PoseStamped   input  (X/Y/psi 측정)
    /centerline_waypoints  WpntArray     input

    drive_topic            AckermannDriveStamped   output (~drive_topic, default nav_1)
    /mpcc_dyna/predicted_path   Marker             viz
    /mpcc_dyna/solve_status     Int8

Driving cmd: speed=vx_pred[1], steering_angle=delta_pred[1].
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Int8
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from mpcc_dyna.vehicle_model import NX, NU, NP   # noqa: E402
from mpcc_dyna.mpcc_ocp import build_mpcc_ocp, solve_once   # noqa: E402
from mpcc_dyna.reference_builder import build_preview, project_to_centerline   # noqa: E402

# Wpnt numpy column indices.
C_X, C_Y, C_Z, C_VX, _, C_S, C_KAPPA, C_PSI, C_AX, C_D, C_MU_RAD = range(11)

# State indices (NX=8).
SX, SY, SPSI, SVX, SVY, SOMEGA, SDELTA, STHETA = range(NX)


def _load_vehicle_params(config_overrides: Optional[dict] = None) -> dict:
    """Load car physical + Pacejka parameters.

    Priority (lowest first → overridden by next):
      1. Hard-coded SRX1 defaults (sim-matched).
      2. ROS rosparam: /vehicle/* and /tire_{front,rear}/*  (실차 전용 yaml).
      3. config_overrides (mpc_config yaml `vehicle_overrides:` section).

    Real-car deployment: stack_master/config/<CAR>/<CAR>_pacejka.yaml 가
    /vehicle/* + /tire_*/* 로 로드되어 있으면 자동 반영. 차량 변경 시 yaml 수정만.
    """
    gp = rospy.get_param
    # 1. Defaults (SRX1 sim, sys-ID 결과)
    vp = {
        "m":    3.54,
        "l_f":  0.162,
        "l_r":  0.145,
        "l_wb": 0.307,
        "I_z":  0.05797,
        "h_cg": 0.014,
        "mu":   0.85,
        "Bf":   4.798, "Cf": 2.164, "Df": 0.650, "Ef": 0.373,
        "Br":  20.000, "Cr": 1.500, "Dr": 0.618, "Er": 0.0,
    }
    # 2. ROS rosparam (실차 셋업이 보통 여기에 다 로드)
    rosparam_keys = [
        ("m",    "/vehicle/m"),
        ("l_f",  "/vehicle/l_f"),
        ("l_r",  "/vehicle/l_r"),
        ("l_wb", "/vehicle/l_wb"),
        ("I_z",  "/vehicle/I_z"),
        ("h_cg", "/vehicle/h_cg"),
        ("mu",   "/vehicle/mu"),
        ("Bf",   "/tire_front/B"), ("Cf", "/tire_front/C"),
        ("Df",   "/tire_front/D"), ("Ef", "/tire_front/E"),
        ("Br",   "/tire_rear/B"),  ("Cr", "/tire_rear/C"),
        ("Dr",   "/tire_rear/D"),  ("Er", "/tire_rear/E"),
    ]
    for vp_key, rp in rosparam_keys:
        try:
            vp[vp_key] = float(gp(rp, vp[vp_key]))
        except Exception:
            pass
    # l_wb sanity (실차 yaml 따라 l_wb 가 명시 안 되어 있으면 l_f+l_r)
    if vp["l_wb"] <= 0:
        vp["l_wb"] = vp["l_f"] + vp["l_r"]
    # 3. yaml mpc_config overrides
    if config_overrides:
        for k, v in config_overrides.items():
            if k in vp:
                try: vp[k] = float(v)
                except Exception: pass
    return vp


class MpccNode:
    def __init__(self):
        rospy.init_node("mpcc_controller", anonymous=False)
        self.name = "mpcc_dyna"
        NS = f"/{self.name}"

        self.lock = threading.Lock()

        # Horizon / dt / loop
        self.N = int(rospy.get_param(f"{NS}/N_horizon", 30))
        self.dt = float(rospy.get_param(f"{NS}/dt", 0.05))
        self.loop_rate = float(rospy.get_param(f"{NS}/loop_rate_hz", 50.0))

        # Limits
        self.v_max = float(rospy.get_param(f"{NS}/v_max", 3.0))
        self.v_min = float(rospy.get_param(f"{NS}/v_min", 0.0))
        self.max_steer = float(rospy.get_param(f"{NS}/max_steer", 0.4))
        self.max_steer_rate = float(rospy.get_param(f"{NS}/max_steer_rate", 3.0))
        self.max_accel = float(rospy.get_param(f"{NS}/max_accel", 3.0))
        self.max_decel = float(rospy.get_param(f"{NS}/max_decel", -3.0))
        self.vs_min = float(rospy.get_param(f"{NS}/vs_min", 0.0))
        self.vs_max = float(rospy.get_param(f"{NS}/vs_max", 8.0))

        # Cost weights (Liniger cost.json 정렬)
        self.mpc_cfg = {
            "N_horizon": self.N, "dt": self.dt,
            "v_max": self.v_max, "v_min": self.v_min,
            "max_steer": self.max_steer, "max_steer_rate": self.max_steer_rate,
            "max_accel": self.max_accel, "max_decel": self.max_decel,
            "vs_min": self.vs_min, "vs_max": self.vs_max,
            # Niraj-style cold seed
            "cold_seed_speed": float(rospy.get_param(f"{NS}/cold_seed_speed", 1.5)),

            # ---- Liniger residual weights ----
            "qC":       float(rospy.get_param(f"{NS}/qC",      0.1)),
            "qL":       float(rospy.get_param(f"{NS}/qL",    500.0)),
            "qMu":      float(rospy.get_param(f"{NS}/qMu",     0.001)),
            "qBeta":    float(rospy.get_param(f"{NS}/qBeta",   0.01)),
            "qR":       float(rospy.get_param(f"{NS}/qR",      0.01)),
            "qDelta":   float(rospy.get_param(f"{NS}/qDelta",  1e-6)),
            "qA":       float(rospy.get_param(f"{NS}/qA",      1e-6)),
            "qDDelta":  float(rospy.get_param(f"{NS}/qDDelta", 5e-3)),
            "qVs":      float(rospy.get_param(f"{NS}/qVs",     0.02)),
            "qC_term":  float(rospy.get_param(f"{NS}/qC_term",  10.0)),
            "qR_term":  float(rospy.get_param(f"{NS}/qR_term",  10.0)),

            # ---- Liniger slack penalties (split quad + linear) ----
            "sc_quad_track": float(rospy.get_param(f"{NS}/sc_quad_track", 100.0)),
            "sc_lin_track":  float(rospy.get_param(f"{NS}/sc_lin_track",    1.0)),
            "sc_quad_tire":  float(rospy.get_param(f"{NS}/sc_quad_tire",    1.0)),
            "sc_lin_tire":   float(rospy.get_param(f"{NS}/sc_lin_tire",     0.1)),

            # friction_margin: codegen-baked
            "friction_margin":
                        float(rospy.get_param(f"{NS}/friction_margin", 0.85)),
            "levenberg_marquardt":
                        float(rospy.get_param(f"{NS}/levenberg_marquardt", 1.0)),
            "sim_method_num_steps":
                        int(rospy.get_param(f"{NS}/sim_method_num_steps", 5)),
            "qp_solver_iter_max":
                        int(rospy.get_param(f"{NS}/qp_solver_iter_max", 200)),
            "nlp_solver_max_iter":
                        int(rospy.get_param(f"{NS}/nlp_solver_max_iter", 2)),
        }

        self.bound_inset = float(rospy.get_param(f"{NS}/bound_inset", 0.10))
        self.startup_delay_s = float(rospy.get_param(f"{NS}/startup_delay_s", 0.3))

        # Vehicle params: defaults → /vehicle/* /tire_*/* → yaml `vehicle_overrides`.
        veh_overrides = rospy.get_param(f"{NS}/vehicle_overrides", {}) or {}
        self.vp = _load_vehicle_params(config_overrides=veh_overrides)

        # Build OCP solver
        codegen_dir = rospy.get_param(f"{NS}/codegen_dir", "/tmp/mpcc_dyna_codegen")
        t0 = time.perf_counter()
        self.solver = build_mpcc_ocp(self.mpc_cfg, self.vp, codegen_dir=codegen_dir)
        rospy.loginfo(f"[{self.name}] OCP built in {(time.perf_counter()-t0):.1f}s")

        # State
        self.odom: Optional[Odometry] = None
        self.pose: Optional[PoseStamped] = None
        self.centerline_np: Optional[np.ndarray] = None
        self.cl_d_right: Optional[np.ndarray] = None
        self.cl_d_left: Optional[np.ndarray] = None
        self.last_delta = 0.0
        self._prev_x_pred: Optional[np.ndarray] = None
        # ### HJ : startup boost 시작 시점은 odom 도착 후. 이전엔 init 시점이라
        # OCP build (1.3s) 동안 startup_delay (1.0s) 다 지나가서 boost 거의 발화 안 됨.
        self._first_odom_t: Optional[float] = None

        # Pub / Sub
        drive_topic = rospy.get_param("~drive_topic",
                                      "/vesc/high_level/ackermann_cmd_mux/input/nav_1")
        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=10)
        self.solve_status_pub = rospy.Publisher(f"{NS}/solve_status", Int8, queue_size=10)
        self.pred_pub = rospy.Publisher(f"{NS}/predicted_path", Marker, queue_size=1)

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=5)
        rospy.Subscriber("/car_state/pose", PoseStamped, self._pose_cb, queue_size=5)
        rospy.Subscriber("/centerline_waypoints", WpntArray, self._cl_cb, queue_size=1)

        rospy.Service(f"{NS}/reload_params", Trigger, self._reload_cb)
        rospy.loginfo(f"[{self.name}] init done. drive_topic={drive_topic}")

    # ---- Callbacks ----
    def _odom_cb(self, msg: Odometry) -> None:
        with self.lock:
            self.odom = msg
            if self._first_odom_t is None:
                self._first_odom_t = rospy.Time.now().to_sec()

    def _pose_cb(self, msg: PoseStamped) -> None:
        with self.lock: self.pose = msg

    def _cl_cb(self, msg: WpntArray) -> None:
        if len(msg.wpnts) < 2:
            return
        arr = np.array([
            [w.x_m, w.y_m, w.z_m, w.vx_mps, 0.0, w.s_m, w.kappa_radpm,
             w.psi_rad, w.ax_mps2, w.d_m, getattr(w, "mu_rad", 0.0)]
            for w in msg.wpnts
        ], dtype=np.float64)
        d_right = np.array([float(getattr(w, "d_right", 0.6)) for w in msg.wpnts])
        d_left = np.array([float(getattr(w, "d_left", 0.6)) for w in msg.wpnts])
        with self.lock:
            self.centerline_np = arr
            self.cl_d_right = d_right
            self.cl_d_left = d_left

    def _reload_cb(self, req):
        """Runtime hot-reload: cost W + bounds. (slack/track_pen needs solver.cost_set)"""
        NS = f"/{self.name}"
        TUNABLE = ("qC", "qL", "qMu", "qBeta", "qR", "qVs",
                   "qDelta", "qDDelta", "qA",
                   "qC_term", "qR_term",
                   "sc_quad_track", "sc_lin_track",
                   "sc_quad_tire", "sc_lin_tire",
                   "v_max", "v_min", "vs_max")
        # NOTE: friction_margin / Pacejka (B/C/D/E) / 차량 기하 (m, l_f, ...) 는
        # codegen-baked 라 runtime 변경 불가 (Phase 2C 에서 tuner.DEFAULT_FORBIDDEN
        # 에 등록). yaml 변경 시 노드 restart + codegen 재빌드 필요.
        applied = []
        try:
            for k in TUNABLE:
                if rospy.has_param(f"{NS}/{k}"):
                    self.mpc_cfg[k] = float(rospy.get_param(f"{NS}/{k}"))
                    applied.append(k)
            with self.lock:
                self._push_runtime_costs()
                self._push_runtime_bounds()
                self._push_runtime_slack()
                self._push_runtime_yref()
            return TriggerResponse(success=True, message=f"updated: {applied}")
        except Exception as e:
            return TriggerResponse(success=False, message=f"err: {e}")

    def _push_runtime_costs(self):
        cfg = self.mpc_cfg
        qC = cfg["qC"]; qL = cfg["qL"]
        # Liniger residual (9): eC, eL, dpsi, beta, omega, delta, u_a, u_ddelta, vs
        W = np.diag([qC, qL,
                     cfg["qMu"], cfg["qBeta"], cfg["qR"],
                     cfg["qDelta"], cfg["qA"], cfg["qDDelta"],
                     cfg["qVs"]])
        # Terminal residual (4): eC, eL, dpsi, omega
        W_e = np.diag([qC * cfg["qC_term"], qL,
                       cfg["qMu"], cfg["qR"] * cfg["qR_term"]])
        for k in range(self.N):
            self.solver.cost_set(k, "W", W)
        self.solver.cost_set(self.N, "W", W_e)

    def _push_runtime_bounds(self):
        cfg = self.mpc_cfg
        lbx = np.array([cfg["v_min"], -self.max_steer])
        ubx = np.array([cfg["v_max"],  self.max_steer])
        lbu = np.array([-self.max_steer_rate, cfg["max_decel"], cfg["vs_min"]])
        ubu = np.array([ self.max_steer_rate, cfg["max_accel"], cfg["vs_max"]])
        for k in range(1, self.N):
            self.solver.constraints_set(k, "lbx", lbx)
            self.solver.constraints_set(k, "ubx", ubx)
            self.solver.constraints_set(k, "lbu", lbu)
            self.solver.constraints_set(k, "ubu", ubu)

    def _push_runtime_yref(self, vx_now: float = 0.0):
        """Liniger progress reward — yref[8] = vs_max → cost qVs·(vs - vs_max)² push vs to bound.

        residual layout (9): [eC, eL, dpsi, beta, omega, delta, u_a, u_ddelta, vs]
        """
        cfg = self.mpc_cfg
        vs_max = float(cfg["vs_max"])
        yref = np.zeros(9, dtype=np.float64)
        yref[8] = vs_max
        for k in range(self.N):
            self.solver.cost_set(k, "yref", yref)

    def _push_runtime_slack(self):
        """Liniger slack penalties: separate quad (Zl/Zu) + linear (zl/zu) per row."""
        cfg = self.mpc_cfg
        sc_track_q = float(cfg.get("sc_quad_track", 100.0))
        sc_track_l = float(cfg.get("sc_lin_track",   1.0))
        sc_tire_q  = float(cfg.get("sc_quad_tire",   1.0))
        sc_tire_l  = float(cfg.get("sc_lin_tire",    0.1))
        Zq = np.array([sc_track_q, sc_track_q, sc_tire_q, sc_tire_q], dtype=np.float64)
        zl = np.array([sc_track_l, sc_track_l, sc_tire_l, sc_tire_l], dtype=np.float64)
        for k in range(self.N + 1):
            try:
                self.solver.cost_set(k, "Zl", Zq)
                self.solver.cost_set(k, "Zu", Zq)
                self.solver.cost_set(k, "zl", zl)
                self.solver.cost_set(k, "zu", zl)
            except Exception:
                pass

    # ---- Warm start helpers ----
    # ### HJ : drift-free anchoring.
    # 매 tick θ_init 을 차의 실제 projection 으로 강제. prev plan 의 절대 θ 값은
    # 무시하고, 상대 progression (Δθ_k = prev[k+1, θ] − prev[1, θ]) 만 가져옴.
    # → drift 개념 자체가 없어지고 cold seed 도 없어짐 (첫 tick 제외).
    def _resolve_theta_warm(self, cl, px, py):
        theta_proj = project_to_centerline(
            (px, py), cl, cl_x_idx=C_X, cl_y_idx=C_Y, cl_s_idx=C_S)
        if self._prev_x_pred is None or not np.all(np.isfinite(self._prev_x_pred)):
            self._prev_x_pred = None
            return theta_proj, None, 0.0

        # ### HJ : Spawn / large position jump detect.
        # spawn_on_waypoint 가 stuck 시 차 재배치 → 이전 plan 의 stage 0 위치와
        # 현재 위치가 1m+ 차이. 그 상태로 warm start 하면 pred path 가 이전 trajectory
        # 그대로 그려져 viz 점프 + 다음 plan 도 misaligned.
        prev_X = float(self._prev_x_pred[0, SX])
        prev_Y = float(self._prev_x_pred[0, SY])
        pos_jump = math.hypot(px - prev_X, py - prev_Y)
        # ### HJ : threshold 는 차의 한 tick 이동거리 (v_max·dt) 의 3배 또는 1m 중 큰 값.
        # 저속 차량 (v_max=3, dt=0.05 → tick 당 0.15m): threshold = max(1, 0.45) = 1m.
        # 고속 차량 (v_max=20 → tick 당 1m): threshold = 3m. false-positive 방지.
        spawn_threshold = max(1.0, 3.0 * self.v_max * self.dt)
        if pos_jump > spawn_threshold:
            rospy.logwarn_throttle(
                2.0, f"[{self.name}] position jump {pos_jump:.2f}m > {spawn_threshold:.2f}m → cold reset")
            self._prev_x_pred = None
            return theta_proj, None, 0.0

        # prev θ sequence shifted by 1 (Liniger SQP shift), last stage 외삽.
        prev_theta = self._prev_x_pred[:, STHETA].copy()
        # ### HJ : Monotonicity 강제. closed track 에서 lap-wrap 시 prev_theta 가
        # [..., 84.5, 84.6, 0.1, 0.2] 처럼 중간에 wrap → shifted 도 깨짐 → rebase
        # 후 theta_pred 의 stage k 사이 큰 jump → acados ODE 가 reference 를 잘못
        # interpolate → pred path endpoint 가 16~19m 점프. unwrap 으로 monotonic 복구.
        try:
            s_total = float(cl[-1, C_S] - cl[0, C_S])
            if s_total > 1e-6:
                d = np.diff(prev_theta)
                # |Δ| > s_total/2 이면 wrap 발생 — backward 면 +s_total, forward 면 -s_total
                wrap = np.where(d < -s_total * 0.5, s_total, 0.0)
                wrap += np.where(d > s_total * 0.5, -s_total, 0.0)
                prev_theta[1:] += np.cumsum(wrap)
        except Exception:
            pass
        last_step = prev_theta[-1] - prev_theta[-2] if len(prev_theta) >= 2 else 0.0
        shifted = np.concatenate([prev_theta[1:], [prev_theta[-1] + last_step]])
        # Anchor at projection: rebase so shifted[0] == theta_proj.
        rebase_offset = float(theta_proj - shifted[0])
        # Theta wrap-around 정규화 (closed track) — projection 이 wrap 후 작은 값일 때.
        try:
            if s_total > 1e-6:
                if rebase_offset > s_total * 0.5:
                    rebase_offset -= s_total
                elif rebase_offset < -s_total * 0.5:
                    rebase_offset += s_total
        except Exception:
            pass
        theta_pred = shifted + rebase_offset
        return theta_proj, theta_pred, rebase_offset

    def _apply_shift_warm_start(self, theta_offset: float = 0.0):
        """Liniger SQP shift + θ rebase by `theta_offset`.

        rebase 가 필요한 이유: x0 의 θ 가 projection 으로 강제되면 stage 1 warm 의
        θ 도 같은 anchor 로 이동시켜야 dynamics 일관 (dθ/dt = vs).
        """
        prev = self._prev_x_pred.copy()
        if abs(theta_offset) > 0.0:
            prev[:, STHETA] = prev[:, STHETA] + theta_offset
        N = self.N
        for k in range(N):
            self.solver.set(k, "x", prev[k + 1])
        self.solver.set(N, "x", prev[N])

    def _apply_cold_seed(self, x0, params=None, theta_init=None):
        """### HJ : Niraj-style cold seed — centerline 따라 vs_seed 로 미리 펼친 trajectory.

        - x0  : 현재 차 state (NX,)
        - params : (N+1, NP) — build_preview 결과. col 0~2 = (x_ref, y_ref, psi_ref).
                   None 이면 단순 x0-hold fallback.
        - theta_init : θ 시작값 (현재 차의 projection). None 이면 x0[STHETA].

        Stage 별 seed (params 있을 때):
          x[0]   = x0  (현재 위치 — 변경 안 함)
          x[k]   = (x_ref[k], y_ref[k], psi_ref[k], vs_seed,   # X, Y, ψ, vx
                    0, 0,                                      # vy, ω (kinematic 가정)
                    0, theta_init + k·dt·vs_seed)              # δ, θ
        u[k]   = (0, 0, vs_seed)                                # u_ddelta, u_a, vs
        → horizon 즉시 centerline 따라 vs_seed 로 진행하는 plan. SQP 한 iter 후
          친션/steering 만 미세 조정. 정지 출발 / spawn 직후 즉시 가속 시작.
        """
        if params is None or theta_init is None:
            # Fallback : x0-hold (기존 단순 방식). spawn 외 시점 (NaN 등).
            zero_u = np.zeros(NU, dtype=np.float64)
            for k in range(self.N + 1):
                self.solver.set(k, "x", x0)
            for k in range(self.N):
                self.solver.set(k, "u", zero_u)
            return

        cfg = self.mpc_cfg
        vs_seed = float(np.clip(cfg.get("cold_seed_speed", 1.5),
                                cfg["vs_min"], cfg["vs_max"]))
        dt = self.dt
        u_seed = np.array([0.0, 0.0, vs_seed], dtype=np.float64)

        # Stage 0: 차의 실제 state 그대로 (acados 가 lbx/ubx=x0 로 강제).
        self.solver.set(0, "x", x0)
        # Stage 1..N: centerline 따라 펼친 trajectory.
        for k in range(1, self.N + 1):
            seed_x = np.zeros(NX, dtype=np.float64)
            seed_x[SX]      = float(params[k, 0])     # x_ref
            seed_x[SY]      = float(params[k, 1])     # y_ref
            seed_x[SPSI]    = float(params[k, 2])     # psi_ref
            seed_x[SVX]     = vs_seed
            seed_x[SVY]     = 0.0
            seed_x[SOMEGA]  = 0.0
            seed_x[SDELTA]  = 0.0
            seed_x[STHETA]  = theta_init + k * dt * vs_seed
            self.solver.set(k, "x", seed_x)
        for k in range(self.N):
            self.solver.set(k, "u", u_seed)

    # ---- Main loop ----
    def run(self):
        rate = rospy.Rate(self.loop_rate)
        while not rospy.is_shutdown():
            self._tick()
            rate.sleep()

    def _tick(self):
        with self.lock:
            odom, pose = self.odom, self.pose
            cl = self.centerline_np
            d_r, d_l = self.cl_d_right, self.cl_d_left
            first_odom_t = self._first_odom_t
        # 입력이 부족하면 정지 (centerline 도 안 와있으면 가속 의미 없음)
        if odom is None or pose is None or cl is None:
            self._publish_drive(0.0, 0.0, 0.0); return

        # ### HJ : startup_delay 동안 cmd=0. MPCC 가 정상 작동하면 자체 가속이라
        # boost 불필요. 단, odom 도착 후 잠깐 (0.3s) 가만히 대기 — OCP 의 첫 solve
        # 가 정상 plan 을 만들 시간 확보.
        now = rospy.Time.now().to_sec()
        if first_odom_t is not None and (now - first_odom_t) < self.startup_delay_s:
            self._publish_drive(0.0, 0.0, 0.0); return

        # x0 from latest measurements
        # ### HJ : NX=8 — vy, omega 도 odom 에서 측정.
        # 실차 deployment 시 carstate_node 가 vy 를 pose-diff + base_link 회전으로,
        # omega 를 EKF/IMU 에서 가져옴. sim 은 f1tenth_simulator 가 직접 publish.
        px = float(pose.pose.position.x); py = float(pose.pose.position.y)
        q = pose.pose.orientation
        psi_now = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        vx_now = float(odom.twist.twist.linear.x)
        vy_now = float(odom.twist.twist.linear.y)
        omega_now = float(odom.twist.twist.angular.z)
        delta_now = float(self.last_delta)

        theta_init, theta_pred, rebase_offset = self._resolve_theta_warm(cl, px, py)
        x0 = np.array([px, py, psi_now, vx_now, vy_now, omega_now,
                       delta_now, theta_init], dtype=np.float64)

        params = build_preview(
            centerline=cl, N=self.N, dt=self.dt,
            theta_0=theta_init, vs_init=max(vx_now, 1.0),
            cl_x_idx=C_X, cl_y_idx=C_Y, cl_s_idx=C_S,
            cl_kappa_idx=C_KAPPA, cl_psi_idx=C_PSI,
            d_right=d_r, d_left=d_l,
            theta_pred=theta_pred, bound_inset=self.bound_inset,
            psi_anchor=psi_now,   # ψ_ref 를 차의 현재 ψ branch 로 정규화
        )
        if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(params)):
            rospy.logerr_throttle(1.0, f"[{self.name}] non-finite x0/params; skip")
            self._prev_x_pred = None
            self._publish_drive(0.0, 0.0, float(self.last_delta)); return

        # Liniger progress reward: yref[8]=vs_max — fixed, push vs to bound.
        try: self._push_runtime_yref(vx_now=vx_now)
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[{self.name}] yref fail: {e}")

        # Warm start (drift-free: prev plan rebased to projection-anchored θ)
        # ### HJ : prev_x_pred=None 이면 cold seed 됨. 이건 spawn 직후라 첫 solve
        # 결과 publish 시 이전 frame 의 pred 와 19m+ jump 발생 → rviz 에서 path 가
        # 화면 가로질러 그어짐. 이 frame 만 prediction publish skip.
        is_cold = (self._prev_x_pred is None)
        if not is_cold and np.all(np.isfinite(self._prev_x_pred)):
            try: self._apply_shift_warm_start(theta_offset=rebase_offset)
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"[{self.name}] shift fail: {e}")
        else:
            try: self._apply_cold_seed(x0, params=params, theta_init=theta_init)
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"[{self.name}] seed fail: {e}")

        try:
            u0, status, info = solve_once(self.solver, x0, params)
        except Exception as e:
            rospy.logerr_throttle(1.0, f"[{self.name}] solve fail: {e}")
            self._prev_x_pred = None; return

        # Stash + status
        try:
            new_x_pred = np.array(
                [self.solver.get(k, "x") for k in range(self.N + 1)], dtype=np.float64)
        except Exception:
            new_x_pred = None
        if status != 0 or new_x_pred is None or not np.all(np.isfinite(new_x_pred)):
            rospy.logwarn_throttle(2.0, f"[{self.name}] solver bad status={status}")
            self._prev_x_pred = None
        else:
            self._prev_x_pred = new_x_pred

        try: self.solve_status_pub.publish(Int8(data=int(status)))
        except Exception: pass

        # Publish cmd
        if status != 0 or new_x_pred is None or not np.all(np.isfinite(new_x_pred)):
            self._publish_drive(0.0, 0.0, float(self.last_delta)); return

        x_next = new_x_pred[1]
        # speed_cmd : MPCC 가 예측한 다음 step 의 vx (실차에선 VESC 가 closed-loop tracking)
        # steer_cmd : 예측한 다음 step 의 δ
        speed_cmd = float(np.clip(x_next[SVX], 0.0, self.v_max))
        steer_cmd = float(np.clip(x_next[SDELTA], -self.max_steer, self.max_steer))
        self.last_delta = steer_cmd
        self._publish_drive(speed_cmd, float(u0[1]), steer_cmd)

        # Spawn 직후 (cold seed) frame 은 pred publish skip — rviz 가 이전 marker
        # 와 새 marker 사이 19m+ jump 보이는 것 방지.
        if not is_cold:
            try: self._publish_prediction()
            except Exception: pass

    def _publish_drive(self, speed: float, accel: float, steer: float):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(speed)
        msg.drive.acceleration = float(accel)
        msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(msg)

    def _publish_prediction(self):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = "mpcc/pred"
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.04
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.9, 0.2, 1.0
        m.pose.orientation.w = 1.0
        for k in range(self.N + 1):
            xk = self.solver.get(k, "x")
            p = Point(); p.x = float(xk[SX]); p.y = float(xk[SY]); p.z = 0.0
            m.points.append(p)
        self.pred_pub.publish(m)


def main():
    MpccNode().run()


if __name__ == "__main__":
    main()
