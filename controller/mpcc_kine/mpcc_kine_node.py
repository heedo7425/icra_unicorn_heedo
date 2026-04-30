#!/usr/bin/env python3
"""
mpcc_node — minimal kinematic MPCC controller (Liniger-style).

State (6):  [X, Y, psi, v, delta, theta]
Input (3):  [u_ddelta, u_a, vs]
NX/NU/NP : see mpcc.vehicle_model

Topics:
    /car_state/odom        Odometry      input
    /car_state/pose        PoseStamped   input
    /centerline_waypoints  WpntArray     input

    /mpcc_kine/cmd              AckermannDriveStamped   output (drive_topic)
    /mpcc_kine/predicted_path   Marker                  viz
    /mpcc_kine/solve_status     Int8

Driving cmd: speed=v_pred[1], steering_angle=delta_pred[1].
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

from mpcc_kine.vehicle_model import NX, NU, NP   # noqa: E402
from mpcc_kine.mpcc_ocp import build_mpcc_ocp, solve_once   # noqa: E402
from mpcc_kine.reference_builder import build_preview, project_to_centerline   # noqa: E402

# Wpnt numpy column indices.
C_X, C_Y, C_Z, C_VX, _, C_S, C_KAPPA, C_PSI, C_AX, C_D, C_MU_RAD = range(11)

# State indices.
SX, SY, SPSI, SV, SDELTA, STHETA = range(NX)


def _load_vehicle_params() -> dict:
    gp = rospy.get_param
    # Kinematic only needs wheelbase. Fallback chain l_wb → l_f+l_r → 0.307.
    l_f = gp("/vehicle/l_f", 0.162)
    l_r = gp("/vehicle/l_r", 0.145)
    l_wb = gp("/vehicle/l_wb", l_f + l_r)
    return {"l_wb": l_wb}


class MpccNode:
    def __init__(self):
        rospy.init_node("mpcc_controller", anonymous=False)
        self.name = "mpcc_kine"
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

        # Cost weights
        self.mpc_cfg = {
            "N_horizon": self.N, "dt": self.dt,
            "v_max": self.v_max, "v_min": self.v_min,
            "max_steer": self.max_steer, "max_steer_rate": self.max_steer_rate,
            "max_accel": self.max_accel, "max_decel": self.max_decel,
            "vs_min": self.vs_min, "vs_max": self.vs_max,
            "vs_target": float(rospy.get_param(f"{NS}/vs_target", self.vs_max)),
            "qC":       float(rospy.get_param(f"{NS}/qC", 1.0)),
            "qL":       float(rospy.get_param(f"{NS}/qL", 100.0)),
            "qMu":      float(rospy.get_param(f"{NS}/qMu", 0.1)),
            "qVs":      float(rospy.get_param(f"{NS}/qVs", 0.5)),
            "qV":       float(rospy.get_param(f"{NS}/qV", 0.5)),
            "qVsv":     float(rospy.get_param(f"{NS}/qVsv", 1.0)),
            "qDelta":   float(rospy.get_param(f"{NS}/qDelta", 0.0)),
            "qDDelta":  float(rospy.get_param(f"{NS}/qDDelta", 0.05)),
            "qA":       float(rospy.get_param(f"{NS}/qA", 0.05)),
            "qC_term":  float(rospy.get_param(f"{NS}/qC_term", 10.0)),
            "qL_term":  float(rospy.get_param(f"{NS}/qL_term", 1.0)),
            "track_slack_penalty":
                        float(rospy.get_param(f"{NS}/track_slack_penalty", 1000.0)),
            "levenberg_marquardt":
                        float(rospy.get_param(f"{NS}/levenberg_marquardt", 1.0)),
        }

        self.bound_inset = float(rospy.get_param(f"{NS}/bound_inset", 0.10))
        self.startup_delay_s = float(rospy.get_param(f"{NS}/startup_delay_s", 1.0))
        # ### HJ : warm_start_max_drift 는 drift-free anchoring 도입 후 사용 안 함.
        # yaml legacy 호환 위해 read 만 하고 무시.
        self.warm_start_max_drift = float(
            rospy.get_param(f"{NS}/warm_start_max_drift", 1.5))   # unused

        self.vp = _load_vehicle_params()

        # Build OCP solver
        codegen_dir = rospy.get_param(f"{NS}/codegen_dir", "/tmp/mpcc_kine_codegen")
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
        self._start_time = rospy.Time.now().to_sec()

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
        with self.lock: self.odom = msg

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
        TUNABLE = ("qC", "qL", "qMu", "qVs", "qV", "qVsv", "qDelta", "qDDelta", "qA",
                   "qC_term", "qL_term", "track_slack_penalty",
                   "v_max", "v_min", "vs_max", "vs_target")
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
            return TriggerResponse(success=True, message=f"updated: {applied}")
        except Exception as e:
            return TriggerResponse(success=False, message=f"err: {e}")

    def _push_runtime_costs(self):
        cfg = self.mpc_cfg
        qC = cfg["qC"]; qL = cfg["qL"]
        # residual order: eC, eL, dpsi, delta, u_ddelta, u_a, vs_target-vs, v_target-v, vs-v
        W = np.diag([qC, qL, cfg["qMu"], cfg["qDelta"],
                     cfg["qDDelta"], cfg["qA"], cfg["qVs"], cfg["qV"], cfg["qVsv"]])
        W_e = np.diag([cfg["qC_term"] * qC, cfg["qL_term"] * qL,
                       cfg["qMu"], cfg["qDelta"]])
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

    def _push_runtime_slack(self):
        track_pen = self.mpc_cfg["track_slack_penalty"]
        z = np.array([track_pen, track_pen], dtype=np.float64)
        for k in range(self.N + 1):
            try:
                self.solver.cost_set(k, "zl", z)
                self.solver.cost_set(k, "zu", z)
                self.solver.cost_set(k, "Zl", z)
                self.solver.cost_set(k, "Zu", z)
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
        # prev θ sequence shifted by 1 (Liniger SQP shift), last stage 외삽.
        prev_theta = self._prev_x_pred[:, STHETA].copy()
        last_step = prev_theta[-1] - prev_theta[-2] if len(prev_theta) >= 2 else 0.0
        shifted = np.concatenate([prev_theta[1:], [prev_theta[-1] + last_step]])
        # Anchor at projection: rebase so shifted[0] == theta_proj.
        rebase_offset = float(theta_proj - shifted[0])
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

    def _apply_cold_seed(self, x0):
        zero_u = np.zeros(NU, dtype=np.float64)
        for k in range(self.N + 1):
            self.solver.set(k, "x", x0)
        for k in range(self.N):
            self.solver.set(k, "u", zero_u)

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
        if odom is None or pose is None or cl is None:
            self._publish_drive(0.0, 0.0, 0.0); return

        if rospy.Time.now().to_sec() - self._start_time < self.startup_delay_s:
            self._publish_drive(0.0, 0.0, 0.0); return

        # x0 from latest measurements
        px = float(pose.pose.position.x); py = float(pose.pose.position.y)
        q = pose.pose.orientation
        psi_now = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        v_now = float(odom.twist.twist.linear.x)
        delta_now = float(self.last_delta)

        theta_init, theta_pred, rebase_offset = self._resolve_theta_warm(cl, px, py)
        x0 = np.array([px, py, psi_now, v_now, delta_now, theta_init], dtype=np.float64)

        params = build_preview(
            centerline=cl, N=self.N, dt=self.dt,
            theta_0=theta_init, vs_init=max(v_now, 1.0),
            cl_x_idx=C_X, cl_y_idx=C_Y, cl_s_idx=C_S,
            cl_kappa_idx=C_KAPPA, cl_psi_idx=C_PSI,
            d_right=d_r, d_left=d_l,
            theta_pred=theta_pred, bound_inset=self.bound_inset,
        )
        if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(params)):
            rospy.logerr_throttle(1.0, f"[{self.name}] non-finite x0/params; skip")
            self._prev_x_pred = None
            self._publish_drive(0.0, 0.0, float(self.last_delta)); return

        # Warm start (drift-free: prev plan rebased to projection-anchored θ)
        if self._prev_x_pred is not None and np.all(np.isfinite(self._prev_x_pred)):
            try: self._apply_shift_warm_start(theta_offset=rebase_offset)
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"[{self.name}] shift fail: {e}")
        else:
            try: self._apply_cold_seed(x0)
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
        speed_cmd = float(np.clip(x_next[SV], 0.0, self.v_max))
        steer_cmd = float(np.clip(x_next[SDELTA], -self.max_steer, self.max_steer))
        self.last_delta = steer_cmd
        self._publish_drive(speed_cmd, float(u0[1]), steer_cmd)

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
