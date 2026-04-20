#!/usr/bin/env python3
"""
μ-adaptation 토글 + 실시간 속도/μ 대시보드 (tkinter).

좌상단 고정, always-on-top. 내용:
  - 속도: /car_state/odom twist.linear.x
  - 명령 속도: /vesc/.../nav_1 drive.speed
  - gt μ / est μ
  - ENABLE/DISABLE 토글 버튼 → /gp_mpc/mu_adapt_enable (Bool, latched)

DISPLAY 없으면 console REPL fallback (Enter=toggle, y/n, Ctrl-C).
"""

from __future__ import annotations

import os

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Empty, Float32, Float32MultiArray


class Dashboard:
    def __init__(self) -> None:
        self.vx = 0.0
        self.cmd_v = 0.0
        self.mu_gt = 0.85
        self.mu_est = 0.85
        self.enabled = True
        # GP 상태
        self.residual = [0.0, 0.0, 0.0]
        self.sigma = [0.0, 0.0, 0.0]
        self.gp_ready = False
        self.train_time_s = 0.0
        self.buffer_size = 0.0
        self.solve_ms = 0.0
        self.cmd_steer = 0.0
        self.cmd_v_raw = 0.0         # /gp_mpc/cmd_raw.speed (pre mu_applier)
        self.cmd_base_speed = 0.0
        self.cmd_base_steer = 0.0

        self.pub = rospy.Publisher("/gp_mpc/mu_adapt_enable", Bool, queue_size=1, latch=True)
        self.reset_pub = rospy.Publisher("/gp_mpc/gp_reset", Empty, queue_size=1)
        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber("/vesc/high_level/ackermann_cmd_mux/input/nav_1",
                         AckermannDriveStamped, self._cmd_cb, queue_size=1)
        rospy.Subscriber("/mu_ground_truth", Float32, self._gt_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/mu_used", Float32, self._est_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/residual", Float32MultiArray, self._res_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/gp_sigma", Float32MultiArray, self._sig_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/gp_ready", Bool, self._ready_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/train_time_s", Float32, self._train_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/buffer_size", Float32, self._buf_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/solve_ms", Float32, self._solve_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/cmd_raw", AckermannDriveStamped,
                         self._cmdraw_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/cmd_base_speed", Float32,
                         lambda m: setattr(self, "cmd_base_speed", float(m.data)),
                         queue_size=1)
        rospy.Subscriber("/gp_mpc/cmd_base_steer", Float32,
                         lambda m: setattr(self, "cmd_base_steer", float(m.data)),
                         queue_size=1)

    def _odom_cb(self, m): self.vx = float(m.twist.twist.linear.x)
    def _cmd_cb(self, m):  self.cmd_v = float(m.drive.speed)
    def _gt_cb(self, m):   self.mu_gt = float(m.data)
    def _est_cb(self, m):  self.mu_est = float(m.data)
    def _res_cb(self, m):
        if len(m.data) >= 3:
            self.residual = [float(m.data[0]), float(m.data[1]), float(m.data[2])]
    def _sig_cb(self, m):
        if len(m.data) >= 3:
            self.sigma = [float(m.data[0]), float(m.data[1]), float(m.data[2])]
    def _ready_cb(self, m):  self.gp_ready = bool(m.data)
    def _train_cb(self, m):  self.train_time_s = float(m.data)
    def _buf_cb(self, m):    self.buffer_size = float(m.data)
    def _solve_cb(self, m):  self.solve_ms = float(m.data)
    def _cmdraw_cb(self, m):
        self.cmd_steer = float(m.drive.steering_angle)
        self.cmd_v_raw = float(m.drive.speed)

    def set_enabled(self, v: bool) -> None:
        self.enabled = v
        self.pub.publish(Bool(data=v))

    def reset_gp(self) -> None:
        self.reset_pub.publish(Empty())


def gui_main(dash: Dashboard):
    import tkinter as tk
    root = tk.Tk()
    root.title("GP-MPC Dashboard")
    root.geometry("460x660+40+40")   # GP vs Base 비교 행 포함
    try:
        root.attributes("-topmost", True)
        root.attributes("-type", "dialog")  # some WMs show dialog above full-screen apps
    except Exception:
        pass
    root.configure(bg="#111")
    # 강제 포커스 + lift
    root.lift()
    root.focus_force()

    lbl_font = ("Helvetica", 12, "bold")
    val_font = ("Helvetica", 18, "bold")

    row_speed = tk.Frame(root, bg="#111"); row_speed.pack(fill="x", pady=2)
    tk.Label(row_speed, text="vx  /  cmd", fg="#bbb", bg="#111", font=lbl_font).pack(side="left", padx=6)
    speed_lbl = tk.Label(row_speed, text="0.00 / 0.00", fg="#6f6", bg="#111", font=val_font)
    speed_lbl.pack(side="right", padx=6)

    row_solve = tk.Frame(root, bg="#111"); row_solve.pack(fill="x", pady=2)
    tk.Label(row_solve, text="OCP solve", fg="#bbb", bg="#111", font=lbl_font).pack(side="left", padx=6)
    solve_lbl = tk.Label(row_solve, text="0.0 ms", fg="#6cf", bg="#111", font=val_font)
    solve_lbl.pack(side="right", padx=6)

    # --- GP vs Base 비교 ---
    sep2 = tk.Frame(root, bg="#333", height=2); sep2.pack(fill="x", padx=6, pady=4)
    tk.Label(root, text="GP  vs  BASE (μ=0.85)", fg="#bbb", bg="#111",
             font=("Helvetica", 11, "bold")).pack(anchor="w", padx=6)
    cmp_font = ("Courier", 14, "bold")
    cmp_frame = tk.Frame(root, bg="#111"); cmp_frame.pack(fill="x", padx=6, pady=2)
    # Speed comparison
    row_speed_cmp = tk.Frame(cmp_frame, bg="#111"); row_speed_cmp.pack(fill="x", pady=1)
    tk.Label(row_speed_cmp, text="speed", fg="#ccc", bg="#111",
             font=("Helvetica", 11, "bold"), width=6).pack(side="left")
    cmp_v_gp = tk.Label(row_speed_cmp, text="0.00", fg="#6f6", bg="#111",
                         font=cmp_font, width=6, anchor="e"); cmp_v_gp.pack(side="left", padx=4)
    tk.Label(row_speed_cmp, text="|", fg="#555", bg="#111").pack(side="left", padx=2)
    cmp_v_base = tk.Label(row_speed_cmp, text="0.00", fg="#aaa", bg="#111",
                           font=cmp_font, width=6, anchor="e"); cmp_v_base.pack(side="left", padx=4)
    tk.Label(row_speed_cmp, text="Δ", fg="#888", bg="#111",
             font=("Helvetica", 11, "bold")).pack(side="left", padx=(10, 2))
    cmp_v_diff = tk.Label(row_speed_cmp, text="+0.00", fg="#fc6", bg="#111",
                           font=cmp_font, width=7, anchor="e"); cmp_v_diff.pack(side="left")
    # Steering comparison (deg)
    row_steer_cmp = tk.Frame(cmp_frame, bg="#111"); row_steer_cmp.pack(fill="x", pady=1)
    tk.Label(row_steer_cmp, text="steer", fg="#ccc", bg="#111",
             font=("Helvetica", 11, "bold"), width=6).pack(side="left")
    cmp_s_gp = tk.Label(row_steer_cmp, text="+0.0°", fg="#6f6", bg="#111",
                         font=cmp_font, width=6, anchor="e"); cmp_s_gp.pack(side="left", padx=4)
    tk.Label(row_steer_cmp, text="|", fg="#555", bg="#111").pack(side="left", padx=2)
    cmp_s_base = tk.Label(row_steer_cmp, text="+0.0°", fg="#aaa", bg="#111",
                           font=cmp_font, width=6, anchor="e"); cmp_s_base.pack(side="left", padx=4)
    tk.Label(row_steer_cmp, text="Δ", fg="#888", bg="#111",
             font=("Helvetica", 11, "bold")).pack(side="left", padx=(10, 2))
    cmp_s_diff = tk.Label(row_steer_cmp, text="+0.0°", fg="#fc6", bg="#111",
                           font=cmp_font, width=7, anchor="e"); cmp_s_diff.pack(side="left")

    # --- GP 패널 ---
    sep = tk.Frame(root, bg="#333", height=2); sep.pack(fill="x", padx=6, pady=6)
    gp_hdr = tk.Frame(root, bg="#111"); gp_hdr.pack(fill="x", pady=2)
    tk.Label(gp_hdr, text="GP", fg="#bbb", bg="#111", font=("Helvetica", 13, "bold")).pack(side="left", padx=6)
    gp_status_lbl = tk.Label(gp_hdr, text="COLD", fg="#aaa", bg="#111", font=val_font)
    gp_status_lbl.pack(side="right", padx=6)

    row_gp_meta = tk.Frame(root, bg="#111"); row_gp_meta.pack(fill="x", pady=2)
    tk.Label(row_gp_meta, text="buf  /  train", fg="#bbb", bg="#111", font=lbl_font).pack(side="left", padx=6)
    gp_meta_lbl = tk.Label(row_gp_meta, text="0 / 0.0s", fg="#9cf", bg="#111", font=val_font)
    gp_meta_lbl.pack(side="right", padx=6)

    # --- GP correction magnitude (큰 숫자 + 색 bar) ---
    row_corr = tk.Frame(root, bg="#111"); row_corr.pack(fill="x", pady=2)
    tk.Label(row_corr, text="correction", fg="#bbb", bg="#111",
             font=("Helvetica", 12, "bold")).pack(side="left", padx=6)
    corr_lbl = tk.Label(row_corr, text="0%", fg="#888", bg="#111",
                        font=("Helvetica", 26, "bold"))
    corr_lbl.pack(side="right", padx=6)

    corr_bar = tk.Canvas(root, height=14, bg="#222", highlightthickness=0)
    corr_bar.pack(fill="x", padx=10, pady=(0, 4))
    corr_bar_fg = corr_bar.create_rectangle(0, 0, 0, 14, fill="#6f6", width=0)

    # Residual — Δvx/Δvy/Δω 각각 한 줄 (값 + σ + 클립 여부 색)
    res_font = ("Courier", 14, "bold")
    res_lbl_font = ("Helvetica", 11, "bold")
    gp_frame = tk.Frame(root, bg="#111"); gp_frame.pack(fill="x", pady=4, padx=6)
    dv_lbls = {}
    for i, name in enumerate(("Δvx", "Δvy", "Δω")):
        row = tk.Frame(gp_frame, bg="#111"); row.pack(fill="x", pady=1)
        tk.Label(row, text=name, fg="#ccc", bg="#111", font=res_lbl_font, width=4).pack(side="left")
        v_lbl = tk.Label(row, text="+0.000", fg="#6f6", bg="#111", font=res_font, width=8, anchor="e")
        v_lbl.pack(side="left", padx=4)
        tk.Label(row, text="σ", fg="#777", bg="#111", font=res_lbl_font).pack(side="left", padx=(10, 2))
        s_lbl = tk.Label(row, text="0.00", fg="#999", bg="#111", font=res_font, width=6, anchor="e")
        s_lbl.pack(side="left")
        dv_lbls[name] = (v_lbl, s_lbl)

    # --- Scrolling residual time-series (last N samples) ---
    PLOT_W = 400
    PLOT_H = 100
    PLOT_N = 150   # ~7.5s at 20Hz
    ts_canvas = tk.Canvas(root, width=PLOT_W, height=PLOT_H, bg="#0a0a0a",
                          highlightthickness=0)
    ts_canvas.pack(padx=10, pady=6)
    # 0-line
    ts_canvas.create_line(0, PLOT_H // 2, PLOT_W, PLOT_H // 2, fill="#444")
    ts_hist = {"Δvx": [], "Δvy": [], "Δω": []}
    ts_colors = {"Δvx": "#6cf", "Δvy": "#fc6", "Δω": "#f6c"}
    ts_lines = {k: ts_canvas.create_line(0, PLOT_H // 2, 0, PLOT_H // 2,
                                          fill=ts_colors[k], width=2) for k in ts_hist}
    # 범례
    lg_frame = tk.Frame(root, bg="#111"); lg_frame.pack(fill="x", padx=10)
    for name, col in ts_colors.items():
        tk.Label(lg_frame, text=f"— {name}", fg=col, bg="#111",
                 font=("Helvetica", 10, "bold")).pack(side="left", padx=6)

    state_lbl = tk.Label(root, text="", bg="#111", font=("Helvetica", 11))
    state_lbl.pack(pady=4)

    def toggle():
        dash.set_enabled(not dash.enabled)
        refresh()

    btn = tk.Button(root, text="", command=toggle, font=("Helvetica", 12, "bold"))
    btn.pack(fill="x", padx=10, pady=6)

    def reset_click():
        dash.reset_gp()
        reset_btn.config(text="GP RESET sent ✓", fg="#fff", bg="#844")
        root.after(1200, lambda: reset_btn.config(text="Reset GP", fg="#fff", bg="#633"))

    reset_btn = tk.Button(root, text="Reset GP", command=reset_click,
                          font=("Helvetica", 11, "bold"), fg="#fff", bg="#633",
                          activebackground="#844")
    reset_btn.pack(fill="x", padx=10, pady=(0, 4))

    # clip 기본값 (config 와 싱크; 초과 시 빨간 표시)
    clip = (10.0, 5.0, 12.0)

    def refresh():
        speed_lbl.config(text=f"{dash.vx:4.2f} / {dash.cmd_v:4.2f}")
        solve_lbl.config(text=f"{dash.solve_ms:4.1f} ms",
                         fg="#6cf" if dash.solve_ms < 15 else ("#fc6" if dash.solve_ms < 25 else "#f55"))

        # GP vs Base 비교 — 양쪽 모두 MPC raw cmd (pre mu_applier scaling).
        # GP reset 후 residual=0 이면 두 값이 같아야 함 (warm-start 수렴 후).
        v_diff = dash.cmd_v_raw - dash.cmd_base_speed
        cmp_v_gp.config(text=f"{dash.cmd_v_raw:5.2f}")
        cmp_v_base.config(text=f"{dash.cmd_base_speed:5.2f}")
        v_diff_color = "#6f6" if v_diff > 0.15 else ("#f66" if v_diff < -0.15 else "#fc6")
        cmp_v_diff.config(text=f"{v_diff:+5.2f}", fg=v_diff_color)
        import math
        gp_deg = math.degrees(dash.cmd_steer)
        bs_deg = math.degrees(dash.cmd_base_steer)
        s_diff = gp_deg - bs_deg
        cmp_s_gp.config(text=f"{gp_deg:+5.1f}°")
        cmp_s_base.config(text=f"{bs_deg:+5.1f}°")
        s_diff_color = "#6cf" if abs(s_diff) > 2.0 else "#fc6"
        cmp_s_diff.config(text=f"{s_diff:+5.1f}°", fg=s_diff_color)

        # --- GP status ---
        if dash.gp_ready:
            gp_status_lbl.config(text="READY", fg="#6f6")
        else:
            gp_status_lbl.config(text="COLD", fg="#aaa")
        gp_meta_lbl.config(text=f"{int(dash.buffer_size)} / {dash.train_time_s:.1f}s")

        # --- Correction magnitude (max channel normalized to clip) ---
        corr = 0.0
        if dash.gp_ready and dash.enabled:
            for i in range(3):
                c = abs(dash.residual[i]) / clip[i]
                if c > corr:
                    corr = c
        pct = min(100.0, corr * 100)
        if pct < 5:
            corr_color = "#888"
        elif pct < 30:
            corr_color = "#6f6"
        elif pct < 70:
            corr_color = "#fc6"
        else:
            corr_color = "#f55"
        corr_lbl.config(text=f"{pct:.0f}%", fg=corr_color)
        bar_w = int(pct / 100.0 * PLOT_W)
        corr_bar.coords(corr_bar_fg, 0, 0, bar_w, 14)
        corr_bar.itemconfig(corr_bar_fg, fill=corr_color)

        # Residual 값별 색: clip 95% 이상 빨강, σ 크면 노랑, 정상 녹색
        names = ("Δvx", "Δvy", "Δω")
        for i, name in enumerate(names):
            v = dash.residual[i]
            s = dash.sigma[i]
            clipped = abs(v) >= 0.95 * clip[i]
            if not dash.gp_ready:
                v_color = "#888"
            elif clipped:
                v_color = "#f55"
            elif s > 2.0:
                v_color = "#fc6"
            else:
                v_color = "#6f6"
            marker = "*" if clipped else ""
            v_lbl, s_lbl = dv_lbls[name]
            v_lbl.config(text=f"{v:+.3f}{marker}", fg=v_color)
            s_color = "#f55" if s > 2.0 else ("#fc6" if s > 1.0 else "#999")
            s_lbl.config(text=f"{s:.2f}", fg=s_color)
            # Push to scrolling history (normalized to clip — 공통 y-axis).
            norm_v = max(-1.0, min(1.0, v / clip[i]))
            ts_hist[name].append(norm_v)
            if len(ts_hist[name]) > PLOT_N:
                ts_hist[name] = ts_hist[name][-PLOT_N:]

        # Redraw time-series lines (scrolling).
        for name in ts_hist:
            pts = ts_hist[name]
            if len(pts) < 2:
                continue
            coords = []
            for k, val in enumerate(pts):
                x = k * PLOT_W / max(1, PLOT_N - 1)
                y = PLOT_H / 2 - val * (PLOT_H / 2 - 4)
                coords.extend([x, y])
            ts_canvas.coords(ts_lines[name], *coords)

        if dash.enabled:
            state_lbl.config(text="adaptation: ENABLED", fg="#6f6")
            btn.config(text="Disable μ adaptation", bg="#aaa")
        else:
            state_lbl.config(text="adaptation: DISABLED", fg="#f66")
            btn.config(text="Enable μ adaptation", bg="#aaa")
        # 주기적으로 lift → rviz 전체화면 뒤로 숨지 않게
        try:
            root.lift()
        except Exception:
            pass
        if not rospy.is_shutdown():
            root.after(200, refresh)
        else:
            root.quit()

    dash.set_enabled(True)
    refresh()
    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), None))
    root.mainloop()


def console_main(dash: Dashboard):
    rospy.loginfo("[mu_toggle_gui] console mode — Enter=toggle, y/n, Ctrl-C=exit")
    dash.set_enabled(True)
    while not rospy.is_shutdown():
        try:
            line = input(f"[adapt={'ON' if dash.enabled else 'OFF'} vx={dash.vx:.2f} gt={dash.mu_gt:.2f} est={dash.mu_est:.2f}] > ").strip().lower()
        except EOFError:
            break
        if line == "":
            dash.set_enabled(not dash.enabled)
        elif line in ("y", "on", "1"):
            dash.set_enabled(True)
        elif line in ("n", "off", "0"):
            dash.set_enabled(False)


def main():
    rospy.init_node("mu_toggle_gui", anonymous=False)
    dash = Dashboard()
    if os.environ.get("DISPLAY", ""):
        try:
            gui_main(dash)
            return
        except Exception as e:
            rospy.logwarn(f"[mu_toggle_gui] GUI failed ({e}); falling back to console")
    console_main(dash)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
