#!/usr/bin/env python3
"""
μ-adaptation 토글 + 실시간 속도/μ 대시보드 (tkinter).

좌상단 고정, always-on-top. 내용:
  - 속도: /car_state/odom twist.linear.x
  - 명령 속도: /vesc/.../nav_1 drive.speed
  - gt μ / est μ
  - ENABLE/DISABLE 토글 버튼 → /ekf_mpc/mu_adapt_enable (Bool, latched)

DISPLAY 없으면 console REPL fallback (Enter=toggle, y/n, Ctrl-C).
"""

from __future__ import annotations

import os

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32


class Dashboard:
    def __init__(self) -> None:
        self.vx = 0.0
        self.cmd_v = 0.0
        self.mu_gt = 0.85
        self.mu_est = 0.85
        self.enabled = True

        self.pub = rospy.Publisher("/ekf_mpc/mu_adapt_enable", Bool, queue_size=1, latch=True)
        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber("/vesc/high_level/ackermann_cmd_mux/input/nav_1",
                         AckermannDriveStamped, self._cmd_cb, queue_size=1)
        rospy.Subscriber("/mu_ground_truth", Float32, self._gt_cb, queue_size=1)
        rospy.Subscriber("/ekf_mpc/mu_used", Float32, self._est_cb, queue_size=1)

    def _odom_cb(self, m): self.vx = float(m.twist.twist.linear.x)
    def _cmd_cb(self, m):  self.cmd_v = float(m.drive.speed)
    def _gt_cb(self, m):   self.mu_gt = float(m.data)
    def _est_cb(self, m):  self.mu_est = float(m.data)

    def set_enabled(self, v: bool) -> None:
        self.enabled = v
        self.pub.publish(Bool(data=v))


def gui_main(dash: Dashboard):
    import tkinter as tk
    root = tk.Tk()
    root.title("MPC-MS Dashboard")
    root.geometry("320x220+40+40")   # 좌상단 고정
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

    row_mu = tk.Frame(root, bg="#111"); row_mu.pack(fill="x", pady=2)
    tk.Label(row_mu, text="gt  /  est", fg="#bbb", bg="#111", font=lbl_font).pack(side="left", padx=6)
    mu_lbl = tk.Label(row_mu, text="0.85 / 0.85", fg="#fc6", bg="#111", font=val_font)
    mu_lbl.pack(side="right", padx=6)

    state_lbl = tk.Label(root, text="", bg="#111", font=("Helvetica", 11))
    state_lbl.pack(pady=4)

    def toggle():
        dash.set_enabled(not dash.enabled)
        refresh()

    btn = tk.Button(root, text="", command=toggle, font=("Helvetica", 12, "bold"))
    btn.pack(fill="x", padx=10, pady=6)

    def refresh():
        speed_lbl.config(text=f"{dash.vx:4.2f} / {dash.cmd_v:4.2f}")
        diff = abs(dash.mu_gt - dash.mu_est)
        fg = "#fc6" if diff < 0.08 else "#f55"
        mu_lbl.config(text=f"{dash.mu_gt:.2f} / {dash.mu_est:.2f}", fg=fg)
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
