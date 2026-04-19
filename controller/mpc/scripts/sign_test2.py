#!/usr/bin/env python3
"""
Sign test v2: sample car state and compute n two ways, compare.

Way A: n_mpc = -sin(psi_ref)*dx + cos(psi_ref)*dy  (MPC's internal convention)
Way B: n_frenet = /car_state/odom_frenet.pose.pose.position.y (repo's converter)

If signs match: same convention.
If signs opposite: my calc uses opposite convention from repo's Frenet.
"""

from __future__ import annotations

import math
import time

import rospy
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


def main():
    rospy.init_node("sign_test2", anonymous=True)
    wp = rospy.wait_for_message("/global_waypoints", WpntArray, timeout=20)
    wpnts = [(w.x_m, w.y_m, w.psi_rad) for w in wp.wpnts]

    samples = []

    def sync_cb():
        # Read latest pose + frenet
        try:
            po = rospy.wait_for_message("/car_state/pose", PoseStamped, timeout=1.0)
            fr = rospy.wait_for_message("/car_state/odom_frenet", Odometry, timeout=1.0)
        except rospy.ROSException:
            return
        cx, cy = po.pose.position.x, po.pose.position.y
        # nearest wpnt
        dists = [(cx - w[0]) ** 2 + (cy - w[1]) ** 2 for w in wpnts]
        i = min(range(len(dists)), key=lambda k: dists[k])
        wpx, wpy, psi = wpnts[i]
        dx, dy = cx - wpx, cy - wpy
        n_mpc = -math.sin(psi) * dx + math.cos(psi) * dy
        n_frenet = fr.pose.pose.position.y
        samples.append((cx, cy, n_mpc, n_frenet, psi))

    rospy.loginfo("[sign_test2] 10 samples over 10s...")
    for _ in range(10):
        sync_cb()
        rospy.sleep(1.0)

    if not samples:
        rospy.logerr("no samples")
        return

    rospy.loginfo("=" * 70)
    rospy.loginfo(f"{'car_xy':<26}{'n_mpc':>10}{'n_frenet':>10}{'signs':>10}")
    agree = 0
    total = 0
    for (cx, cy, nm, nf, psi) in samples:
        same = "OK" if (nm * nf > 0) else ("ZERO" if nm * nf == 0 else "FLIP")
        if nm * nf != 0:
            total += 1
            if nm * nf > 0:
                agree += 1
        rospy.loginfo(f"({cx:+.2f},{cy:+.2f})            {nm:+.3f}   {nf:+.3f}   {same}")
    rospy.loginfo("=" * 70)
    if total > 0:
        rospy.loginfo(f"[sign_test2] agreement {agree}/{total} → " + ("SAME SIGN" if agree == total else "OPPOSITE SIGN"))


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
