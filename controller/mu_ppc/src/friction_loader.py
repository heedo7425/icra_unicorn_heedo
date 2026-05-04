#!/usr/bin/env python3
"""friction_loader — stack_master/maps/<map>/friction_scaling.yaml 직접 로드해
/friction_map_params/{n_sectors, global_friction_limit, Sector{i}/...} 로 publish.
sector 갯수 무관 (dyn_reconfigure 우회). /global_waypoints 도착 시 idx → s_m 변환.
"""
from __future__ import annotations

import os
from typing import List

import rospkg
import rospy
import yaml
from f110_msgs.msg import WpntArray
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class FrictionLoader:
    def __init__(self):
        rospy.init_node("friction_loader")
        self.map_name = rospy.get_param("~map", rospy.get_param("/map", ""))
        if not self.map_name:
            rospy.logwarn("[friction_loader] no map param; aborting")
            return
        self.yaml_path = self._resolve_yaml(self.map_name)
        self.sectors: List[dict] = []
        self.global_limit = 1.0
        self._wpts_done = False

        self._loaded_pub = rospy.Publisher("/mu_ppc/sectors_loaded", Bool,
                                           queue_size=1, latch=True)
        self._marker_pub = rospy.Publisher(
            "/mu_ppc/friction_sectors_loaded_markers", MarkerArray,
            queue_size=1, latch=True)

        self._load_yaml()
        self._publish_idx_only_params()
        rospy.Subscriber("/global_waypoints", WpntArray, self._wpnts_cb, queue_size=1)
        rospy.loginfo("[friction_loader] %d sectors loaded from %s",
                      len(self.sectors), self.yaml_path)

    def _resolve_yaml(self, m: str) -> str:
        try:
            base = rospkg.RosPack().get_path("stack_master")
        except Exception:
            base = "/home/hmcl/catkin_ws/src/race_stack/stack_master"
        return os.path.join(base, "maps", m, "friction_scaling.yaml")

    def _load_yaml(self) -> None:
        if not os.path.isfile(self.yaml_path):
            rospy.logwarn("[friction_loader] %s not found — empty (mu=1.0)", self.yaml_path)
            return
        with open(self.yaml_path) as f:
            data = yaml.safe_load(f) or {}
        self.global_limit = float(data.get("global_friction_limit",
                                           data.get("global_friction", 1.0)))
        n = int(data.get("n_sectors", 0))
        for i in range(n):
            sec = data.get(f"Sector{i}")
            if not isinstance(sec, dict):
                continue
            self.sectors.append({
                "start": int(sec.get("start", 0)),
                "end": int(sec.get("end", 0)),
                "friction": float(sec.get("friction", 1.0)),
                "s_start": -1.0,
                "s_end": -1.0,
            })

    def _publish_idx_only_params(self) -> None:
        rospy.set_param("/friction_map_params/n_sectors", len(self.sectors))
        rospy.set_param("/friction_map_params/global_friction_limit", self.global_limit)
        for i, sec in enumerate(self.sectors):
            k = f"/friction_map_params/Sector{i}"
            rospy.set_param(f"{k}/start", sec["start"])
            rospy.set_param(f"{k}/end", sec["end"])
            rospy.set_param(f"{k}/friction", sec["friction"])
            rospy.set_param(f"{k}/s_start", -1.0)
            rospy.set_param(f"{k}/s_end", -1.0)

    def _wpnts_cb(self, msg: WpntArray) -> None:
        if self._wpts_done or len(msg.wpnts) < 2:
            return
        N = len(msg.wpnts)
        s_arr = [w.s_m for w in msg.wpnts]
        xy = [(w.x_m, w.y_m) for w in msg.wpnts]
        for sec in self.sectors:
            a = max(0, min(sec["start"], N - 1))
            b = max(0, min(sec["end"], N - 1))
            sec["s_start"] = float(s_arr[a])
            sec["s_end"] = float(s_arr[b])
            k = f"/friction_map_params/Sector{self.sectors.index(sec)}"
            rospy.set_param(f"{k}/s_start", sec["s_start"])
            rospy.set_param(f"{k}/s_end", sec["s_end"])
        self._wpts_done = True
        self._loaded_pub.publish(Bool(data=True))
        self._publish_markers(xy)
        rospy.loginfo("[friction_loader] s_start/s_end set; /mu_ppc/sectors_loaded fired")

    def _publish_markers(self, xy) -> None:
        ma = MarkerArray()
        for i, sec in enumerate(self.sectors):
            a = sec["start"]
            if not (0 <= a < len(xy)):
                continue
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = "friction_loader_text"; m.id = i
            m.type = Marker.TEXT_VIEW_FACING; m.action = Marker.ADD
            m.pose.position.x = float(xy[a][0])
            m.pose.position.y = float(xy[a][1])
            m.pose.position.z = 0.7
            m.pose.orientation.w = 1.0
            m.scale.z = 0.45
            m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            m.text = f"S{i} [{a}-{sec['end']}] μ={sec['friction']:.2f}"
            ma.markers.append(m)
        if ma.markers:
            self._marker_pub.publish(ma)


if __name__ == "__main__":
    FrictionLoader()
    rospy.spin()
