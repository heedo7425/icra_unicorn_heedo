#! /usr/bin/env python3

import os

import rospy
import yaml
from ackermann_msgs.msg import AckermannDriveStamped


class MPCController:
    """wuee_mpcc output relay.

    솔버 (`mpcc_controller.py`) 가 `/mpc_controller/next_input` 으로 매 solve 마다 1 개의
    stage-1 명령을 publish → 이 노드가 `MPC_freq` (40Hz) 로 VESC 에 재발행.

    solver stall / deadline miss 로 next_input 이 비었을 때는 **마지막으로 발행한 cmd 를
    재사용**. 과거에는 `/mpc_controller/input_stream` (horizon stage 2~5, 즉 67~167ms 후의
    미래 state) 을 fallback 으로 pop 해서 VESC 에 쏘던 코드가 있었는데, 이는 "현재 cmd"
    가 아닌 "미래 horizon step" 을 그대로 실어 보내는 심각한 버그였음 (steer/speed 가
    실제 필요 값보다 앞서감). 해당 경로 제거.
    """

    def __init__(self, conf_file) -> None:
        rospy.loginfo(f"[wuee_mpcc output] loading config: {conf_file}")
        with open(conf_file, "r") as file:
            cfg = yaml.safe_load(file)
            for key in cfg.keys():
                if type(cfg[key]) is list:
                    cfg[key] = [float(i) for i in cfg[key]]

        self.Hz = cfg["MPC_freq"]

        self.next_input = []
        self.last_cmd = None  # solver stall 시 재사용할 마지막 명령

        rospy.Subscriber(
            "/mpc_controller/next_input", AckermannDriveStamped, self._next_input_cb
        )
        self.drive_pub = rospy.Publisher(
            "/vesc/high_level/ackermann_cmd_mux/input/nav_1", AckermannDriveStamped, queue_size=10
        )

    def _next_input_cb(self, data: AckermannDriveStamped) -> None:
        # 최신 solve 결과 1 개만 유지. 40Hz loop 가 32Hz solve 보다 빨라 자주 드레인됨.
        self.next_input = [data]

    def send_ackermann_cmd(self) -> None:
        if self.next_input:
            ack_msg = self.next_input.pop()
            self.last_cmd = ack_msg
        else:
            rospy.logwarn_throttle(1.0, "No new solution this tick — re-publishing last cmd")
            ack_msg = self.last_cmd

        rospy.logdebug(
            f"Publish ackermann msg: {ack_msg.drive.speed}, {ack_msg.drive.steering_angle}"
        )
        self.drive_pub.publish(ack_msg)

    def loop(self) -> None:
        rate = rospy.Rate(self.Hz)
        while not rospy.is_shutdown():
            if self.last_cmd is None and not self.next_input:
                # 솔버가 아직 한 번도 publish 하지 않음 → cmd 자체가 없어 skip.
                rate.sleep()
                continue
            self.send_ackermann_cmd()
            rate.sleep()


if __name__ == "__main__":
    # init_node 를 먼저 호출해야 `~config_file` private param 이 정상 resolve 됨.
    rospy.init_node("controller_node", anonymous=True, log_level=rospy.DEBUG)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.join(dir_path, "mpc", "param_config.yaml")
    conf_file = rospy.get_param("~config_file", default_cfg)
    controller = MPCController(conf_file=conf_file)
    controller.loop()
