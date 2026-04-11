#!/usr/bin/env python3
"""
GP Manager Node — rqt에서 bag 녹화/학습/모델 업데이트 관리.

rqt dynamic_reconfigure:
    record: True/False → bag 녹화 시작/중지
    train:  True → bag에서 학습 실행 (자동 False 복귀)

학습 시 controller의 gp_steer_enabled를 자동으로 끔.
학습 완료 후 사용자가 직접 gp_steer_enabled를 켜야 함.
"""

import rospy
import subprocess
import os
import signal
import threading

from dynamic_reconfigure.server import Server
from controller.cfg import gp_managerConfig


class GPManager:
    def __init__(self):
        rospy.init_node('gp_manager', anonymous=False)

        # paths
        self.gp_root = os.path.join(os.path.dirname(__file__), '..')
        self.bag_path = os.path.join(self.gp_root, 'bag', 'gp_data.bag')
        self.csv_path = os.path.join(self.gp_root, 'data', 'gp_data.csv')
        self.model_path = os.path.join(self.gp_root, 'models', 'gp_model.pkl')
        self.extract_script = os.path.join(self.gp_root, 'scripts', 'gp_extract.py')
        self.train_script = os.path.join(self.gp_root, 'scripts', 'gp_train.py')

        # state
        self.recording = False
        self.training = False
        self.bag_proc = None

        # topics to record
        self.record_topics = [
            '/car_state/odom',
            '/car_state/pose',
            '/imu/data',
            '/vesc/high_level/ackermann_cmd_mux/input/nav_1',
            '/behavior_strategy',
            '/car_state/odom_frenet',
        ]

        # dynamic reconfigure server
        self.dyn_srv = Server(gp_managerConfig, self._dyn_cb)

        rospy.loginfo("[GPManager] Ready. Use rqt to record/train.")

    def _dyn_cb(self, config, level):
        # ── Record ──
        if config.record and not self.recording:
            self._start_record()
        elif not config.record and self.recording:
            self._stop_record()

        # ── Train ──
        if config.train:
            config.train = False  # auto-reset
            if self.training:
                rospy.logwarn("[GPManager] Training already in progress")
            elif self.recording:
                rospy.logwarn("[GPManager] Stop recording before training")
            else:
                # run training in separate thread to not block rqt
                t = threading.Thread(target=self._run_training)
                t.daemon = True
                t.start()

        return config

    def _start_record(self):
        """Start rosbag record."""
        os.makedirs(os.path.dirname(self.bag_path), exist_ok=True)

        # remove old bag if exists
        for f in [self.bag_path, self.bag_path + '.active']:
            if os.path.exists(f):
                os.remove(f)

        cmd = ['rosbag', 'record',
               '-O', self.bag_path,
               '--buffsize=256'] + self.record_topics

        self.bag_proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
        self.recording = True
        rospy.loginfo(f"[GPManager] Recording started → {self.bag_path}")

    def _stop_record(self):
        """Stop rosbag record."""
        if self.bag_proc is not None:
            # send SIGINT to process group for clean shutdown
            try:
                os.killpg(os.getpgid(self.bag_proc.pid), signal.SIGINT)
                self.bag_proc.wait(timeout=5)
            except Exception as e:
                rospy.logwarn(f"[GPManager] Bag stop issue: {e}")
                self.bag_proc.kill()
            self.bag_proc = None

        self.recording = False
        if os.path.exists(self.bag_path):
            size_mb = os.path.getsize(self.bag_path) / (1024 * 1024)
            rospy.loginfo(f"[GPManager] Recording stopped. Bag: {size_mb:.1f} MB")
        else:
            rospy.logwarn("[GPManager] Recording stopped but bag file not found")

    def _run_training(self):
        """Run extract → train pipeline."""
        self.training = True

        # 1. Check bag exists
        if not os.path.exists(self.bag_path):
            rospy.logerr(f"[GPManager] No bag file found at {self.bag_path}")
            rospy.logerr("[GPManager] Record a bag first, then train.")
            self.training = False
            return

        # 2. Disable GP in controller during training
        try:
            rospy.set_param('dyn_controller/gp_steer_enabled', False)
            rospy.loginfo("[GPManager] gp_steer_enabled → False (training)")
        except Exception:
            pass

        # 3. Extract CSV from bag
        rospy.loginfo("[GPManager] Step 1/2: Extracting data from bag...")
        try:
            result = subprocess.run(
                ['python3', self.extract_script,
                 '--bag', self.bag_path,
                 '--output', self.csv_path],
                capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                rospy.logerr(f"[GPManager] Extract failed:\n{result.stderr}")
                self.training = False
                return
            rospy.loginfo(f"[GPManager] Extract done: {result.stdout.strip()}")
        except Exception as e:
            rospy.logerr(f"[GPManager] Extract error: {e}")
            self.training = False
            return

        # 4. Train
        rospy.loginfo("[GPManager] Step 2/2: Training GP model...")
        try:
            result = subprocess.run(
                ['python3', self.train_script,
                 '--data', self.csv_path,
                 '--output', self.model_path],
                capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                rospy.logerr(f"[GPManager] Training failed:\n{result.stderr}")
                self.training = False
                return
            rospy.loginfo(f"[GPManager] Training done:\n{result.stdout}")
        except Exception as e:
            rospy.logerr(f"[GPManager] Training error: {e}")
            self.training = False
            return

        # 5. Done — controller will hot-reload the model
        rospy.loginfo("[GPManager] Model updated. Enable gp_steer_enabled in controller rqt.")
        self.training = False

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        manager = GPManager()
        manager.run()
    except rospy.ROSInterruptException:
        pass
