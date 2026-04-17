#!/usr/bin/env python3
### HJ : FBGA-based 3D velocity planner ROS node
### global_waypoints 토픽에서 waypoint를 받아 FBGA로 속도 재계산 후 publish

import rospy
import os
import subprocess
import tempfile
import numpy as np
import json
import yaml
import struct

## IY : hot-reload support (threading lock + Trigger service)
import threading
from std_srvs.srv import Trigger, TriggerResponse
## IY : end

from f110_msgs.msg import WpntArray, Wpnt
import trajectory_planning_helpers as tph


## IY(0416) : per-sector friction support for FBGA
#   FBGA C++ binary only accepts a single gg.bin, so multi-GGV is handled
#   by running FBGA once per unique friction and selecting per-waypoint results.
def _read_friction_sectors_from_yaml(maps_dir, map_name):
    """Read friction sectors from friction_scaling.yaml. Returns [] if unavailable."""
    yaml_path = os.path.join(maps_dir, map_name, 'friction_scaling.yaml')
    if not os.path.exists(yaml_path):
        return []
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        sectors = []
        for i in range(data.get('n_sectors', 0)):
            sec = data.get(f'Sector{i}', {})
            fric = sec.get('friction', -1.0)
            if fric > 0:
                sectors.append({'start': sec.get('start', 0),
                                'end': sec.get('end', 0),
                                'friction': float(fric)})
        return sectors
    except Exception:
        return []


def _build_wpnt_sector_map(sectors, n_waypoints, base_p_Dx_1):
    """Build per-waypoint friction value array.
    Returns friction_per_wpnt (np.array) or None if single-GGV is sufficient.
    """
    if not sectors:
        return None
    unique = set(s['friction'] for s in sectors)
    if len(unique) == 1 and abs(list(unique)[0] - base_p_Dx_1) < 1e-4:
        return None
    fric_arr = np.full(n_waypoints, base_p_Dx_1)
    for sec in sectors:
        s, e = sec['start'], min(sec['end'] + 1, n_waypoints)
        fric_arr[s:e] = sec['friction']
    return fric_arr
## IY(0416) : end


class FBGAVelocityPlanner:

    def __init__(self):
        rospy.loginfo("[FBGA] Initializing...")

        # === 경로 설정 (컨테이너 내부 절대경로) ===
        race_stack = '/home/unicorn/catkin_ws/src/race_stack'
        self.fbga_bin = rospy.get_param(
            "~fbga_bin",
            os.path.join(race_stack, 'f110_utils', 'libs', 'FBGA', 'bin', 'GIGI_test_unicorn.exe'))

        ## IY : default paths → *_latest (auto-updated by gg_tuner_node)
        #       gg_tuner creates v<N> files and copies/symlinks to *_latest,
        #       so these defaults always resolve to the most recent GGV + params.
        # (previous: hardcoded to rc_car_10th_fast4 — stale after gg_tuner runs)
        # gg_bin_default = os.path.join(
        #     race_stack, 'planner', '3d_gb_optimizer', 'global_line', 'data',
        #     'gg_diagrams', 'rc_car_10th_fast4', 'velocity_frame', 'gg.bin')
        # params_yml_default = os.path.join(
        #     race_stack, 'planner', '3d_gb_optimizer', 'global_line', 'data',
        #     'vehicle_params', 'params_rc_car_10th_fast4.yml')
        gg_bin_default = os.path.join(
            race_stack, 'planner', '3d_gb_optimizer', 'global_line', 'data',
            'gg_diagrams', 'rc_car_10th_latest', 'velocity_frame', 'gg.bin')
        self.gg_bin = rospy.get_param("~gg_bin", gg_bin_default)

        params_yml_default = os.path.join(
            race_stack, 'planner', '3d_gb_optimizer', 'global_line', 'data',
            'vehicle_params', 'params_rc_car_10th_latest.yml')
        params_yml = rospy.get_param("~params_yml", params_yml_default)
        ## IY : end

        # 경로 검증 (gg.bin 없으면 npy에서 자동 생성)
        for name, path in [('fbga_bin', self.fbga_bin), ('params_yml', params_yml)]:
            if not os.path.exists(path):
                rospy.logerr(f"[FBGA] File not found: {name}={path}")
                raise FileNotFoundError(path)

        if not os.path.exists(self.gg_bin):
            self._generate_gg_bin(self.gg_bin)

        # === params.txt 생성 (tmp) ===
        self.params_txt = os.path.join(tempfile.gettempdir(), 'fbga_params.txt')
        self._convert_params_yml(params_yml)

        # === FBGA 설정 ===
        self.n_laps = rospy.get_param("~n_laps", 3)
        self.max_iter = rospy.get_param("~max_iter", 50)
        self.tol = rospy.get_param("~tol", 0.05)        # m/s
        self.alpha = rospy.get_param("~alpha", 1.0)      # under-relaxation
        self.v0 = rospy.get_param("~v0", 1.0)
        ### HJ : mu 보정 on/off (True: ax_tilde + Vmax mu 보정, False: g_tilde만)
        self.enable_mu = rospy.get_param("~enable_mu", True)

        # GGV g_list 범위 (clamp용)
        self.g_min, self.g_max = self._read_g_range()

        # === Pub/Sub ===
        self.processed = False  ### HJ : 한번만 처리 (자기 출력 재수신 방지)
        self.pub = rospy.Publisher('/global_waypoints', WpntArray, queue_size=10)
        rospy.Subscriber('/global_waypoints', WpntArray, self.wpnts_callback)

        ## IY : hot-reload — cache last input wpnts + mutex + /fbga/reload service
        #       gg_tuner 가 GGV 재생성 후 프로세스 재시작 없이
        #       갈아끼우기 위해 사용 (Python boot ~3-4s 절감).
        self.last_wpnts_msg = None
        self.process_lock = threading.Lock()
        self.reload_srv = rospy.Service('/fbga/reload', Trigger, self.reload_cb)
        ## IY : end

        ## IY(0416) : per-sector friction — load friction sectors + build gg.bin map
        self.race_stack = race_stack
        self.gg_base_dir = os.path.dirname(os.path.dirname(self.gg_bin))  # e.g. gg_diagrams/rc_car_10th_latest
        self.friction_gg_bins = {}  # friction_value → gg.bin path
        self._load_friction_sectors()
        ## IY(0416) : end

        rospy.loginfo(f"[FBGA] Ready. bin={self.fbga_bin}")
        rospy.loginfo(f"[FBGA] gg.bin={self.gg_bin}")
        rospy.loginfo(f"[FBGA] n_laps={self.n_laps}, max_iter={self.max_iter}, tol={self.tol}")

    def _generate_gg_bin(self, bin_path):
        """npy 파일에서 gg.bin 자동 생성"""
        npy_dir = os.path.dirname(bin_path)
        rospy.loginfo(f"[FBGA] gg.bin not found, generating from {npy_dir}")

        v_list = np.load(os.path.join(npy_dir, 'v_list.npy')).astype(np.float64)
        g_list = np.load(os.path.join(npy_dir, 'g_list.npy')).astype(np.float64)
        ax_max = np.load(os.path.join(npy_dir, 'ax_max.npy')).astype(np.float64)
        ax_min = np.load(os.path.join(npy_dir, 'ax_min.npy')).astype(np.float64)
        ay_max = np.load(os.path.join(npy_dir, 'ay_max.npy')).astype(np.float64)
        gg_exp = np.load(os.path.join(npy_dir, 'gg_exponent.npy')).astype(np.float64)

        nv, ng = len(v_list), len(g_list)
        with open(bin_path, 'wb') as f:
            f.write(struct.pack('II', nv, ng))
            for arr in [v_list, g_list, ax_max, ax_min, ay_max, gg_exp]:
                arr.tofile(f)

        rospy.loginfo(f"[FBGA] gg.bin generated: nv={nv}, ng={ng}, size={os.path.getsize(bin_path)} bytes")

    def _convert_params_yml(self, yml_path):
        """params YAML → params.txt (FBGA C++ runner 입력)"""
        with open(yml_path) as f:
            cfg = yaml.safe_load(f)
        vp = cfg['vehicle_params']
        tp = cfg['tire_params']
        with open(self.params_txt, 'w') as f:
            f.write(f"m={vp['m']}\n")
            f.write(f"P_max={vp['P_max']}\n")
            f.write(f"mu_x={tp['p_Dx_1']}\n")
            f.write(f"mu_y={tp['p_Dy_1']}\n")
            f.write(f"v_max={vp['v_max']}\n")
        rospy.loginfo(f"[FBGA] params.txt saved: m={vp['m']}, P_max={vp['P_max']}, v_max={vp['v_max']}")

    ## IY(0416) : load friction sectors and map gg.bin paths per friction
    def _load_friction_sectors(self):
        """Read friction sectors and locate per-friction gg.bin files."""
        map_name = rospy.get_param('/map', '')
        maps_dir = os.path.join(self.race_stack, 'stack_master', 'maps')
        sectors = _read_friction_sectors_from_yaml(maps_dir, map_name)

        # read base p_Dx_1 from params
        try:
            with open(rospy.get_param('~params_yml',
                      os.path.join(self.race_stack, 'planner', '3d_gb_optimizer',
                                   'global_line', 'data', 'vehicle_params',
                                   'params_rc_car_10th_latest.yml'))) as f:
                self.base_p_Dx_1 = yaml.safe_load(f).get(
                    'tire_params', {}).get('p_Dx_1', 0.56)
        except Exception:
            self.base_p_Dx_1 = 0.56

        self.friction_per_wpnt = None
        self.friction_gg_bins = {}

        if not sectors:
            rospy.loginfo("[FBGA] No friction sectors → single GGV mode")
            return

        # build per-waypoint friction (will be resized when waypoints arrive)
        self._friction_sectors_raw = sectors

        # locate gg.bin per unique friction
        gg_parent = os.path.dirname(self.gg_base_dir)  # gg_diagrams/
        base_name = os.path.basename(self.gg_base_dir)  # rc_car_10th_latest
        unique_frics = sorted(set(s['friction'] for s in sectors))
        all_found = True
        for fric in unique_frics:
            if abs(fric - self.base_p_Dx_1) < 1e-4:
                self.friction_gg_bins[fric] = self.gg_bin
                continue
            fric_int = int(round(fric * 100))
            fric_bin = os.path.join(gg_parent,
                                    f'{base_name}_f{fric_int:03d}',
                                    'velocity_frame', 'gg.bin')
            if not os.path.exists(fric_bin):
                rospy.logwarn(f"[FBGA] friction gg.bin not found: {fric_bin}")
                all_found = False
                break
            self.friction_gg_bins[fric] = fric_bin

        if not all_found or len(self.friction_gg_bins) <= 1:
            self.friction_gg_bins = {}
            rospy.loginfo("[FBGA] Friction GGVs incomplete → single GGV mode")
        else:
            rospy.loginfo(f"[FBGA] Multi-GGV friction: {self.friction_gg_bins}")
    ## IY(0416) : end

    def _read_g_range(self):
        """gg.bin에서 g_list 범위 읽기"""
        with open(self.gg_bin, 'rb') as f:
            nv, ng = struct.unpack('II', f.read(8))
            v_list = np.frombuffer(f.read(nv * 8), dtype=np.float64)
            g_list = np.frombuffer(f.read(ng * 8), dtype=np.float64)
        rospy.loginfo(f"[FBGA] GGV range: v=[{v_list.min():.1f},{v_list.max():.1f}], "
                      f"g=[{g_list.min():.2f},{g_list.max():.2f}]")
        return float(g_list.min()), float(g_list.max())

    def _compute_g_tilde(self, mu, v, dmu_ds):
        """g_tilde = 9.81*cos(mu) - v^2 * dmu/ds, clamped to GGV range"""
        gt = 9.81 * np.cos(mu) - v**2 * dmu_ds
        return np.clip(gt, self.g_min, self.g_max)

    def _initial_speed_estimate(self, kappa, mu, dmu_ds):
        """XY곡률 + 수직곡률 결합 초기 속도 추정"""
        ay_max = 4.5  # TODO: GGV에서 읽기
        v_max = 12.0

        # XY 곡률 한계
        radius = np.where(np.abs(kappa) > 1e-4, 1.0 / np.abs(kappa), 1e4)
        v_lat = np.clip(np.sqrt(ay_max * radius), 0, v_max)

        # 수직 곡률 한계 (crest에서 g_tilde > 0 조건)
        v_vert = np.full_like(kappa, v_max)
        crest = dmu_ds > 1e-4
        v_vert[crest] = np.clip(
            np.sqrt(9.81 * np.cos(mu[crest]) / dmu_ds[crest]), 0.5, v_max)

        return np.minimum(v_lat, v_vert)

    def _stack_laps(self, s, kappa, g_tilde, mu, dmu_ds):
        """N-laps 이어붙이기 (closed loop 보완)"""
        n_pts = len(s)
        ### HJ : 첫 두 점 간격 사용
        ds = s[1] - s[0] if n_pts > 1 else 0.1
        lap_length = s[-1] - s[0] + ds

        s_stack = np.concatenate([s + i * lap_length for i in range(self.n_laps)])
        k_stack = np.tile(kappa, self.n_laps)
        g_stack = np.tile(g_tilde, self.n_laps)
        mu_stack = np.tile(mu, self.n_laps)
        dmu_stack = np.tile(dmu_ds, self.n_laps)  ### HJ : dmu_ds도 같이 stack
        return s_stack, k_stack, g_stack, mu_stack, dmu_stack, lap_length, n_pts

    # --- (original _run_fbga signature, kept for reference) ---
    # def _run_fbga(self, s, kappa, g_tilde, mu, dmu_ds, v0):
    # --- (end) ---
    ## IY(0416) : add gg_bin_override for per-friction FBGA runs
    def _run_fbga(self, s, kappa, g_tilde, mu, dmu_ds, v0, gg_bin_override=None):
        """Run FBGA C++ binary. gg_bin_override selects friction-specific GGV."""
        input_csv = os.path.join(tempfile.gettempdir(), 'fbga_input.csv')
        output_csv = os.path.join(tempfile.gettempdir(), 'fbga_output.csv')

        ### HJ : enable_mu flag로 mu 보정 여부 제어
        with open(input_csv, 'w') as f:
            if self.enable_mu:
                f.write('s,kappa,g_tilde,mu,dmu_ds\n')
                for i in range(len(s)):
                    f.write(f'{s[i]:.6f},{kappa[i]:.8f},{g_tilde[i]:.6f},{mu[i]:.8f},{dmu_ds[i]:.8f}\n')
            else:
                f.write('s,kappa,g_tilde\n')
                for i in range(len(s)):
                    f.write(f'{s[i]:.6f},{kappa[i]:.8f},{g_tilde[i]:.6f}\n')

        ## IY(0416) : use override gg.bin if provided (multi-friction)
        # --- (original cmd, kept for reference) ---
        # cmd = [self.fbga_bin, '--model', 'lookup', '--input', input_csv,
        #        '--params', self.params_txt, '--gg', self.gg_bin,
        #        '--output', output_csv, '--v0', f'{v0:.4f}']
        # --- (end) ---
        gg_file = gg_bin_override if gg_bin_override else self.gg_bin
        cmd = [
            self.fbga_bin,
            '--model', 'lookup',
            '--input', input_csv,
            '--params', self.params_txt,
            '--gg', gg_file,
            '--output', output_csv,
            '--v0', f'{v0:.4f}',
        ]
        ## IY(0416) : end

        try:
            rospy.loginfo(f"[FBGA] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
            rospy.loginfo(f"[FBGA] exe stdout: {result.stdout[-200:]}")
        except subprocess.CalledProcessError as e:
            rospy.logerr(f"[FBGA] exe failed (rc={e.returncode}): {e.stderr[:500]}")
            return None
        except subprocess.TimeoutExpired:
            rospy.logerr("[FBGA] exe timeout")
            return None
        except Exception as e:
            rospy.logerr(f"[FBGA] exe error: {e}")
            return None

        # 결과 읽기
        v_out = []
        ax_out = []
        with open(output_csv) as f:
            for line in f:
                if line.startswith('#') or line.startswith('s,'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    v_out.append(float(parts[1]))
                    ax_out.append(float(parts[2]))

        ### HJ : tmp 파일 정리 (디버깅용 비활성화)
        try:
            os.remove(input_csv)
            os.remove(output_csv)
        except OSError:
            pass

        return np.array(v_out), np.array(ax_out)

    def _extract_middle_lap(self, v_full, ax_full, n_pts_per_lap):
        """N-laps 결과에서 중간 바퀴 추출"""
        middle = self.n_laps // 2
        start = middle * n_pts_per_lap
        end = start + n_pts_per_lap
        return v_full[start:end], ax_full[start:end]

    ## IY : split callback → cache input + delegate to _process_and_publish.
    #       self.processed 는 self-echo 방지용으로 유지.
    #       reload_cb 에서 캐시된 입력으로 재계산할 수 있도록 last_wpnts_msg 저장.
    # --- (기존 wpnts_callback 원본, 보존용 주석) ---
    # def wpnts_callback(self, msg):
    #     if self.processed:
    #         return
    #     self.processed = True
    #
    #     wpnts = msg.wpnts
    #     n = len(wpnts)
    # --- (원본 끝) ---
    def wpnts_callback(self, msg):
        with self.process_lock:
            if self.processed:
                return
            self.last_wpnts_msg = msg
            self.processed = True
            self._process_and_publish(msg)

    def reload_cb(self, req):
        """Hot-reload: re-read gg.bin + params.yml + enable_mu, reprocess cached wpnts.

        gg_tuner_node 가 새 GGV 생성 후 호출. Python import/boot 없이
        0.5초 이내에 새 파라미터로 /global_waypoints 갱신.
        """
        try:
            new_gg = rospy.get_param('~gg_bin', self.gg_bin)
            new_params = rospy.get_param('~params_yml', None)
            new_enable_mu = rospy.get_param('~enable_mu', self.enable_mu)

            with self.process_lock:
                self.gg_bin = new_gg
                self.enable_mu = new_enable_mu
                self.g_min, self.g_max = self._read_g_range()
                if new_params and os.path.exists(new_params):
                    self._convert_params_yml(new_params)

                ## IY(0416) : reload friction sectors on hot-reload
                self.gg_base_dir = os.path.dirname(os.path.dirname(new_gg))
                self._load_friction_sectors()
                ## IY(0416) : end

                rospy.loginfo(
                    f"[FBGA] Reloaded: gg={new_gg}, enable_mu={new_enable_mu}")

                if self.last_wpnts_msg is not None:
                    self._process_and_publish(self.last_wpnts_msg)
                    return TriggerResponse(
                        success=True, message="reloaded and reprocessed")
                else:
                    return TriggerResponse(
                        success=True, message="reloaded (no cached waypoints)")
        except Exception as e:
            rospy.logerr(f"[FBGA] reload failed: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return TriggerResponse(success=False, message=str(e)[:200])

    def _process_and_publish(self, msg):
        """FBGA 반복 계산 + /global_waypoints publish. 기존 wpnts_callback 본문."""
        wpnts = msg.wpnts
        n = len(wpnts)
    ## IY : end

        # waypoint 데이터 추출
        s = np.array([wp.s_m for wp in wpnts])
        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        mu = np.array([wp.mu_rad for wp in wpnts])
        v_existing = np.array([wp.vx_mps for wp in wpnts])

        ### HJ : periodic central difference (run_fwbw.py와 동일)
        ### HJ : ds를 첫 두 점 간격으로 계산
        ds_grid = s[1] - s[0] if n > 1 else 0.1
        mu_wrap = np.concatenate([[mu[-1]], mu, [mu[0]]])
        dmu_ds = (mu_wrap[2:] - mu_wrap[:-2]) / (2.0 * ds_grid)

        # 초기 속도: waypoint에 이미 있으면 사용, 없으면 추정
        if np.any(v_existing > 0.1):
            v_prev = v_existing.copy()
            rospy.loginfo("[FBGA] Using existing waypoint speeds as initial estimate")
        else:
            v_prev = self._initial_speed_estimate(kappa, mu, dmu_ds)
            rospy.loginfo("[FBGA] Using curvature+slope initial speed estimate")

        ## IY(0416) : build per-waypoint friction map (resize to actual wpnt count)
        friction_per_wpnt = None
        if self.friction_gg_bins and hasattr(self, '_friction_sectors_raw'):
            friction_per_wpnt = _build_wpnt_sector_map(
                self._friction_sectors_raw, n, self.base_p_Dx_1)
        ## IY(0416) : end

        # === Fixed-point iteration ===
        for it in range(self.max_iter):
            # g_tilde 계산
            g_tilde = self._compute_g_tilde(mu, v_prev, dmu_ds)

            # N-laps stack (mu, dmu_ds도 같이)
            s_stack, k_stack, g_stack, mu_stack, dmu_stack, lap_length, n_pts = self._stack_laps(s, kappa, g_tilde, mu, dmu_ds)

            v0 = max(float(v_prev[0]), 1.0)

            # --- (original single-FBGA run, kept for reference) ---
            # result = self._run_fbga(s_stack, k_stack, g_stack, mu_stack, dmu_stack, v0)
            # if result is None:
            #     rospy.logwarn("[FBGA] Failed, keeping existing speeds")
            #     return
            # v_full, ax_full = result
            # v_new, ax_new = self._extract_middle_lap(v_full, ax_full, n_pts)
            # --- (end) ---

            ## IY(0416) : multi-friction FBGA
            #   Run FBGA once per unique friction (full track each time).
            #   Each friction run's BW pass naturally creates braking zones
            #   before low-grip sections. Per-waypoint, pick the result from
            #   the run matching that waypoint's sector friction.
            if friction_per_wpnt is not None and len(self.friction_gg_bins) > 1:
                # run FBGA per unique friction
                fric_results = {}  # friction → (v_1lap, ax_1lap)
                for fric, gg_bin in self.friction_gg_bins.items():
                    res = self._run_fbga(s_stack, k_stack, g_stack,
                                         mu_stack, dmu_stack, v0,
                                         gg_bin_override=gg_bin)
                    if res is None:
                        rospy.logwarn(f"[FBGA] friction={fric:.3f} run failed")
                        continue
                    v_f, ax_f = res
                    v_1, ax_1 = self._extract_middle_lap(v_f, ax_f, n_pts)
                    fric_results[fric] = (v_1, ax_1)

                if not fric_results:
                    rospy.logwarn("[FBGA] All friction runs failed")
                    return

                # per-waypoint: select from matching friction run
                v_new = np.zeros(n)
                ax_new = np.zeros(n)
                for i in range(n):
                    fric_i = friction_per_wpnt[i]
                    # find closest available friction
                    best_fric = min(fric_results.keys(),
                                    key=lambda f: abs(f - fric_i))
                    v_new[i] = fric_results[best_fric][0][i]
                    ax_new[i] = fric_results[best_fric][1][i]
            else:
                # single GGV (original path)
                result = self._run_fbga(s_stack, k_stack, g_stack,
                                        mu_stack, dmu_stack, v0)
                if result is None:
                    rospy.logwarn("[FBGA] Failed, keeping existing speeds")
                    return
                v_full, ax_full = result
                v_new, ax_new = self._extract_middle_lap(v_full, ax_full, n_pts)
            ## IY(0416) : end

            # NaN 처리
            nan_mask = np.isnan(v_new)
            if nan_mask.any():
                n_nan = nan_mask.sum()
                if n_nan / n > 0.05:
                    rospy.logwarn(f"[FBGA] Too many NaNs: {n_nan}/{n}")
                    return
                valid = np.where(~nan_mask)[0]
                v_new[nan_mask] = np.interp(np.where(nan_mask)[0], valid, v_new[valid])

            # 수렴 체크
            delta = float(np.max(np.abs(v_new - v_prev)))
            rospy.loginfo(f"[FBGA] iter {it}: max|dv|={delta:.4f} m/s, "
                          f"g_tilde=[{g_tilde.min():.2f},{g_tilde.max():.2f}]")

            if delta < self.tol:
                rospy.loginfo(f"[FBGA] Converged at iter {it}")
                break

            v_prev = self.alpha * v_new + (1.0 - self.alpha) * v_prev

        # === waypoint update ===
        ax_nan_mask = np.isnan(ax_new)
        if ax_nan_mask.any():
            valid_ax = np.where(~ax_nan_mask)[0]
            ax_new[ax_nan_mask] = np.interp(np.where(ax_nan_mask)[0], valid_ax, ax_new[valid_ax])

        for i in range(n):
            wpnts[i].vx_mps = float(v_new[i])
            wpnts[i].ax_mps2 = float(ax_new[i])

        msg.wpnts = wpnts
        rospy.loginfo(f"[FBGA] Publishing: v=[{v_new.min():.2f},{v_new.max():.2f}] m/s")
        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node("fbga_velocity_planner")
    node = FBGAVelocityPlanner()
    rospy.spin()
