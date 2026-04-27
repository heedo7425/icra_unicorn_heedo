# Phase 0 — `upenn_mpc` 인벤토리

소스: `controller/upenn_mpc/{upenn_mpc_node.py, mpcc_ocp_upenn.py, vehicle_model.py, config/upenn_mpc_srx1.yaml}`

## 1. ROS 토픽 (튜너 데몬이 알아야 할 것)

### 노드가 구독 (in)
| 토픽 | 타입 | 비고 |
|---|---|---|
| `/car_state/odom` | Odometry | vx, vy |
| `/car_state/pose` | PoseStamped | yaw, xy |
| `/car_state/odom_frenet` | Odometry | s, n |
| `/imu/data` | Imu | omega (yaw rate) |
| `/global_waypoints_scaled` | WpntArray | active raceline (x,y,vx,s,κ,ψ,ax) |
| `/global_waypoints` | WpntArray | raceline raw (fallback) |
| `/centerline_waypoints` | WpntArray | (centerline 모드만) |
| `/mu_ground_truth` | Float32 | μ-aware vx scaling |
| `/upenn_mpc/mu_estimate` | Float32 | runtime μ (rls/gp/patch 라우팅) |
| `/upenn_mpc/mu_adapt_enable` | Bool | μ adaptation 토글 |
| `/upenn_mpc/residual` | Float32MultiArray | GP 잔차 [Δvx,Δvy,Δω] |
| `/upenn_mpc/gp_ready` | Bool | |
| `/upenn_mpc/gp_reset` | Empty | |

### 노드가 발행 (out, **튜너가 활용**)
| 토픽 | 타입 | 튜너 용도 |
|---|---|---|
| `/vesc/.../nav_1` | AckermannDriveStamped | u_steer, u_speed (제어 거칠기) |
| `/upenn_mpc/prediction` | MarkerArray | (시각화, 튜너 무시) |
| `/upenn_mpc/reference` | MarkerArray | (시각화, 튜너 무시) |
| `/upenn_mpc/solve_ms` | Float32 | **solve_time 메트릭** |
| `/upenn_mpc/mu_used` | Float32 | μ 트래킹 |
| `/upenn_mpc/cmd_base_speed` | Float32 | base solver 비교 |
| `/upenn_mpc/cmd_base_steer` | Float32 | base solver 비교 |

### Lap 이벤트 (외부 노드)
- `lap_analyser` 가 `lap_data` (LapData{`lap_count`, `lap_time`, ...}) 발행
  - 위치: `f110_utils/nodes/lap_analyser/src/lap_analyser/lap_analyser.py:61`
  - 데몬은 이 토픽으로 lap 경계 + lap_time 직접 수신
  - `/lap_analyser/start` (Empty) 로 시작 트리거 가능

### 추가 필요 토픽 (튜너 데몬 ↔ 노드)
- `/upenn_mpc/reload_params` (svc, std_srvs/Trigger) — Phase 3 에서 노드에 신설
- `/upenn_mpc/solver_status` (선택, Float32MultiArray [status, qp_iter]) — solve_ms 만으로는 infeasible 카운트 어려움. 노드에 추가하거나 stderr/log 파싱

## 2. 파생 채널 (튜너가 자체 계산)

| 채널 | 계산식 | 용도 |
|---|---|---|
| `slip_f` | atan2(vy + lf*ω, vx) - δ | 전륜 슬립 |
| `slip_r` | atan2(vy - lr*ω, vx) | 후륜 슬립 |
| `ay` | vx·ω + vy_dot (또는 IMU.linear_accel.y) | 횡가속 |
| `ay_usage` | \|ay\| / (μ·g) | friction circle 사용률 |
| `n_signed_at_corner_exit` | n 샘플 (코너 출구 idx) | 언더/오버스티어 판별 |
| `dpsi_at_turn_in` | dpsi 샘플 (코너 진입 idx) | turn-in lag |
| `omega_oscillation` | omega FFT peak / mean | yaw 진동 |
| `u_steer_rate_rms` | rms(Δδ/dt) | 제어 거칠기 |

## 3. yaml 키 분류 (Hot / Warm / Cold / Static)

`controller/upenn_mpc/config/upenn_mpc_srx1.yaml` 기준.

### HOT — 노드 코드 변경 없이 acados 인스턴스 메서드만 호출
solver `cost_set(stage, "W", ...)` / `cost_set_e` 로 즉시 갱신. acados rebuild 불필요.

| 키 | 갱신 경로 |
|---|---|
| `w_d`, `w_dpsi`, `w_vx`, `w_vy`, `w_omega`, `w_steer`, `w_u_steer_rate`, `w_u_accel` | `solver.cost_set(k, "W", W_stage)` for k=0..N-1 |
| `w_terminal_scale` (W_N 전체 스케일) | `solver.cost_set(N, "W", W_terminal_scale * W_subset)` |
| `friction_slack_penalty` | `solver.cost_set(k, "Zl"/"Zu"/"zl"/"zu", ...)` |

### WARM — acados constraints_set 으로 갱신 (rebuild 불필요)
| 키 | 갱신 경로 |
|---|---|
| `v_max`, `v_min` | `solver.constraints_set(k, "ubx"/"lbx", ...)` for stage bounds |
| `max_steer` (state x[6]=δ bound) | `solver.constraints_set(k, "ubx"/"lbx", ...)` |
| `max_steer_rate`, `max_accel`, `max_decel` | `solver.constraints_set(k, "ubu"/"lbu", ...)` |
| `friction_margin` | **CAVEAT**: `mpcc_ocp_upenn.py:155-156` 에서 `(mu*margin*Nf)^2` 가 con_h_expr 안에 **상수로 baked in**. 현재 코드 그대로면 COLD. → 권장: 코드 수정해서 `margin` 을 stage param 으로 빼고 build_preview 에서 매 stage 주입 (NP 1개 추가). 그 후 HOT 으로 강등 |

### Runtime-only — solver 와 무관, 노드 attribute 만 갱신
| 키 | 적용 경로 |
|---|---|
| `mu_default` | `self.mu_default = ...` (다음 tick 부터 반영) |
| `speed_boost` | `self.speed_boost` |
| `mu_scale_exp` | `self.mu_scale_exp` |
| `mu_margin_k` | (코드 검색 필요, 현 노드 미사용으로 보임) |
| `mu_estimate_topic` | (resub 비용 → 정적 취급 권장) |
| `friction_circle` (bool) | con_h_expr 활성/비활성은 build 단계 결정 → **COLD** |

### COLD — acados 코드젠 재빌드 필요 (자동화 가능, 단 30~60s)
| 키 | 이유 |
|---|---|
| `N_horizon` | OCP 차원 변경 |
| `dt` | discretization 변경 |
| `friction_circle` (on/off) | con_h_expr 구조 변경 |
| `gp/residual_enable` | NP_GP 차원 변경 (현재 코드에서 NP_GP 는 build 시 결정) |
| Vehicle params (`m, lf, lr, Iz, B, C, D` 등 `vehicle_model.py`) | 동역학 RHS 변경 → "측정 후 고정" 그룹과 일치 |

### STATIC — 노드 시작 시 한 번만, 변경 시 노드 재시작 필요
| 키 | 비고 |
|---|---|
| `loop_rate_hz` | rospy.Rate 객체 재생성 필요 |
| `window_size` | 메모리 슬라이스 크기 |
| `codegen_dir` | 빌드 캐시 키 |
| `line_source` (raceline/centerline) | Subscriber 결정 |
| `mu_source` | Subscriber 결정 |
| `startup_delay_s`, `warmup_*`, `crash_stuck_sec`, `reset_jump_thres_m`, `stuck_status_thres` | 안전/세이프 로직 상수 |
| `test_mode` | 빌드 스킵 분기 |

## 4. Hot-reload 노드 측 작업 (Phase 3 에서 구현)

`upenn_mpc_node.py` 에 추가:

```python
from std_srvs.srv import Trigger, TriggerResponse

def _reload_cb(self, req):
    NS = "upenn_mpc"
    # 1. ROS param 다시 읽기 (HOT + WARM 만)
    new_cfg = { k: rospy.get_param(f"{NS}/{k}", v) for k,v in self.mpc_cfg.items() }
    # 2. W_stage / W_N 재구성 후 acados cost_set
    # 3. constraints_set 으로 bound 갱신
    # 4. self.mu_default / speed_boost / mu_scale_exp 갱신
    return TriggerResponse(success=True, message=f"reloaded {len(diffs)} params")

rospy.Service("/upenn_mpc/reload_params", Trigger, self._reload_cb)
```

CAVEAT: `friction_margin` 핫리로드 원하면 `mpcc_ocp_upenn.py` 의 `con_h_expr`
정의에서 margin 을 stage parameter 로 분리 (NP +1, build_preview 에 column 추가)
하는 작업이 선행되어야 함. **Phase 3 시작 시 결정 필요**:
- (A) margin 도 핫으로 — OCP 코드 손대고 재코드젠 1회 (이후 영구 핫)
- (B) margin 은 cold 로 두고 자주 안 만짐 — 코드 변경 0
→ 현재 권고: **(B)**. friction_margin 은 안전 마진이라 자주 튜닝 대상 아님.

## 5. 데몬이 트리거할 reload 흐름

```
1. yaml 패치 (in-place edit)
2. rosparam load <patched.yaml> /upenn_mpc
3. rosservice call /upenn_mpc/reload_params
4. 다음 lap 메트릭 수신 → 효과 평가
```

코드젠 재빌드가 필요한 COLD 키는 데몬이 자동 적용 **금지** (룰 yaml 의 action
대상에서 제외). COLD 변경은 사용자가 명시적으로 `tune_rebuild` 같은 별도
커맨드로만 트리거.
