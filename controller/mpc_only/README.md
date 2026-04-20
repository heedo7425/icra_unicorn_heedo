# `controller/mpc_only` — Minimal global-raceline MPC

> State machine / planner 체인 없이 **전역 raceline 을 직접 추종** 하는 경량 MPC.

---

## 1. 이론적 배경

### Problem
`controller/mpc` 는 `/behavior_strategy.local_wpnts` (state_machine 출력) 를 구독 →
그 위에 behavior_controller → 3d_state_machine → planner_spline → behavior_strategy
체인이 있음. 각 단계 버그가 MPC 성능을 가림 (예: `planner_spline` 의 v=2 하드코딩,
`spline_bound_mindist` 벽 접촉, smart_static_avoidance 의 offline 의존성).

빈 트랙 / raceline 추종 단계에서 이런 복잡도는 불필요 overhead.

### Approach: Direct global raceline tracking
- `/global_waypoints_scaled` (sector velocity-scaling 된 glob raceline) 직접 구독
- 차량 현재 s 로부터 nearest-waypoint index → horizon N stage slice (wrap-around 포함)
- OCP 는 Frenet contouring cost (ETH MPCC / Liniger 2015 류)
- State machine / planner 전부 **우회**

### MPC OCP (controller/mpc 와 공유)
- State (7): `[s, n, dpsi, vx, vy, omega, delta]` Frenet-frame dynamic bicycle
- Pacejka lateral force + friction circle (soft slack)
- acados SQP-RTI (max_iter=1, 50Hz real-time)

---

## 2. 기능 / Components

| 노드 | 역할 | 주기 |
|---|---|---|
| `mpc_only_node.py` | raceline slice → OCP params → solve → cmd publish | 50Hz |
| `controller/mpc/scripts/spawn_on_waypoint.py` (재사용) | 시작 위치 snap + stuck supervisor | event |

**의존**: `controller/mpc/vehicle_model.py`, `mpcc_ocp.py`, `reference_builder.py` 를 import

---

## 3. 목표

- 상위 파이프라인 버그로부터 **격리** → MPC 자체 성능만 측정
- ETH MPCC / Liniger 패턴과 일치 (글로벌 reference + OCP contouring)
- state machine 없이도 2D sim 에서 **안정 주행 + 랩타임** 유지
- 이후 `ekf_mpc` / `gp_mpc` 의 **baseline** 역할

---

## 4. 파일 구조

```
controller/mpc_only/
├── mpc_only_node.py
├── config/
│   └── mpc_only_srx1.yaml            # /mpc_only/* namespace 튜닝
├── launch/mpc_only_sim.launch        # minimal sim (state_machine/planner 제외)
└── README.md
```

---

## 5. 파라미터 (`config/mpc_only_srx1.yaml`)

### MPC 공통
| 키 | 의미 | 기본값 |
|---|---|---|
| `N_horizon`, `dt` | 20 × 0.05s | 1s horizon |
| `window_size` | 로컬 slice 크기 | 200 |
| `v_max`/`v_min`, `max_steer`, `max_accel`, `max_decel` | 입력/상태 한계 | |
| `w_d`, `w_dpsi`, `w_vx`, `w_vy`, `w_omega` | cost 가중 | |
| `w_steer`, `w_u_steer_rate`, `w_u_accel` | input 가중 | |
| `friction_circle`, `friction_margin`, `friction_slack_penalty` | friction circle | |
| `mu_default` | 마찰 상수 (모든 stage 동일) | 0.85 |

### Warmup / Startup
| 키 | 의미 |
|---|---|
| `warmup_vx_min`, `warmup_speed_cmd`, `startup_delay_s` | 출발 kick |
| `reset_jump_thres_m`, `stuck_status_thres` | 리셋/stuck 감지 |
| `codegen_dir` | `/tmp/mpc_only_c_generated` (충돌 방지) |

### Launch 인자 (`mpc_only_sim.launch`)
| arg | 기본 | 설명 |
|---|---|---|
| `map` | `f` | 맵 이름 |
| `rviz` | `true` | rviz 표시 |

---

## 6. 실행 (2D sim)

```bash
CAR_NAME=SIM roslaunch controller mpc_only_sim.launch map:=f
```

**런치 구성** — 명시적으로 **제외**되는 것: `state_machine`, `dyn_statemachine`,
`planner_spline`, `recovery_spliner`, `start_planner`, `planner_sqp`,
`waypoint_updater`, `gp_traj_predictor`. `base_system.launch sim:=true` 는
dyn_planner param server 만 올리고 실제 planner 노드는 안 기동 → 깨끗.

---

## 7. 실차 포팅 (→ `Brojoon-10/ICRA2026_HJ/main`)

### Step 1: 파일 복사
```bash
cp -r controller/mpc_only/ <main-repo>/controller/mpc_only/
# 의존: controller/mpc/vehicle_model.py, mpcc_ocp.py, reference_builder.py,
#       scripts/spawn_on_waypoint.py
```

### Step 2: 의존성
- `acados` + `acados_template`: 없으면 `thirdparty/setup_acados.sh` 실행
- env: `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib`
- ROS1 Noetic + `f110_msgs`
- 순수 numpy + acados → GPU·ML lib 불필요

### Step 3: 차량 파라미터
- `shared_vehicle/vehicle_<car>.yaml` 의 `m`, `l_f`, `l_r`, `I_z`, `Bf..Er` 실측치
- `mu_default` 를 현장 측정 μ 로

### Step 4: 실차 런치 수정
`launch/mpc_only_sim.launch` 에서:
- `base_system.launch sim:=true` → 실차 런치 (e.g. `headtohead.launch`)
- `/vesc/...nav_1` 출력 유지

### Step 5: 안전
- 첫 주행 `v_max=5` 정도 → raceline 추종 확인 → 점진 증속
- `friction_margin=0.85` 시작 → 점진 완화

---

## 8. 한계 / 후속

**자체 한계**:
- μ 상수 고정 → patch 많은 트랙에서 부정확
- Obstacle / 동적 장애물 고려 없음 (raceline 만)

**후속** (같은 MPC 모델 재사용):
- `controller/rls_mpc` — RLS 로 μ online 추정 (peak-ay, patch 구분 약함)
- `controller/ekf_mpc` — Pacejka-EKF μ 추정 (권장 경량)
- `controller/gp_mpc` — GP residual 학습 (UPenn Nagy 2023)
