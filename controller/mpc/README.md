# `controller/mpc` — Tracking MPC baseline (초기 스캐폴딩)

> ICRA2026 RoboRacer 용 MPC 의 **최초 tracking baseline**. acados 기반
> Frenet-frame dynamic bicycle + Pacejka lateral force. 이 시점에서는
> friction circle 제약은 아직 없고 weight 튜닝 + 속도 프로파일 passthrough 까지 작업됨.

---

## 1. 이론적 배경 (현재 단계)

### Model: Frenet-frame dynamic bicycle (Liniger 2015)

State (7):
```
x = [s, n, dpsi, vx, vy, omega, delta]
```

Input (2): `u = [u_ddelta, u_ax]`

### Pacejka lateral force (μ-dependent)
```
F_y = μ · N · D · sin(C · atan(B·α − E·(B·α − atan(B·α))))
```
- α = 슬립각 (전후륜 각각)
- N = 수직하중
- B, C, D, E = tire fitting 상수 (vehicle_srx1.yaml)

### acados OCP
- SQP-RTI, HPIPM QP
- `max_iter=1` (50Hz 실시간)
- Cost: `[n−n_ref, dpsi, vx−vx_ref, vy, omega, delta, u_ddelta, u_ax]` NONLINEAR_LS

### 이 단계의 한계
- **Friction circle 제약 없음** — tire saturation 고려 안 됨 (다음 브랜치에서 추가)
- μ 상수 고정
- sim Pacejka 값은 tuning 덜 됨

---

## 2. 기능 / Components

| 파일 | 역할 |
|---|---|
| `mpc_node.py` | `/behavior_strategy.local_wpnts` 추종 MPC 노드 |
| `vehicle_model.py` | Pacejka dynamic bicycle (CasADi) + acados `AcadosModel` |
| `mpcc_ocp.py` | `build_tracking_ocp()` + `solve_once()` |
| `reference_builder.py` | `build_preview()` — local_wpnts → OCP param matrix |
| `scripts/spawn_on_waypoint.py` | sim 시작 시 wpnt[0] 로 snap + stuck supervisor |

---

## 3. 목표 (이 단계)

- acados 기반 Pacejka MPC skeleton 구성
- `/behavior_strategy.local_wpnts` 입력으로 tracking 동작
- 속도 프로파일 passthrough (vx_ref 를 OCP cost 에 반영)
- sim 에서 lap 완주 가능한 weight 튜닝
- **이후 단계** (friction-circle → mu_only → rls → ekf → gp) 의 기반

---

## 4. 파일 구조

```
controller/mpc/
├── mpc_node.py
├── vehicle_model.py
├── mpcc_ocp.py
├── reference_builder.py
├── config/
│   └── mpc_srx1.yaml           # 초기 튜닝
├── launch/
│   └── mpc_sim.launch          # 2D sim (state_machine 포함)
├── rviz/mpc_sim.rviz
└── scripts/
    └── spawn_on_waypoint.py
```

---

## 5. 파라미터 (`config/mpc_srx1.yaml`)

### MPC 공통
| 키 | 의미 |
|---|---|
| `N_horizon` / `dt` | horizon 길이 / 제어 주기 |
| `v_max`, `v_min`, `max_steer`, `max_accel`, `max_decel` | 입력/상태 한계 |
| `w_d`, `w_dpsi`, `w_vx`, `w_vy`, `w_omega` | cost 가중 (측방/heading/속도/slip/yaw) |
| `w_steer`, `w_u_steer_rate`, `w_u_accel` | input 가중 |
| `w_terminal_scale` | 터미널 cost scale |
| `mu_default` | 마찰 상수 |

### 차량 파라미터 (외부 yaml)
`/vehicle/*`, `/tire_front/*`, `/tire_rear/*` → `shared_vehicle/vehicle_*.yaml`

---

## 6. 실행 (2D sim)

```bash
CAR_NAME=SIM roslaunch controller mpc_sim.launch map:=f
```

이 시점 sim 은 `base_system.launch sim:=true` + state_machine 체인을 포함.

---

## 7. 실차 포팅 (→ `Brojoon-10/ICRA2026_HJ/main`)

**이 단계** 는 baseline 이라 실차 직접 포팅 가능성 있음:

### Step 1: 의존성
- `acados` + `acados_template` (`thirdparty/setup_acados.sh` 참조)
- env: `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib`
- ROS1 Noetic + `f110_msgs`

### Step 2: 차량 파라미터
- `shared_vehicle/vehicle_<car>.yaml` 의 `m`, `l_f`, `l_r`, `I_z`, `Bf..Er` 실측치로 교체
- Pacejka 상수는 skid-pad / step-steer bench test 로 fit

### Step 3: 출력 토픽
- `/vesc/high_level/ackermann_cmd_mux/input/nav_1` 유지

### Step 4: 주의
- **Friction circle 없음** → 이 버전은 실차 이식 시 코너 과속 위험
- **다음 브랜치** (friction-circle) 까지 기다리거나, `friction_circle: true` 수동 추가 권장

---

## 8. 후속 개선 (이후 브랜치)

순차적으로:
1. **friction-circle + tire/weight tuning** (`dev/20260419-2151`)
2. **mpc_only + multisurface_mpc** (`dev/20260419-2318`) — state_machine 우회 + RLS μ
3. **ekf_mpc** (`dev/20260420-0011`) — Pacejka-EKF scalar μ
4. **gp_mpc** (`feat/20260420-2101-upenn-gp-mpc`) — UPenn 3D residual GP
