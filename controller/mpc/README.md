# `controller/mpc` — Pacejka dynamic-bicycle MPC + friction-circle

> ICRA2026 RoboRacer baseline MPC. Frenet-frame dynamic bicycle + Pacejka
> lateral force + friction circle soft constraint. 모든 후속 패키지
> (`mpc_only`, `rls_mpc`, `ekf_mpc`, `gp_mpc`) 의 공통 기반.

---

## 1. 이론적 배경

### Model: Dynamic bicycle in Frenet frame (Liniger 2015 / Rajamani 류)

State (7):
```
x = [s, n, dpsi, vx, vy, omega, delta]
    │  │    │     │   │    │      └ steering angle (body)
    │  │    │     └───┴────┴──── body-frame velocities + yaw rate
    │  │    └── heading error w.r.t. Frenet tangent
    │  └── lateral offset from reference
    └── arc length along reference
```

Input (2): `u = [u_ddelta, u_ax]` (steer rate, longitudinal accel)

### Pacejka magic formula (lateral)
슬립각 `α_f = δ − atan((vy + l_f·ω)/vx_reg)`, `α_r = −atan((vy − l_r·ω)/vx_reg)`
(vx_reg = √(vx² + 1.5²) — low-speed singularity 회피)

```
F_y = μ · N · D · sin(C · atan(B·α − E·(B·α − atan(B·α))))
```

B, C, D, E 는 차량별 타이어 fitting 상수. μ 는 노면.

### Friction circle (soft nonlinear constraint)
각 axle 마다:
```
F_x² + F_y² ≤ (μ · margin · N)²
```
- Front (무구동): `F_x ≈ 0` → `|F_yf| ≤ μ·N_f`
- Rear (RWD): `(m·u_ax)² + F_yr² ≤ (μ·N_r)²`
- **Soft slack** (`friction_slack_penalty`) 으로 transient 초과 허용 → OCP infeasibility 방지

### acados 설정
- SQP-RTI, HPIPM QP solver
- `max_iter=1` (50Hz real-time budget)
- `levenberg_marquardt=0.01` (Pacejka low-speed singularity 주변 Hessian 안정화)
- `integrator_type=ERK`

### Cost (NONLINEAR_LS)
```
y = [n − n_ref, dpsi, vx − vx_ref, vy, omega, delta, u_ddelta, u_ax]
l = y^T W y
```
- 측방 오차 `n`, heading `dpsi`, 속도 tracking `vx−vx_ref` 를 추종
- `vy`, `omega`, `delta` 는 regularize (slip / yaw rate / steer 억제)
- Input rate (`u_ddelta`, `u_ax`) regularize

---

## 2. 기능 / Components

| 파일 | 역할 |
|---|---|
| `mpc_node.py` | `/behavior_strategy.local_wpnts` 추종 MPC (state_machine 기반 파이프라인) |
| `vehicle_model.py` | Pacejka dynamic bicycle CasADi + acados `AcadosModel` |
| `mpcc_ocp.py` | `build_tracking_ocp()` — OCP 빌드 + `solve_once()` |
| `reference_builder.py` | `build_preview()` — local_wpnts → (N+1, 6) param matrix |
| `scripts/spawn_on_waypoint.py` | sim 시작 시 wpnt[0] 로 snap + stuck supervisor |

후속 패키지는 `vehicle_model`, `mpcc_ocp`, `reference_builder` 를 **import 공유**.
`mpc_node.py` 는 state_machine 기반 원본. `mpc_only` 가 글로벌 raceline 버전.

---

## 3. 목표

- **ICRA2026 RoboRacer baseline** MPC
- Multi-friction / μ 적응 알고리즘 (rls, ekf, gp) 의 공통 모델 제공
- acados + HPIPM 50Hz 실시간 보장
- Friction circle 제약으로 tire 한계 명시적 인식

---

## 4. 파일 구조

```
controller/mpc/
├── mpc_node.py                   # state_machine 기반 MPC (원본)
├── vehicle_model.py              # Pacejka dynamic bicycle
├── mpcc_ocp.py                   # acados OCP 빌더
├── reference_builder.py          # preview param 생성
├── config/
│   └── mpc_srx1.yaml             # 튜닝
├── launch/
│   └── mpc_sim.launch            # 2D sim (state_machine 포함)
├── rviz/mpc_sim.rviz
└── scripts/
    └── spawn_on_waypoint.py      # spawn/supervisor
```

---

## 5. 파라미터 (`config/mpc_srx1.yaml`)

### MPC 공통
| 키 | 의미 | 기본값 |
|---|---|---|
| `N_horizon` | OCP horizon | 20 |
| `dt` | 제어 주기 | 0.05 (50Hz) |
| `v_max`/`v_min` | 속도 경계 | 12.0 / 0.0 |
| `max_steer`/`max_steer_rate` | 조향 한계 | 0.4 rad / 2.0 rad/s |
| `max_accel`/`max_decel` | 가속 한계 | 3.0 / -3.0 m/s² |
| `w_d` | 측방 오차 가중 | 8.0 |
| `w_dpsi`, `w_vx`, `w_vy`, `w_omega` | cost 가중 | 6 / 3 / 1 / 3 |
| `w_steer`, `w_u_steer_rate`, `w_u_accel` | input 가중 | 0.5 / 3 / 0.1 |
| `w_terminal_scale` | 터미널 cost scale | 1.0 |

### Friction circle
| 키 | 의미 | 기본값 |
|---|---|---|
| `friction_circle` | 제약 on/off | true |
| `friction_margin` | 사용 한도 (0.95 = 95%) | 0.95 |
| `friction_slack_penalty` | slack weight | 1000 |
| `mu_default` | 마찰 상수 | 0.85 |

### 차량 파라미터 (`/vehicle/*`, `/tire_front/*`, `/tire_rear/*`)
ROS param 으로 외부 yaml (`shared_vehicle/vehicle_*.yaml`) 에서 로드:
| 키 | 의미 |
|---|---|
| `m`, `l_f`, `l_r`, `l_wb`, `I_z` | 질량, wheel base |
| `Bf, Cf, Df, Ef` | 전륜 Pacejka |
| `Br, Cr, Dr, Er` | 후륜 Pacejka |

---

## 6. 실행 (2D sim)

```bash
CAR_NAME=SIM roslaunch controller mpc_sim.launch map:=f
```

**의존**: state_machine / planner_spline / behavior_strategy 체인 전체. 깨끗한
성능 비교 원하면 `controller/mpc_only` 사용 권장.

---

## 7. 실차 포팅 (→ `Brojoon-10/ICRA2026_HJ/main`)

이 패키지는 실차 레포에 **그대로 존재** 할 가능성 높음 (baseline). 수정 포팅 시:

### Step 1: 차량 파라미터 교체
- `shared_vehicle/vehicle_<car>.yaml` 의 `m`, `l_f`, `l_r`, `I_z` 실측치
- **Pacejka B, C, D, E** 는 bench-test 로 fitting 필요 (skid-pad 또는 step-steer)
- 잘못된 값은 MPC 예측 오차 → 코너 성능 저하

### Step 2: 의존성
- `acados` + `acados_template`: 없으면 `thirdparty/setup_acados.sh` 실행
- env: `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib`
- ROS1 Noetic + `f110_msgs`

### Step 3: 출력 토픽
- `/vesc/high_level/ackermann_cmd_mux/input/nav_1` 유지 (VESC mux 입력)

### Step 4: 실차 튜닝 요령
- **`friction_margin`**: 실차에선 0.80–0.85 추천 (sim 값 0.95 는 공격적)
- **`w_d`**: 트랙 coning 고려. 14–16 이 안전, 8 은 공격적
- **`max_accel`/`max_decel`**: 실차 VESC 한계 반영
- **`w_terminal_scale`**: horizon 끝 stability 중요하면 ↑

### Step 5: 안전
- `v_max` 를 실차 역량 기준으로 설정 (10 m/s 가 SRX1 upper)
- `friction_slack_penalty` 를 1e4 정도 높이면 slack 사용 억제 → tire 한계 안쪽에서만 동작
- `startup_delay_s` 로 센서 안정화 시간 확보

---

## 8. 한계 / 후속 개선

### 본 MPC 자체 한계
- μ 가 **상수** — 노면 변화 대응 불가
- State_machine 체인 의존 → 파이프라인 버그에 취약 (`mpc_only` 가 이 문제 해결)
- Pacejka 파라미터 sim vs 실차 mismatch 시 Tire force 오차

### 후속 패키지 (동일 MPC 공유)
- `controller/mpc_only` — state_machine 체인 우회, 글로벌 raceline 직접 추종
- `controller/rls_mpc` (→ multisurface_mpc 원래 이름) — peak-ay RLS μ 추정 (한계 확인용)
- `controller/ekf_mpc` — Pacejka-EKF scalar μ 추정 + lap-to-lap memory
- `controller/gp_mpc` — 3D residual GP 학습 (UPenn Nagy 2023 IFAC WC)
