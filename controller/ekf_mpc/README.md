# `controller/ekf_mpc` — Pacejka-EKF scalar-μ MPC

> 주행 중 노면 마찰 μ 를 **scalar EKF** 로 온라인 추정해서 MPC OCP 에 주입.

---

## 1. 이론적 배경

### Problem
마찰계수 μ 는 노면 패치별로 다르며 (race track 의 먼지/물기/노면 변화),
Pacejka base 모델의 μ 가 틀리면 friction circle 제약이 부정확 → 코너 성능 저하
또는 과도 보수화.

### Approach: Pacejka-based EKF
- **상태변수**: μ (scalar)
- **관측식**: `a_y = μ · K(v_x, v_y, ω, δ) + noise`
  - K 는 Pacejka μ-free 식 `D·sin(C·atan(B·α − E·(...)))` 에 정상하중 (`N_f`, `N_r`) 과
    코사인 보정 (`cos(δ)`) 과 질량 역수 (`1/m`) 를 포함한 선형 계수
- **업데이트 조건**: `v_x > min_speed AND |K| > min_K` — 직선 (α≈0) 에선 K 작아 skip

### Level 2 확장 (L2.3 튜닝)
1. **Longitudinal channel** — `|u_ax| > long_ax_thresh AND |u_ax| − |a_x_meas| > long_slip_thresh` 이면
   `z_long = |a_x_meas|` 로 두 번째 관측. 급가속 상황에서 μ 정보 보강.
2. **s-memory prior** — Frenet s 를 `mem_bin_width` (2m) 로 나눈 bin 별 μ̂ 를 EMA 로 저장.
   업데이트 없는 직선 구간에서 이전 lap 의 해당 bin μ̂ 로 soft pull → lap-to-lap 복원.

### 한계 (close-loop bias)
MPC 가 μ̂ 를 써서 보수 주행 → ay 작아짐 → EKF 가 μ 더 낮게 추정 → self-fulfilling.
(`controller/gp_mpc` 가 이 한계를 잔차 학습으로 극복하는 후속 패키지)

---

## 2. 기능 / Components

| 노드 | 역할 | 주기 |
|---|---|---|
| `ekf_mpc_node.py` | acados OCP 에 runtime μ 주입. `/ekf_mpc/mu_estimate` 구독 | 50Hz |
| `scripts/mu_estimator_ekf.py` | Pacejka-EKF + longitudinal channel + s-memory prior | 50Hz |
| `scripts/ekf_mu_patch_publisher.py` | 2D sim μ patch → `/mu_ground_truth` + rviz marker | event |
| `scripts/ekf_mu_applier.py` | μ_gt 기반 cmd scaling (2D sim 가상 마찰) | 50Hz |
| `scripts/ekf_mu_hud.py` | rviz HUD: gt/est μ 구체 + 텍스트 + Δ 경고 | 10Hz |
| `scripts/ekf_mu_toggle_gui.py` | tkinter 대시보드 — μ 적응 on/off | 5Hz |
| `scripts/ekf_mu_estimator_gp.py` | GP ensemble stub (실험용) | 50Hz |

---

## 3. 목표

- **ICRA2026 RoboRacer** multi-friction 트랙 대응
- 경량 (순수 numpy EKF, NUC 14 Pro 상 <1ms)
- Patch 경계 인지 + lap-to-lap 복원
- 코너 통과 성공률 (icy_hairpin 포함)

---

## 4. 파일 구조

```
controller/ekf_mpc/
├── ekf_mpc_node.py                   # MPC 노드
├── config/
│   ├── ekf_mpc_srx1.yaml             # 튜닝 (MPC + EKF L2.3)
│   └── mu_patches_f.yaml             # 2D sim μ patch 정의
├── launch/ekf_mpc_sim.launch         # 2D sim 런치
├── rviz/ekf_mpc.rviz
└── scripts/
    ├── mu_estimator_ekf.py
    ├── ekf_mu_patch_publisher.py
    ├── ekf_mu_applier.py
    ├── ekf_mu_hud.py
    ├── ekf_mu_toggle_gui.py
    └── ekf_mu_estimator_gp.py
```

---

## 5. 파라미터 (`config/ekf_mpc_srx1.yaml`)

### MPC 공통
| 키 | 의미 | 기본값 |
|---|---|---|
| `N_horizon`, `dt` | 20 × 0.05s (1s horizon) | |
| `v_max`/`v_min` | 속도 경계 | 12.0 / 0.0 |
| `max_steer`, `max_accel`, `max_decel` | 입력 한계 | 0.4 / 3.0 / -3.0 |
| `w_d`, `w_dpsi`, `w_vx`, `w_vy`, `w_omega` | cost 가중 | 8 / 6 / 3 / 1 / 3 |
| `friction_margin`, `friction_slack_penalty` | friction circle | 0.95 / 1000 |
| `mu_default` | static μ / base 상수 | 0.85 |

### EKF 세부 (`ekf:` 섹션)
| 키 | 의미 | 기본값 |
|---|---|---|
| `init_mu`, `init_sigma` | 초기값 / 초기 불확실성 | 0.85 / 0.30 |
| `proc_sigma` | process noise σ | 0.03 (L2.3) |
| `meas_sigma` | ay 측정 noise σ | 0.50 (L2.3) |
| `min_speed`, `min_K` | 업데이트 조건 | 1.0 / 0.7 |
| `mu_min`, `mu_max` | 추정치 clip | 0.35 / 1.3 |
| `vy_smooth_alpha` | dvy/dt LPF | 0.35 |
| `prior_pull_rate` | skip 시 init_mu 로 pull | 0.005 |
| `long_enable`, `long_ax_thresh`, `long_slip_thresh`, `long_meas_sigma` | longitudinal channel | true / 1.5 / 0.5 / 0.45 |
| `mem_enable`, `mem_bin_width`, `mem_ema_alpha`, `mem_pull_rate` | s-memory | true / 2.0m / 0.15 / 0.015 |

### Warmup / Startup
| 키 | 의미 | 기본값 |
|---|---|---|
| `warmup_vx_min`, `warmup_speed_cmd`, `warmup_exit_vx` | 출발 kick + 1회 disarm | 0.2 / 2.0 / 0.8 |
| `startup_delay_s` | 초기 hold | 7.0 |

### Launch 인자 (`ekf_mpc_sim.launch`)
| arg | 기본 | 설명 |
|---|---|---|
| `map` | `f` | 맵 이름 |
| `mu_source` | `static` | `static` / `ground_truth` / `ekf` / `gp` |
| `apply_mu_to_sim` | `true` | cmd scaling 활성 |
| `rviz`, `toggle_gui` | `true` | viz 토글 |

---

## 6. 실행 (2D sim)

```bash
CAR_NAME=SIM roslaunch controller ekf_mpc_sim.launch map:=f mu_source:=ekf
```

`mu_source`:
- `static`: `mu_default=0.85` 고정 (baseline)
- `ground_truth`: patch μ 를 직접 relay (이상적 상한)
- `ekf`: Pacejka-EKF 추정값 (권장)
- `gp`: GP ensemble stub (실험적)

대시보드 **ENABLE/DISABLE** 버튼으로 μ 적응 on/off 토글 → A/B 비교.

---

## 7. 실차 포팅 (→ `Brojoon-10/ICRA2026_HJ/main`)

**대상 레포**: <https://github.com/Brojoon-10/ICRA2026_HJ/tree/main>

### Step 1: 파일 복사
```bash
cp -r controller/ekf_mpc/ <main-repo>/controller/ekf_mpc/
# 의존: controller/mpc/vehicle_model.py, mpcc_ocp.py, reference_builder.py (import 공유)
```

### Step 2: 의존성
- `acados` + `acados_template` (Python): 없으면 `thirdparty/setup_acados.sh` 실행
- env: `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib`
- ROS1 Noetic + `f110_msgs`
- (EKF 는 numpy only → 별도 ML 의존성 없음. GPU 불필요)

### Step 3: 차량 파라미터
- `shared_vehicle/vehicle_<car>.yaml` 의 `m`, `l_f`, `l_r`, `I_z`, `Bf..Er` 실측치로 교체
- **bench-test 로 Pacejka B, C, D, E 를 새차량에 맞게 fit** 필요
  (sim 값 그대로 쓰면 K 계수 오차 누적 → EKF 편향)

### Step 4: 실차 런치 수정
`launch/ekf_mpc_sim.launch` 에서:
- `base_system.launch sim:=true` → 실차 런치로 교체 (`headtohead.launch` 류)
- `ekf_mu_applier.py` 제거 (실차 마찰은 실물)
- `ekf_mu_patch_publisher.py` 제거 (patch 는 sim 전용)
- `/vesc/...nav_1` 출력 토픽 유지

### Step 5: 실차 튜닝 check
- `proc_sigma`: 실차 slip dynamics 빠르면 0.05 까지 ↑
- `meas_sigma`: IMU 노이즈 수준 반영 (실측)
- `min_K`: 실차 Pacejka K 범위 측정 후 조정
- `s-memory`: 트랙 고정이면 유지. 매번 트랙 바뀌면 `mem_enable=false`

### Step 6: 안전 장치
- `mu_min` 을 0.35 이상 (너무 낮으면 OCP 극단 감속)
- `apply_mu_to_sim=false` 실차 탑재 (sim 전용 스케일링 끄기)
- `mu_source=static` 으로 시작 → safe lap 확인 → `ekf` 로 전환

### 주의
- sim vs 실차 Pacejka mismatch → K 추정 오차 → EKF 가 μ 를 mismatch 보상으로 씀
- 저마찰 구간 편향: Race 중 `Reset` 가능한 UI 권장

---

## 8. 한계 / 후속

**Scalar μ 구조적 한계**:
- ay-only 관측 → patch 구분 해상도 부족 (icy/normal/grippy 평균 err ~0.2)
- Closed-loop bias (보수화 → 관측 약화 → 편향 심화)
- 단일 μ 로 longitudinal / lateral 비대칭 못 잡음

**후속 (`controller/gp_mpc`)**:
- 3D residual GP `(Δvx, Δvy, Δω)` 로 bias-free 잔차 학습
- UPenn Nagy et al. 2023 IFAC WC 구현
- s-feature 추가로 patch 자동 구분
