# `controller/gp_mpc` — GP residual MPC (UPenn-style)

> Base Pacejka dynamic bicycle MPC 위에 **3D 잔차 GP** `(Δvx, Δvy, Δω)` 를 얹어
> 온라인으로 노면 특성을 적응.

---

## 1. 이론적 배경

### Problem
Friction coefficient μ 는 트랙 위치/노면 재질마다 다르고 단일 scalar 로는 묘사 부족
(`controller/ekf_mpc` 에서 확인). Pacejka base model 만으로는 patch 이동 시 오차 큼.

### Approach: Nagy et al. 2023 (arxiv 2303.13694, IFAC WC)
**"Ensemble Gaussian Processes for Adaptive Autonomous Driving on Multi-friction Surfaces"**
<https://github.com/mlab-upenn/multisurface-racing>

- GP 가 base dynamic model 의 **잔차** 를 학습: `Δẋ = ẋ_measured − f_pacejka(x, u)`
- OCP 동역학 RHS 에 residual 을 additive 주입 → 예측 궤적이 GP 보정 반영
- μ 는 base 용 상수로 고정. GP 가 모든 적응 책임

### 수학 구조
```
State  (7):  x = [s, n, dpsi, vx, vy, omega, delta]
Input  (2):  u = [u_ddelta, u_ax]
Params (9):  p = [kappa, theta, kappa_z, mu, vx_ref, n_ref,   Δvx, Δvy, Δω]
                 └──────── base OCP ──────────┘  └ GP residual ┘

ẋ = f_pacejka(x, u; p[:6]) + [0, 0, 0, p[6], p[7], p[8], 0]
```

GP features (7D): `(vx, vy, ω, δ, u_ax, u_ddelta, s)` — 마지막 `s` (Frenet 호장)
이 트랙 위 절대위치를 인코딩 → 같은 (속도, 조향) 입력이 다른 patch 에서 다른 residual 학습 가능.

GP outputs (3 독립 task): `(Δvx, Δvy, Δω)` 각각 ExactGP, RBF+ARD kernel, GPyTorch
`BatchIndependentMultitaskGPModel`.

---

## 2. 기능 / Components

| 노드 | 역할 | 주기 |
|---|---|---|
| `gp_mpc_node.py` | acados OCP + residual 주입, 두 solver (active/base) 병렬 solve | 50Hz |
| `scripts/gp_trainer.py` | lap-end 마다 ExactGP 3-task 학습 | lap event |
| `scripts/gp_residual_publisher.py` | 50Hz 로 GP 평가 → `/gp_mpc/residual` | 50Hz |
| `scripts/gp_mu_patch_publisher.py` | μ patch 로부터 ground-truth μ publish (sim 가상 마찰용) | event |
| `scripts/gp_mu_applier.py` | μ_gt 기반 cmd scaling (2D sim 의 가상 마찰 효과) | 50Hz |
| `scripts/gp_mu_hud.py` | rviz HUD: ground-truth μ + GP correction strength viz | 10Hz |
| `scripts/gp_mu_toggle_gui.py` | tkinter 대시보드: vx/cmd/solve/Δ residual/scroll plot/Reset/Toggle 버튼 | 5Hz |

**핵심 기능 요약**:
- **Online incremental 학습** — 사전 surface library 불필요
- **σ-attenuation** — GP 불확실성 ∝ 1/(1+(σ/σ0)²) 로 자동 감쇠 → OCP infeasibility 방지
- **μ-aware speed boost** — `v_ref ← v_ref · speed_boost · (μ_gt/μ_default)^mu_scale_exp` (grippy 에서 더 빠르게)
- **Crash recovery** — vx<0.2 가 crash_stuck_sec 지속 시 warmup 재무장 → 자동 복구
- **Parallel base OCP** — GP on/off 즉시 A/B 비교 (residual=0 solver 도 항상 돌림)
- **Reset/Toggle** — 대시보드 버튼으로 모델 삭제 또는 base 강제 → 자동 respawn

---

## 3. 목표

- ICRA2026 RoboRacer 대회 **multi-friction track** 대응
- **사전 노면 데이터 수집 없이** 경쟁 lap time 유지
- 실차 (NUC 14 Pro, Core Ultra) 에서 50Hz 실시간 유지
- ekf_mpc 의 scalar-μ 한계 극복 (closed-loop bias, patch 구분 해상도)

---

## 4. 파일 구조

```
controller/gp_mpc/
├── gp_mpc_node.py                  # MPC 메인 — acados OCP + residual 주입
├── mpcc_ocp_gp.py                  # build_tracking_ocp_gp (NP_GP = NP + 3)
├── config/
│   ├── gp_mpc_srx1.yaml            # 전체 튜닝 파라미터
│   └── mu_patches_f.yaml           # 2D sim μ patch (s-range, d-range, μ 값)
├── launch/gp_mpc_sim.launch        # 2D sim 런치 (base_system + 모든 노드)
├── rviz/gp_mpc.rviz                # rviz config (prediction, reference, patches, HUD)
└── scripts/
    ├── gp_trainer.py
    ├── gp_residual_publisher.py
    ├── gp_mu_patch_publisher.py
    ├── gp_mu_applier.py
    ├── gp_mu_hud.py
    └── gp_mu_toggle_gui.py
```

---

## 5. 파라미터 (`config/gp_mpc_srx1.yaml`)

### MPC 공통
| 키 | 의미 | 기본값 |
|---|---|---|
| `N_horizon` | OCP horizon 길이 | 20 |
| `dt` | 제어 주기 | 0.05 |
| `v_max` / `v_min` | 속도 경계 | 12.0 / 0.0 |
| `max_steer` / `max_steer_rate` | 조향 한계 | 0.4 rad / 2.0 rad/s |
| `max_accel` / `max_decel` | 가속/감속 한계 | 3.0 / -3.0 m/s² |
| `w_d` | 측방 오차 `n` 가중 | **14.0** (inside cut 방지용 상향) |
| `w_dpsi`, `w_vx`, `w_vy`, `w_omega` | cost 가중 | 6 / 3 / 1 / 3 |
| `w_steer`, `w_u_steer_rate`, `w_u_accel` | input 가중 | 0.5 / 3 / 0.1 |
| `friction_margin` | friction circle 여유 (0.85 = 85% 만 사용) | **0.85** |
| `friction_slack_penalty` | slack cost | 1000 |

### μ source
| 키 | 의미 | 기본값 |
|---|---|---|
| `mu_default` | base OCP μ (GP 모드에선 고정) | 0.85 |
| `mu_source` | `static` / `ground_truth` / `gp` | gp (런치 arg 로 override) |
| `speed_boost` | raceline `vx_ref` 배수 | **1.3** |
| `mu_scale_exp` | μ-aware scaling (v ∝ μ^exp) | **0.3** (0=off, 0.5=√μ) |

### GP 세부 (`gp:` 섹션)
| 키 | 의미 | 기본값 |
|---|---|---|
| `model_path` | 체크포인트 경로 | `/tmp/gp_mpc_models/latest.pth` |
| `buffer_size` | 학습용 rolling FIFO | **10000** (≈ 5 lap) |
| `train_min_samples` | 첫 학습 전 최소 샘플 | 400 |
| `train_max_samples` | ExactGP O(N³) 보호용 subsample | **700** (7D ARD 고려) |
| `train_epochs` | Adam iter | 100 |
| `train_lr` | 학습률 | 0.05 |
| `skip_first_sec` | warmup 구간 데이터 제외 | 3.0 |
| `residual_clip` | `[Δvx, Δvy, Δω]` hard clip | `[10.0, 5.0, 12.0]` |
| `enable_stagewise` | Phase 2 — stage-wise GP eval | false |
| `residual_enable` | OCP 주입 on/off (debug) | true |
| `torch_num_threads` | PyTorch 스레드 수 | 2 |

### MPC Warmup / Crash
| 키 | 의미 | 기본값 |
|---|---|---|
| `warmup_vx_min` | warmup 진입 기준 | 0.2 |
| `warmup_speed_cmd` | warmup 시 발행 속도 | 2.0 |
| `warmup_exit_vx` | 1회 달성 후 warmup 영구 disarm | 0.8 |
| `crash_stuck_sec` | 이 시간 stuck 시 warmup 재무장 | 1.5 |

### Launch 인자 (`gp_mpc_sim.launch`)
| arg | 기본 | 설명 |
|---|---|---|
| `map` | `f` | 맵 이름 (`stack_master/maps/<name>/`) |
| `mu_source` | `gp` | `static` / `ground_truth` / `gp` |
| `apply_mu_to_sim` | `true` | `gp_mu_applier` 의 cmd scaling |
| `rviz` | `true` | rviz 표시 |
| `toggle_gui` | `true` | tkinter 대시보드 표시 |

---

## 6. 실행 (2D sim)

```bash
CAR_NAME=SIM roslaunch controller gp_mpc_sim.launch map:=f mu_source:=gp
```

첫 lap 은 cold-start (GP off). lap 끝나면 trainer 가 학습 → lap 2 부터 GP residual 활성.

대시보드 버튼:
- **Disable μ adaptation** → base solver 로 전환 + 차 respawn
- **Reset GP** → 모델 삭제 + 버퍼 clear + 첫 랩부터 재학습

---

## 7. 실차 포팅 (→ `Brojoon-10/ICRA2026_HJ/main`)

**대상 레포**: <https://github.com/Brojoon-10/ICRA2026_HJ/tree/main>

### Step 1: 파일 복사
```bash
# 이 브랜치에서 controller/gp_mpc/ 전체를 main 레포로 복사.
cp -r controller/gp_mpc/ <main-repo>/controller/gp_mpc/
# mpc 쪽 base 모듈 (vehicle_model, mpcc_ocp, reference_builder) 은 import 로 공유됨.
```

### Step 2: 의존성 (컨테이너/차량)
- `acados` + `acados_template` (Python). 실차 컨테이너에 이미 있으면 skip. 없으면:
  - `thirdparty/setup_acados.sh` (repo 포함) 실행
  - env: `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH`
- `gpytorch >= 1.8`, `torch >= 2.0` (pip install)
- ROS1 Noetic + `f110_utils` + `f110_msgs`

### Step 3: 차량 파라미터
- `shared_vehicle/vehicle_<YourCar>.yaml` 의 `m`, `l_f`, `l_r`, `I_z`, `Bf/Cf/Df/Ef`, `Br/Cr/Dr/Er` 을 해당 차량 실측값으로 교체
- `gp_mpc_srx1.yaml` 의 `mu_default` 를 실차 측정 μ 로 설정

### Step 4: 런치 수정 (실차용)
- `launch/gp_mpc_sim.launch` 에서:
  - `base_system.launch` 의 `sim:=true` → 실차 런치로 교체 (예: `headtohead.launch` 류)
  - `gp_mu_applier` 는 **제거** (실차에선 실물 마찰이 작동)
  - `gp_mu_patch_publisher` 도 제거 (patch 개념은 2D sim 전용)
  - `/vesc/...nav_1` 토픽은 동일하게 유지 (VESC mux 입력)
  - RViz config 는 실차 맵 tf 에 맞게 조정

### Step 5: 실시간 검증 (bench)
```bash
# NUC 에서 MPC + GP 동시 실행 CPU 부하 측정
stress-ng --cpu 4 --timeout 60s &
roslaunch controller gp_mpc_sim.launch
rostopic hz /gp_mpc/solve_ms   # <25ms 유지 확인
rostopic hz /vesc/.../nav_1    # 50Hz 유지 확인
```

### Step 6: GP 학습 코어 분리 (권장)
실차 안정성을 위해 `torch.set_num_threads(2)` + trainer 는 **별도 프로세스** (multiprocessing.Process) 로 격리. 현재 구현은 threading — 실차 이전 전 검증 필요.

### Step 7: 대회 1-2 lap 예열
- Practice/Quali 에서 GP 가 충분 데이터 수집해 수렴 (buffer 10000 = 5 lap)
- Race day 는 이미 학습된 모델로 시작 → Reset 금지
- 새 노면 감지 시 `Reset GP` 버튼으로 재학습 가능

### 주의 / Risk
- **acados 없는 차량**: IPOPT fallback 쓰려면 `mpcc_ocp_gp.py` 의 solver 설정 교체 필요 (p-solver 변경). 실시간성 약화.
- **gpytorch 없는 차량**: pip install. 또는 numpy-only sparse GP 로 포팅 (~50 LOC).
- **친구/동료 레포와 병합**: namespace `/gp_mpc/*` 는 고유하므로 충돌 없음. codegen dir `/tmp/gp_mpc_c_generated` 도 격리.

---

## 8. 한계 / 후속 (Phase 2+)

- **Stagewise residual**: 현재는 x0 에서만 GP eval → N+1 stage broadcast. Stage-wise 평가로 horizon 끝단까지 정확도 ↑ (OCP solve +3-8ms)
- **Active learning**: variance-max 기반 250-pt subsample (UPenn 원본 방식)
- **Hybrid-learning 결합**: 별도 데스크톱 폴더의 bruteforce (B, C, μ, I_z) parameter ID → base model 자체 갱신 (Phase 2 로 계획)
- **Multi-surface library**: 사전 학습된 surface-별 GP 라이브러리 + weight blending (UPenn 논문 본래 ensemble 접근)
