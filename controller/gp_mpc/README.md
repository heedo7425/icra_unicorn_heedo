# `controller/gp_mpc`

UPenn-style **online GP residual MPC** — 2D f1tenth sim 용.

Base Pacejka dynamic bicycle MPC 위에 **3D residual GP** `(Δvx, Δvy, Δω)` 를 추가해
런타임에 base model 의 예측 오차를 학습/보정한다.

- Paper: Nagy et al. 2023, *Ensemble Gaussian Processes for Adaptive Autonomous Driving
  on Multi-friction Surfaces*, IFAC WC (arxiv: 2303.13694)
- Repo ref: <https://github.com/mlab-upenn/multisurface-racing>

`controller/ekf_mpc` 의 single-scalar μ EKF 와 병행 실험 (같은 base acados OCP 공유).
완전 격리: 별도 네임스페이스 `/gp_mpc/*`, 별도 codegen dir `/tmp/gp_mpc_c_generated`.

---

## 1. 요약

**ekf_mpc 와 차이**

| | ekf_mpc | gp_mpc |
|---|---|---|
| 추정 | scalar μ (Pacejka EKF) | 3D residual `(Δvx, Δvy, Δω)` (GPyTorch ExactGP) |
| 신호 | `a_y = μ·K(x,u)` 관측 | trajectory residual = `dx/dt_measured - pacejka(x,u)` |
| Closed-loop bias | 있음 (ay-feedback) | 없음 (잔차 직접 fit) |
| Cadence | 50Hz online | lap-end batch 학습 + 50Hz inference |
| MPC 주입 | stage param `mu` (N+1, 6) | stage param `(μ, Δvx, Δvy, Δω)` (N+1, 9) |

**MPC μ 역할**: GP 가 잔차로 마찰 효과 흡수 → MPC base model 의 `mu` 는 `mu_default=0.85` 로 **고정**.
`mu_source=gp` 모드에서는 μ estimate 를 직접 주입하지 않음. GP 가 그 역할.

**2D sim 가상 마찰**: `gp_mu_applier.py` 가 ground-truth μ (patch lookup) 으로 cmd scaling — sim 에
"마찰 낮음" 효과 유도. 학습 데이터의 residual 이 이 효과를 흡수 (명령 vs 실제 거동 차이).

---

## 2. 파일 구조

```
controller/gp_mpc/
├── gp_mpc_node.py                      # MPC 노드 (acados OCP + residual 주입)
├── mpcc_ocp_gp.py                      # build_tracking_ocp_gp (NP+3 extended)
├── config/
│   ├── gp_mpc_srx1.yaml
│   └── mu_patches_f.yaml
├── launch/gp_mpc_sim.launch
├── rviz/gp_mpc.rviz
└── scripts/
    ├── gp_mu_patch_publisher.py        # /mu_ground_truth + rviz patch viz
    ├── gp_mu_applier.py                # cmd scaling (sim 가상 마찰)
    ├── gp_mu_hud.py                    # rviz HUD (gt vs MPC 사용 μ)
    ├── gp_mu_toggle_gui.py             # tkinter on/off 대시보드
    ├── gp_trainer.py                   # NEW — lap-end GP 학습 (ExactGP, 3-task)
    └── gp_residual_publisher.py        # NEW — 50Hz GP eval → /gp_mpc/residual
```

---

## 3. 데이터 흐름

```
Lap 1: gp_trainer 가 /car_state/odom + /gp_mpc/cmd_raw buffer 축적
       gp_ready=False → gp_residual_publisher 는 zeros publish → MPC residual=0 (base only)
  │
  ↓ s wrap-around (track_length*0.75 → *0.15)
gp_trainer._train_and_save():
   pairs (x_k, u_k, x_{k+1}) → residual = (x_{k+1}-x_k)/dt - pacejka_numpy(...)
   ExactGP 3-task (Δvx, Δvy, Δω), ScaleKernel·RBF(ard=6), Adam 300 iter
   torch.save → /tmp/gp_mpc_models/latest.pth (atomic rename)
   publish /gp_mpc/gp_ready=True, /gp_mpc/train_time_s
  │
  ↓
gp_residual_publisher (50Hz):
   mtime watch → 핫리로드
   eval (vx, vy, ω, δ, u_ax, u_ddelta) → mean·variance
   residual_clip hard-clip
   publish /gp_mpc/residual (Float32MultiArray), /gp_mpc/gp_sigma
  │
  ↓
gp_mpc_node._solve():
   build_preview() → params (N+1, 6) → 3 residual 열 broadcast → (N+1, 9)
   solve_once_gp(solver, x0, params)
   xdot_corrected = xdot_pacejka + [0,0,0,Δvx,Δvy,Δω,0]
```

---

## 4. 실행

```bash
CAR_NAME=SIM roslaunch controller gp_mpc_sim.launch \
    map:=f mu_source:=<static|ground_truth|gp>
```

`mu_source`:
- `static`: `mu_default=0.85` base model, GP off (잔차 학습은 여전히 돌지만 OCP 주입 없음)
- `ground_truth`: `/mu_ground_truth` → `/gp_mpc/mu_estimate` relay (sanity check)
- `gp` (권장): base μ = mu_default 고정, GP residual 가 적응

Cold start (lap 1) 은 어느 모드든 residual=0. Lap 끝난 후 gp_trainer 가 model publish → lap 2 부터 residual 활성.

---

## 5. 주요 토픽

| Topic | Type | 의미 |
|---|---|---|
| `/gp_mpc/cmd_raw` | AckermannDriveStamped | MPC 원 명령 (mu_applier scaling 전) |
| `/gp_mpc/prediction` | MarkerArray | 예측 horizon |
| `/gp_mpc/reference` | MarkerArray | 슬라이스된 raceline |
| `/gp_mpc/solve_ms` | Float32 | OCP solve time |
| `/gp_mpc/mu_used` | Float32 | OCP 에 넣은 base μ |
| `/gp_mpc/mu_estimate` | Float32 | (static/gt 모드) 외부 추정 μ |
| `/gp_mpc/residual` | Float32MultiArray | `[Δvx, Δvy, Δω]` (GP mean) |
| `/gp_mpc/gp_sigma` | Float32MultiArray | `[σ_vx, σ_vy, σ_ω]` (GP posterior std) |
| `/gp_mpc/gp_ready` | Bool | GP 학습 완료 (latched) |
| `/gp_mpc/train_time_s` | Float32 | 마지막 학습 소요 시간 |
| `/gp_mpc/buffer_size` | Float32 | 사용한 학습 샘플 수 |
| `/gp_mpc/mu_adapt_enable` | Bool | GP residual on/off 토글 (tkinter GUI) |
| `/mu_ground_truth` | Float32 | patch 기반 gt |
| `/mu_patches/markers` | MarkerArray | patch viz |

---

## 6. 파라미터 (`/gp_mpc/gp:`)

| 키 | 기본 | 의미 |
|---|---|---|
| `model_path` | `/tmp/gp_mpc_models/latest.pth` | trainer/publisher 공유 checkpoint |
| `buffer_size` | 5000 | trainer rolling buffer (FIFO) |
| `train_min_samples` | 500 | 첫 lap 미숙 학습 방지 |
| `train_max_samples` | 1500 | ExactGP O(N³) 보호용 subsample |
| `train_epochs` | 300 | Adam |
| `train_lr` | 0.05 | |
| `skip_first_sec` | 3.0 | warmup 제외 |
| `residual_clip` | `[10.0, 5.0, 8.0]` | Δvx, Δvy, Δω hard clip |
| `residual_enable` | true | OCP 에 실제 주입 여부 (debug 용 off 가능) |
| `enable_stagewise` | false | Phase 2 — stage 별 GP eval |

---

## 7. 한계 / 후속

**Phase 1 (이 버전)**:
- Residual constant across horizon (Option A). OCP solve time 영향 ≤1ms 예상
- ExactGP: buffer 5000 cap + max 1500 subsample 로 lap-end 학습 <15s 목표

**Phase 2 예정**:
- Stagewise GP eval: OCP 외부 rollout 으로 stage 별 residual 제공
- Active learning (variance-max 250-pt): buffer 우선순위 재조정
- Hybrid-learning desktop 폴더의 bruteforce (B,C,μ,Iz) parameter ID 와 결합 — base 자체 refinement
- Multi-surface GP library (UPenn 원 논문의 ensemble blending)
