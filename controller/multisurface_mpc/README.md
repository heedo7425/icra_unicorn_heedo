# `controller/multisurface_mpc` — Peak-tracking RLS μ MPC (prototype)

> 가장 단순한 접근: **peak lateral acceleration** 으로 saturation 을 가정한
> scalar μ 추정. **patch 구분 실패** 한계 확인 후 `ekf_mpc` 로 대체됨.
> *(이 패키지는 이후 `rls_mpc` 로 이름 변경됨 — 이 브랜치에는 원래 이름으로 보존)*

---

## 1. 이론적 배경

### Idea: Peak-Ay / g ≈ μ assumption
Friction circle 가정: `|a_y_max| ≈ μ · g` (saturation 가정). 따라서
`μ ≈ |a_y|_peak / g`.

### RLS 추정
관측 `z_k = |a_y|_peak / g`. Recursive Least Squares 로 scalar μ:
```
μ̂_{k+1} = μ̂_k + K_k · (z_k − μ̂_k)
K_k = P_k / (λ + P_k)
P_{k+1} = (1 − K_k) · P_k / λ
```
`λ` 는 forgetting factor (0.98 쯤).

### 왜 실패하나
- **모든 patch 에서 비슷한 ay_peak (2-3 m/s²) 만 측정**: MPC 가 safe 하게 주행하면
  tire 가 saturation 에 도달 안 함 → peak μ 관측 불가
- **Non-saturation 구간 정보 0**: RLS 업데이트 할 데이터가 없음
- 결과: icy_long 0.40 → est 0.80 (err 0.40), patch 구분 불가

이 한계가 **`controller/ekf_mpc` 로 전환하는 동기**. Pacejka 모델을 관측 방정식에
넣으면 non-saturation 구간에서도 μ 정보 추출 가능.

---

## 2. 기능 / Components

| 노드 | 역할 |
|---|---|
| `multisurface_mpc_node.py` | MPC (acados), runtime μ 수신 |
| `scripts/mu_estimator_rls.py` | peak-ay RLS |
| `scripts/mu_estimator_gp.py` | GP stub (MA fallback) |
| `scripts/mu_patch_publisher.py` | 2D sim μ patch → `/mu_ground_truth` + rviz |
| `scripts/mu_applier.py` | μ_gt 기반 cmd scaling (sim 가상 마찰) |
| `scripts/mu_hud.py` | rviz gt vs est μ 비교 |
| `scripts/mu_toggle_gui.py` | tkinter 대시보드 |

---

## 3. 목표 (원래 설계)

- 2D sim multi-friction patch 인식 실험의 **최소 scalar μ MPC**
- RLS 경량 (수 μs inference) → 실차 이식성 데모
- 이후 더 정교한 EKF / GP 방법과 비교 baseline

---

## 4. 파일 구조

```
controller/multisurface_mpc/
├── multisurface_mpc_node.py
├── config/
│   ├── multisurface_mpc_srx1.yaml       # /mpc_ms/* namespace
│   └── mu_patches_f.yaml                # patch 정의
├── launch/multisurface_mpc_sim.launch
├── rviz/multisurface_mpc.rviz
└── scripts/
    ├── mu_estimator_rls.py
    ├── mu_estimator_gp.py
    ├── mu_patch_publisher.py
    ├── mu_applier.py
    ├── mu_hud.py
    └── mu_toggle_gui.py
```

---

## 5. 파라미터 (`config/multisurface_mpc_srx1.yaml`)

### MPC 공통
controller/mpc_only 와 동일한 MPC 구조 + runtime μ 주입

### RLS (`rls:` 섹션)
| 키 | 의미 | 기본값 |
|---|---|---|
| `forgetting_factor` (λ) | 과거 가중 감쇠 | 0.98 |
| `init_mu` | 초기 μ̂ | 0.85 |
| `init_P` | 초기 공분산 | 1.0 |
| `peak_window_s` | ay peak 검출 윈도우 | 0.5 |
| `mu_min`, `mu_max` | clip | 0.2 / 1.3 |

### μ source enum
`static` / `ground_truth` / `rls` / `gp`

---

## 6. 실행 (2D sim)

```bash
CAR_NAME=SIM roslaunch controller multisurface_mpc_sim.launch \
    map:=f mu_source:=<static|ground_truth|rls|gp>
```

---

## 7. 실차 포팅 (→ `Brojoon-10/ICRA2026_HJ/main`)

**비추** — 이 패키지는 한계 확인용 prototype 이고 후속 `ekf_mpc` / `gp_mpc` 가 권장.
그럼에도 포팅 시:

### Step 1: 파일 복사
```bash
cp -r controller/multisurface_mpc/ <main-repo>/controller/multisurface_mpc/
# 의존: controller/mpc/vehicle_model.py, mpcc_ocp.py, reference_builder.py
```

### Step 2: 의존성
- `acados` + `acados_template`, ROS1 Noetic + `f110_msgs`
- 순수 numpy → GPU 불필요

### Step 3: 실차 런치 수정
- `mu_applier.py` 제거 (실차 실마찰)
- `mu_patch_publisher.py` 제거 (sim 전용)
- `mu_source=static` 으로 시작 → `rls` 로 전환

### Step 4: 한계 인지
- 실차에서도 **tire saturation 안 일어나는 주행 스타일**이면 RLS 안 움직임
- 저마찰에서 오히려 추정치가 과대 (MPC 보수화 → ay 작아짐 → saturation 없음)
- 긴급 출동 상황 (skid) 에서만 유의미 관측

---

## 8. 한계 / 후속

**본 접근의 본질적 한계**:
- Non-saturation observability 0
- Single scalar, closed-loop bias (ekf_mpc 와 같은 이슈)
- patch 구분 실패 (2D sim 실측 확인됨)

**후속**:
- `controller/ekf_mpc` — Pacejka 모델 기반 관측으로 non-saturation 도 μ 추출
- `controller/gp_mpc` — 3D residual GP 로 μ 개념 자체 우회 + patch 구분 성공
