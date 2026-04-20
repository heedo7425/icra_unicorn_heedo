# `controller/rls_mpc` — Peak-tracking RLS μ MPC (preserved)

> `controller/multisurface_mpc` 의 이름 변경본. peak-ay RLS 기반 scalar μ 추정.
> **이력 보존용** — patch 구분 실패가 확인되어 `controller/ekf_mpc` 로 대체됨.

---

## 1. 이론적 배경

### Idea
Friction circle 가정: `|a_y_max| ≈ μ·g` (타이어 saturation). → `μ ≈ |a_y|_peak/g`.
이를 online 관측으로 scalar μ RLS.

### RLS 업데이트
```
z_k = |a_y|_peak / g
μ̂_{k+1} = μ̂_k + K_k · (z_k − μ̂_k)
K_k = P_k / (λ + P_k),  P_{k+1} = (1 − K_k)·P_k/λ
λ ≈ 0.98 (forgetting)
```

### 왜 실패하나 (실측 확인)
- 2D sim 에서 모든 patch 가 비슷한 `ay_peak` (2-3 m/s²) 만 보임
- 이유: MPC 가 μ̂ 보고 safe 하게 주행 → tire saturation 도달 안 함
- 결과 patch 별 err:
  - icy_long (0.40): est 0.80 (err 0.40)
  - icy_hairpin (0.40): est 0.42 (err 0.10, 코너에서만 saturate)
  - grippy_top (1.20): est 1.09 (err 0.14)
  - grippy_corner (1.20): est 0.89 (err 0.32)
- **Patch 구분 사실상 실패**

**핵심 결함**: Saturation 가정이 ay-only 관측에선 non-saturation 구간 정보 0.
`controller/ekf_mpc` 는 Pacejka 모델을 관측 방정식에 넣어 이 한계 극복.

---

## 2. 기능 / Components

| 노드 | 역할 |
|---|---|
| `rls_mpc_node.py` | MPC, runtime μ 수신 |
| `scripts/mu_estimator_rls.py` | peak-ay RLS |
| `scripts/mu_estimator_gp.py` | GP stub (MA fallback) |
| `scripts/mu_patch_publisher.py` | sim μ patch → `/mu_ground_truth` |
| `scripts/mu_applier.py` | μ_gt 기반 cmd scaling |
| `scripts/mu_hud.py` | rviz 비교 viz |
| `scripts/mu_toggle_gui.py` | tkinter dashboard |

---

## 3. 목표 (원 설계)

- Scalar μ 온라인 추정 최소 솔루션
- 실차 이식성 (RLS 수 μs inference, 순수 numpy)
- 후속 EKF / GP 와 비교 baseline

**실제**: 2D sim 에서 한계 명확 → preservation.

---

## 4. 파일 구조

```
controller/rls_mpc/
├── rls_mpc_node.py
├── config/
│   ├── rls_mpc_srx1.yaml             # /rls_mpc/* namespace
│   └── mu_patches_f.yaml
├── launch/rls_mpc_sim.launch
├── rviz/rls_mpc.rviz
└── scripts/
    ├── mu_estimator_rls.py
    ├── mu_estimator_gp.py
    ├── mu_patch_publisher.py
    ├── mu_applier.py
    ├── mu_hud.py
    └── mu_toggle_gui.py
```

---

## 5. 파라미터 (`config/rls_mpc_srx1.yaml`)

### MPC 공통
controller/mpc 와 같음. 추가로 `mu_estimate_topic`, `codegen_dir=/tmp/rls_mpc_c_generated`.

### RLS (`rls:` 섹션)
| 키 | 의미 | 기본값 |
|---|---|---|
| `forgetting_factor` | λ | 0.98 |
| `init_mu`, `init_P` | 초기값 | 0.85 / 1.0 |
| `peak_window_s` | ay peak 검출 윈도우 | 0.5 |
| `mu_min`, `mu_max` | clip | 0.2 / 1.3 |

### μ source enum
`static` / `ground_truth` / `rls` / `gp`

---

## 6. 실행 (2D sim)

```bash
CAR_NAME=SIM roslaunch controller rls_mpc_sim.launch \
    map:=f mu_source:=<static|ground_truth|rls|gp>
```

---

## 7. 실차 포팅 (→ `Brojoon-10/ICRA2026_HJ/main`)

**비추** — 한계 확인된 prototype. 실차엔 `ekf_mpc` / `gp_mpc` 권장.

### 만약 포팅한다면
```bash
cp -r controller/rls_mpc/ <main-repo>/controller/rls_mpc/
# 의존: controller/mpc/vehicle_model.py, mpcc_ocp.py, reference_builder.py
```

### 주의
- **실차 saturation 도달 빈도 낮으면 RLS 정체** — μ̂ 가 init 값 근처에서 안 움직임
- sim 전용 `mu_applier.py`, `mu_patch_publisher.py` 제거 필수
- `mu_source=static` 으로 시작 → `rls` 전환 신중

---

## 8. 한계 / 후속

**구조적 한계**:
- Non-saturation observability 0
- Single scalar + closed-loop bias
- patch 구분 2D sim 실측 실패

**후속**:
- `controller/ekf_mpc` — Pacejka-EKF (non-saturation 에서도 관측)
- `controller/gp_mpc` — 3D residual GP (patch 구분 성공, UPenn Nagy 2023)
