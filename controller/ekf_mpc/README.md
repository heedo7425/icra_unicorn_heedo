# `controller/ekf_mpc`

**Pacejka 기반 EKF** 를 이용한 online μ 추정 + MPC raceline 추종. 2D f1tenth sim 에서 multi-friction 패치 인식 실험용 **독립 패키지**.

`controller/mpc`, `controller/mpc_only`, `controller/rls_mpc` 와 완전 격리:
- 네임스페이스: `/ekf_mpc/*`
- Codegen dir: `/tmp/ekf_mpc_c_generated`
- 스크립트 prefix: `ekf_mu_*.py` (중복 이름 충돌 방지)

---

## 1. 왜 EKF (RLS 대신)?

`controller/rls_mpc` 의 peak-ay RLS 는 모든 patch 에서 비슷한 ay_peak (2-3 m/s²) 만 측정해 **patch 구분 실패**. 이유: ay/g 관측은 saturation 가정이라 non-saturation 구간에서 μ 정보 극빈약.

EKF 는 Pacejka 모델을 관측 방정식에 반영해 **non-saturation 에서도 μ 관측 가능**:

```
a_y = μ · K(v_x, v_y, ω, δ) + noise,   K = (N_f·pf·cos(δ) + N_r·pr) / m

여기서 pf, pr 는 Pacejka "μ-free" factor:
pi = D_i · sin(C_i · atan(B_i·α_i − E_i·(B_i·α_i − atan(B_i·α_i))))
```

- μ 가 K 에 선형으로 들어가 Jacobian H = K 명시적으로 계산 가능
- 슬립각 α_f, α_r 작을 때 K 도 작음 → 자동으로 업데이트 억제 (무정보 자동 인지)
- 코너에서 K 크면 적극적 업데이트

---

## 2. Level 2 확장 (단순 Pacejka-EKF 에 추가)

### (a) Longitudinal channel — 직선 구간 커버
a_y 관측만 쓰면 직진 시 K ≈ 0 → 업데이트 불가. `|u_ax|` 와 `|a_x_measured|` 비교로 **saturation 감지**:

```
if |u_ax| ≥ long_ax_thresh AND (|u_ax| − |a_x_meas|) ≥ long_slip_thresh:
    z_long = |a_x_measured|                # 도달한 최대 가속도
    EKF update with H = g (∂z/∂μ ≈ g)
```

**효과**: 급가속/급감속 시 μ 추정 가능. icy 직선 구간 err 0.56 → 0.19 로 대폭 개선.

### (b) s-memory prior — lap-to-lap 학습
2m 간격으로 s-축 bin 에 EMA-smoothed μ̂ 저장. Update 없는 구간에선 같은 s_bin 의 이전 값으로 pull:

```
if no lateral/long update this tick:
    target = s_memory[s_bin] if exists else init_mu
    μ̂ ← μ̂ + mem_pull_rate · (target − μ̂)
if lateral update happened:
    s_memory[s_bin] ← (1−α)·s_memory[s_bin] + α·μ̂
```

**효과**: 같은 patch 반복 방문 시 값 안정화. 랩 간 일관성 개선.

---

## 3. 파일 구조

```
controller/ekf_mpc/
├── ekf_mpc_node.py                    # MPC 메인 (런타임 μ 수신)
├── config/
│   ├── ekf_mpc_srx1.yaml              # MPC + EKF 파라미터
│   └── mu_patches_f.yaml              # μ 패치 yaml
├── launch/ekf_mpc_sim.launch          # 통합 런치
├── rviz/ekf_mpc.rviz                  # 전용 rviz config
└── scripts/
    ├── mu_estimator_ekf.py            # Pacejka EKF + long + s-memory
    ├── ekf_mu_patch_publisher.py      # patch → /mu_ground_truth + 마커
    ├── ekf_mu_applier.py              # cmd scaling (가상 마찰)
    ├── ekf_mu_hud.py                  # rviz gt/est 비교 구·텍스트
    ├── ekf_mu_toggle_gui.py           # tkinter 대시보드
    └── ekf_mu_estimator_gp.py         # GP stub (추후 훈련 대비)
```

---

## 4. 실행

### 기본 커맨드
```bash
CAR_NAME=SIM roslaunch controller ekf_mpc_sim.launch \
    map:=f racecar_version:=SIM rviz:=true mu_source:=<MODE>
```

### `mu_source` 모드

| 모드 | 동작 |
|---|---|
| `static` | MPC 가 `mu_default=0.85` 고정 사용. 추정기 미기동. |
| `ground_truth` | `topic_tools/relay` 가 `/mu_ground_truth → /ekf_mpc/mu_estimate` 직결. Sanity check 용. |
| **`ekf`** | **Pacejka EKF + long channel + s-memory** 추정 (권장). |
| `gp` | GP stub (MA fallback, 사전훈련 pkl 없으면). |

### 런치 arg

| arg | 기본값 | 설명 |
|---|---|---|
| `map` | `f` | 맵 이름 |
| `rviz` | `true` | rviz 기동 |
| `mu_source` | `static` | 위 표 |
| `patches_file` | `$(find controller)/ekf_mpc/config/mu_patches_$(arg map).yaml` | patch yaml |
| `apply_mu_to_sim` | `true` | `ekf_mu_applier` 로 가상 마찰 cmd scaling 적용 |
| `toggle_gui` | `true` | tkinter 대시보드 기동 |

### 예시
```bash
# EKF 추정 (권장)
CAR_NAME=SIM roslaunch controller ekf_mpc_sim.launch mu_source:=ekf

# Patch gt 직접 주입 (EKF 우회, sanity 용)
CAR_NAME=SIM roslaunch controller ekf_mpc_sim.launch mu_source:=ground_truth

# Applier 끄기 (MPC planning 만 관찰)
CAR_NAME=SIM roslaunch controller ekf_mpc_sim.launch mu_source:=ekf apply_mu_to_sim:=false

# 정적 baseline
CAR_NAME=SIM roslaunch controller ekf_mpc_sim.launch mu_source:=static
```

---

## 5. 토픽 레퍼런스

### Pub / Sub 구조

```
[ekf_mu_patch_publisher]
  sub:  /global_waypoints, /car_state/odom_frenet
  pub:  /mu_ground_truth (Float32), /mu_patches/markers (MarkerArray)

[mu_gt_relay]  (ground_truth 모드)
  /mu_ground_truth → /ekf_mpc/mu_estimate

[mu_estimator_ekf]  (ekf 모드)
  sub:  /car_state/odom, /car_state/odom_frenet, /ekf_mpc/cmd_raw
  pub:  /ekf_mpc/mu_estimate, /ekf_mpc/mu_sigma,
        /ekf_mpc/ekf_K, /ekf_mpc/ekf_innov, /ekf_mpc/ekf_long_active

[ekf_mpc_controller]
  sub:  /car_state/{odom,pose,odom_frenet,imu}, /global_waypoints_scaled,
        /ekf_mpc/mu_estimate, /ekf_mpc/mu_adapt_enable
  pub:  /ekf_mpc/cmd_raw, /ekf_mpc/mu_used,
        /ekf_mpc/prediction, /ekf_mpc/reference, /ekf_mpc/solve_ms

[ekf_mu_applier]
  sub:  /ekf_mpc/cmd_raw, /mu_ground_truth, /ekf_mpc/mu_used
  pub:  /vesc/high_level/ackermann_cmd_mux/input/nav_1,
        /ekf_mpc/cmd_scaled_debug, /ekf_mpc/slip_indicator

[ekf_mu_hud]
  sub:  /car_state/pose, /ekf_mpc/mu_used, /mu_ground_truth
  pub:  /mu_hud/markers

[ekf_mu_toggle_gui]
  sub:  /car_state/odom, /vesc/.../nav_1, /mu_ground_truth, /ekf_mpc/mu_used
  pub:  /ekf_mpc/mu_adapt_enable (Bool, latched)
```

### 주요 토픽 의미

| 토픽 | 타입 | 의미 |
|---|---|---|
| `/mu_ground_truth` | Float32 | 차량 현재 (s, d) 의 patch μ (yaml 기반). patch 외면 `default_mu=0.85`. |
| `/mu_patches/markers` | MarkerArray | rviz raceline 위 색 밴드 + μ 라벨 |
| `/ekf_mpc/mu_estimate` | Float32 | MPC 가 구독하는 **최종 μ 입력**. mu_source 에 따라 estimator 가 publish |
| `/ekf_mpc/mu_used` | Float32 | MPC OCP 에 실제 주입된 μ. `mu_adapt_enable=False` 면 `mu_default` |
| `/ekf_mpc/mu_sigma` | Float32 | EKF 추정 √P (확신도) |
| `/ekf_mpc/ekf_K` | Float32 | Jacobian H=K. 절댓값 작으면 업데이트 skip |
| `/ekf_mpc/ekf_innov` | Float32 | Innovation `a_y − μ̂·K` |
| `/ekf_mpc/ekf_long_active` | Float32 | Longitudinal channel 업데이트 여부 (0/1) |
| `/ekf_mpc/mu_adapt_enable` | Bool latched | 대시보드 토글. False 면 MPC 는 μ_default 사용 |
| `/ekf_mpc/cmd_raw` | AckermannDriveStamped | MPC 원출력 (applier 이전) |
| `/ekf_mpc/cmd_scaled_debug` | Float32 | applier scaling ratio |
| `/ekf_mpc/slip_indicator` | Float32 | MPC 과신 시 slip 주입 강도 |
| `/ekf_mpc/prediction` | MarkerArray | 녹색 horizon 궤적 (rviz) |
| `/ekf_mpc/reference` | MarkerArray | 주황 raceline window (rviz) |
| `/mu_hud/markers` | MarkerArray | 차 위 gt/est 구 + 텍스트 + 불일치 badge |

### 토픽 활용 레시피

**추정 성능 실시간 모니터링**
```bash
# HUD 텍스트 체크 (gt=X est=Y Δ)
rostopic echo /mu_hud/markers | grep text
# 최근 mu_estimate stream
rostopic echo /ekf_mpc/mu_estimate | head -30
```

**Adaptation 토픽으로 토글** (GUI 없이)
```bash
rostopic pub -1 /ekf_mpc/mu_adapt_enable std_msgs/Bool "data: false"
rostopic pub -1 /ekf_mpc/mu_adapt_enable std_msgs/Bool "data: true"
```

**EKF 업데이트 확인**
```bash
# K 값 (코너 진입 시 증가)
rostopic echo /ekf_mpc/ekf_K
# Long channel 활성화 시점
rostopic echo /ekf_mpc/ekf_long_active | grep "1.0"
```

**μ applier scaling 추적**
```bash
rostopic echo /ekf_mpc/cmd_scaled_debug   # ratio (low-μ 에서 <1)
rostopic echo /ekf_mpc/slip_indicator     # MPC 과신 시 >0
```

---

## 6. 대시보드 GUI (좌상단 창, always-on-top)

- **`vx / cmd`**: 실제 속도 / MPC 명령 속도 (초록)
- **`gt / est`**: patch gt / MPC 주입 μ (Δ>0.08 이면 빨강)
- **상태 라벨**: `ENABLED`/`DISABLED`
- **토글 버튼**: `/ekf_mpc/mu_adapt_enable` publish

### Toggle OFF 효과
MPC 가 `mu_default=0.85` 고정 → patch gt 와 불일치 → `ekf_mu_applier` 가 slip 주입 → 차량 조향 노이즈 + 속도 감쇄 (2D sim 상 "가상 슬립" 시각화).

---

## 7. μ Patch 포맷 (`config/mu_patches_<map>.yaml`)

```yaml
patches:
  - name: icy_long_straight
    s_start: 6.0
    s_end:   14.0
    d_min:  -0.9
    d_max:   0.9
    mu:      0.40
    color:   [0.3, 0.6, 1.0, 0.55]
```

- `s_start > s_end` 시 wrap-around 자동 처리
- 바깥 구역: `ekf_mu_patch_publisher` 의 `~default_mu` (0.85)
- 여러 patch overlap 시 yaml 순서상 먼저 나오는 것 우선

### 맵 f 현 패치

| 이름 | s | μ | 특성 |
|---|---|---|---|
| icy_long_straight | 6~14 | 0.40 | 저마찰 직선 |
| grippy_top_straight | 30~37 | 1.20 | 고마찰 직선 |
| grippy_right_corner | 42~48 | 1.20 | 고마찰 우회전 |
| icy_hairpin | 55~62 | 0.40 | 저마찰 헤어핀 |

---

## 8. EKF 파라미터 (`config/ekf_mpc_srx1.yaml`)

```yaml
ekf_mpc:
  ekf:
    # Kalman
    init_mu:        0.85
    init_sigma:     0.30
    proc_sigma:     0.03      # Q. 작을수록 안정, 클수록 반응 빠름
    meas_sigma:     0.50      # R_lat. 크면 업데이트 per-pull 완화
    min_speed:      1.0       # v_x ≤ 이값: skip
    min_K:          0.7       # |K| ≤ 이값: skip (no-info)
    mu_min:         0.2
    mu_max:         1.3
    vy_smooth_alpha: 0.35     # dvy/dt LPF
    prior_pull_rate: 0.005    # memory 없고 K 작을 때 init 으로 서서히

    # Level 2a: Longitudinal channel
    long_enable:        true
    long_ax_thresh:     1.5   # |u_ax| 가 이값 이상일 때 check
    long_slip_thresh:   0.5   # |u_ax|-|a_x_meas| 이 이값 이상 slip 간주
    long_meas_sigma:    0.45

    # Level 2b: s-memory prior
    mem_enable:         true
    mem_bin_width:      2.0   # 2m bin
    mem_ema_alpha:      0.15  # 기존 memory 와 새 값 EMA
    mem_pull_rate:      0.015 # K-skip 시 memory 로 pull 강도
```

---

## 9. 현재 성능 (맵 f, 45초 × 평균 1 랩)

튜닝 단계별 patch 별 \|err\| = |mean(est) − gt|:

| Patch | gt | Init (L0) | L1 | L2.1 | L2.2 | **L2.3 (current)** |
|---|---|---|---|---|---|---|
| icy_long_straight | 0.40 | 0.56 | 0.40 | 0.24 | 0.27 | **0.19** ✓ |
| grippy_top_straight | 1.20 | 0.20 | 0.14 | 0.16 | 0.17 | **0.10** ✓ |
| grippy_right_corner | 1.20 | 0.36 | 0.32 | 0.28 | 0.31 | **0.34** |
| icy_hairpin | 0.40 | 0.20 | 0.10 | 0.18 | 0.18 | **0.17** |
| normal | 0.85 | 0.31 | 0.19 | 0.29 | 0.27 | **0.23** |

**해석**:
- icy_long 성공 (0.56 → 0.19) — longitudinal channel 기여
- grippy_top 고정확도 (mean 1.15 vs gt 1.20)
- grippy_corner 여전히 bias 0.34 — closed-loop feedback loop 기인 구조적 한계

---

## 10. 구조적 한계

1. **Closed-loop μ feedback**: MPC 가 μ̂ 낮게 믿음 → 보수적 planning → vx·ω 작음 → ay 작음 → μ̂ 더 낮게 확인. Positive feedback 로 under-confident bias 생성.

2. **직선 구간 관측성**: 직진 시 K≈0, long channel fire 가 있어도 간헐적. s-memory 로 완화하나 첫 방문에는 불가피.

3. **2D sim 물리 고정**: f1tenth_simulator 는 friction_coeff=1.0 고정. `ekf_mu_applier` 가 cmd scaling 으로 가상 마찰 효과. EKF 가 학습하는 것은 엄밀히는 "patch × applier 합성 효과" 이지 pure tire μ 아님.

4. **단일 스칼라 추정**: μ_f, μ_r 분리 아님. 실차처럼 비대칭 grip (예: 한쪽 바퀴만 젖은 경우) 표현 불가.

**개선 방향 (미구현)**:
- μ_f, μ_r 분리 2-state EKF
- Excitation injection (warmup 시 steer sine sweep)
- `gp` 모드 실제 GP 학습 파이프라인
- 실차 IMU + wheel-slip 측정으로 관측 채널 증가

---

## 11. 관련 패키지 비교

| 패키지 | μ 처리 | patch 식별 |
|---|---|---|
| `controller/mpc` | state_machine 경유 local_wpnts 추종, μ 는 MPC 내부 정적 | N/A |
| `controller/mpc_only` | global raceline 직접 추종, μ 정적 | N/A |
| `controller/rls_mpc` | peak-ay RLS | **실패** (모든 patch 유사) |
| **`controller/ekf_mpc`** | **Pacejka EKF + long + s-memory** | **성공** (grippy/icy/normal 분리 가능) |

---

## 12. 주의사항

- drive 토픽 `/vesc/high_level/ackermann_cmd_mux/input/nav_1` 공유. `controller/mpc`, `mpc_only`, `rls_mpc` 와 **동시 실행 금지**.
- `ekf_` prefix 스크립트 이름 은 `rls_mpc` 와 ROS 중복 해결용. `rosrun controller mu_hud.py` 같이 접두사 없는 name 으로 실행하면 임의 픽업됨.
- EKF 추정값 의 절대 정확도 보다 **patch 간 상대 변화** 를 중시 (RLS 대비 patch 구분 되는지).
