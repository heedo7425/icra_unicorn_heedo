# `controller/multisurface_mpc`

2D f1tenth sim 에서 **multi-friction 노면 대응 MPC** 실험용 **독립 패키지**. `controller/mpc`, `controller/mpc_only` 와 물리적 격리 (별도 rosparam 네임스페이스 `/mpc_ms/*`, 별도 codegen dir `/tmp/mpc_ms_c_generated`).

---

## 1. 개요

기본 MPC (`controller/mpc_only`) 는 μ 를 **정적 상수** 로 가정. 본 패키지는:

- **μ 패치**: 맵 raceline 에 s-range 기반 영역별 μ 정의 (yaml)
- **μ 추정기** (3 모드 switchable):
  - `static` — 추정 안 함, `mu_default` 고정
  - `ground_truth` — 패치 퍼블리셔 출력 직접 주입 (sanity check 용)
  - `rls` — IMU-less 환경용 peak-tracking RLS (body-frame ay proxy = vx·ω)
  - `gp` — GP ensemble stub (사전훈련 pkl 없으면 MA fallback)
- **μ applier**: 2D sim 의 고정 friction 물리를 우회해 cmd speed/steer 를 μ 비율로 스케일링 + MPC 과신 시 조향 노이즈 주입
- **HUD**: rviz 에서 gt vs est μ 실시간 비교, 불일치 시 빨간 배지
- **대시보드 GUI**: 좌상단 고정 tkinter 창, 속도/μ 표시 + adaptation 토글 버튼

---

## 2. 파일 트리

```
controller/multisurface_mpc/
├── __init__.py
├── mpc_ms_node.py                    # MPC 메인 노드 (런타임 μ 수신)
├── README.md                         # (본 파일)
├── config/
│   ├── mpc_ms_srx1.yaml              # /mpc_ms/* 파라미터
│   └── mu_patches_f.yaml             # 맵 f 의 μ 패치 정의 (s-range)
├── launch/
│   └── mpc_ms_sim.launch             # 통합 런치
├── rviz/
│   └── mpc_ms.rviz                   # 전용 rviz config
└── scripts/
    ├── mu_patch_publisher.py         # s-range → /mu_ground_truth + MarkerArray
    ├── mu_estimator_rls.py           # peak-tracking RLS
    ├── mu_estimator_gp.py            # GP ensemble (+ MA fallback)
    ├── mu_applier.py                 # cmd scaling + slip 주입
    ├── mu_hud.py                     # rviz gt/est 비교 시각화
    └── mu_toggle_gui.py              # tkinter 대시보드 + adapt on/off 토글
```

---

## 3. 실행

### 기본 형태
```bash
CAR_NAME=SIM roslaunch controller mpc_ms_sim.launch \
    map:=f racecar_version:=SIM rviz:=true mu_source:=<MODE>
```

`mu_source` 값:

| 값 | 동작 |
|---|---|
| `static` | MPC 항상 `mu_default=0.85` 사용. 추정기·릴레이 노드 기동 안 함. |
| `ground_truth` | `topic_tools/relay` 가 `/mu_ground_truth → /mpc_ms/mu_estimate` 포워딩. MPC 는 패치 gt 값 직접 사용 (sanity 테스트). |
| `rls` | `mu_estimator_rls` 노드 기동. odom 기반 peak-ay 로 μ̂ 추정. |
| `gp` | `mu_estimator_gp` 노드 기동. `/tmp/gp_ensemble_srx1.pkl` 존재 시 GP 앙상블, 없으면 moving-average fallback. |

### 런치 arg

| arg | 기본값 | 설명 |
|---|---|---|
| `map` | `f` | 맵 이름 (`stack_master/maps/<map>`) |
| `rviz` | `true` | rviz 기동 여부 |
| `mu_source` | `static` | 위 표 |
| `patches_file` | `$(find controller)/multisurface_mpc/config/mu_patches_$(arg map).yaml` | μ 패치 yaml 경로 |
| `apply_mu_to_sim` | `true` | `mu_applier` 활성. false 로 두면 MPC 원명령 그대로 sim 투입 (MPC planning 만 관찰). |
| `toggle_gui` | `true` | 대시보드 창 기동 여부 |

### 예시
```bash
# 정적 μ (baseline)
roslaunch controller mpc_ms_sim.launch mu_source:=static

# 패치 ground truth 주입 (릴레이만, 추정 없음)
roslaunch controller mpc_ms_sim.launch mu_source:=ground_truth

# RLS 온라인 추정 (peak ay 기반)
roslaunch controller mpc_ms_sim.launch mu_source:=rls

# GP stub (실제 학습 없이 MA fallback)
roslaunch controller mpc_ms_sim.launch mu_source:=gp

# applier 꺼서 MPC planning 만 확인
roslaunch controller mpc_ms_sim.launch mu_source:=rls apply_mu_to_sim:=false
```

---

## 4. 토픽 레퍼런스

### Pub / Sub 맵

```
[mu_patch_publisher]
    sub:  /global_waypoints, /global_waypoints_scaled, /car_state/odom_frenet
    pub:  /mu_ground_truth (Float32)
          /mu_patches/markers (MarkerArray)

[mu_gt_relay]  (ground_truth 모드만)
    sub:  /mu_ground_truth
    pub:  /mpc_ms/mu_estimate

[mu_estimator_rls]  (rls 모드만)
    sub:  /car_state/odom
    pub:  /mpc_ms/mu_estimate, /mpc_ms/mu_sigma, /mpc_ms/rls_ay_peak

[mu_estimator_gp]  (gp 모드만)
    sub:  /car_state/odom, /imu/data, /vesc/.../nav_1
    pub:  /mpc_ms/mu_estimate, /mpc_ms/gp_weights

[mpc_ms_controller]
    sub:  /car_state/odom, /pose, /odom_frenet, /imu/data,
          /global_waypoints_scaled, /global_waypoints,
          /mpc_ms/mu_estimate, /mpc_ms/mu_adapt_enable
    pub:  /mpc_ms/cmd_raw (AckermannDriveStamped)
          /mpc_ms/mu_used (Float32)
          /mpc_ms/prediction (MarkerArray, horizon)
          /mpc_ms/reference (MarkerArray, raceline window)
          /mpc_ms/solve_ms (Float32)

[mu_applier]
    sub:  /mpc_ms/cmd_raw, /mu_ground_truth, /mpc_ms/mu_used
    pub:  /vesc/high_level/ackermann_cmd_mux/input/nav_1
          /mpc_ms/cmd_scaled_debug (Float32, ratio)
          /mpc_ms/slip_indicator (Float32, 슬립 강도)

[mu_hud]
    sub:  /car_state/pose, /mpc_ms/mu_used, /mu_ground_truth
    pub:  /mu_hud/markers (MarkerArray)

[mu_toggle_gui]
    sub:  /car_state/odom, /vesc/.../nav_1, /mu_ground_truth, /mpc_ms/mu_used
    pub:  /mpc_ms/mu_adapt_enable (Bool, latched)
```

### 토픽별 의미·사용법

| 토픽 | 타입 | 발행자 | 의미 |
|---|---|---|---|
| `/mu_ground_truth` | Float32 | patch_publisher | 차량 현재 (s, d) 가 속한 패치 μ (없으면 `default_mu=0.85`) |
| `/mu_patches/markers` | MarkerArray | patch_publisher | rviz 용 raceline 위 색깔 밴드 + 패치 라벨 |
| `/mpc_ms/mu_estimate` | Float32 | 현 estimator (ground_truth/rls/gp) | MPC 가 구독하는 **최종 μ 입력**. source 에 따라 제공자 바뀜. |
| `/mpc_ms/mu_used` | Float32 | mpc_ms_controller | **MPC 가 실제 OCP 에 주입한 μ**. `mu_adapt_enable=False` 면 `mu_default`, 아니면 `mu_estimate`. 검증용. |
| `/mpc_ms/mu_sigma` | Float32 | mu_estimator_rls | RLS 분산 √P (확신도 역지표). |
| `/mpc_ms/rls_ay_peak` | Float32 | mu_estimator_rls | 최근 1 초 윈도우 내 peak\|ay\| (m/s²). threshold 이상이면 μ̂ 업데이트. |
| `/mpc_ms/gp_weights` | Float32 | mu_estimator_gp | GP ensemble 첫 surface 가중치 (stub 모드 = 1.0) |
| `/mpc_ms/mu_adapt_enable` | Bool latched | mu_toggle_gui | 대시보드 버튼 상태. False 시 MPC 는 추정 무시하고 `mu_default` 사용. |
| `/mpc_ms/cmd_raw` | AckermannDriveStamped | mpc_ms_controller | MPC 원 출력 (applier 거치기 전). 디버깅용. |
| `/mpc_ms/cmd_scaled_debug` | Float32 | mu_applier | applier 가 현재 걸고 있는 scaling ratio. |
| `/mpc_ms/slip_indicator` | Float32 | mu_applier | MPC belief > gt 불일치 강도. 0=정상, 클수록 슬립 효과 강해짐. |
| `/mpc_ms/prediction` | MarkerArray | mpc_ms_controller | 녹색 LINE_STRIP — 예측 horizon xy. |
| `/mpc_ms/reference` | MarkerArray | mpc_ms_controller | 주황 LINE_STRIP — 슬라이스된 raceline window. |
| `/mpc_ms/solve_ms` | Float32 | mpc_ms_controller | solve 시간 (ms). |
| `/mu_hud/markers` | MarkerArray | mu_hud | 차 위 gt/est 구 + 텍스트 + 불일치 badge. |

### 토픽 활용 레시피

**추정 성능 모니터링**:
```bash
# 현재 gt vs est 실시간 비교
rostopic echo /mu_ground_truth /mpc_ms/mu_used --filter 'm.data'
# 또는 HUD 한 번에
rostopic echo /mu_hud/markers | grep text
```

**MPC cmd scaling 추적**:
```bash
rostopic hz /mpc_ms/cmd_raw                       # MPC 원 cmd 주기
rostopic echo /mpc_ms/cmd_scaled_debug | grep data # applier ratio
rostopic echo /mpc_ms/slip_indicator | grep data   # slip 주입 강도
```

**adaptation 토글을 topic 으로 제어 (GUI 없이)**:
```bash
rostopic pub -1 /mpc_ms/mu_adapt_enable std_msgs/Bool "data: false"
rostopic pub -1 /mpc_ms/mu_adapt_enable std_msgs/Bool "data: true"
```

**RLS 업데이트 순간 포착**:
```bash
# ay_peak 가 threshold(default 2.0) 넘기면 그 다음 tick 에 μ̂ 업데이트됨
rostopic echo /mpc_ms/rls_ay_peak
```

**μ applier 끄기 (MPC planning 만 보고 싶을 때)**:
```bash
roslaunch controller mpc_ms_sim.launch mu_source:=rls apply_mu_to_sim:=false
# 또는 런타임에
rosparam set /mu_applier/enable false
# applier 재시작 필요
```

---

## 5. 대시보드 GUI (좌상단 창)

항상 보이는 (topmost) tkinter 창:

- **`vx / cmd`**: 실제 차 속도 / MPC 명령 속도. 명령이 클수록 `vx` 가 뒤쳐짐 = 동력 부족 또는 slip.
- **`gt / est`**: 패치 ground truth / 추정기 출력. 둘 차이가 `Δ>0.08` 이면 빨간색.
- **상태**: `ENABLED` (초록) 또는 `DISABLED` (빨강). 버튼으로 토글.
- **버튼**: `Disable μ adaptation` / `Enable μ adaptation` — `/mpc_ms/mu_adapt_enable` publish.

### 토글 OFF 시 동작
MPC 가 `mu_default=0.85` 고정 → 실제 gt 와 불일치 → `mu_applier` 가 `mu_used - mu_gt` 차이에 비례해 조향 노이즈 + 속도 추가 감쇄 주입 → 차량 비틀거림 (슬립 시뮬).

---

## 6. μ 패치 포맷 (`config/mu_patches_<map>.yaml`)

```yaml
patches:
  - name: icy_long_straight         # 식별자 (rviz 라벨)
    s_start: 6.0                    # Frenet s [m]
    s_end:   14.0
    d_min:  -0.9                    # 좌/우 경계 (raceline 수직 오프셋)
    d_max:   0.9
    mu:      0.40                   # 패치 마찰계수
    color:   [0.3, 0.6, 1.0, 0.55]  # rviz 색 RGBA
```

- `s_start > s_end` 이면 wrap-around 자동 처리
- patch 외 구역: `mu_patch_publisher` 의 `~default_mu` (기본 0.85)
- 여러 patch overlap 시 **yaml 순서상 먼저 나오는 것** 우선

### 현 맵 f 패치 (참고)

| 이름 | s 구간 | μ | 특성 |
|---|---|---|---|
| icy_long_straight | 6~14 | 0.40 | 장가속 직선 |
| grippy_top_straight | 30~37 | 1.20 | 상단 직선 |
| grippy_right_corner | 42~48 | 1.20 | 강한 우회전 |
| icy_hairpin | 55~62 | 0.40 | 헤어핀 |

---

## 7. 주요 파라미터 (`config/mpc_ms_srx1.yaml`)

```yaml
mpc_ms:
  N_horizon: 20            # 1 s preview @ dt=0.05
  v_max: 12.0              # clip 비활성 수준 (raceline max ≈ 11.3)
  mu_default: 0.85         # static 모드 or adaptation OFF 시 사용
  max_accel: 3.0           # SRX1 spec
  friction_circle: true    # OCP 소프트 제약
  friction_margin: 0.95
  warmup_vx_min: 0.2       # warmup 이탈 임계 속도
  warmup_speed_cmd: 2.0    # warmup 시 강제 speed

  mu_source: static        # launch arg 로 오버라이드
  mu_estimate_topic: /mpc_ms/mu_estimate

  rls:                     # mu_source=rls 때만 사용
    forgetting: 0.95
    init_mu: 0.85
    init_sigma: 0.18
    meas_sigma: 0.22
    ay_threshold: 2.0
    mu_min: 0.2
    mu_max: 1.3
    peak_window_sec: 1.0

  gp:                      # mu_source=gp 때만 사용
    ensemble_path: /tmp/gp_ensemble_srx1.pkl
    init_mu: 0.85
    blend_temperature: 5.0
    buffer_size: 50
```

---

## 8. 현재 알려진 한계

- **2D sim 은 실제 friction_coeff=1.0 고정**. 패치 μ 변화는 `mu_applier` cmd scaling 으로만 근사. 실제 물리 grip 은 변하지 않음 → 실차·Gazebo 수준 정확도 기대 불가.
- **단순 RLS from ay 단독으로는 patch 구분 어려움**. ay_peak 측정이 모든 patch 에서 유사하게 나옴 (~2-3 m/s²). 이는 estimator 버그가 아니라 *saturation 미발생 + 단일 관측 채널* 의 본질적 한계. 실제 적용에서는 slip-angle 측정 or GP ensemble 이 필요.
- **GP 모드는 stub** — 사전훈련 pkl 없으므로 moving-average fallback 동작. 실제 GP 훈련 파이프라인 (data collection + offline training) 은 추후.
- **`mu_source=ground_truth` 는 "답지 보기"** — HUD 의 gt=est 완전 일치가 `topic_tools/relay` 결과. 추정기 성능 평가 지표 아님.
- 같은 drive 토픽 (`/vesc/.../nav_1`) 공유 → `controller/mpc` 또는 `controller/mpc_only` 와 **동시 실행 금지**.

---

## 9. 후속 작업 후보

- 실제 GP ensemble 훈련 스크립트 (`scripts/collect_data.py`, `train_offline.py`)
- f1tenth_simulator 의 friction_coeff runtime 수정 → 물리 grip 진짜 변화시키기
- Slip-angle 기반 μ sensor 추가 (wheel speed vs vehicle kinematics 차이)
- Gazebo 3D `carpet_zones` plugin 통합 (실 phase 2 목표)
