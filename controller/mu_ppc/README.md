# `mu_ppc` — Friction-aware Pure Pursuit Controller

ICRA2026 RoboRacer 트랙처럼 마찰력이 구간마다 다른 환경에서 쓰는
**Pure Pursuit (PPC)** 변형. 기존 race_stack 의 `combined.Controller` 와
같은 인터페이스로 꽂을 수 있고, 단독 노드로도 돌릴 수 있다.

> **설계 원칙 한 줄**:
> "지도(prior)는 *조심해* 라는 귓속말, 실제 판단은 *차가 직접 미끄러짐을 느끼고* 한다.
> 속도 프로파일은 절대 건드리지 않고, 컨트롤러측 신호(steer / accel / jerk)만 적응시킨다."

---

## 1. 왜 만들었나

대회 트랙 중간에 **저μ 구간 / 고μ 구간** 이 섞여 있다는 정보가 사전에 주어진다.
순수한 PPC 한 세트의 게인으로는:

- 저μ 구간에서 **너무 빠르게 핸들·가속 명령** → 슬립
- 고μ 구간에서 **공격적으로 못 가서 손해**

그렇다고 sector 별 속도 프로파일을 조정하는 건 우리 워크플로 규칙상 금지
(이미 차량 한계 반영된 프로파일이라 손대면 다른 곳 깨짐). 그래서:

- **속도 명령은 그대로 둔다** (`v_ref` = local waypoints `vx_mps`)
- 대신 **컨트롤러 게인**을 마찰 상태에 따라 실시간으로 조여/풀어준다

게인을 움직이는 입력은 두 종류:

1. **Prior** — 트랙 μ-zone 맵 (Frenet `s` 기반, 사전에 라벨링)
2. **Measurement** — 슬립 추정기 (yaw rate, 휠 슬립, 횡G 잔차)

prior 는 코너 *진입 직전* 안전 마진 주려고, measurement 는 진입 후 *실제 그립*에 맞춰
풀어주려고 쓴다. 두 신호를 비대칭 시상수로 융합 — slip 쪽으로 빠르게 조이고
grip 쪽으로 천천히 푼다.

---

## 2. 폴더 구조

```
controller/mu_ppc/
├── __init__.py
├── README.md                  ← 이 파일
├── src/
│   ├── __init__.py
│   ├── mu_zone_map.py         μ-zone Frenet-s 인덱싱 + 엣지 블렌딩
│   ├── slip_estimator.py      3가지 슬립 잔차 융합 (LP filter, asym tau)
│   ├── gain_scheduler.py      prior + 측정 → Ld·steer rate·a_x 캡
│   ├── MuPPC.py               메인 컨트롤러 클래스 (main_loop)
│   └── mu_ppc_node.py         standalone ROS 노드
├── launch/
│   └── mu_ppc.launch          standalone 노드 launch
├── config/
│   └── mu_ppc.yaml            모든 파라미터
└── cfg/
    └── MuPPC.cfg              dynamic_reconfigure 정의 (rqt 튜닝용)
```

---

## 3. 데이터 흐름

```
                ┌────────────────────────────────────────────────┐
 Frenet s ─────►│  MuZoneMap.mu_ahead(s, lookahead)              │
                │   • zones from /friction_map_params/*          │─► mu_prior  (~0.4..1.2)
                │   • cosine ramp at edges                       │
                └────────────────────────────────────────────────┘

   yaw_rate    ┐
   v_body      │   ┌────────────────────────────────────────────┐
   v_wheel     │──►│  SlipEstimator.update(...)                 │─► slip_lvl  (-1..+1)
   ay_meas     │   │   • 3 residuals (yaw / wheel / ay)         │   +1 = slipping
   steer_cmd   ┘   │   • soft sat + LP, asym tau                │   -1 = grip excess
                   └────────────────────────────────────────────┘

mu_prior + slip_lvl
                   ┌────────────────────────────────────────────┐
                   │  GainScheduler.update(...)                 │   ld_scale       (0.8..1.6)
                   │   • alpha = blend(prior, measurement)      │─► steer_rate_lim (rad/s)
                   │   • slow relax / fast tighten              │   ax_max         (m/s²)
                   └────────────────────────────────────────────┘   ax_min         (m/s²)

position, waypoints, gains
                   ┌────────────────────────────────────────────┐
                   │  MuPPC.main_loop(...)                      │
                   │   1. baseline Ld(v) = q + m*v              │─► (speed, accel,
                   │   2. Ld *= ld_scale                        │   jerk, steer,
                   │   3. find lookahead pt → atan(2 L y / Ld²) │   L1 marker, ...)
                   │   4. clip steer rate to steer_rate_lim     │
                   │   5. clip ax_ref to [ax_min, ax_max]       │
                   │   6. publish v_ref unchanged               │
                   └────────────────────────────────────────────┘
```

---

## 4. 모듈 상세

### 4.1 `mu_zone_map.py` — `MuZoneMap`

Frenet `s` (예: 트랙 시작점부터의 호 길이) 를 입력받아 **유효 마찰계수**를 돌려주는
사전 지도. zone 데이터는 race_stack 의 `friction_sector_tuner` 가 띄우는
`/friction_map_params/*` rosparam 그대로 재사용한다.

```yaml
/friction_map_params/n_sectors: 3
/friction_map_params/global_friction_limit: 1.0
/friction_map_params/Sector0/s_start: 12.5
/friction_map_params/Sector0/s_end:   18.0
/friction_map_params/Sector0/friction: 0.55
...
```

**엣지 블렌딩**: 각 zone 경계 ±`mu_zone_edge_blend` (default 2 m) 구간에서
cosine 램프로 default(=1.0) 와 zone 값 사이를 부드럽게 보간 → 게인 점프 방지.

**`mu_ahead(s, lookahead)`**: `[s, s + lookahead]` 8개 샘플 중 **가장 작은 μ** 반환.
`MuPPC` 에서는 `lookahead = 1.5 * Ld_base` 로 호출 → 코너 진입 *전에* 저μ를 인지.

```python
m = MuZoneMap(track_length=50.0, edge_blend=2.0)
m.set_zones([(10, 20, 0.6), (35, 42, 0.5)])
m.mu_at(15)   # 0.6  (zone 중심)
m.mu_at(11)   # ~0.83 (엣지 블렌딩)
m.mu_at(5)    # 1.0  (zone 밖)
m.mu_ahead(8, lookahead=4)   # 0.6 (4 m 앞에 zone)
```

### 4.2 `slip_estimator.py` — `SlipEstimator`

세 개의 **무차원 잔차**를 만든 뒤 융합한다.

| 잔차 | 식 | 의미 |
|---|---|---|
| `r_yaw`  | `(\|ω_meas\| − \|ω_cmd\|)/\|ω_cmd\|` (부호반전) | 명령보다 덜 돌면 언더스티어 → 양수 |
| `r_long` | `(v_wheel − v_body) / v_wheel` | 구동축 헛돌면 양수 |
| `r_ay`   | `(\|ay_exp\| − \|ay_meas\|) / \|ay_exp\|`     | 모델보다 덜 받으면 슬립 → 양수 |

여기서:
- `ω_cmd = v · tan(δ_cmd) / L` (자전거 모델)
- `ay_exp = v · ω_cmd`
- `δ_cmd` 는 **이전 사이클**의 publish 한 steer (실제 차량 명령)
- `v_wheel` 은 `/vesc/odom` linear.x, `v_body` 는 `/car_state/odom` linear.x

각 잔차를 자기 임계값으로 나눠 단위 슬립 척도로 만들고 ±1.5 saturate.

**융합**: slip 신호 (`max`) 가 양수면 그대로, 아니면 grip 신호 (`min`) 사용.
즉 셋 중 **하나라도** 슬립을 가리키면 그게 우선.

**비대칭 LP**:
- `tau_up` (default 0.05 s) — 슬립이 *증가*할 때 (위험 인지) 빠르게 추적
- `tau_down` (default 0.30 s) — 회복 시 천천히

저속(`v < v_min = 0.8 m/s`)에서는 분모가 작아져 노이즈가 폭주하므로 강제로 0 출력.

### 4.3 `gain_scheduler.py` — `GainScheduler`

prior(`mu_prior`)와 측정(`slip_lvl`) 을 alpha ∈ [0, 1] 로 환산해 **하나의 슬라이더**로
만든다. alpha = 1 이면 그립 좋음, 0 이면 미끄러짐.

```
a_prior = clip( (mu_prior − 0.4) / (1.2 − 0.4), 0, 1 )
a_meas  = clip( 0.5 − 0.5 * slip_lvl,           0, 1 )

# 보수적 융합: 슬립이 감지되면 측정이 무조건 우선
alpha   = min( w*a_prior + (1−w)*a_meas,
               a_meas if slip_lvl > 0 else a_prior )
```

(`w = prior_weight`, default 0.5)

이 alpha 로 4개 출력을 선형 보간:

| 출력 | low (slip side) | high (grip side) |
|---|---|---|
| `ld_scale` | 1.6 (lookahead 길게 → 핸들 부드럽게) | 0.8 (짧게 → 타이트) |
| `steer_rate_lim` [rad/s] | 2.5 | 8.0 |
| `ax_max` [m/s²] | 2.0 | 6.0 |
| `\|ax_min\|` [m/s²] | 2.5 | 7.0 |

각 출력은 1차 LP 필터링되고, **방향에 따라 다른 시상수** 적용:
- 조이는 방향 (`tau_tighten = 0.08 s`) — 안전쪽으로 빠르게
- 푸는 방향 (`tau_relax = 0.6 s`) — 그립 확신 후 천천히

### 4.4 `MuPPC.py` — `MuPPC`

`combined.Controller` 의 `main_loop` 와 같은 시그니처:

```python
speed, accel, jerk, steer, L1_point, L1_dist, idx_nearest, kappa, future_pos = \
    MuPPC.main_loop(state, position_in_map, waypoint_array, speed_now,
                    opponent, position_in_map_frenet, acc_now, track_length)
```

내부 흐름:

1. baseline Ld: `Ld_base = clip(q + m·v, ld_base_min, ld_base_max)`
2. **prior**: `mu_ahead(s_now, 1.5·Ld_base)`
3. **measurement**: 직전 publish 한 `δ_cmd` 와 현재 IMU/odom/VESC odom 으로 slip lvl 갱신
4. **gain**: scheduler → `ld_scale, steer_rate_lim, ax_max, ax_min`
5. `Ld = Ld_base · ld_scale`, lookahead point 찾기
6. PP 공식 (body frame): `δ = atan(2 L · y_b / Ld²)`
7. steer **rate limit**: `δ ← clip(δ, δ_prev ± steer_rate_lim · dt)`
8. 속도 명령 `v_ref` 그대로, **accel 만 [ax_min, ax_max] 로 클립**
9. 다음 cycle 의 slip 추정에 쓰려고 `δ_cmd` 저장

> **중요**: `v_ref` 는 *waypoints 의 vx_mps* 그대로 publish. 우리 stack 의 속도
> 프로파일은 차량 한계 기준이라 위로는 절대 안 올린다. 고μ 구간에서 "더 빨리"
> 라는 건 *프로파일 속도를 놓치지 않고 정확히 따라간다* 는 뜻 — `ax_max` 캡이
> 풀려서 가속 망설임이 없어지는 식으로 표현된다.

### 4.5 `mu_ppc_node.py` — `MuPPCNode`

ROS 래퍼. controller_manager 와 같은 토픽을 구독해서 `MuPPC.main_loop` 을 50 Hz 로
돌리고 Ackermann 명령 publish.

#### 구독
| 토픽 | 타입 | 용도 |
|---|---|---|
| `/behavior_strategy` | `f110_msgs/BehaviorStrategy` | local waypoints + state |
| `/car_state/odom` | `nav_msgs/Odometry` | `v_body` |
| `/car_state/pose` | `geometry_msgs/PoseStamped` | (x, y, θ) |
| `/car_state/odom_frenet` | `nav_msgs/Odometry` | `s` |
| `/imu/data` | `sensor_msgs/Imu` | yaw rate, ay |
| `/vesc/odom` | `nav_msgs/Odometry` | `v_wheel` |

#### 발행
| 토픽 | 타입 | 용도 |
|---|---|---|
| `/vesc/high_level/ackermann_cmd_mux/input/nav_1` | `AckermannDriveStamped` | 차량 명령 |
| `/mu_ppc/lookahead` | `Marker` | RViz lookahead 점 |
| `/mu_ppc/alpha` | `Float32` | 현재 ld_scale (튜닝 모니터링) |
| `/mu_ppc/debug` | `Point` | (slip_lvl, ax_max, steer_rate) |

---

## 5. 빌드

이미 `controller/CMakeLists.txt` 의 `generate_dynamic_reconfigure_options` 에
`mu_ppc/cfg/MuPPC.cfg` 가 등록되어 있다. 컨테이너 안에서 평소처럼:

```bash
docker exec icra2026 bash -c "
  source /opt/ros/noetic/setup.bash &&
  source /home/hmcl/catkin_ws/devel/setup.bash &&
  cd /home/hmcl/catkin_ws &&
  catkin build controller
"
```

확인:
```
devel/include/controller/MuPPCConfig.h
devel/lib/python3/dist-packages/controller/cfg/MuPPCConfig.py
```

---

## 6. 실행

### 6.1 standalone

전제: base_system / state_machine / friction_sector_tuner 가 이미 떠 있어서
`/behavior_strategy`, `/car_state/*`, `/friction_map_params/*` 등이 publish 되는
상태.

```bash
roslaunch controller mu_ppc.launch
# 옵션:
roslaunch controller mu_ppc.launch \
    racecar_version:=UNICORN2 \
    drive_topic:=/vesc/high_level/ackermann_cmd_mux/input/nav_1 \
    rate:=50 \
    config:=$(rospack find controller)/mu_ppc/config/mu_ppc.yaml
```

기존 `controller_manager` 와 토픽이 충돌하므로 **둘 중 하나만** 띄울 것.
headtohead.launch 와 함께 쓰려면 거기서 `controller_manager` 노드 부분을 주석
처리하거나 `mu_ppc_standalone` 변형 launch 를 따로 만든다.

### 6.2 controller_manager 통합 (TODO)

`combined/Controller.py` 의 `__init__` 에서 `ctrl_algo == 'MU_PPC'` 분기를 추가하고
`main_loop` 안에서 `MuPPC.main_loop` 로 위임하면 된다. 본 패키지는 *분기에
얹기 좋게* main_loop 시그니처와 반환값을 그대로 맞춰뒀다.

---

## 7. 파라미터 사전

| 파라미터 | 기본값 | 의미 |
|---|---:|---|
| `rate` | 50 | 제어 주기 [Hz] |
| `ld_base_min` / `ld_base_max` | 0.6 / 3.0 | baseline lookahead 범위 [m] |
| `ld_speed_slope` / `ld_speed_intercept` | 0.30 / 0.20 | `Ld_base = q + m·v` |
| `max_steer` | 0.42 | 조향 절대 한계 [rad] |
| `mu_zone_edge_blend` | 2.0 | zone 경계 cosine 램프 폭 [m] |
| `mu_default` | 1.0 | zone 밖 기본 μ |
| `prior_weight` | 0.5 | 융합 시 prior 가중 (1=prior만, 0=측정만) |
| `slip_yaw_thr` | 0.15 | yaw 잔차 임계 (단위 슬립 = 잔차 / 임계) |
| `slip_long_thr` | 0.10 | 휠 슬립비 임계 |
| `slip_ay_thr` | 0.20 | 횡G 잔차 임계 |
| `slip_tau_up` / `slip_tau_down` | 0.05 / 0.30 | 슬립 LP 비대칭 시상수 [s] |
| `ld_scale_low` / `ld_scale_high` | 1.6 / 0.8 | Ld 배율 (slip / grip 끝점) |
| `steer_rate_low` / `steer_rate_high` | 2.5 / 8.0 | 조향 rate 캡 [rad/s] |
| `ax_max_low` / `ax_max_high` | 2.0 / 6.0 | 종방향 가속 캡 [m/s²] |
| `brake_low` / `brake_high` | 2.5 / 7.0 | \|ax_min\| 캡 [m/s²] |
| `gain_tau_tighten` / `gain_tau_relax` | 0.08 / 0.6 | 게인 LP 비대칭 시상수 [s] |

---

## 8. 튜닝 절차 (권장 순서)

> "한 번에 한 노브씩, 슬립 측정과 prior 를 분리해서 본다."

1. **base PPC 동작 먼저 검증** (zone / slip 둘 다 OFF 상태)
   - `prior_weight = 1.0`, `mu_default = 1.0`, friction_map_params 비움
   - `slip_*_thr` 매우 크게 (예: 100) → slip lvl 항상 0
   - `ld_speed_slope`, `ld_speed_intercept` 로 lap time 맞추기
   - `ld_scale_low = ld_scale_high = 1.0` 으로 두면 game inactive

2. **prior only** — 트랙 라벨 정확도 확인
   - `friction_map_params` 채우기, `prior_weight = 1.0`
   - slip 임계는 여전히 크게 → 측정 영향 0
   - 저μ zone 진입 직전에 `ld_scale` 이 커지는지 RViz / `/mu_ppc/alpha` 로 확인
   - zone 경계에서 라인이 흔들리면 `mu_zone_edge_blend` 키움

3. **measurement only** — 슬립 검출 보정
   - `friction_map_params` 비우거나 `prior_weight = 0`
   - 일부러 저μ 구간에서 약간 빠르게 진입시켜 슬립 발생
   - 셋 중 어떤 잔차가 먼저 트리거되는지 `/mu_ppc/debug` 로 보고
     `slip_*_thr` 균형 맞춤
   - 회복 시 차가 멈칫거리면 `slip_tau_down` 줄임

4. **융합 + 비대칭 시상수**
   - `prior_weight = 0.5` 가 일반적으로 좋음
   - 진입 직전 살짝 조여지고, 벗어나자마자 풀리면 OK
   - 진입 전부터 너무 겁먹으면 `gain_tau_relax` 줄이거나 `prior_weight` 낮춤
   - 슬립 후에도 천천히 풀리면 `gain_tau_relax` 늘림

5. **steer rate / accel 캡 최종 튜닝**
   - 풀 grip 구간에서 조향이 부드럽기만 하면 `steer_rate_high` 키움
   - 가속에서 wheel spin 보이면 `ax_max_high` 줄임

---

## 9. 디버깅

### 9.1 RViz
- `/mu_ppc/lookahead` Marker — 현재 lookahead point. 라인을 따라 이동해야 함.
- `/lookahead_point` (controller_manager 가 켜져 있다면 충돌 체크용)

### 9.2 토픽
```bash
rostopic echo /mu_ppc/alpha          # 현재 ld_scale (1.0 = baseline)
rostopic echo /mu_ppc/debug          # x=slip_lvl, y=ax_max, z=steer_rate
rostopic hz /mu_ppc/lookahead        # 50 Hz 나와야 정상
```

### 9.3 흔한 증상

| 증상 | 가능 원인 | 해결 |
|---|---|---|
| 출발 안 함 | `/behavior_strategy` 없음 | base_system + state_machine 켜기 |
| `mu_ppc/alpha` 가 항상 1.0 | zone 비어있고 slip 임계 큼 | `friction_map_params` 또는 `slip_*_thr` 점검 |
| 코너에서 outside 로 흘러감 | `ld_scale_low` 가 너무 큼 → 핸들 너무 부드러움 | `ld_scale_low` 줄이고 `steer_rate_low` 키움 |
| 직선에서도 `slip_lvl > 0` | yaw 임계가 너무 작음 | `slip_yaw_thr` 키움 |
| 게인이 너무 늦게 풀림 | `gain_tau_relax` 큼 | 줄임 |
| 게인이 너무 빨리 풀려 슬립 재발생 | 위 반대 | 키움 |

### 9.4 단위 테스트 (no ROS)

```bash
docker exec icra2026 bash -c "cd /home/hmcl/catkin_ws/src/race_stack/controller && python3 -c '
from mu_ppc.src.mu_zone_map import MuZoneMap
from mu_ppc.src.slip_estimator import SlipEstimator
from mu_ppc.src.gain_scheduler import GainScheduler

m = MuZoneMap(track_length=50.0); m.set_zones([(10,20,0.6)])
print(\"mu @ 5  :\", round(m.mu_at(5),3))
print(\"mu @ 15 :\", round(m.mu_at(15),3))

s = SlipEstimator()
print(\"slip lvl:\", round(s.update(0.0, 3.0, 3.0, 0.0, 0.0, 0.0),3))

g = GainScheduler()
print(\"gains  :\", g.update(0.0, 1.0, 0.0))
'"
```

---

## 10. 한계 / TODO

- **검출 지연**: 슬립을 측정으로 감지하려면 미끄러져야 한다. 진입 직전엔 prior 에 의존
  → zone 라벨이 너무 헐겁거나 빠지면 사고. zone 정확도가 가장 중요하다.
- **저속 영역 (`v < 0.8 m/s`)** 에서는 slip 신호 강제 0. 출발 직후엔 prior 만 작동.
- **GLIL pitch / vy 보정 없음**. 3D 트랙에서는 `/car_state/pitch` 를 받아 ay 보정 필요할 수
  있음 (현재 IMU 의 −y 직접 사용). 향후 확장 포인트.
- **타이어 모델 사용 안 함**. 일부러 model-free 로 잡았다. SQP/MPC 와 같은 μ 맵을 공유
  하면서 더 정교한 PPC 가 필요하면 `slip_estimator` 를 Pacejka 잔차로 바꾸는 게 다음 단계.
- **controller_manager 통합 분기 미작성**. 현재는 standalone 노드만 동작. 통합은
  `combined/Controller.py` 에 `MU_PPC` 분기 추가하는 한 PR 로 마무리 가능.

---

## 11. 라이선스 / 주의사항

- race_stack 의 다른 부분과 동일 라이선스.
- 본 모듈은 *컨트롤러측 적응만* 수행한다. **속도 프로파일을 절대 수정하지 않는다**
  (프로젝트 규칙). raceline / sector 속도를 바꾸고 싶으면 `sector_tuner` 쪽에서
  처리.
