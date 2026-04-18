# Sampling Planner × State Machine Integration

> **작성일:** 2026-04-18  
> **작성자:** HJ (Claude Opus 4.7 협업)  
> **목적:** 세션 인수인계. 다른 PC / 다음 세션에서 이 문서 하나만 열어도 맥락 복구하고 이어서 작업할 수 있게.  
> **상태:** Step 0~1 완료, Step 2 진행 중

---

## 0. 빠른 복귀 가이드 (다음 세션 첫 10분)

다음 세션 시작 시 이 순서대로 보면 됨:

1. 이 문서 전체 1회 읽기
2. `git status` 로 현재 작업 트리 상태 확인
3. 아래 **"6. 진행 상황 체크리스트"** 에서 어디까지 끝났는지 확인
4. **"7. 다음 세션이 바로 할 일"** 부터 이어서 작업

복원이 필요하면 백업 파일 위치:
- `state_machine/src/3d_state_machine_node_backup_20260418.py`
- `stack_master/launch/3d_headtohead_backup_20260418.launch`

---

## 1. 배경 & 목표

### 1.1 왜 이 작업을 하는가

현재 `planner/3d_sampling_based_planner` 는 **observation mode 전용**:
- `/car_state/odom` 만 구독, `~best_trajectory` (WpntArray) 만 발행 → 제어 파이프라인과 단절
- cost 가중치 YAML 은 init 시점 1회만 읽힘
  ([sampling_planner_node.py:78-80](../planner/3d_sampling_based_planner/node/sampling_planner_node.py#L78-L80))
- **tick 마다 argmin 선택 → 궤적이 왔다갔다 튀는 연속성 문제**

기존 state machine ([3d_state_machine_node.py](../state_machine/src/3d_state_machine_node.py)) 의 로컬 경로 합성 흐름:
- OVERTAKE state → `/planner/avoidance/otwpnts` (OTWpntArray) 를 `local_waypoints` 로 합성
- RECOVERY state → `/planner/recovery/wpnts` (WpntArray) 를 `local_waypoints` 로 합성
- state machine 은 "토픽 선택"이 아니라 **"local_waypoints 에 들어갈 경로를 state 별로 고른다"**
- 합성 함수: `_pub_local_wpnts()` ([3d_state_machine_node.py:1644 근방](../state_machine/src/3d_state_machine_node.py#L1644))

### 1.2 목표 (8개)

1. sampling planner 를 OVERTAKE / RECOVERY 두 역할로 주입 가능하게.
2. cost 가중치를 rqt / YAML / ROS 파라미터로 **런타임 튜닝 + 저장 + 초기화(reset)**.
3. **OVERTAKE/RECOVERY 2개 인스턴스 병렬 기동**, state machine 은 기존 로직대로 state 별 선택.
4. 기존 `3d_headtohead.launch` 에 `sampling_planner_enable` arg 추가 → true 면 sampling planner 켜지고 기존 spliner/sqp/recovery_spliner 비활성.
5. 샘플링 궤적의 **연속성/안정성 확보** (tick 간 진동 억제) + **local_waypoints 합성 시 이음새 매끄럽게** (frenet 등간격 resampling + raceline tail-blending).
6. **원본 파일 보존** — 수정 대상은 `_backup_` 복사 후 자유롭게 변경.
7. 디버깅 편의성 — Claude 가 직접 기동·검증 가능하게 상태/마커/로그 풍부.
8. 이 문서로 다음 세션 인수인계.

---

## 2. 설계 요약

### 2.1 결정 1: 원본 보존 = "복사 백업 후 자유 수정"

| 파일 | 처리 |
|---|---|
| `node/sampling_planner_node.py` (observe 전용) | **불변**, 그대로 둠 |
| `launch/sampling_planner_observe.launch` | **불변**, 그대로 둠 |
| `config/default.yaml` | **불변**, 그대로 둠 |
| `src/sampling_based_planner.py` (코어) | **불변**, import 로만 재사용 |
| `state_machine/src/3d_state_machine_node.py` | `_backup_20260418.py` 로 복사 후 **수정** (`### HJ :` 주석 래핑) |
| `stack_master/launch/3d_headtohead.launch` | `_backup_20260418.launch` 로 복사 후 **수정** (`<!-- ### HJ : -->`) |

### 2.2 결정 2: 1차 구현 = OVERTAKE + RECOVERY

| 인스턴스 이름 | state | 발행 토픽 | 타입 | 대체 대상 |
|---|---|---|---|---|
| `sampling_planner_ot` | OVERTAKE | `/planner/avoidance/otwpnts` | OTWpntArray | spliner / sqp_planner / lane_change_planner / spliner_planner |
| `sampling_planner_recovery` | RECOVERY | `/planner/recovery/wpnts` | WpntArray | recovery_spliner |

START 는 추후 확장. 1차 스코프 제외.

### 2.3 결정 3: 노드 이름 = `sampling_planner_state_node.py`

동일 파일이 `~state` 파라미터 (`overtake` / `recovery` / `observe`) 에 따라 거동 분기.  
런치 인스턴스 이름 = `sampling_planner_ot`, `sampling_planner_recovery`.

### 2.4 결정 4: 런치 통합 = `3d_headtohead.launch` + `sampling_planner_enable`

```xml
<arg name="sampling_planner_enable" default="false"
     doc="true → sampling_based_planner_3d (OVERTAKE+RECOVERY) 사용,
          기존 spliner/sqp/recovery_spliner 비활성"/>

<!-- 기존 planner 들은 unless 로 제외 -->
<include file="..." unless="$(arg sampling_planner_enable)"/>
<!-- 신규 sampling planner 는 if 로 포함 -->
<include file="$(find sampling_based_planner_3d)/launch/sampling_planner_multi.launch"
         if="$(arg sampling_planner_enable)">
  <arg name="map_name"     value="$(arg map_name)"/>
  <arg name="vehicle_name" value="$(arg vehicle_name)"/>
</include>
```

→ `roslaunch stack_master 3d_headtohead.launch sampling_planner_enable:=true` 한 줄로 전환.

### 2.5 결정 5: cost 파라미터 = 자유 확장 + rqt 저장/초기화

`cfg/SamplingCost.cfg` (완료) 항목:
- 코어: `raceline_weight / velocity_weight / prediction_weight` (기존)
- 추가: `continuity_weight` (이전 tick best 와의 L2 편차 페널티), `boundary_weight` (트랙 경계 소프트 페널티)
- 연속성 필터: `filter_alpha` (1-pole EMA, 1=no filter)
- MPPI: `mppi_enable / mppi_temperature / mppi_temporal_weight`
- 출력: `resample_enable / resample_ds_m` (공간 등간격 resampling)
- 트리거: `save_params` / `reset_params`

rqt save/reset 패턴은 [dynamic_statemachine_server.py](../state_machine/src/dynamic_statemachine_server.py) 참고.

### 2.6 결정 6: 연속성 확보 (핵심)

사용자 지적: "왔다갔다 너무 심해서 주행하다가 오히려 안하느니만 못하게 불안정."

**3단 접근:**

**(a) 노드 내부 tick 간 진동 억제**
- MPPI 기본 활성 (`mppi_enable=true`), `mppi_temporal_weight` 양수로 상향
- 1-pole EMA: `d_new = α·d_raw + (1-α)·d_prev` (cfg `filter_alpha`)
- `continuity_weight`: 이전 tick best 와의 L2 거리 페널티를 cost 에 직접 추가
  → 원본 `LocalSamplingPlanner.calc_trajectory` 의 cost 함수 밖에서 후처리로 post-add 해야 할 수 있음 (upstream 건드리지 않고)

**(b) 노드 출력: frenet 등간격 resampling**
- 원본 출력은 시간 등간격 → 공간 간격 가변
- 기존 waypoints 표준은 공간 등간격 (~0.1m)
- 노드 출력 직전 s-축 등간격 resampling (`np.interp`)

**(c) state machine 이음새 블렌딩**
- sampling 경로(~15m) 끝 ↔ global raceline 을 cosine ramp 로 블렌딩
- 끝 1~2m 에서 `d_sampling → d_raceline` smoothstep
- 위치: `_pub_local_wpnts()` 내부 sampling enable 분기
- 주석: `### HJ : sampling tail-blending for continuous local_waypoints`

### 2.7 결정 7: 디버깅 친화

- latched `~status` (String): `INIT_OK`, `WAITING_ODOM`, `NO_FEASIBLE`, `USING_MPPI`, `FALLBACK_RACELINE` 등
- `~timing_ms` (Float32)
- 마커: best trajectory (색=속도), candidate fan, **이전 tick best** (반투명 그레이) — 진동 육안 확인
- 기동 시 최종 weight 값 INFO 로그 덤프
- 단독 기동: `roslaunch sampling_based_planner_3d sampling_planner_state.launch state:=overtake`
- bag 재생만으로도 검증 가능 (`/car_state/odom` 만 있으면 됨)

---

## 3. 탐색 결과 요약 (한번에 볼 수 있게)

### 3.1 원본 sampling planner

**노드 진입:** [sampling_planner_node.py](../planner/3d_sampling_based_planner/node/sampling_planner_node.py)  
**클래스:** `SamplingPlannerNode`  
**코어 모듈:** [sampling_based_planner.py](../planner/3d_sampling_based_planner/src/sampling_based_planner.py) `LocalSamplingPlanner`

**현 파라미터 (L55-85):**
```
~track_csv_path, ~gg_dir_path, ~vehicle_params_path, ~raceline_csv_path
~rate_hz, ~frame_id
~horizon, ~num_samples, ~n_samples, ~v_samples
~safety_distance, ~gg_abs_margin, ~gg_margin_rel, ~friction_check_2d
~s_dot_min, ~kappa_thr, ~relative_generation
~cost/raceline_weight, ~cost/velocity_weight, ~cost/prediction_weight
~mppi/enable, ~mppi/temperature_rel, ~mppi/temporal_weight
```

**구독:** `/car_state/odom` (Odometry)  
**발행 (~namespace):**
- `~best_trajectory` (WpntArray), `~best_sample` (Path)
- `~best_sample/markers`, `~best_sample/vel_markers` (MarkerArray)
- `~candidates` (MarkerArray), `~status` (String, latched), `~timing_ms` (Float32)

**cost 함수:** [sampling_based_planner.py `get_optimal_trajectory_idx()` L341-400](../planner/3d_sampling_based_planner/src/sampling_based_planner.py#L341-L400)
- velocity_cost: `((V - V_raceline)/V_raceline)²` 시간적분
- raceline_cost: `(n - n_raceline)²` 시간적분
- prediction_cost: opponent 위치 exp Gaussian (현재 prediction={} 이므로 비활성)

**핵심 제약:** 코어는 import 해서 재사용, **weight 는 `calc_trajectory()` 인자로 전달** ([sampling_planner_node.py:485-487](../planner/3d_sampling_based_planner/node/sampling_planner_node.py#L485-L487)) — runtime 에 갱신된 self.w_* 를 넘기면 됨.

### 3.2 State Machine

**3D 버전 활성:** [3d_state_machine_node.py](../state_machine/src/3d_state_machine_node.py)  
(2D 는 `state_machine_node.py`, 백업은 `_original.py`)

**State 정의:** [states_types.py](../state_machine/src/states_types.py)
- `GB_TRACK`, `TRAILING`, `OVERTAKE`, `FTGONLY`, `RECOVERY`, `ATTACK`, `START`, `LOSTLINE`, `SMART_STATIC`

**WaypointData 클래스 (L84-L113):**
- `node_name = "/dyn_planners_statemachine/" + planner_name`
- `/dyn_planners_statemachine/<name>/parameter_updates` 구독
- `rospy.get_param(node_name + "/min_horizon")` 등 호출 (이 rosparam 은 `dynamic_planners_server.py` 가 제공)

**현 planner 인스턴스 (L229-L238):**
```python
self.cur_gb_wpnts               = WaypointData('global_tracking', True)
self.cur_recovery_wpnts         = WaypointData('recovery_planner', False)
self.cur_avoidance_wpnts        = WaypointData('dynamic_avoidance_planner', False)
self.cur_static_avoidance_wpnts = WaypointData('static_avoidance_planner', False)
self.cur_start_wpnts            = WaypointData('start_planner', False)
self.cur_smart_static_avoidance_wpnts = WaypointData('static_avoidance_planner', True)
```

**구독 토픽 (L365-L397):**
- `/planner/recovery/wpnts` (WpntArray) ← recovery planner
- `/planner/avoidance/otwpnts` (OTWpntArray) ← 오버테이킹 planner
- `/planner/avoidance/smart_static_otwpnts` (OTWpntArray)
- `/planner/avoidance/static_otwpnts` (OTWpntArray)
- `/planner/start_wpnts` (OTWpntArray)

**발행:** `local_waypoints` (WpntArray) — 컨트롤러가 구독할 최종 로컬 경로

### 3.3 기존 planner 매핑

| State | 현행 planner | 발행 토픽 | 대체 대상? |
|---|---|---|---|
| OVERTAKE (static) | `planner/spliner` | `/planner/avoidance/otwpnts` | ✅ |
| OVERTAKE (dynamic SQP) | `planner/sqp_planner` | `/planner/avoidance/otwpnts` | ✅ |
| OVERTAKE (dynamic Spline) | `planner/spliner_planner` | `/planner/avoidance/otwpnts` | ✅ |
| OVERTAKE (Change) | `planner/lane_change_planner` | `/planner/avoidance/otwpnts` | ✅ |
| RECOVERY | `planner/recovery_spliner` | `/planner/recovery/wpnts` (WpntArray!) | ✅ |
| START | `planner/spliner/start_spline_node_v2` | `/planner/start_wpnts` | ❌ 1차 제외 |
| SMART_STATIC | `planner/spliner` | `/planner/avoidance/smart_static_otwpnts` | ❌ 1차 제외 |

**ot_planner 선택 파라미터:** `state_machine/ot_planner` ([3d_state_machine_node.py:203](../state_machine/src/3d_state_machine_node.py#L203))

### 3.4 메시지 타입 요약

**Wpnt.msg** ([f110_msgs/msg/Wpnt.msg](../f110_utils/libs/f110_msgs/msg/Wpnt.msg)):
- `id, s_m, d_m, x_m, y_m, z_m` (3D), `d_right, d_left`, `psi_rad, kappa_radpm, vx_mps, ax_mps2, mu_rad` (3D)

**OTWpntArray.msg**:
- `header, last_switch_time (time), side_switch (bool), ot_side (string), ot_line (string), wpnts (Wpnt[])`
- OVERTAKE 노드는 `side_switch=False, ot_side='right|left', ot_line='spline|...'` 정도로 채우면 state machine 이 받아들임

**WpntArray.msg**: `header, wpnts (Wpnt[])` — recovery 용

---

## 4. 파일 구조 (생성 계획 vs 현재 상태)

### 4.1 신규 파일

| 경로 | 상태 | 역할 |
|---|---|---|
| `planner/3d_sampling_based_planner/cfg/SamplingCost.cfg` | ✅ 완료 | dynamic_reconfigure 정의 |
| `planner/3d_sampling_based_planner/node/sampling_planner_state_node.py` | ⏳ **Step 2 진행 중** | 상태별 동작 sampling 노드 |
| `planner/3d_sampling_based_planner/config/state_overtake.yaml` | ⬜ 미착수 | OVERTAKE weight preset |
| `planner/3d_sampling_based_planner/config/state_recovery.yaml` | ⬜ 미착수 | RECOVERY weight preset |
| `planner/3d_sampling_based_planner/launch/sampling_planner_state.launch` | ⬜ 미착수 | 단일 인스턴스 템플릿 |
| `planner/3d_sampling_based_planner/launch/sampling_planner_multi.launch` | ⬜ 미착수 | OT+RC 2개 include |
| `stack_master/config/planners/sampling_ot.yaml` | ⬜ 미착수 | WaypointData rosparam 제공 |
| `stack_master/config/planners/sampling_recovery.yaml` | ⬜ 미착수 | WaypointData rosparam 제공 |
| `HJ_docs/sampling_planner_state_machine_integration.md` | ✅ 이 문서 | 인수인계 문서 |

### 4.2 백업본 (Step 0 생성 완료)

| 백업 | 원본 |
|---|---|
| `state_machine/src/3d_state_machine_node_backup_20260418.py` | `state_machine/src/3d_state_machine_node.py` |
| `stack_master/launch/3d_headtohead_backup_20260418.launch` | `stack_master/launch/3d_headtohead.launch` |

### 4.3 수정 파일 (원본 보존: 위 백업본 먼저 생성된 상태)

| 경로 | 상태 | 변경 내용 |
|---|---|---|
| `planner/3d_sampling_based_planner/CMakeLists.txt` | ✅ 완료 | `dynamic_reconfigure`, `generate_dynamic_reconfigure_options`, state node install |
| `planner/3d_sampling_based_planner/package.xml` | ✅ 완료 | `dynamic_reconfigure` depend 추가 |
| `state_machine/src/3d_state_machine_node.py` | ⬜ 미착수 | sampling 분기 + tail-blending |
| `stack_master/launch/3d_headtohead.launch` | ⬜ 미착수 | `sampling_planner_enable` arg, conditional include |

### 4.4 불변 파일

- `planner/3d_sampling_based_planner/node/sampling_planner_node.py`
- `planner/3d_sampling_based_planner/launch/sampling_planner_observe.launch`
- `planner/3d_sampling_based_planner/config/default.yaml`
- `planner/3d_sampling_based_planner/src/sampling_based_planner.py`

---

## 5. 재사용할 기존 코드

**코어 로직 import:**
- [sampling_based_planner.py `LocalSamplingPlanner.calc_trajectory()` L341-400](../planner/3d_sampling_based_planner/src/sampling_based_planner.py#L341-L400)
  - cost 이미 모듈화, weight 인자 주입으로 충분
- [track3D.py `Track3D`](../planner/3d_sampling_based_planner/src/track3D.py)
- [ggManager.py `GGManager`](../planner/3d_gb_optimizer/global_line/src/ggManager.py) (shared from 3d_gb_optimizer)

**패턴 모방:**
- [dynamic_statemachine_server.py](../state_machine/src/dynamic_statemachine_server.py) — rqt save_params trigger 패턴
- [smart_static_avoidance_node.py](../planner/spliner/src/smart_static_avoidance_node.py) — OTWpntArray 필드 포맷
- [recovery_spliner_node.py](../planner/recovery_spliner/src/recovery_spliner_node.py) — 레이스라인 복귀 / 블렌딩
- [3d_state_machine_node.py WaypointData L84-L113](../state_machine/src/3d_state_machine_node.py#L84-L113) — planner_name 지정 재사용

**원본 노드의 거의 전체 로직** ([sampling_planner_node.py](../planner/3d_sampling_based_planner/node/sampling_planner_node.py)) 를 새 state 노드에 **복사**해서 시작:
- 초기화 (L50-167)
- `_load_raceline_dict` (L172-...)
- `_cb_odom`
- `_cart_to_cl_frenet_exact`
- main loop (L390 근방)
- `_publish_trajectory`, `_publish_candidates`, `_mppi_blend`
- `_publish_status`

그 위에 추가:
- `~state` 파라미터 분기
- dynamic_reconfigure Server
- save/reset callback
- continuity_weight post-cost (원본 cost_array 에 가산)
- 1-pole filter on (d, V)
- s-축 등간격 resampling
- OTWpntArray / WpntArray 분기 발행 with `~out/otwpnts` or `~out/wpnts` topic
- 이전 tick best marker (반투명)

---

## 6. 진행 상황 체크리스트

- [x] **Step 0:** 백업 복사 (2026-04-18)
  - `3d_state_machine_node_backup_20260418.py` ✅
  - `3d_headtohead_backup_20260418.launch` ✅
- [x] **Step 1:** dynamic_reconfigure 인프라
  - `cfg/SamplingCost.cfg` ✅
  - `CMakeLists.txt` 수정 ✅
  - `package.xml` 수정 ✅
- [ ] **Step 2:** `sampling_planner_state_node.py` 신규 작성 ⏳ **현재 지점**
  - 원본 노드 전체 분석 완료
  - 파일 스켈레톤 작성 미완
- [ ] **Step 3:** state 별 YAML 2개
- [ ] **Step 4:** launch 파일 2개 (state.launch / multi.launch)
- [ ] **Step 5:** `3d_headtohead.launch` 에 `sampling_planner_enable` arg
- [ ] **Step 6:** state machine 연동 (WaypointData 추가 + tail-blending)
- [ ] **Step 7:** Docker 빌드 + 기동 검증
- [ ] **Step 8:** 이 문서 최종화 (구현 완료 반영)

---

## 7. 다음 세션이 바로 할 일

### 7.1 Step 2: `sampling_planner_state_node.py` 작성

**파일 경로:** `planner/3d_sampling_based_planner/node/sampling_planner_state_node.py`  
**시작 방식:** 원본 [sampling_planner_node.py](../planner/3d_sampling_based_planner/node/sampling_planner_node.py) 를 **파일 레벨 복사**한 뒤 아래 변경:

**추가 import:**
```python
import rospkg
from dynamic_reconfigure.server import Server
from sampling_based_planner_3d.cfg import SamplingCostConfig
from f110_msgs.msg import OTWpntArray
```

**__init__ 추가 내용:**
1. `self.state = rospy.get_param('~state', 'overtake')`  →  {'overtake','recovery','observe'} 검증
2. `self.instance_yaml_path = rospy.get_param('~instance_yaml', '')`  (save/reset 대상)
3. cost 파라미터 초기값을 `rospy.get_param('~cost/*', default)` 로 로드 (기존 3개 + continuity + boundary)
4. `self.filter_alpha = rospy.get_param('~filter_alpha', 0.7)`
5. `self.resample_enable, self.resample_ds = ...`
6. `self._prev_best_s = None; self._prev_best_n = None; self._prev_best_V = None`  (continuity + EMA)
7. dynamic_reconfigure:
   ```python
   self._dyn_srv = Server(SamplingCostConfig, self._weight_cb)
   ```
8. 발행 토픽 분기 (launch 에서 `<remap>` 로 외부 경로 매핑):
   ```python
   if self.state == 'overtake':
       self.pub_out = rospy.Publisher('~out/otwpnts', OTWpntArray, queue_size=1)
   elif self.state == 'recovery':
       self.pub_out = rospy.Publisher('~out/wpnts', WpntArray, queue_size=1)
   else:  # observe
       self.pub_out = rospy.Publisher('~best_trajectory', WpntArray, queue_size=1)
   ```

**`_weight_cb` 구현:**
```python
def _weight_cb(self, config, level):
    # 트리거 우선
    if config.save_params:
        self._save_yaml(config)
        config.save_params = False
    if config.reset_params:
        new_cfg = self._reload_yaml()
        if new_cfg is not None:
            self._dyn_srv.update_configuration(new_cfg)
        config.reset_params = False
    # 일반 파라미터
    self.w_raceline     = float(config.raceline_weight)
    self.w_velocity     = float(config.velocity_weight)
    self.w_prediction   = float(config.prediction_weight)
    self.w_continuity   = float(config.continuity_weight)
    self.w_boundary     = float(config.boundary_weight)
    self.filter_alpha   = float(config.filter_alpha)
    self.mppi_enable    = bool(config.mppi_enable)
    self.mppi_temperature_rel = float(config.mppi_temperature)
    self.mppi_temporal_weight = float(config.mppi_temporal_weight)
    self.resample_enable = bool(config.resample_enable)
    self.resample_ds     = float(config.resample_ds_m)
    rospy.loginfo_throttle(2.0, '[sampling][%s] cost=(rl=%.2f v=%.1f p=%.1f c=%.1f b=%.1f) α=%.2f mppi=%s/%.3f/%.1f rs=%s/%.2f',
                            rospy.get_name(), self.w_raceline, self.w_velocity, self.w_prediction,
                            self.w_continuity, self.w_boundary, self.filter_alpha,
                            self.mppi_enable, self.mppi_temperature_rel, self.mppi_temporal_weight,
                            self.resample_enable, self.resample_ds)
    return config
```

**continuity 페널티:** `LocalSamplingPlanner.calc_trajectory` 호출 후, `self.planner.candidates['s']` / `['n']` 와 이전 tick best 를 비교해 `self.planner.cost_array` 에 post-add:
```python
if self.w_continuity > 0.0 and self._prev_best_s is not None:
    s_all = np.asarray(self.planner.candidates['s'])
    n_all = np.asarray(self.planner.candidates['n'])
    m = min(self._prev_best_s.shape[0], s_all.shape[1])
    ds = s_all[:, :m] - self._prev_best_s[:m]
    dn = n_all[:, :m] - self._prev_best_n[:m]
    L2 = np.sum(ds * ds + dn * dn, axis=1)
    self.planner.cost_array = self.planner.cost_array + self.w_continuity * L2
    # 그 후 best_idx 를 다시 argmin 해야 함
    valid = np.asarray(self.planner.candidates['valid'], dtype=bool)
    if valid.any():
        cost_masked = np.where(valid, self.planner.cost_array, np.inf)
        best_idx = int(np.argmin(cost_masked))
        # self.planner.trajectory 재구성 필요 — candidates[i] 에서 뽑아서
        self.planner.trajectory = self._extract_traj_from_candidate(best_idx)
```
(정확한 `_extract_traj_from_candidate` 는 원본에 없음 — candidates dict 에서 `[best_idx]` 슬라이싱해서 dict 만들기)

**1-pole EMA on (d, V):** `_publish_trajectory` 직전에:
```python
if self._prev_best_n is not None and 0.0 < self.filter_alpha < 1.0:
    α = self.filter_alpha
    m = min(len(self._prev_best_n), len(traj['n']))
    traj['n'][:m] = α * traj['n'][:m] + (1 - α) * self._prev_best_n[:m]
    traj['V'][:m] = α * traj['V'][:m] + (1 - α) * self._prev_best_V[:m]
    # s, x, y, z 재계산
    s_mod = np.clip(np.mod(traj['s'], L), 1e-6, L - 1e-6)
    xyz = self.track.sn2cartesian(s=s_mod, n=np.asarray(traj['n']))
    traj['x'], traj['y'], traj['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
```

**s-등간격 resampling:** `_publish_trajectory` 내부에서 Wpnt 생성 전:
```python
if self.resample_enable:
    s_unwr_uniform = np.arange(s_unwr[0], s_unwr[-1], self.resample_ds)
    n_resampled = np.interp(s_unwr_uniform, s_unwr, n_arr)
    V_resampled = np.interp(s_unwr_uniform, s_unwr, traj['V'][:len(s_unwr)])
    # 그 외 필드도 동일, x/y/z 는 sn2cartesian 재계산
    ...
```

**role별 발행 분기:** `_publish_trajectory` 끝에서 OTWpntArray 또는 WpntArray 중 적절한 것을 `self.pub_out` 으로 발행.

**OTWpntArray 필드:**
```python
ot = OTWpntArray()
ot.header = header
ot.last_switch_time = rospy.Time.now()
ot.side_switch = False
ot.ot_side = 'right'  # 추후 동적 판정 가능
ot.ot_line = 'sampling'
ot.wpnts = wp_arr.wpnts  # 재사용
self.pub_out.publish(ot)
```

**save/reset YAML 헬퍼:**
```python
def _save_yaml(self, config):
    if not self.instance_yaml_path or not os.path.exists(self.instance_yaml_path):
        rospy.logwarn('[sampling][save] skip — instance_yaml not set or missing')
        return
    with open(self.instance_yaml_path) as f:
        data = yaml.safe_load(f)
    data.setdefault('cost', {})
    data['cost']['raceline_weight']   = float(config.raceline_weight)
    data['cost']['velocity_weight']   = float(config.velocity_weight)
    data['cost']['prediction_weight'] = float(config.prediction_weight)
    data['cost']['continuity_weight'] = float(config.continuity_weight)
    data['cost']['boundary_weight']   = float(config.boundary_weight)
    data['filter_alpha'] = float(config.filter_alpha)
    data.setdefault('mppi', {})
    data['mppi']['enable']          = bool(config.mppi_enable)
    data['mppi']['temperature_rel'] = float(config.mppi_temperature)
    data['mppi']['temporal_weight'] = float(config.mppi_temporal_weight)
    data['resample_enable'] = bool(config.resample_enable)
    data['resample_ds_m']   = float(config.resample_ds_m)
    with open(self.instance_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    rospy.loginfo('[sampling][save] YAML updated: %s', self.instance_yaml_path)

def _reload_yaml(self):
    if not self.instance_yaml_path or not os.path.exists(self.instance_yaml_path):
        return None
    with open(self.instance_yaml_path) as f:
        d = yaml.safe_load(f)
    return {
        'raceline_weight':   float(d.get('cost', {}).get('raceline_weight', 0.1)),
        'velocity_weight':   float(d.get('cost', {}).get('velocity_weight', 100.0)),
        'prediction_weight': float(d.get('cost', {}).get('prediction_weight', 5000.0)),
        'continuity_weight': float(d.get('cost', {}).get('continuity_weight', 50.0)),
        'boundary_weight':   float(d.get('cost', {}).get('boundary_weight', 0.0)),
        'filter_alpha':      float(d.get('filter_alpha', 0.7)),
        'mppi_enable':          bool(d.get('mppi', {}).get('enable', True)),
        'mppi_temperature':     float(d.get('mppi', {}).get('temperature_rel', 0.25)),
        'mppi_temporal_weight': float(d.get('mppi', {}).get('temporal_weight', 0.0)),
        'resample_enable': bool(d.get('resample_enable', True)),
        'resample_ds_m':   float(d.get('resample_ds_m', 0.1)),
        'save_params': False,
        'reset_params': False,
    }
```

### 7.2 Step 3: state 별 YAML

**`config/state_overtake.yaml`** (튜닝 초안):
```yaml
rate_hz: 30.0
frame_id: map
horizon:      1.0
num_samples:  30
n_samples:    11
v_samples:    5
safety_distance: 0.20
gg_abs_margin:   0.0
gg_margin_rel:   0.0
friction_check_2d: false
s_dot_min: 2.5
kappa_thr: 1.5
relative_generation: true
cost:
  raceline_weight:   0.05    # overtake: 레이스라인 고집 덜 함
  velocity_weight:   100.0
  prediction_weight: 5000.0  # 장애물 회피 민감
  continuity_weight: 50.0
  boundary_weight:   0.0
filter_alpha: 0.7
mppi:
  enable:          true
  temperature_rel: 0.25
  temporal_weight: 100.0
resample_enable: true
resample_ds_m:   0.1
```

**`config/state_recovery.yaml`**:
```yaml
rate_hz: 30.0
frame_id: map
horizon:      2.0        # recovery: 좀 더 긴 수평선으로 복귀 경로 확보
num_samples:  40
n_samples:    11
v_samples:    3          # 속도 선택지 축소
safety_distance: 0.30
# ...
s_dot_min: 1.5
kappa_thr: 1.2
relative_generation: true
cost:
  raceline_weight:   10.0   # 복귀 최우선
  velocity_weight:   20.0   # 속도 덜 고집
  prediction_weight: 1000.0
  continuity_weight: 100.0  # 연속성 더 강하게
  boundary_weight:   10.0   # 트랙 경계 안전 마진
filter_alpha: 0.5
mppi:
  enable:          true
  temperature_rel: 0.3
  temporal_weight: 200.0
resample_enable: true
resample_ds_m:   0.1
```

### 7.3 Step 4: launch 파일

**`launch/sampling_planner_state.launch`**:
```xml
<?xml version="1.0"?>
<!-- ### HJ : state-aware sampling planner (single instance template) -->
<launch>
  <arg name="state"          default="overtake"/>
  <arg name="instance_name"  default="sampling_planner_$(arg state)"/>
  <arg name="config_file"    default="$(find sampling_based_planner_3d)/config/state_$(arg state).yaml"/>
  <arg name="output_topic"   default=""/>
  <arg name="map_name"       default="experiment_3d_2"/>
  <arg name="vehicle_name"   default="rc_car_10th_latest"/>

  <arg name="track_csv"      default="$(find stack_master)/maps/$(arg map_name)/$(arg map_name)_3d_smoothed.csv"/>
  <arg name="raceline_csv"   default="$(find stack_master)/maps/$(arg map_name)/$(arg map_name)_3d_$(arg vehicle_name)_timeoptimal.csv"/>
  <arg name="gg_dir"         default="$(find global_line_3d)/data/gg_diagrams/$(arg vehicle_name)/velocity_frame"/>
  <arg name="vehicle_params" default="$(find global_line_3d)/data/vehicle_params/params_$(arg vehicle_name).yml"/>

  <node pkg="sampling_based_planner_3d" type="sampling_planner_state_node.py"
        name="$(arg instance_name)" output="screen">
    <rosparam command="load" file="$(arg config_file)"/>
    <param name="state"               value="$(arg state)"/>
    <param name="instance_yaml"       value="$(arg config_file)"/>
    <param name="track_csv_path"      value="$(arg track_csv)"/>
    <param name="raceline_csv_path"   value="$(arg raceline_csv)"/>
    <param name="gg_dir_path"         value="$(arg gg_dir)"/>
    <param name="vehicle_params_path" value="$(arg vehicle_params)"/>
    <remap from="~out/otwpnts" to="$(arg output_topic)" if="$(eval output_topic != '')"/>
    <remap from="~out/wpnts"   to="$(arg output_topic)" if="$(eval output_topic != '')"/>
  </node>
</launch>
```

**`launch/sampling_planner_multi.launch`**: OVERTAKE + RECOVERY 를 각각 include.

### 7.4 Step 5: `3d_headtohead.launch` 수정

백업본 참조해서 `sampling_planner_enable` arg 추가, 기존 planner include 에 `unless`, sampling multi include 에 `if` 달기.

`stack_master/config/planners/sampling_ot.yaml`, `sampling_recovery.yaml` 을 새로 만들고 **dynamic_planners_server.py 가 요구하는 필드** (min_horizon, max_horizon, lateral_width_m, free_scaling_reference_distance_m, latest_threshold, hyst_timer_sec, killing_timer_sec, on_spline_front_horizon_thres_m, on_spline_min_dist_thres_m) 를 채워두기.

### 7.5 Step 6: state machine 연동

`3d_state_machine_node.py` 수정 (원본은 `_backup_20260418.py` 로 보존됨):
1. `sampling_planner_enable` rosparam 읽기
2. enable 시 `WaypointData('sampling_ot', False)` / `WaypointData('sampling_recovery', False)` 사용
3. `_pub_local_wpnts()` 에 tail-blending 로직 추가 (sampling 경로 끝 ↔ global raceline cosine ramp)
4. 모두 `### HJ :` 주석 래핑

### 7.6 Step 7: 빌드 & 검증

```bash
docker exec icra2026 bash -c \
  "source /opt/ros/noetic/setup.bash && source /home/unicorn/catkin_ws/devel/setup.bash && \
   cd /home/unicorn/catkin_ws && catkin build sampling_based_planner_3d state_machine stack_master"
```

**단계별 기동:**
1. 단독: `roslaunch sampling_based_planner_3d sampling_planner_state.launch state:=observe`  → 원본 회귀 확인
2. Multi: `roslaunch sampling_based_planner_3d sampling_planner_multi.launch`  → 2 노드 정상 발행
3. 통합:
   ```bash
   roslaunch glim_ros glil_cpu.launch
   roslaunch stack_master 3d_headtohead.launch map:=experiment_3d_2 sampling_planner_enable:=true
   ```
4. rqt_reconfigure 로 `sampling_planner_ot/prediction_weight` 변경 → 즉시 반영 확인
5. `save_params` 트리거 → YAML 업데이트 확인, `reset_params` → 초기값 복귀

**회귀:** `sampling_planner_enable:=false` (default) 로 기존 동작 100% 유지.

### 7.7 Step 8: 이 문서 최종화

구현 완료 후 섹션 6 체크박스, 섹션 4 파일 상태, 섹션 9 튜닝값을 반영.

---

## 8. 알려진 이슈 / 판단 보류

### 8.1 `continuity_weight` 후처리 방식

원본 `LocalSamplingPlanner.calc_trajectory` 의 cost 식을 건드리지 않고 post-add 해야 함. 순서:
1. `calc_trajectory` 호출 → `self.planner.cost_array` 생성됨
2. post-add: `cost_array += w_continuity * L2_to_prev_best`
3. 다시 argmin 해서 best idx 결정
4. `self.planner.trajectory` 재구성 (candidates[best_idx] 에서 슬라이싱)

**리스크:** `self.planner.trajectory` 의 정확한 구조 확인 필요 — candidates dict 에서 슬라이싱만으로 재구성되는지. 안 되면 `_extract_traj_from_candidate` helper 를 직접 작성.

### 8.2 `_prev_best_s` 저장 지점

- 매 tick 최종 선택 후 `self._prev_best_s = blended_or_best['s'].copy()` 등으로 저장
- MPPI 가 활성이면 blended 를, 아니면 argmin 을 사용
- 필터링 이전 값 vs 이후 값 중 어느 것을 저장할지 → 필터링 이전(raw) 값이 "궤적 자체의 튀는 정도" 를 측정하는 데 더 맞음

### 8.3 state machine WaypointData rosparam

`dynamic_planners_server.py` 가 모든 rosparam 을 제공 — sampling planner 용으로 `sampling_ot.yaml`, `sampling_recovery.yaml` 을 `stack_master/config/planners/` 에 추가하고 `dynamic_planners_server` 를 기존 `dynamic_avoidance_planner` 와 동일하게 띄워야 함 (3d_headtohead.launch 에 추가).

대안: WaypointData 의 `update_param` 을 sampling 전용 분기 추가해서 rosparam 없이 기본값 채움. 덜 깨끗하지만 확실함.

### 8.4 OTWpntArray `side_switch` / `ot_side`

- state machine 에서 `side_switch` 를 감지해 어떻게 반응하는지 추적 필요
- 초기 구현: `side_switch=False, ot_side='right', ot_line='sampling'` 고정
- 추후: sampling planner 결과의 평균 n 부호를 보고 `ot_side` 동적 설정

### 8.5 tail-blending 길이

- 1m ~ 2m 사이에서 실험
- cosine ramp vs linear vs quintic spline 비교 필요
- sampling 궤적 끝점에서의 한 번의 불연속만 해결하면 되므로 과하게 길 필요 없음

### 8.6 prediction 연동

현재 `prediction={}` 로 고정 ([sampling_planner_node.py:467](../planner/3d_sampling_based_planner/node/sampling_planner_node.py#L467)). 실 장애물 예측을 쓰려면 `/opponent_prediction/obstacles_pred` 구독해 dict-of-dicts 로 변환 필요. **2차 작업.**

---

## 9. 튜닝 로그 (주행 후 채움)

| 날짜 | state | raceline | velocity | prediction | continuity | filter_α | mppi_tw | 결과/메모 |
|---|---|---|---|---|---|---|---|---|
| 2026-04-18 | overtake | 0.05 | 100.0 | 5000.0 | 50.0 | 0.7 | 100.0 | 초안 (미검증) |
| 2026-04-18 | recovery | 10.0 | 20.0 | 1000.0 | 100.0 | 0.5 | 200.0 | 초안 (미검증) |

---

## 10. 확장 포인트 (1차 이후)

- **START state sampling planner** — `~out/otwpnts` role 을 `start` 로 확장, `/planner/start_wpnts` 발행
- **성능별 변형 (aggressive/defensive)** — 같은 state 에 여러 인스턴스 + state machine 분기 (예: `ot_aggressive`, `ot_defensive`)
- **실 장애물 prediction 연동** — `/opponent_prediction/obstacles_pred` → `prediction` dict 변환
- **SMART_STATIC role** — `/planner/avoidance/smart_static_otwpnts` 추가
- **cost term 추가** — `boundary_weight` 튜닝, curvature smoothness term, ego prediction consistency term
- **tail-blending 고도화** — cosine → spline, 길이 동적 (속도 비례)

---

## 11. 원칙 (CLAUDE.md 에서)

- 한국어로 대화, 코드 주석은 영어
- 3D 확장 수정은 `### HJ :` 주석 래핑, 제거 대신 주석 처리 권장 (`## IY :` / `## IY : end` 패턴 가능)
- Docker 컨테이너 `icra2026` 안에서 빌드/실행
- 빌드: `docker exec icra2026 bash -c "source /opt/ros/noetic/setup.bash && source /home/unicorn/catkin_ws/devel/setup.bash && cd /home/unicorn/catkin_ws && catkin build"`
- 로컬 git 영향 docker 명령(fetch/checkout 등) 절대 금지
- 판단 근거 제시 (파일 경로, 라인 번호, 토픽명)
- 이슈/해결을 `TODO_HJ.md` 에 즉시 업데이트
- commit 에 Co-Authored-By Claude 넣지 않음
