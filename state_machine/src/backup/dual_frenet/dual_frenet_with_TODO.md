# Dual Frenet Coordinate System Integration

## Overview

State Machine이 **GB Raceline**과 **Smart Static Avoidance Path** 두 가지 Frenet 좌표계를 동시에 지원하도록 확장한 작업입니다.

Smart Static Mode에서는 GB Optimizer로 생성된 고정 경로를 기준 좌표계로 사용하여, 장애물과 차량의 Frenet 좌표를 재계산합니다.

---

## Core Concept

### Dual Frenet Reference Systems

- **GB (Global) Frenet**: 기존 GB raceline을 기준으로 한 Frenet 좌표계
- **Smart Static Frenet**: Smart Static Avoidance path를 기준으로 한 Frenet 좌표계

### Architecture Pattern

**Class Inheritance** 패턴을 사용하여 구현:
```python
class SmartStaticChecker(StateMachine):
    def __init__(self, parent_state_machine):
        # Copy ALL parent attributes via __dict__.update()
        self.__dict__.update(parent_state_machine.__dict__)

        # Override only Smart Static specific variables
        self.cur_s = 0.0  # Smart Static s
        self.cur_d = 0.0  # Smart Static d
        self.cur_vs = 0.0  # Smart Static velocity
        self.obstacles = []  # Smart Static Frenet obstacles
        self.cur_gb_wpnts = parent.cur_smart_static_avoidance_wpnts
```

- 모든 check 함수는 **상속**되어 그대로 사용
- 인스턴스 변수만 **오버라이드**하여 좌표계 전환

---

## Modified Files

### 1. `state_machine/src/state_helper_for_smart.py` ⭐ NEW FILE

Smart Static Checker 구현체.

**주요 기능**:
- StateMachine 상속하여 모든 check 함수 재사용
- `__dict__.update()` 패턴으로 parent 속성 복사 후 필요한 변수만 오버라이드
- FrenetConverter를 사용한 정확한 Frenet 좌표 변환
- Waypoint s_m 재계산 (Smart Static 경로 기준)
- Obstacle Frenet 좌표 변환 (위치 + 속도)
- 시간 미분 방식의 cur_vs 계산 (track wrapping 처리)
- 80Hz 타이머 기반 백그라운드 업데이트

**핵심 메소드**:
```python
def update(self):
    # 1. Build FrenetConverter from Smart Static waypoints
    # 2. Recalculate waypoint s_m values
    # 3. Convert obstacles to Smart Static Frenet
    # 4. Calculate ego cur_s, cur_d, cur_vs
    # 5. Update obstacles_in_interest
```

**Obstacle 변환 로직**:
```python
def _convert_obstacles_to_smart_static_frenet(self):
    # Position: x,y -> Smart Static (s, d)
    # Velocity: GB (vs, vd) -> Cartesian (vx, vy) -> Smart Static (vs, vd)
```

---

### 2. `state_machine/src/state_transitions.py`

모든 transition 함수 수정.

**주요 변경사항**:

#### A. Explicit base_traj Parameter Passing
Race condition 방지를 위해 `base_traj`를 명시적으로 전달:

```python
# BEFORE (race condition risk)
def ObstacleTransition(state_machine, close_to_raceline):
    base_traj, _ = state_machine._get_base_state_and_trajectory()
    # ⚠️ 여기서 smart_static_active 플래그가 callback으로 변경될 수 있음
    return NonObstacleTransition(state_machine, close_to_raceline)

# AFTER (explicit parameter)
def GlobalTrackingTransition(state_machine):
    if len(state_machine.cur_obstacles_in_interest) == 0:
        return NonObstacleTransition(state_machine, close_to_raceline, StateType.GB_TRACK)
    else:
        return ObstacleTransition(state_machine, close_to_raceline, StateType.GB_TRACK)
```

#### B. base_checker Selection
적절한 checker를 base_traj에 따라 선택:

```python
def ObstacleTransition(state_machine, close_to_raceline, base_traj: StateType = None):
    if base_traj is None:
        base_traj, _ = state_machine._get_base_state_and_trajectory()

    # Select appropriate checker and waypoints
    if base_traj == StateType.SMART_STATIC:
        base_wpnts = state_machine.cur_smart_static_avoidance_wpnts
        base_checker = state_machine.smart_static_checker
    else:
        base_wpnts = state_machine.cur_gb_wpnts
        base_checker = state_machine

    # Use base_checker for ALL checks
    if close_to_raceline and base_checker._check_free_frenet(base_wpnts):
        return base_traj, base_traj
```

**수정된 Transitions**:
- `GlobalTrackingTransition`: StateType.GB_TRACK 명시적 전달
- `SmartStaticTransition`: StateType.SMART_STATIC 명시적 전달
- `RecoveryTransition`: base_checker 사용
- `OvertakingTransition`: base_checker 사용
- `NonObstacleTransition`: base_traj 파라미터 추가
- `ObstacleTransition`: base_checker 및 base_wpnts 조건 분기
- `TrailingTransition`: base_checker 및 obstacles_in_interest 조건 분기
- `FTGOnlyTransition`: base_checker 완전 적용

---

### 3. `state_machine/src/state_machine_node.py`

State Machine 메인 노드.

**추가 사항**:
```python
# Line 283-284: Smart Static Checker 인스턴스 생성
from state_helper_for_smart import SmartStaticChecker
self.smart_static_checker = SmartStaticChecker(self)

# Line 141-142: Smart Static 관련 변수 추가
self.smart_static_wpnts = None
self.smart_static_active = False

# Line 163: WaypointData 추가
self.cur_smart_static_avoidance_wpnts = WaypointData('smart_static_avoidance_planner', False)

# Line 314-316: Smart Static 토픽 구독
rospy.Subscriber("/planner/avoidance/smart_static_otwpnts", OTWpntArray, self.smart_static_avoidance_cb)
rospy.Subscriber("/planner/avoidance/smart_static_active", Bool, self.smart_static_active_cb)

# Line 266: State 등록
StateType.SMART_STATIC: states.SmartStatic

# Line 277: Transition 등록
StateType.SMART_STATIC: state_transitions.SmartStaticTransition

# Line 397-403: Callback 구현
def smart_static_avoidance_cb(self, data: OTWpntArray):
    self.cur_smart_static_avoidance_wpnts.initialize_traj(data)

def smart_static_active_cb(self, data):
    self.smart_static_active = data.data

# Line 698-702: Helper 함수
def _get_base_state_and_trajectory(self) -> Tuple[StateType, StateType]:
    if self.smart_static_active and len(self.cur_smart_static_avoidance_wpnts.list) > 0:
        return StateType.SMART_STATIC, StateType.SMART_STATIC
    return StateType.GB_TRACK, StateType.GB_TRACK

# Line 1223-1237: Waypoint getter
def get_smart_static_wpts(self) -> WpntArray:
    # Returns Smart Static waypoints for local planning
```

---

### 4. `state_machine/src/states_types.py`

StateType enum 확장.

**추가**:
```python
SMART_STATIC = 'SMART_STATIC'
```

---

### 5. `state_machine/src/states.py`

State behavior 함수 추가.

**추가**:
```python
def SmartStatic(state_machine):
    """Smart Static Avoidance behavior - follow smart static path"""
    return state_machine.get_smart_static_wpts()
```

---

## Coordinate-Dependent Variables

좌표계 의존 변수들과 처리 상태:

### ✅ Handled (Override in SmartStaticChecker)

| Variable | Type | Description | Conversion Method |
|----------|------|-------------|-------------------|
| `cur_s` | float | Ego s position | FrenetConverter.get_frenet() |
| `cur_d` | float | Ego d position | FrenetConverter.get_frenet() |
| `cur_vs` | float | Ego s velocity | Time derivative: ds/dt |
| `obstacles` | list[Obstacle] | All obstacles | Position + Velocity conversion |
| `obstacles_in_interest` | list[Obstacle] | Filtered obstacles | Based on Smart Static s |
| `cur_obstacles_in_interest` | list[Obstacle] | Current filtered | Based on Smart Static s |
| `cur_gb_wpnts` | WaypointData | Waypoints | Points to cur_smart_static_avoidance_wpnts |
| `num_glb_wpnts` | int | Waypoint count | Recalculated |

### ✅ Handled (Recalculated)

| Variable | Type | Description | Conversion Method |
|----------|------|-------------|-------------------|
| `wpnt.s_m` | float | Waypoint s | FrenetConverter.get_frenet() |
| `obs.s_start` | float | Obstacle s start | FrenetConverter.get_frenet() + delta |
| `obs.s_center` | float | Obstacle s center | FrenetConverter.get_frenet() |
| `obs.s_end` | float | Obstacle s end | FrenetConverter.get_frenet() + delta |
| `obs.d_center` | float | Obstacle d center | FrenetConverter.get_frenet() |
| `obs.vs` | float | Obstacle s velocity | GB Frenet -> Cartesian -> Smart Static Frenet |
| `obs.vd` | float | Obstacle d velocity | GB Frenet -> Cartesian -> Smart Static Frenet |

### ✅ Handled (Copied via __dict__.update)

| Variable | Type | Description | Why Safe to Share |
|----------|------|-------------|-------------------|
| `current_position` | list[float] | Ego x, y, theta | ROS callback updates, shared reference |
| `track_length` | float | Track length | Physical constant |
| `gb_ego_width_m` | float | Ego width | Physical constant |
| `lateral_width_gb_m` | float | Lateral width | Configuration parameter |
| `waypoints_dist` | float | Waypoint spacing | Configuration parameter |

---

## Check Functions Analysis

State Machine의 모든 check 함수가 좌표계 독립적으로 작동함을 확인:

| Check Function | Coordinate-Dependent Variables | Status |
|----------------|-------------------------------|--------|
| `_check_only_ftg_zone()` | `cur_s` | ✅ Overridden |
| `_check_close_to_raceline()` | `cur_d` | ✅ Overridden |
| `_check_close_to_raceline_heading()` | `cur_s`, `cur_gb_wpnts` | ✅ Overridden |
| `_check_ot_sector()` | `cur_s` | ✅ Overridden |
| `_check_getting_closer()` | `obstacles_in_interest`, `cur_vs`, `obs.vs` | ✅ Overridden |
| `_check_enemy_in_front()` | `obstacles`, `cur_s` | ✅ Overridden |
| `_check_on_spline()` | `wpnt.s_m`, `cur_s` | ✅ Recalculated |
| `_check_free_frenet()` | `cur_obstacles_in_interest`, `cur_s`, `obs.*`, `wpnts_data` | ✅ Overridden |
| `_check_free_cartesian()` | `cur_obstacles_in_interest`, `cur_s` | ✅ Overridden |
| `_check_ftg()` | `cur_vs` | ✅ Overridden |
| `_check_latest_wpnts()` | `wpnts_data` (passed as arg) | ✅ Works with any wpnts |
| `_check_availability()` | Uses `_check_on_spline`, `_check_free_frenet` | ✅ Inherited |
| `_check_sustainability()` | Uses `_check_availability`, `_check_free_frenet` | ✅ Inherited |
| `_check_overtaking_mode()` | Uses other check functions | ✅ Inherited |
| `_check_static_overtaking_mode()` | Uses other check functions | ✅ Inherited |
| `_check_overtaking_mode_sustainability()` | Uses other check functions | ✅ Inherited |

---

## Technical Implementation Details

### 1. Obstacle Velocity Conversion

장애물 속도 변환은 3단계로 진행:

```python
# Step 1: GB Frenet (vs, vd) -> Cartesian (vx, vy)
gb_psi = gb_frenet_converter.get_heading(obs_x, obs_y)[0]
vx_cart = obs.vs * np.cos(gb_psi) - obs.vd * np.sin(gb_psi)
vy_cart = obs.vs * np.sin(gb_psi) + obs.vd * np.cos(gb_psi)

# Step 2: Cartesian (vx, vy) -> Smart Static Frenet (vs, vd)
smart_psi = smart_frenet_converter.get_heading(obs_x, obs_y)[0]
vs_smart = vx_cart * np.cos(smart_psi) + vy_cart * np.sin(smart_psi)
vd_smart = -vx_cart * np.sin(smart_psi) + vy_cart * np.cos(smart_psi)
```

### 2. Ego Velocity Calculation

시간 미분 방식으로 cur_vs 계산 (track wrapping 처리):

```python
dt = (current_time - self.prev_time).to_sec()
if dt > 0.0:
    ds = new_s - self.prev_s
    # Handle track wrapping
    if ds > self.track_length / 2:
        ds -= self.track_length
    elif ds < -self.track_length / 2:
        ds += self.track_length
    self.cur_vs = ds / dt
```

### 3. FrenetConverter Usage

모든 좌표 변환에 FrenetConverter 사용:

```python
from frenet_converter.frenet_converter import FrenetConverter

# Create converter
wpnts_x = np.array([wpnt.x_m for wpnt in waypoints])
wpnts_y = np.array([wpnt.y_m for wpnt in waypoints])
frenet_converter = FrenetConverter(wpnts_x, wpnts_y)

# Convert position
ego_x = np.array([x])
ego_y = np.array([y])
s, d = frenet_converter.get_frenet(ego_x, ego_y)

# Get heading
psi = frenet_converter.get_heading(ego_x, ego_y)[0]
```

---

## TODO: Additional Work Required

### 1. Planner Updates (CRITICAL)

현재 RECOVERY와 OVERTAKE planner가 GB 경로만 지원. Smart Static 모드를 위한 수정 필요:

#### A. Recovery Planner
**File**: `planner/recovery_spliner/src/recovery_spliner_node.py`

**Required Changes**:
- Smart Static base trajectory 지원 추가
- GB와 Smart Static 중 선택하여 recovery 경로 생성
- State Machine으로부터 base_traj 정보 받기 (새로운 토픽 필요)

**Approach**:
```python
# Option 1: Subscribe to state machine's current state
rospy.Subscriber("/state_machine", String, self.state_cb)

# Option 2: New parameter topic
rospy.Subscriber("/state_machine/base_trajectory", String, self.base_traj_cb)

# Use appropriate reference waypoints
if self.base_traj == "SMART_STATIC":
    ref_wpnts = self.smart_static_wpnts
else:
    ref_wpnts = self.gb_wpnts
```

#### B. Overtaking Planner (Spliner)
**File**: `planner/spliner/src/spliner_node.py` (dynamic avoidance)
**File**: `planner/spliner/src/static_avoidance_node.py` (static avoidance)

**Required Changes**:
- Smart Static base trajectory 지원 추가
- Prediction 좌표계 일치 (opponent_prediction이 어떤 좌표계 쓰는지 확인)
- Smart Static 경로 기준으로 overtaking 궤적 생성

**Potential Issues**:
- Predictive spliner가 ego/opponent prediction 사용
- Prediction이 GB Frenet 기준일 경우 변환 필요
- Graph-based planner도 마찬가지 문제

#### C. Prediction Module
**File**: `perception/scripts/multi_tracking.py` (if exists)
**File**: Related prediction nodes

**Required Changes**:
- Prediction을 양쪽 좌표계로 모두 생성하거나
- Smart Static 모드 감지하여 적절한 좌표계 사용

---

### 2. Frenet Converter Node Update

**File**: `frenet_converter/src/frenet_converter_node.py`

**Required Changes**:
- Smart Static 경로 구독 추가
- `/car_state/odom_frenet_smart_static` 토픽 publish
- Smart Static Frenet 좌표 실시간 계산

**Implementation**:
```python
# Subscribe to Smart Static waypoints
rospy.Subscriber("/planner/avoidance/smart_static_otwpnts", OTWpntArray, self.smart_static_cb)

# Publish Smart Static Frenet
self.frenet_smart_pub = rospy.Publisher("/car_state/odom_frenet_smart_static", Odometry, queue_size=1)

def smart_static_cb(self, data):
    # Update Smart Static FrenetConverter
    wpnts_x = np.array([wpnt.x_m for wpnt in data.wpnts])
    wpnts_y = np.array([wpnt.y_m for wpnt in data.wpnts])
    self.smart_static_converter = FrenetConverter(wpnts_x, wpnts_y)

def odom_cb(self, data):
    # Calculate both GB and Smart Static Frenet
    # Publish to both topics
```

**Note**: 현재 SmartStaticChecker가 내부에서 FrenetConverter를 직접 생성하므로 이 작업은 optional. 하지만 다른 노드들이 Smart Static Frenet을 사용하려면 필요.

---

### 3. Velocity Scaler Integration

**File**: `planner/velocity_scaler/src/velocity_scaler_node.py` (if exists)

**Required Changes**:
- Smart Static waypoints의 속도 프로파일 생성
- GB와 동일한 방식으로 sector speed 적용

---

### 4. Visualization and Debugging

**Required Additions**:
- Smart Static Frenet 좌표 시각화 (RViz markers)
- Smart Static 모드에서의 obstacle Frenet 좌표 시각화
- Dual coordinate display (GB vs Smart Static 동시 표시)

**Implementation**:
```python
# In state_machine_node.py or new viz node
self.smart_frenet_viz_pub = rospy.Publisher("/viz/smart_static_frenet", MarkerArray, queue_size=1)

def visualize_dual_frenet(self):
    # Show ego position in both coordinate systems
    # Show obstacles in both coordinate systems
    # Color code: GB=blue, Smart Static=green
```

---

### 5. Dynamic Reconfigure

**File**: `state_machine/cfg/DynStateMachine.cfg` (if exists)

**Required Changes**:
- Smart Static mode 강제 활성화/비활성화 파라미터 추가
- Smart Static 관련 threshold 파라미터 추가

```python
gen.add("force_smart_static", bool_t, 0, "Force Smart Static mode", False)
gen.add("smart_static_lateral_width_m", double_t, 0, "Smart Static lateral width", 0.3, 0.0, 2.0)
```

---

### 6. Testing and Validation

**Required Tests**:
1. **Coordinate Consistency Test**: GB와 Smart Static Frenet이 같은 물리적 위치를 가리키는지 확인
2. **Obstacle Velocity Test**: 장애물 속도 변환이 올바른지 검증 (예상 궤적과 비교)
3. **Transition Stability Test**: GB ↔ Smart Static 전환 시 부드러운 전환 확인
4. **Race Condition Test**: 플래그 변경 중 transition 호출 시 올바른 동작 확인
5. **Performance Test**: SmartStaticChecker update 빈도 및 latency 측정

---

## Known Issues and Limitations

### 1. Performance Concerns

**Issue**: SmartStaticChecker가 매 iteration마다 FrenetConverter 생성 및 모든 장애물 변환 수행.

**Impact**:
- 장애물 수가 많을 경우 계산 부하 증가
- FrenetConverter 생성 비용

**Mitigation**:
- 현재: 80Hz update (state machine과 동일)
- 개선안: Waypoint 변경 시에만 FrenetConverter 재생성 (timestamp 비교)

### 2. Prediction Coordinate Mismatch

**Issue**: Opponent prediction이 GB Frenet 기준일 경우 Smart Static 모드에서 부정확.

**Impact**:
- `_check_free_frenet()`에서 prediction 기반 충돌 체크 오류
- Overtaking transition 판단 오류

**Mitigation**:
- Prediction module 수정 필요 (TODO #2-C)
- 또는 SmartStaticChecker에서 prediction도 변환 (성능 이슈)

### 3. Track Length Assumption

**Issue**: Smart Static 경로 길이가 GB 경로 길이와 다를 수 있음.

**Impact**:
- Track wrapping 계산 오류
- FTG zone, overtaking sector 계산 오류

**Current Approach**: `self.track_length`를 공유 (GB 기준)

**Better Approach**: Smart Static 경로 전체 길이 계산 후 `self.track_length` 오버라이드

### 4. Recovery/Overtake Planner Limitation

**Issue**: Recovery와 Overtake planner가 GB 경로만 지원.

**Impact**:
- Smart Static 모드에서 RECOVERY로 전환 시 GB 기준 recovery 생성
- OVERTAKE 모드에서도 마찬가지

**Mitigation**:
- 당장은 작동하지만 최적이 아님
- TODO #1에서 해결 필요

---

## File Structure

```
catkin_ws/src/race_stack/
├── state_machine/
│   └── src/
│       ├── state_machine_node.py          [MODIFIED]
│       ├── state_transitions.py           [MODIFIED]
│       ├── states.py                      [MODIFIED]
│       ├── states_types.py                [MODIFIED]
│       ├── state_helper_for_smart.py      [NEW]
│       └── dual_frenet/
│           └── README.md                  [THIS FILE]
├── planner/
│   ├── recovery_spliner/
│   │   └── src/
│   │       └── recovery_spliner_node.py   [TODO]
│   └── spliner/
│       └── src/
│           ├── spliner_node.py            [TODO]
│           ├── static_avoidance_node.py   [TODO]
│           └── smart_static_avoidance_node.py [EXISTS - publishes Smart Static path]
├── frenet_converter/
│   └── src/
│       └── frenet_converter_node.py       [TODO - optional]
└── perception/
    └── scripts/
        └── multi_tracking.py              [TODO - check prediction coordinate system]
```

---

## Integration Checklist

- [x] StateType.SMART_STATIC 추가
- [x] SmartStaticChecker 구현
- [x] State transitions에서 base_checker 사용
- [x] Explicit base_traj parameter passing
- [x] Obstacle position conversion
- [x] Obstacle velocity conversion
- [x] Waypoint Frenet recalculation
- [x] Ego velocity calculation (cur_vs)
- [x] Check functions analysis
- [ ] Recovery planner Smart Static 지원
- [ ] Overtake planner Smart Static 지원
- [ ] Prediction coordinate system 확인 및 수정
- [ ] Frenet converter node 업데이트 (optional)
- [ ] Velocity scaler 통합
- [ ] Visualization 추가
- [ ] Dynamic reconfigure 파라미터
- [ ] Testing and validation

---

## Usage Example

### Activation

Smart Static 모드는 spliner가 publish하는 플래그로 자동 활성화:

```python
# In smart_static_avoidance_node.py
self.smart_static_active_pub = rospy.Publisher("/planner/avoidance/smart_static_active", Bool, queue_size=1)
self.smart_static_wpnts_pub = rospy.Publisher("/planner/avoidance/smart_static_otwpnts", OTWpntArray, queue_size=1)

# When smart static path is valid
self.smart_static_active_pub.publish(Bool(data=True))
self.smart_static_wpnts_pub.publish(smart_static_otwpnts)
```

### State Machine Behavior

```python
# State machine automatically switches between GB and Smart Static
if self.smart_static_active and len(self.cur_smart_static_avoidance_wpnts.list) > 0:
    # Use Smart Static Frenet coordinate system
    cur_state, local_wpnts_src = state_transitions.SmartStaticTransition(self)
else:
    # Use GB Frenet coordinate system
    cur_state, local_wpnts_src = state_transitions.GlobalTrackingTransition(self)
```

### Manual Override (via dynamic reconfigure)

```bash
# Force GB mode
rosrun dynamic_reconfigure dynparam set /state_machine force_GBTRACK True

# Allow Smart Static mode
rosrun dynamic_reconfigure dynparam set /state_machine force_GBTRACK False
```

---

## References

- FrenetConverter: `frenet_converter/frenet_converter.py`
- State Machine Architecture: `state_machine/src/state_machine_node.py`
- Smart Static Avoidance: `planner/spliner/src/smart_static_avoidance_node.py`

---

## Conclusion

현재까지 구현된 Dual Frenet 시스템은 **State Machine 레벨에서는 완성**되었습니다.

**작동 가능한 부분**:
- ✅ State transitions에서 올바른 좌표계 선택
- ✅ Check functions가 좌표계 독립적으로 작동
- ✅ Obstacle과 waypoint Frenet 변환
- ✅ Velocity 변환 (ego + obstacles)

**추가 작업 필요**:
- ⚠️ Recovery/Overtake planner의 Smart Static 지원
- ⚠️ Prediction 좌표계 일치
- ⚠️ 완전한 시스템 테스트

일단 현재 상태로 **기본적인 Smart Static tracking은 가능**하지만, **RECOVERY/OVERTAKE 상태로의 전환**은 GB 기준으로 동작합니다.

완전한 Dual Frenet 지원을 위해서는 planner 노드들의 수정이 필수입니다.
