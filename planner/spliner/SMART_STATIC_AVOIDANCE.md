# Smart Static Avoidance System

> GB Optimizer 기반 정적 장애물 회피 시스템 - 학습된 장애물 위치를 활용한 최적 경로 생성

**파일**: `smart_static_avoidance_node.py` (2,838 lines)
**작성**: HJ & Team | **버전**: v2.0

---

## 1. 시스템 목적

트랙에 **고정된 정적 장애물**이 있을 때:
- 실시간 Spline 회피로는 **최적 경로를 찾기 어렵다** (매 순간 즉각 반응)
- GB Optimizer를 사용하여 **장애물 위치를 고려한 최적 경로**를 생성
- 장애물이 사라지면 **자동으로 원래 GB raceline으로 복귀**

**핵심 아이디어**: "장애물 위치를 학습한 후, 최적화된 고정 경로를 생성하자"

---

## 2. 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT                                                         │
├─────────────────────────────────────────────────────────────┤
│ • /tracking/obstacles        - 감지된 장애물 (LiDAR)         │
│ • /car_state/odom_frenet     - 차량 위치 (s, d)             │
│ • /global_waypoints   - GB raceline                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: PRE-FIX (장애물 학습)                                │
├─────────────────────────────────────────────────────────────┤
│ 1. 장애물 감지 → Memory에 (s, d) 좌표 누적                     │
│ 2. 안정성 검증 (obs_count≥5, std<0.05m)                       │
│ 3. GB raceline 간섭 체크 (lateral_width_gb_m)                │
│ 4. Fixed Path 생성 조건 충족?                                 │
│    - Stage 1: 2개 이상 장애물 섹터에 안정적 장애물 각 1개 이상 존재               │
│    - Stage 2: 그 중 1개 이상이 GB raceline 간섭               │
│    → 둘 다 YES: GB Optimizer 실행 (background thread)        │
│                                                               │
│ OUTPUT: /planner/avoidance/otwpnts (20Hz, 실시간 Spline 회피)       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: POST-FIX (Fixed Path 사용)                          │
├─────────────────────────────────────────────────────────────┤
│ 1. use_fixed_path = True → State Machine이 Fixed Path 선택   │
│ 2. Zone Passage Monitoring 시작                              │
│    - 간섭 장애물이 있던 섹터만 모니터링                         │
│    - 차량이 해당 섹터 통과할 때 장애물 확인                     │
│ 3. GB 복귀 조건 체크                                          │
│    - 섹터의 지정 % 지점까지 간섭 장애물 없음?                   │
│    → YES: use_fixed_path = False (GB 복귀!)                  │
│                                                               │
│ OUTPUT: /planner/avoidance/smart_static_otwpnts (0.5Hz)      │
│         /planner/avoidance/smart_static_active (Bool)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 개념

### 3.1 Memory System (장애물 학습)

**목적**: 장애물의 안정적인 위치를 학습

```python
self.static_obs_memory = {
    (sector_id, obs_id): {
        's_history': [s1, s2, ...],      # GB Frenet s 좌표 이력
        'd_history': [d1, d2, ...],      # GB Frenet d 좌표 이력
        'obs_count': 10,                  # 관측 횟수
        'interferes': True,               # GB raceline 간섭 여부
        'interferes_fixed': False,        # Fixed path 간섭 여부 (POST-FIX)
        'last_seen': rospy.Time.now()
    }
}
```

**안정성 검증 기준**:
- `obs_count ≥ 5`: 최소 5회 이상 관측
- `std(s_history) < 0.05m` 및 `std(d_history) < 0.05m`: 위치 변동 5cm 이내

**타임아웃**: **없음** (Line 122: `memory_timeout_decay_amount = 0.0`)
- 경기 중 정지/재출발 시에도 메모리 유지

---

### 3.2 PRE-FIX Mode (실시간 회피 + 학습)

차량이 처음 장애물을 만났을 때의 모드:

#### 1) Spline 회피 (`do_spline()`, Line 1363-1584)
- 실시간으로 감지된 장애물 즉각 회피
- `/planner/avoidance/otwpnts` 20Hz 발행

#### 2) 동시에 메모리 학습
- 장애물 위치 (s, d) 누적
- 안정성 검증 완료된 장애물만 사용

#### 3) Fixed Path 생성 트리거 (Line 650-703)

**2-Stage Condition**:

```python
# Stage 1: is_ready
# - 2개 이상 섹터에 안정적(verified) 장애물 존재
# - 간섭 여부는 무관
is_ready = len(sectors_with_verified_obs) >= 2

# Stage 2: should_generate
# - 그 중 1개 이상이 GB raceline에 간섭
should_generate = interfering_count > 0

# 둘 다 True일 때만 Fixed Path 생성
if is_ready and should_generate:
    self._generate_fixed_path()
```

**왜 2-Stage인가?**
- 장애물이 각 섹터에 1개 이상씩 있어야 안정적
- 하지만 GB에 간섭하지 않으면 최적화 불필요

---

### 3.3 POST-FIX Mode (Fixed Path 사용)

GB Optimizer로 최적 경로가 생성된 후:

#### 1) Fixed Path 사용
- `use_fixed_path = True`
- State Machine이 `/planner/avoidance/smart_static_otwpnts` 사용
- 0.5Hz로 발행 (계산 부하 최소화, Line 514)
- Timestamp는 매 publish마다 업데이트 (Line 514-516)

#### 2) Zone Passage Monitoring (Line 906-1016)

**목적**: 장애물이 사라졌는지 감지하여 GB로 복귀

**Detection Zone 정의**:
```python
# Line 919-920
sector_start_margin = 3.0           # 섹터 시작 전 margin (m)
sector_check_threshold = 0.8        # 섹터의 80%까지 체크 (기본값, 가변)

# Zone 범위 계산
check_start = (s_start - sector_start_margin) % track_length
check_end = (s_start + sector_length * sector_check_threshold) % track_length
```

**Zone 진입 (-3m 지점)**:
```python
# Line 960-988
if in_zone and not monitor['in_zone']:  # 진입 순간
    monitor['in_zone'] = True
    monitor['had_obstacles'] = False  # 리셋
    monitor['entered_at_s'] = cur_s_norm
```

**Zone 통과 중**:
```python
# Line 972-976: 현재 섹터의 간섭 장애물만 체크
interfering_obs = [obs for obs in self.latest_obstacles
                  if obs.in_static_obs_sector
                  and obs.is_static
                  and obs.sector_id == sector_id  # 이 섹터만!
                  and self._check_obstacle_interference(obs)]

# Line 991-994: 장애물 발견 시
if interfering_obs:
    monitor['had_obstacles'] = True
```

**Zone 퇴장 (지정% 지점)**:
```python
# Line 1000-1015
if not vehicle_in_zone and monitor['in_zone']:  # 퇴장 순간
    monitor['in_zone'] = False

    if not monitor['had_obstacles']:  # 장애물 없었다!
        self.use_fixed_path = False  # GB 복귀!
```

---

### 3.4 Sector-Specific Tracking

**왜 필요한가?**

트랙에 여러 섹터(0, 1, 2, ...)가 있을 때:
- **섹터 0, 2**: 간섭 장애물 있음 → Fixed Path 생성 원인
- **섹터 1**: 장애물은 있지만 간섭 안함 (경로에서 멀리 떨어짐)

만약 **모든 섹터를 모니터링**하면:
- 섹터 1 통과 시 "간섭 장애물 없음"으로 오판 → **잘못된 GB 복귀**

**해결책**: Fixed Path 생성 시 **간섭 장애물이 있던 섹터 ID만 저장**

```python
# Line 615, 638, 643 (_get_verified_obstacles)
interfering_sectors = set()

for (sector_id, obs_id), mem in self.static_obs_memory.items():
    if mem['interferes']:  # GB raceline 간섭하는 장애물만
        # ... 안정성 검증 ...
        interfering_sectors.add(sector_id)

self.sectors_with_interfering_obs = interfering_sectors  # {0, 2}
```

나중에 Zone Monitoring에서:
```python
# Line 928-931
if sector_id not in self.sectors_with_interfering_obs:
    rospy.loginfo_throttle(5.0, "Skipping sector X (no interfering obstacles)")
    continue  # 간섭 없던 섹터는 스킵!
```

---

## 4. 주요 함수

### 4.1 `_process_obstacles()` (Line 744-1082)

**역할**: 메인 처리 루프 - 매 obstacle topic마다 실행 (20Hz)

**주요 작업**:
1. 메모리 업데이트 (`_update_obstacle_memory()`)
2. 메모리 타임아웃 체크 (현재는 비활성화)
3. Zone passage monitoring (POST-FIX일 때)
4. `use_fixed_path` 플래그 관리

---

### 4.2 `_check_obstacle_interference()` (Line 526-570)

**역할**: 장애물이 GB raceline을 간섭하는지 판단

**알고리즘**:
```python
# 차량 전방 10m 구간의 raceline waypoints 추출
ego_path = raceline[ego_s - 1.0 : ego_s + 10.0]

# 각 waypoint에서 장애물까지 최소 거리 계산
for waypoint in ego_path:
    free_dist = min_distance(waypoint, obstacle_corners)

    if free_dist < lateral_width_gb_m:  # 기본 0.1m
        return True  # 간섭!

return False  # 간섭 없음
```

**파라미터**:
- `gb_ego_width_m`: 0.3m (차량 폭)
- `lateral_width_gb_m`: 0.1m (간섭 판단 threshold)

---

### 4.3 `_get_verified_obstacles()` (Line 603-647)

**역할**: 메모리에서 **안정적이고 간섭하는** 장애물만 추출

**검증 기준**:
1. `interferes == True` (GB raceline 간섭)
2. `obs_count ≥ min_stable_observations` (기본 5)
3. `std(s_history) < 0.05m` 및 `std(d_history) < 0.05m`

**중요**: 여기서 `sectors_with_interfering_obs` 업데이트! (Line 643)

**반환값**: `[(sector_id, obs_id, s_mean, d_mean), ...]`

---

### 4.4 `_check_obs_is_ready_for_path_gen()` (Line 650-703)

**역할**: Fixed Path 생성 조건 체크 (2-Stage)

**Stage 1 - is_ready**:
```python
# 2개 이상 섹터에 안정적(verified) 장애물 존재
is_ready = len(sectors_with_verified_obs) >= self.min_sectors_with_stable_obs  # 기본 2
```

**Stage 2 - should_generate**:
```python
# 1개 이상의 간섭 장애물 존재
should_generate = interfering_count > 0
```

**최종 판단**:
```python
return is_ready and should_generate
```

---

### 4.5 `_generate_fixed_path()` (Line 1660-1822)

**역할**: GB Optimizer를 실행하여 Fixed Path 생성

**입력**:
- `verified_obs`: 검증된 간섭 장애물 리스트

**Retry 전략** (Line 1683):
```python
safety_width_ratios = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]  # 7단계

for ratio in safety_width_ratios:
    safety_width_adjusted = self.safety_width * ratio
    result = gb_optimizer.optimize(obstacles, safety_width=safety_width_adjusted)
    if result.success:
        break
```

**출력**:
- `self.fixed_path_wpnts` (OTWpntArray)
- `self.fixed_path_markers` (초록색 시각화)
- `self.fixed_path_generated = True` (영구 플래그)

**특이사항**: 마지막 waypoint 제거 (Line 1744-1748, closed loop 방지)

---

### 4.6 `do_spline()` (Line 1363-1584)

**역할**: 실시간 Spline 회피 경로 생성

**PRE-FIX vs POST-FIX**:
- **PRE-FIX**: 현재 감지된 장애물 (`self.latest_obstacles`)
- **POST-FIX**: 메모리의 간섭 장애물 (`verified_obstacles`)

**알고리즘**:
1. 차량 전방 장애물 필터링
2. Lateral offset 계산 (장애물 회피 방향)
3. Cubic spline fitting
4. Waypoint 생성

---

## 5. 데이터 구조

### 5.1 Memory (Line 133)
```python
self.static_obs_memory = {}  # (sector_id, obs_id) → obstacle info
```

### 5.2 Flags
```python
self.fixed_path_generated = False  # Static: 한 번 True면 영구
self.use_fixed_path = False         # Dynamic: 상황에 따라 toggle
```

### 5.3 Sector Tracking (Line 130-133)
```python
self.sectors_with_interfering_obs = set()  # {0, 2, 5, ...}

self.sector_monitoring = {
    sector_id: {
        'in_zone': bool,        # 차량이 zone 내부?
        'had_obstacles': bool,  # Zone 통과 중 장애물 감지?
        'entered_at_s': float   # 진입 시 s 좌표
    }
}
```

---

## 6. 파라미터

### 6.1 Memory 관련
```python
self.min_stable_observations = 5          # 최소 관측 횟수
self.position_std_threshold = 0.05        # 위치 표준편차 threshold (m)
self.memory_timeout_decay_amount = 0.0   # 타임아웃 비활성화! (Line 122)
```

### 6.2 Interference 판단
```python
self.gb_ego_width_m = 0.3                # 차량 폭 (m)
self.lateral_width_gb_m = 0.1            # GB 간섭 threshold (m)
self.lateral_width_fixed_m = 0.1         # Fixed path 간섭 threshold
```

### 6.3 Zone Monitoring (Line 919-920)
```python
sector_start_margin = 0.0                # 섹터 시작 margin (m)
sector_check_threshold = 0.8             # 섹터 80% 지점까지 체크 (가변)
```

### 6.4 Path Generation 조건 (Line 686)
```python
self.min_sectors_with_stable_obs = 2     # 최소 섹터 개수
```

### 6.5 GB Optimizer
```python
safety_width_ratios = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]  # Retry 7단계
```

---

## 7. ROS Topics

### 7.1 Subscribed
| Topic | Type | Frequency | 용도 |
|-------|------|-----------|------|
| `/tracking/obstacles` | `ObstacleArray` | 40Hz | LiDAR 장애물 정보 |
| `/car_state/odom_frenet` | `OdomFrenet` | 100Hz | 차량 위치 (s, d) |
| `/global_waypoints` | `OTWpntArray` | 1Hz | GB raceline |

### 7.2 Published
| Topic | Type | Frequency | 용도 |
|-------|------|-----------|------|
| `/planner/avoidance/otwpnts` | `OTWpntArray` | 20Hz | 실시간 Spline 경로 |
| `/planner/avoidance/smart_static_otwpnts` | `OTWpntArray` | 0.5Hz | Fixed Path |
| `/planner/avoidance/smart_static_active` | `Bool` | 20Hz | use_fixed_path 플래그 |
| `/planner/avoidance/markers` | `MarkerArray` | 20Hz | Spline 시각화 (보라색) |
| `/planner/avoidance/smart_markers` | `MarkerArray` | 0.5Hz | Fixed Path 시각화 (초록색) |

---

## 8. State Machine 통합

### 8.1 State Machine Node (`state_machine_node.py`)

**Callback** (Line 94-97):
```python
def smart_static_avoidance_cb(self, data: OTWpntArray):
    """Smart static avoidance waypoints from GB optimizer fixed path"""
    self.smart_static_wpnts = data  # 메시지 저장!
    self.cur_smart_static_avoidance_wpnts.initialize_traj(data)
```

**State Transition**:
```python
# GB_TRACK → SMART_STATIC 조건
if wpnts_valid and active:
    return State.SMART_STATIC

# SMART_STATIC → GB_TRACK 조건
if not active or not wpnts_valid:
    return State.GB_TRACK
```

**Validation**:
- `wpnts_valid`: Timestamp 5초 이내 (Line 514-516에서 매번 업데이트)
- `active`: `/planner/avoidance/smart_static_active == True`

---

## 9. Rviz

### 9.1 RViz 시각화

**Markers**:
- **보라색** (`/planner/avoidance/markers`): 실시간 Spline 경로
- **파란색** (`/planner/avoidance/smart_markers`): Fixed Path

**Obstacles**:
- `/tracking/obstacle_markers`: 감지된 장애물

### 9.2 Topic 모니터링

```bash
# use_fixed_path 상태 확인
rostopic echo /planner/avoidance/smart_static_active

# Fixed Path waypoints 확인
rostopic echo /planner/avoidance/smart_static_otwpnts

# State Machine 상태 확인
rostopic echo /state_machine/current_state
```


## 10. 주요 코드 위치

| 기능 | 함수/위치 | 라인 | 설명 |
|------|-----------|------|------|
| **메인 루프** | `_process_obstacles()` | 744-1082 | 메모리 업데이트 + Zone monitoring |
| **간섭 체크** | `_check_obstacle_interference()` | 526-570 | GB raceline 간섭 판단 |
| **검증 장애물 추출** | `_get_verified_obstacles()` | 603-647 | 안정적 + 간섭 장애물 필터링 |
| **생성 조건 체크** | `_check_obs_is_ready_for_path_gen()` | 650-703 | 2-Stage 조건 확인 |
| **Fixed Path 생성** | `_generate_fixed_path()` | 1660-1822 | GB Optimizer 실행 |
| **Spline 회피** | `do_spline()` | 1363-1584 | 실시간 회피 경로 생성 |
| **Zone Monitoring** | (in `_process_obstacles()`) | 906-1016 | 섹터별 장애물 감지 + GB 복귀 |
| **Sector 저장** | (in `_get_verified_obstacles()`) | 643 | `sectors_with_interfering_obs` 업데이트 |
| **타임아웃 설정** | `__init__()` | 122 | `memory_timeout_decay_amount = 0.0` |
| **Timestamp 업데이트** | `_fixed_path_pub_timer_cb()` | 514-516 | 매 publish마다 갱신 |

---

## 11. 동작 예시

### 시나리오: 트랙에 3개 섹터, 2개 장애물

```
트랙 구조:
┌─────────────────────────────────────────┐
│ Sector 0: 장애물 A (GB raceline 간섭)     │
│ Sector 1: 장애물 없음                     │
│ Sector 2: 장애물 B (GB 간섭 없음, 경로 밖) │
└─────────────────────────────────────────┘
```

**Lap 1 (PRE-FIX)**:
1. 차량이 Sector 0 진입 → 장애물 A 감지
2. Spline 회피 + 메모리 학습 시작
3. Sector 2 진입 → 장애물 B 감지
4. 조건 체크:
   - `is_ready`: 2개 섹터 (0, 2) → **True**
   - `should_generate`: 1개 간섭 (A만 간섭) → **True**
5. GB Optimizer 실행 → Fixed Path 생성 완료
6. `sectors_with_interfering_obs = {0}`  (섹터 0만!)

**Lap 2-10 (POST-FIX)**:
1. `use_fixed_path = True` → Fixed Path 사용
2. Zone Monitoring 시작:
   - **Sector 0 모니터링**: 장애물 A 여전히 있음 → `had_obstacles = True`
   - **Sector 1 스킵**: 애초에 장애물 없었음
   - **Sector 2 스킵**: `sectors_with_interfering_obs`에 없음 (간섭 안했음)
3. Fixed Path 계속 사용

**Lap 11-20 (장애물 제거됨)**:
1. Sector 0 진입: `-3m` 지점에서 `in_zone = True`
2. Sector 0 통과 (0% → 80%): 장애물 A 없음 → `had_obstacles = False`
3. Sector 0 퇴장 (80% 지점):
   - `had_obstacles == False` → **use_fixed_path = False**
4. GB raceline으로 복귀!

---

## 12. 성능 특징

### 12.1 장점
- **최적 경로**: GB Optimizer로 전역 최적화 (Spline보다 빠름)
- **안정성**: 메모리 기반이라 노이즈에 강함 (5회 관측 + 표준편차 체크)
- **자동 복귀**: 장애물 제거 시 즉시 GB로 전환
- **선택적 모니터링**: 간섭 섹터만 추적 (효율적, false positive 방지)
- **정지/재출발 대응**: 시간이라는 개념 없이 위치와 인지로 해결

### 12.2 제약사항
- **정적 장애물 전용**: 움직이는 장애물은 Spline로 처리
- **학습 시간 필요**: 최소 5회 관측 필요 (첫 lap은 Spline)
- **GB Optimizer 의존**: Retry 실패 시 Spline 유지

---

## 13. TODO

### 13.1 우선순위

### 형준 
- 오프라인 경로로 스플라인 대신 활용
- More Space 코드 보기(스플라인용 png 제작 등등)
- Spliner 튜닝 연습
- 멀티 장애물 대응 for Overtaking (etc. FTG, 지능적 Filtering)
- Velocity Profile 튜닝 연습
- 차량 하드웨어 점검(직진성 등등) 

### 인영

- State Machine / State Transition 빈틈없게 하기
- Planner Reference 상황따라 바뀌게 수정
- Spliner 튜닝 연습
- 멀티 장애물 대응 for Overtaking (etc. FTG, 지능적 Filtering)

### 한교

- Detect(차 주변 필터링, 장애물 사이즈), Tracking Noise 실험
- Mapping, Localization, Adaptive odom 튜닝 연습

### 희도

- Velocity Profile 튜닝 연습
- 차량 하드웨어 점검(직진성 등등)
---
### 13.2 차순위

- Dual Frenet 시도(state machine, odom publisher, planner) (인영, 형준)

