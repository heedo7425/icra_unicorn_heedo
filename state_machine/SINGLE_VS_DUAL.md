# Single Frenet vs Dual Frenet Approach

## 문제 정의

Smart Static Avoidance에서 **Fixed path 생성 후 새로운 장애물**이 출현했을 때:
1. 새로운 장애물이 **Fixed path에 간섭하는지** 판단
2. 간섭하면 **Fixed path 기준으로 회피 경로** 생성 (`do_spline`)
3. State machine에서 **Fixed path vs Real-time spline** 선택

### 핵심 문제
- **기존**: GB raceline 기준 Frenet 좌표계 1개만 사용
- **새로운 요구사항**: Fixed path 기준으로도 장애물 간섭 판단 필요

---

## Option 1: Dual Frenet (2개의 Frenet 좌표계)

### 개념
Tracking 노드에서 **두 좌표계 모두 변환**하여 전달

```python
# Tracking 노드에서
tracking_obs = {
    'x': x, 'y': y,
    's_gb': s_gb,           'd_gb': d_gb,        # GB raceline 기준
    's_fixed': s_fixed,     'd_fixed': d_fixed   # Fixed path 기준
}
```

### 구현 세부사항

#### 1. Tracking 노드 수정
```python
class TrackingNode:
    def __init__(self):
        # Dual FrenetConverter
        self.gb_converter = FrenetConverter(gb_raceline_x, gb_raceline_y)
        self.fixed_converter = None  # 초기에는 None

        # Fixed path subscribe
        rospy.Subscriber("/planner/avoidance/smart_static_otwpnts",
                        OTWpntArray, self.fixed_path_cb)

    def fixed_path_cb(self, data: OTWpntArray):
        """Fixed path 수신 시 converter 업데이트"""
        if len(data.wpnts) > 0:
            x = [w.x_m for w in data.wpnts]
            y = [w.y_m for w in data.wpnts]
            self.fixed_converter = FrenetConverter(x, y)
            rospy.loginfo("[Tracking] Fixed path converter updated")

    def publish_obstacles(self, obstacles):
        """장애물 publish with dual Frenet"""
        obs_array = ObstacleArray()

        for obs in obstacles:
            # GB raceline Frenet
            s_gb, d_gb = self.gb_converter.get_frenet(obs.x, obs.y)

            # Fixed path Frenet (없으면 GB와 동일)
            if self.fixed_converter is not None:
                s_fixed, d_fixed = self.fixed_converter.get_frenet(obs.x, obs.y)
            else:
                s_fixed, d_fixed = s_gb, d_gb

            obs_msg = Obstacle()
            obs_msg.x = obs.x
            obs_msg.y = obs.y
            obs_msg.s_gb = s_gb
            obs_msg.d_gb = d_gb
            obs_msg.s_fixed = s_fixed      # 추가
            obs_msg.d_fixed = d_fixed      # 추가

            obs_array.obstacles.append(obs_msg)

        self.pub.publish(obs_array)
```

#### 2. Smart Static Node 수정
```python
def _process_obstacles(self):
    """장애물 처리 - Frenet 좌표계 선택"""

    for obs in self.latest_obstacles:
        # Fixed path 생성 여부에 따라 좌표계 선택
        if self.fixed_path_generated:
            s, d = obs.s_fixed, obs.d_fixed  # Fixed path 기준
        else:
            s, d = obs.s_gb, obs.d_gb        # GB raceline 기준

        # 간섭 체크 (선택된 좌표계 기준)
        interferes = self._check_interference(s, d)

        # Memory 업데이트
        key = (obs.sector_id, obs.id)
        if key not in self.static_obs_memory:
            self.static_obs_memory[key] = {
                's_history': [],
                'd_history': [],
                'interferes': interferes,
                ...
            }

        self.static_obs_memory[key]['s_history'].append(s)
        self.static_obs_memory[key]['d_history'].append(d)

def do_spline(self, obs, gb_wpnts):
    """Spline 생성 - 선택된 좌표계 기준"""
    if self.fixed_path_generated:
        # Fixed path 기준으로 spline 생성
        reference_wpnts = self.fixed_path_wpnts
        obs_s = obs.s_fixed
        obs_d = obs.d_fixed
    else:
        # GB raceline 기준으로 spline 생성
        reference_wpnts = gb_wpnts
        obs_s = obs.s_gb
        obs_d = obs.d_gb

    # Spline 생성 로직 (기존과 동일, reference만 변경)
    ...
```

#### 3. State Machine 수정
```python
class StateMachine:
    def __init__(self):
        self.cur_s_gb = 0.0
        self.cur_d_gb = 0.0
        self.cur_s_fixed = 0.0     # 추가
        self.cur_d_fixed = 0.0     # 추가

        self.using_fixed_path = False

    def frenet_pose_cb(self, data: Odometry):
        """Dual Frenet pose callback"""
        # GB raceline 기준 (기존)
        self.cur_s_gb = data.pose.pose.position.x
        self.cur_d_gb = data.pose.pose.position.y

        # Fixed path 기준 (새로 추가, 별도 토픽 필요)
        # TODO: car_state에서 dual frenet publish 필요

    def _check_free_frenet(self, wpnts):
        """Frenet free 체크 - 좌표계 선택"""
        if self.using_fixed_path:
            return abs(self.cur_d_fixed) < threshold
        else:
            return abs(self.cur_d_gb) < threshold
```

### 장점
✅ **정확도**: 각 경로의 실제 Frenet 좌표 사용
✅ **성능**: Tracking에서 한 번만 변환 (20Hz)
✅ **명확성**: 코드가 깔끔 - "어떤 좌표계 쓸지" 선택만
✅ **확장성**: 새로운 경로 추가 시 같은 패턴 적용 가능

### 단점
❌ **수정 범위 넓음**: Tracking, Smart Static, State Machine, car_state 모두 수정
❌ **메시지 크기**: ObstacleArray 메시지 크기 2배 (s, d가 2세트)
❌ **불필요한 연산**: Fixed path 없을 때도 변환 (초기 단계)
❌ **복잡도 증가**: 시스템 전체 복잡도 증가

---

## Option 2: Single Frenet (1개 + 경로 변환)

### 개념
**GB raceline Frenet만 사용**, Fixed path와의 lateral distance를 **매번 계산**

```python
# Tracking 노드는 그대로 (GB raceline 기준만)
tracking_obs = {
    'x': x, 'y': y,
    's_gb': s_gb, 'd_gb': d_gb
}

# Smart Static Node에서 변환
if self.fixed_path_generated:
    # Fixed path를 GB raceline 기준으로 해석
    lateral_offset = calculate_lateral_offset_to_fixed_path(obs.s_gb, obs.d_gb)
```

### 구현 세부사항

#### 1. Fixed Path Lateral Distance 계산
```python
def _get_lateral_distance_to_fixed_path(self, obs_s_gb: float, obs_x: float, obs_y: float) -> float:
    """
    Obstacle의 GB raceline s 좌표에서 Fixed path까지의 lateral distance 계산.

    Args:
        obs_s_gb: Obstacle의 GB raceline s 좌표
        obs_x, obs_y: Obstacle의 XY 좌표

    Returns:
        lateral_distance: Fixed path로부터의 lateral distance (왼쪽 +, 오른쪽 -)
    """
    # 1. obs_s_gb에 가장 가까운 fixed path waypoint 찾기
    min_dist = float('inf')
    closest_idx = 0

    for i, wpnt in enumerate(self.fixed_path_wpnts.wpnts):
        # GB raceline s 좌표 차이 (wraparound 고려)
        s_diff = abs(wpnt.s_m - obs_s_gb)
        s_diff_wrap = min(s_diff, self.gb_max_s - s_diff)

        if s_diff_wrap < min_dist:
            min_dist = s_diff_wrap
            closest_idx = i

    # 2. Closest waypoint에서 obstacle까지의 벡터
    wpnt = self.fixed_path_wpnts.wpnts[closest_idx]
    dx = obs_x - wpnt.x_m
    dy = obs_y - wpnt.y_m

    # 3. Waypoint의 heading 기준 lateral distance 계산
    # psi_rad: waypoint의 heading
    # Lateral = dx * sin(psi) - dy * cos(psi)
    # (왼손 법칙: 왼쪽 +, 오른쪽 -)
    lateral = dx * np.sin(wpnt.psi_rad) - dy * np.cos(wpnt.psi_rad)

    return lateral
```

#### 2. 간섭 체크 (Fixed Path 기준)
```python
def _check_interference_fixed_path(self, obs_s_gb: float, obs_x: float, obs_y: float) -> bool:
    """
    Fixed path 기준으로 장애물 간섭 체크.

    GB raceline의 간섭 체크와 유사하지만, Fixed path를 기준으로 판단.
    """
    # Fixed path까지의 lateral distance
    lateral_dist = self._get_lateral_distance_to_fixed_path(obs_s_gb, obs_x, obs_y)

    # 간섭 threshold (차량 폭 + lateral margin)
    threshold = self.gb_ego_width_m + self.lateral_width_gb_m

    # Lateral distance가 threshold 이내면 간섭
    return abs(lateral_dist) < threshold
```

#### 3. Do Spline 수정 (Fixed Path Reference)
```python
def do_spline(self, obs, gb_wpnts):
    """
    Spline 생성 - Fixed path 생성 여부에 따라 reference 선택.
    """
    if self.fixed_path_generated:
        # Fixed path를 reference로 사용
        reference_wpnts = self.fixed_path_wpnts

        # Obstacle의 lateral distance (Fixed path 기준)
        obs_lateral = self._get_lateral_distance_to_fixed_path(
            obs.s_center, obs.x_center, obs.y_center
        )

        # Spline 생성 시 obs_lateral 사용
        # (기존 d_m 대신 lateral distance 사용)
        spline_points = self._generate_spline_points(
            obs_s=obs.s_center,
            obs_lateral=obs_lateral,
            reference=reference_wpnts
        )
    else:
        # GB raceline을 reference로 사용 (기존 로직)
        reference_wpnts = gb_wpnts
        obs_lateral = obs.d_center

        spline_points = self._generate_spline_points(
            obs_s=obs.s_center,
            obs_lateral=obs_lateral,
            reference=reference_wpnts
        )

    return spline_points
```

#### 4. Memory 업데이트 (Fixed Path 기준)
```python
def _process_obstacles(self):
    """장애물 처리 - Single Frenet with lateral distance"""

    for obs in self.latest_obstacles:
        # GB raceline Frenet (기존)
        s_gb = obs.s_center
        d_gb = obs.d_center

        # 간섭 체크 (Fixed path 고려)
        if self.fixed_path_generated:
            # Fixed path 기준 간섭 체크
            interferes = self._check_interference_fixed_path(s_gb, obs.x_center, obs.y_center)
        else:
            # GB raceline 기준 간섭 체크 (기존)
            interferes = abs(d_gb) < (self.gb_ego_width_m + self.lateral_width_gb_m)

        # Memory 업데이트 (s_gb, d_gb 그대로 사용)
        key = (obs.sector_id, obs.id)
        if key not in self.static_obs_memory:
            self.static_obs_memory[key] = {
                's_history': [],
                'd_history': [],
                'interferes': interferes,
                ...
            }

        self.static_obs_memory[key]['s_history'].append(s_gb)
        self.static_obs_memory[key]['d_history'].append(d_gb)
```

### 정확도 고려사항

#### 곡선 구간에서의 오차
```
GB raceline:  ─────╮
                   │ <── 곡선
Fixed path:        ╰─────

Obstacle: ●

GB s=10m 위치에서:
- GB raceline 기준 lateral: 0.5m
- Fixed path 기준 lateral: 0.3m (실제)
- Single Frenet 계산: ~0.35m (근사치)
```

**오차 발생 원인:**
- Fixed path의 waypoint가 discrete (0.1m 간격)
- GB s 위치와 Fixed path waypoint s 위치가 정확히 일치하지 않음
- 곡선 구간에서 heading 차이로 인한 projection 오차

**오차 최소화 방법:**
1. Linear interpolation 사용
```python
# Closest waypoint 2개 찾아서 보간
wpnt_prev = self.fixed_path_wpnts.wpnts[closest_idx]
wpnt_next = self.fixed_path_wpnts.wpnts[(closest_idx + 1) % len(self.fixed_path_wpnts.wpnts)]

# s 좌표 기준 보간 비율
ratio = (obs_s_gb - wpnt_prev.s_m) / (wpnt_next.s_m - wpnt_prev.s_m)

# Interpolated position and heading
x_interp = wpnt_prev.x_m + ratio * (wpnt_next.x_m - wpnt_prev.x_m)
y_interp = wpnt_prev.y_m + ratio * (wpnt_next.y_m - wpnt_prev.y_m)
psi_interp = wpnt_prev.psi_rad + ratio * (wpnt_next.psi_rad - wpnt_prev.psi_rad)

# 보간된 위치 기준으로 lateral distance 계산
```

2. Denser waypoint sampling
- Fixed path waypoint 간격을 0.05m로 줄이기 (현재 0.1m)

### 장점
✅ **수정 범위 최소**: Smart Static Node만 수정
✅ **메시지 크기 동일**: 기존 ObstacleArray 그대로 사용
✅ **단순성**: Tracking, State Machine 수정 불필요
✅ **빠른 구현**: 즉시 테스트 가능

### 단점
❌ **매 루프 연산**: 20Hz로 lateral distance 계산 (성능 이슈 가능)
❌ **정확도**: 곡선 구간에서 근사치 (1-2cm 오차 예상)
❌ **복잡한 로직**: Lateral distance 계산 로직 추가
❌ **확장성**: 새로운 경로 추가 시 계산 복잡도 증가

---

## 성능 비교

### Computational Cost

| Operation | Dual Frenet | Single Frenet |
|-----------|-------------|---------------|
| **Tracking (20Hz)** | 2 × FrenetConverter.get_frenet() | 1 × FrenetConverter.get_frenet() |
| **Smart Static (20Hz)** | if-else 분기 (O(1)) | find_closest_waypoint (O(N))<br>+ lateral_distance (O(1)) |
| **Total per obstacle** | ~0.1ms | ~0.5ms (N=400 waypoints) |

**예상 성능:**
- **Dual**: 10개 장애물 × 0.1ms = 1ms
- **Single**: 10개 장애물 × 0.5ms = 5ms

20Hz (50ms per loop) 기준 **둘 다 충분히 빠름**.

### Memory Usage

| Item | Dual Frenet | Single Frenet |
|------|-------------|---------------|
| **ObstacleArray** | 2× (s, d 필드 추가) | 1× |
| **FrenetConverter** | 2개 (GB + Fixed) | 1개 (GB) |
| **Total overhead** | +~10KB per message | +0KB |

---

## 추천 결론

### **Phase 1: Single Frenet (현재 구현)**

**이유:**
1. ✅ **빠른 개발**: Smart Static Node만 수정
2. ✅ **충분한 성능**: 5ms << 50ms (20Hz)
3. ✅ **충분한 정확도**: 1-2cm 오차는 실용적으로 무시 가능
4. ✅ **검증 가능**: 즉시 테스트하여 동작 확인

**구현 우선순위:**
1. `_get_lateral_distance_to_fixed_path()` - Lateral distance 계산
2. `_check_interference_fixed_path()` - Fixed path 기준 간섭 체크
3. `do_spline()` 수정 - Fixed path reference 지원
4. 테스트 및 정확도 검증

---

### **Phase 2: Dual Frenet (성능/정확도 부족 시)**

**전환 조건:**
1. ❌ Single Frenet 성능이 부족 (>10ms per loop)
2. ❌ 정확도 이슈 발생 (곡선 구간에서 오작동)
3. ✅ 다른 경로 추가 필요 (예: 다중 경로 선택)

**전환 시 수정 필요:**
- Tracking 노드: Dual FrenetConverter
- Smart Static Node: Frenet 좌표계 선택 로직
- State Machine: Dual cur_s, cur_d
- car_state: Dual Frenet pose publish

---

## 구현 가이드

### Single Frenet 구현 체크리스트

- [ ] `_get_lateral_distance_to_fixed_path()` 함수 구현
  - [ ] Closest waypoint 찾기 (wraparound 고려)
  - [ ] Linear interpolation 추가 (정확도 향상)
  - [ ] Lateral distance 계산 (heading 기준)

- [ ] `_check_interference_fixed_path()` 함수 구현
  - [ ] Lateral threshold 설정
  - [ ] 간섭 여부 반환

- [ ] `_process_obstacles()` 수정
  - [ ] Fixed path 생성 여부에 따라 간섭 체크 분기
  - [ ] Memory 업데이트 (기존 s_gb, d_gb 유지)

- [ ] `do_spline()` 수정
  - [ ] Fixed path를 reference로 사용
  - [ ] Lateral distance 기준 spline 생성

- [ ] 테스트
  - [ ] 직선 구간에서 정확도 검증
  - [ ] 곡선 구간에서 정확도 검증
  - [ ] 성능 측정 (latency)
  - [ ] 실제 주행 테스트

---

## 참고: Frenet Coordinate System

### GB Raceline Frenet
```
        d (lateral)
        ↑
        │
        │  ● Obstacle (s=10.5m, d=0.3m)
        │
────────┼────────────→ s (along track)
        │
        │
    GB Raceline
```

### Fixed Path Frenet (Dual 방식)
```
        d_fixed
        ↑
        │  ● Obstacle (s_fixed=10.3m, d_fixed=0.2m)
        │
────────┼────────────→ s_fixed
        │
    Fixed Path
    (장애물 회피 경로)
```

### Single Frenet + Lateral Distance
```
    GB Raceline
        │
        │  ● Obstacle (s_gb=10.5m, d_gb=0.3m)
        │
        ╰─────╮  Fixed Path
              │
              │  lateral_dist = calculate_distance(obs, fixed_path)
              ↓
```

---

## 버전 히스토리

- **v1.0** (2025-01-XX): Initial design document
  - Single vs Dual Frenet 비교 분석
  - Phase 1 (Single) → Phase 2 (Dual) 전략 수립
