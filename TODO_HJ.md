# TODO - HJ (3D Pipeline Migration)

## 완료

- [x] **Wpnt.msg에 mu_rad 추가** — 맨 뒤에 `float64 mu_rad` 추가, 빌드 확인
- [x] **export_global_waypoints.py: CubicSpline 보간** — np.interp → CubicSpline(periodic), psi/kappa 해석적 계산
- [x] **export_global_waypoints.py: mu_rad 계산** — 레이싱라인 xyz에서 직접 `mu = -arcsin(dz/ds)` (센터라인 매칭 불필요)
- [x] **export_global_waypoints.py: IQP = SP** — 3D는 단일 raceline, IQP/SP 동일 데이터
- [x] **export_global_waypoints.py: speed sphere + cylinder markers** — 구(속도색상) + 실린더(속도높이) 두 종류
- [x] **Track3D.resample(step_size)** — smooth와 최적화 그리드 분리
- [x] **gen_global_racing_line.py: step_size_opt** — 최적화 그리드 0.2m 독립 설정
- [x] **C++ CalcFrenetPoint 3D 접선 투영** — 인접 waypoint xyz로 3D 접선 계산, dz 포함
- [x] **C++ CalcFrenetVelocity 판단** — GLIL base_odom twist는 body frame (`T_world_base.T * v_world`) 확인 → 기존 2D 로직 유지가 정확
- [x] **gazebo_wall export + 테스트 완료** — 867pts, 0.1m 등간격, s/d/v_s 정상, 2분 8바퀴 anomaly 0
- [x] **3d_base_system.launch** — sim:=true 시 센서 안 킴, odom_relay → /car_state/odom
- [x] **test_3d_frenet.launch** — fake odom + waypoints + frenet 최소 테스트 구성
- [x] **d_right 음수 수정** — export에서 abs() 처리
- [x] **psi_rad normalize** — arctan2 사용 → [-π,π] 자동
- [x] **Track3D 소스 확보** — planner/3d_racing_line/src/
- [x] **Controller 3D 점검 + 수정** (0330)
  - L1 point: `waypoint_at_distance_before_car` → `[:3]` xyz 반환 (누적 거리는 이미 3D)
  - Future position z: spline 보간 (`converter.spline_z(s)`)으로 추정
  - Lateral error: `get_frenet_3d` 적용 (future_position_z 활용)
  - AEB: 원본 활성화 (전수 검색 방식), emergency 버전 주석 처리
  - 마커 z 적용: lookahead(L1 xyz), future position(spline z), prediction sphere(spline z), trailing opponent(`get_cartesian_3d`)
  - Nearest waypoint: local waypoints 내 검색이라 2D로 충분 → 스킵
  - position_in_map z: Controller 내에서 z 직접 사용 없음 → 구조 변경 불필요

## 즉시 (시스템 돌아가게 하는 것)

- [ ] **1. car_race.launch `/car_state/odom` 이중 발행 해결**
  - 현상: carstate_node와 odom_relay가 동시에 `/car_state/odom` publish
  - 3d_base_system.launch에서는 carstate_node 제거로 해결됨
  - car_race.launch도 같은 방식으로 정리 필요

## 즉시 (Frenet 남은 작업)

- [x] **2. Python frenet_converter.py 3D 대응 + 보호 로직** (0330)
  - `build_raceline()`: 3D 거리 누적으로 s 계산, `waypoints_s[index]`로 반환 (하드코딩 제거)
  - `waypoints_distance_m`: 실제 간격 중앙값으로 계산 (0.1 하드코딩 제거)
  - Height filter (`d_height`): CalcHeightOffset 수식 벡터화, threshold 0.10m
  - Boundary check: `set_track_bounds`에서 선분 시작/끝/z평균 미리 계산, numpy 벡터화 교차 판정
  - 회전 검색: 벽 넘으면 90°/180°/270° 방향 탐색 → s 전진/후진으로 최단 waypoint
  - Fallback: height filter 전멸 → 단순 3D nearest, 회전 실패 → 벽 무시 nearest
  - Track bounds 자동 로드: 생성자에서 `/trackbounds/markers` 토픽 1회 수신 (5s timeout)
  - psi/mu spline 미분으로 자체 계산 (Wpnt 배열 불필요)
- [x] **2-1. Python frenet_converter.py 호출부 3D 적용** (0330)
  - Controller lateral error: `get_frenet_3d` 적용 (future_position_z 활용)
  - Controller trailing opponent 마커: `get_cartesian_3d` 적용
  - **미완료**: 플래너/perception 등 ~20개 노드 호출부는 점진적 전환 필요
- [ ] **3. Glob2Frenet.srv에 z 추가** — frenet_conversion_server에서 3D 서비스 지원

## 단기 (경로 품질)

- [ ] **4. 0327 최적화 결과를 export에 통과시키기** — 센터라인 smooth 필요 (iy 의존)
- [ ] **5. 경사면 속도 보상** — pitch(mu_rad) 기반 속도 커맨드 보정

## 단기 (Perception 대체 — Gazebo 환경)

- [x] **10. Gazebo static obstacle → planner 직접 연결** (0330)
  - `gazebo_static_obstacle_publisher.py` 생성 — poses + radii 구독 → `get_frenet_3d` → ObstacleArray publish
  - `gazebo_static_obstacle_publisher.launch` 생성 — in/out 토픽 configurable
  - Obstacle.msg에 `z_m` 필드 추가
  - srv 4개에 z 필드 추가 (Glob2Frenet, Glob2FrenetArr, Frenet2Glob, Frenet2GlobArr)
  - `GetFrenetPoint` 시그니처에 z 추가, `GetGlobalPoint`에 z 출력 추가
  - frenet_conversion_server 콜백에서 req.z/res.z 활용
- [ ] **10-1. detect/tracking C++ 코드 3D 대응** (detection 확정 후)
  - `GetFrenetPoint`, `GetGlobalPoint` 시그니처 변경으로 인한 호출부 수정 필요:
    - `abp-detection/detect.cpp:128,132` — GetGlobalPoint에 &p.z 추가
    - `abp-detection/detect.cpp:632,684` — GetFrenetPoint에 center_z 추가
    - `2.5d_detection/tracking_node.cpp:486` — GetFrenetPoint에 z 추가
    - `2.5d_detection/detection_node.cpp:338` — GetFrenetPoint에 z 추가
  - Obstacle.msg z_m 필드 채우기 (tracking 결과에서 z 반영)

## 중기 (Closest point 개선)

- [x] **6. UpdateClosestIndex 교차 구간 robustness** (0330)
  - C++: height filter + boundary raycast + 회전 검색(90°/180°/270°) + s 전진/후진 + fallback 추가
  - C++: 음수 s wrapping 수정 (`fmod` 후 `+= length`)
  - C++: `ForceFullSearch()` public 메서드 추가
  - C++: Interactive marker (rviz 클릭 → full search 강제 트리거)
  - C++: height_filter_threshold 0.10m, z_boundary_margin 0.10m
  - C++: trackbounds 로드 확인 로그 → ROS_WARN
  - Python도 동일 보호 로직 적용 (TODO #2에서 완료)
- [ ] **7. Controller nearest_waypoint XY only → 3D**
  - `Controller.py:190` — `nearest_waypoint(position[:2], waypoints[:2])`
  - local waypoints 내 검색이라 당장 문제없음, 경로 겹침 시 필요

## 장기

- [ ] **8. 시각화 마커 z 반영** — sector_server 등 z=0 하드코딩 수정 (controller 마커는 0330 완료)
- [ ] **9. 하드코딩 제거 → launch/yaml 인자 통합**

## 핵심 판단 기록

### CalcFrenetPoint — 3D 접선 투영 필요
- position은 global frame (x,y,z) → xy만 투영하면 경사에서 s가 cos(mu) 만큼 줄어듦
- 인접 waypoint xyz로 3D 접선 벡터 계산, dz 포함해서 투영

### CalcFrenetVelocity — 기존 2D body frame 로직 유지
- GLIL base_odom twist는 **body frame** (코드 확인: `v_base = T_world_base.T * v_world_imu`)
- body frame linear.x = 노면 위 전진 속도 (경사 포함)
- delta_psi 회전만으로 v_s, v_d 계산 완료
- vz 추가는 이중 계산 → 롤백함

### 차가 노면에서 안 떨어진다
- d는 xy 평면 lateral offset으로 충분 — 노면 위에서 z는 종속 변수
- s만 3D 접선 투영이 필요 (보정량에 dz 포함)

## 참고: 현재 3D 파이프라인 상태

| 단계 | 상태 | 비고 |
|------|------|------|
| Localization (GLIL) | OK | x,y,z,roll,pitch,yaw, body frame twist |
| Waypoint (Wpnt.msg) | OK | z_m, mu_rad 필드 있음 |
| global_waypoints.json | OK | CubicSpline, mu_rad, IQP=SP, gazebo_wall 테스트 완료 |
| Frenet C++ (closest point) | **3D** | height filter + boundary raycast + 회전 검색 + fallback |
| Frenet C++ (s,d 계산) | **3D** | 3D 접선 투영 적용 완료, 음수 s wrapping 수정 |
| Frenet C++ (v_s,v_d 계산) | 2D | body frame이라 기존 로직이 정확 |
| Frenet C++ (interactive) | OK | rviz 버튼 → ForceFullSearch |
| Frenet Conversion Server | 2D | srv에 z 없음 (TODO #3) |
| Python FrenetConverter | **3D** | height filter + boundary(벡터화) + 회전 검색 + fallback, trackbounds 자동 로드 |
| Sector Servers | 무관 | 인덱스 기반 |
| Velocity Scaler | 무관 | 인덱스 기반 |
| State Machine | 무관 | s,d만 사용 |
| Controller (L1/PP) | **3D** | L1 xyz, future z spline, lateral error 3D frenet, 마커 z, AEB 활성화 |
| 3d_base_system.launch | OK | sim/real 분리 |
| test_3d_frenet.launch | OK | fake odom 테스트 검증 완료 |
