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

## 즉시 (시스템 돌아가게 하는 것)

- [ ] **1. car_race.launch `/car_state/odom` 이중 발행 해결**
  - 현상: carstate_node와 odom_relay가 동시에 `/car_state/odom` publish
  - 3d_base_system.launch에서는 carstate_node 제거로 해결됨
  - car_race.launch도 같은 방식으로 정리 필요

## 즉시 (Frenet 남은 작업)

- [ ] **2. Python frenet_converter.py 호출부 3D 수정 및 검증**
  - FrenetConverter 클래스 3D 대응 완료 (z 입력, 3D s, spline_z, get_frenet_3d/get_cartesian_3d)
  - ~20개 노드의 생성자에 z 전달 완료 (`[wpnt.x_m, wpnt.y_m, wpnt.z_m]`)
  - **미완료**: 각 노드의 메서드 호출부 검증 필요
    - `get_frenet(x, y)` → `get_frenet_3d(x, y, z)` 전환 필요 여부 (교차 구간 nearest search)
    - `get_cartesian(s, d)` → `get_cartesian_3d(s, d)` 전환 필요 여부 (RViz 마커 z 반영)
    - 플래너 spline 결과 마커가 z=0에 찍히는 문제
  - 노드별로 3D 전환 필요성 판단 후 점진적 수정
- [ ] **3. Python frenet_converter.py 3D 대응**
  - `build_raceline()`: xy 2D 거리로 s 자체 재계산 → export한 3D s_m 무시
  - `__init__`: z 입력 없음 (`waypoints_x, waypoints_y`만)
  - `get_approx_s`: xy only nearest search → 교차 구간에서 틀림
  - ~20개 노드가 사용 (planner, controller, prediction, perception, sector tuner)
  - 파일: `f110_utils/libs/frenet_conversion/src/frenet_converter/frenet_converter.py`
- [ ] **3. Glob2Frenet.srv에 z 추가** — frenet_conversion_server에서 3D 서비스 지원

## 단기 (경로 품질)

- [ ] **4. 0327 최적화 결과를 export에 통과시키기** — 센터라인 smooth 필요 (iy 의존)
- [ ] **5. 경사면 속도 보상** — pitch(mu_rad) 기반 속도 커맨드 보정

## 중기 (Closest point 개선)

- [ ] **6. UpdateClosestIndex 교차 구간 robustness**
  - 현재 3D 유클리드 거리로 동작, 교차 구간에서 위험
  - 접근법: z_weight / mu_rad 비교 / surface segment
- [ ] **7. Controller nearest_waypoint XY only → 3D**
  - `Controller.py:190` — `nearest_waypoint(position[:2], waypoints[:2])`

## 장기

- [ ] **8. 시각화 마커 z 반영** — sector_server 등 z=0 하드코딩 수정
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
| Frenet C++ (closest point) | OK | 3D Euclidean distance (교차 구간 개선 필요) |
| Frenet C++ (s,d 계산) | **3D** | 3D 접선 투영 적용 완료 |
| Frenet C++ (v_s,v_d 계산) | 2D | body frame이라 기존 로직이 정확 |
| Frenet Conversion Server | 2D | srv에 z 없음 (TODO #3) |
| Python FrenetConverter | **2D** | z 없음, ~20개 노드 사용 (TODO #2) |
| Sector Servers | 무관 | 인덱스 기반 |
| Velocity Scaler | 무관 | 인덱스 기반 |
| State Machine | 무관 | s,d만 사용 |
| Controller (L1/PP) | 2D | Python FrenetConverter 의존 |
| 3d_base_system.launch | OK | sim/real 분리 |
| test_3d_frenet.launch | OK | fake odom 테스트 검증 완료 |
