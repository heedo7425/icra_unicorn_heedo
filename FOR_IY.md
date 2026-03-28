# HJ 수정사항 전달 (2026-03-28~29)

## 변경 의도

2D gb_optimizer와 동등한 정밀도를 3D 파이프라인에서 확보 + Frenet 변환 3D 대응.

## 수정 파일 요약

| 파일 | 변경 |
|------|------|
| `f110_msgs/msg/Wpnt.msg` | `mu_rad` 필드 추가 (맨 뒤) |
| `frenet_conversion/src/frenet_conversion.cc` | `CalcFrenetPoint` 3D 접선 투영 |
| `frenet_conversion/include/frenet_conversion.h` | `CalcFrenetPoint` 시그니처 z 추가 |
| `3d_racing_line/src/track3D.py` | `resample(step_size)` 메서드 추가 |
| `3d_racing_line/.../gen_global_racing_line.py` | `step_size_opt` 파라미터 추가 |
| `3d_racing_line/.../export_global_waypoints.py` | CubicSpline 보간, mu_rad, IQP=SP, speed markers |
| `stack_master/launch/3d_base_system.launch` | 3D용 base system (sim/real) |
| `stack_master/launch/test_3d_frenet.launch` | frenet 테스트용 최소 launch |
| `stack_master/scripts/fake_odom_publisher.py` | waypoint 따라가는 fake odom |

## 상세 변경 내용

### 1. Wpnt.msg — mu_rad 추가
- 맨 뒤에 `float64 mu_rad` 추가 (기존 인덱싱 영향 없음)
- export에서 레이싱라인 xyz의 CubicSpline 미분으로 직접 계산: `mu = -arcsin(dz/ds_3d)`
- 센터라인 mu와 매칭하지 않음 — 레이싱라인 자체 경사각

### 2. CalcFrenetPoint — 3D 접선 투영
- 기존: `s = dx*cos(psi) + dy*sin(psi) + wpt.s_m` (2D)
- 변경: 인접 waypoint xyz로 3D tangent 벡터 계산, `s = dot(d_vec, tangent) + wpt.s_m`
- d는 수평 법선 방향 투영 유지 (차가 노면 위에 있으므로)
- 이유: 경사면에서 xy만 투영하면 s가 cos(mu) 만큼 줄어듦

### 3. CalcFrenetVelocity — 기존 2D 유지 (변경 없음)
- GLIL base_odom twist는 **body frame** 확인 완료
  - 코드: `v_base = T_world_base.linear().transpose() * v_world_imu`
  - 파일: `GLIL_unicorn_racing/glim_ros1/src/glim_ext/imu_prediction_module.cpp:240`
- body frame linear.x = 노면 위 전진 속도 (경사 포함)
- delta_psi 회전만으로 v_s, v_d 계산 완료 — vz 불필요

### 4. Track3D.resample(step_size)
- smooth 출력과 최적화 그리드를 독립적으로 설정
- 2D의 `stepsize_prep` → `stepsize_reg` 분리와 동일 목적

### 5. export_global_waypoints.py 개선
- `np.interp` → `CubicSpline(bc_type='periodic')` (C² 연속)
- psi: 스플라인 1차 미분으로 해석적 계산
- kappa: `(x'y'' - y'x'') / (x'²+y'²)^1.5` 해석적 계산
- mu_rad: `cs_z` 1차 미분으로 직접 계산
- IQP = SP = 동일 waypoints (3D는 단일 raceline)
- markers: speed sphere (빨강-초록) + velocity cylinder (빨강)

### 6. gazebo_wall 테스트 결과
- 867pts, 0.1m 등간격, s_m ↔ 3D 누적오차 0.007m
- fake odom으로 2분 8바퀴 테스트: anomaly 0, wrap-around 정상
- frenet_odom_republisher segfault 원인: IQP wpnts 빈 배열 → 수정 완료

## 실행 순서

```
1. smooth_track(step_size=0.1)    → smoothed CSV
2. gen_global_racing_line.py      → optimization CSV (step_size_opt=0.2)
3. export_global_waypoints.py     → global_waypoints.json (0.1m CubicSpline, mu_rad 포함)
```

## Python FrenetConverter 호출부 — 수정 및 검증 필요

FrenetConverter 클래스 자체는 3D 대응 완료 (z 입력, 3D s 계산, spline_z).
~20개 노드의 생성자에 z 전달 완료 (`[wpnt.x_m, wpnt.y_m, wpnt.z_m]`).

**미완료 — 각 노드의 메서드 호출부 검증 필요:**
- `get_frenet(x, y)` → 교차 구간이 있는 트랙에서는 `get_frenet_3d(x, y, z)` 전환 필요
- `get_cartesian(s, d)` → RViz 마커를 3D로 찍으려면 `get_cartesian_3d(s, d)` 전환 필요
- 현재는 기존 2D 메서드로도 동작함 (build_raceline의 s가 3D로 계산되므로)
- 노드별로 3D 전환 필요성 판단 후 점진적 수정

## 확인/시도 필요 사항

- [ ] `data/` 디렉토리 구조 셋업 (README 참조)
- [ ] `gen_global_racing_line.py` params를 새 트랙으로 지정
- [ ] `smooth_3d_track.py`에서 `step_size=0.05`로 smooth 돌려보기
  - smooth NLP가 무거워지지만 오프라인이라 괜찮음
  - 0.1과 결과 비교 → 차이 없으면 0.1 유지
- [ ] reverse trajectory — centerline 반전 후 전체 파이프라인 재실행 필요
  - export에서 뒤집으면 dynamics 깨짐
- [ ] `catkin build` 필요 — Wpnt.msg 변경 + frenet_conversion C++ 변경
