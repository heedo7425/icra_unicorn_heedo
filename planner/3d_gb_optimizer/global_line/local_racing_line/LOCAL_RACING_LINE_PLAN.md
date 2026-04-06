# Local Racing Line Mux Node 구현 계획

## Context

현재 3D 스택은 global waypoint 추종(GlobalTracking)만 가능한 상태.
차량이 global racing line에서 크게 벗어났을 때(횡방향/속도), 기존 recovery_planner 등 별도 플래너 없이 **local racing line 하나로** 시간최적 복귀 경로를 계산하여 커버하는 것이 목표.

event-triggered 방식으로, 평소에는 global waypoints를 그대로 pass-through하고, 이탈 시에만 acados solver를 실행.
launch 파일에서 `use_local_racing_line:=true/false`로 on/off 가능.

---

## sim_local_racing_line.py와의 관계

`sim_local_racing_line.py`는 **직접 사용하지 않는다.** 오프라인 시뮬레이터(while 1 + matplotlib + perfect tracking 가정)이기 때문.

**하지만 그 파일의 초기화 패턴을 그대로 따른다:**

```
sim_local_racing_line.py (레퍼런스)     →  local_raceline_mux_node.py (새 노드)
─────────────────────────────────────────────────────────────────────────
track_handler = Track3D(path=...)       →  동일하게 생성
gg_handler = GGManager(gg_path=...)     →  동일하게 생성
model = export_point_mass_ode_model()   →  동일하게 생성
planner = LocalRacinglinePlanner(...)   →  동일하게 생성 (파라미터만 RC car용으로 변경)
─────────────────────────────────────────────────────────────────────────
while 1: (무한루프)                      →  ROS timer (10Hz)
  상태를 자체 계산 (perfect tracking)    →  /car_state/odom_frenet에서 수신
  raceline = planner.calc_raceline()    →  동일하게 호출 (이탈 시에만)
  visualizer.update()                   →  RViz 마커 publish
```

4개 클래스(`Track3D`, `GGManager`, `export_point_mass_ode_model`, `LocalRacinglinePlanner`)는 수정 없이 라이브러리로 import.

---

## Waypoint 수와 Lookahead 분석

### 현재 스택 수치 (gazebo_wall_2_0402)

| 항목 | 값 |
|------|-----|
| 트랙 전장 | 86.0m |
| Global waypoints 총 수 | 860개 |
| Waypoint 간격 | 0.1m |
| `n_loc_wpnts` (state machine) | 80개 |
| **현재 lookahead 거리** | **80 × 0.1m = 8.0m** |
| 평균 속도 기준 lookahead 시간 | 8.0m / 5.9m/s ≈ 1.35초 |

### Solver 출력 vs State Machine 기대값

| 항목 | Solver 출력 | State Machine 기대 |
|------|------------|-------------------|
| 포인트 수 | N_steps = 40 | n_loc_wpnts = 80 |
| 간격 | 0.375m (15m/40) | 0.1m |
| 전체 거리 | 15.0m | 8.0m |

**문제**: solver는 40개 포인트를 0.375m 간격으로 생성하지만, state machine은 0.1m 간격 80개를 기대.

### 해결: solver 출력을 global waypoint 그리드에 보간

```
Solver 출력 (40pts, 0.375m 간격)
    ↓ np.interp으로 0.1m 그리드에 리샘플링
Global WpntArray (860pts 중 해당 구간 ~80개 덮어쓰기)
    ↓ State Machine이 80개 선택
Controller
```

- Solver horizon 15m는 약 150개 global waypoint에 해당
- State machine은 그 중 앞쪽 80개(8m)만 사용
- **solver가 8m 이상을 계산하므로 충분** → 80개 waypoint를 모두 커버
- Solver의 뒷부분(8~15m)은 수렴 체크용으로만 사용

### Optimization Horizon 조정

15m는 충분하지만 계산 효율을 위해 **10m**으로 줄여도 됨 (80개 wpnt = 8m를 커버하고도 여유):
- `optimization_horizon`: 10.0m
- `N_steps`: 30 (0.33m/step)
- 이렇면 solver 부하 감소 + 여전히 state machine lookahead 커버

→ **일단 10m, N=30으로 시작. 필요 시 15m으로 확장.**

---

## 아키텍처

```
[use_local_racing_line:=false — 기존과 동일]
vel_scaler → /global_waypoints_scaled → state_machine → controller

[use_local_racing_line:=true]
vel_scaler → /global_waypoints_scaled_raw (launch에서 remap)
    → local_raceline_mux_node (새 노드)
        구독: /global_waypoints_scaled_raw (원본 global wpnts)
              /car_state/odom_frenet (s, d, vs)
              /car_state/odom (yaw → chi 계산용)
              /ekf/imu/data (실차만, ax/ay)
        발행: /global_waypoints_scaled (state_machine이 그대로 구독)
              /local_raceline/status (String: "PASSTHROUGH" / "ACTIVE")
    → state_machine (변경 없음) → controller (변경 없음)
```

---

## 생성/수정할 파일

### 1. 새 파일: `planner/gb_optimizer/src/local_raceline_mux_node.py`

**초기화 (`__init__`)**:
- ROS 파라미터 로딩 (threshold, horizon, N_steps 등)
- `sys.path.append`로 `3d_gb_optimizer/global_line/src/` 추가
- sim_local_racing_line.py 패턴 따라 4개 클래스 인스턴스 생성:
  - Track3D(smoothed CSV 경로)
  - GGManager(gg_diagrams 경로)
  - export_point_mass_ode_model(vehicle_params, track_handler, gg_handler, ...)
  - LocalRacinglinePlanner(params, ..., nlp_solver_type='SQP_RTI', N_steps=30, optimization_horizon=10.0)
- Global raceline CSV 로딩 → v_ref(s), n_ref(s) 보간기 구축
- 모드: PASSTHROUGH (초기)

**메인 루프 (`timer_cb`, 10Hz)**:

```python
PASSTHROUGH 모드:
  1. deviation 계산:
     - d_err = |cur_d - n_ref(cur_s)|
     - v_err = |cur_vs - v_ref(cur_s)|
  2. 트리거 체크: d_err > d_threshold OR v_err > v_threshold → ACTIVE 전환
  3. global waypoints 그대로 publish

ACTIVE 모드:
  1. chi = normalize(cur_yaw - track_handler.theta_interpolator(cur_s))
  2. ax, ay: sim이면 prev_solution/0, 실차면 IMU
  3. solver 호출: planner.calc_raceline(s, V, n, chi, ax, ay, ...)
  4. solver 출력을 global wpnt 그리드에 보간 → WpntArray 생성
  5. 수렴 체크: solver 끝부분의 (n, V)가 global (n_ref, v_ref)과 근접 → PASSTHROUGH 복귀
  6. 수정된 WpntArray publish
```

**핵심 메서드: `_raceline_to_wpntarray(raceline)`**:

```python
1. Global WpntArray를 복사
2. Solver 출력 (s[], V[], n[], chi[], ax[], ay[], x[], y[], z[])을
   global waypoint의 s_m 그리드에 np.interp으로 보간
3. 보간 결과로 해당 구간 waypoint 덮어쓰기:
   - x_m, y_m, z_m ← raceline['x','y','z'] 보간값
   - vx_mps ← V 보간값
   - d_m ← n 보간값
   - psi_rad ← theta_track(s) + chi 보간값
   - kappa_radpm ← ay / V² 보간값
   - ax_mps2 ← ax 보간값
   - d_right, d_left ← 트랙폭 기반 재계산
4. 구간 밖 waypoints는 원본 유지
```

**상태 변환:**

| Solver 변수 | ROS 소스 | 계산 |
|-------------|----------|------|
| s | odom_frenet.pose.position.x | 직접 |
| V | odom_frenet.twist.linear.x (= vs) | 직접 |
| n | odom_frenet.pose.position.y (= d) | 직접 |
| chi | odom의 yaw − track_handler.theta_interpolator(s) | 직접 계산, [-π, π] wrap |
| ax, ay | sim: 0→prev_solution, 실차: /ekf/imu/data | 조건부 |

### 2. 수정: `stack_master/launch/3d_base_system.launch`

```xml
<!-- 추가할 arg -->
<arg name="use_local_racing_line" default="false"
  doc="Enable event-triggered local racing line replanning" />

<!-- vel_scaler에 조건부 remap -->
<node pkg="sector_tuner" type="vel_scaler_node.py" name="velocity_scaler" output="screen">
  <param name="enable_smart_scaling" value="$(arg enable_smart_scaling)" />
  <remap if="$(arg use_local_racing_line)"
         from="/global_waypoints_scaled" to="/global_waypoints_scaled_raw" />
</node>

<!-- local racing line mux (조건부 실행) -->
<group if="$(arg use_local_racing_line)">
  <node pkg="gb_optimizer" type="local_raceline_mux_node.py"
        name="local_raceline_mux" output="screen">
    <param name="d_threshold" value="0.3" />
    <param name="v_threshold" value="1.5" />
    <param name="convergence_d_threshold" value="0.1" />
    <param name="convergence_v_threshold" value="0.5" />
    <param name="optimization_horizon" value="10.0" />
    <param name="N_steps" value="30" />
    <param name="gg_mode" value="diamond" />
    <param name="safety_distance" value="0.05" />
  </node>
</group>
```

---

## 재사용하는 기존 코드 (수정 안 함)

| 파일 | 역할 |
|------|------|
| `3d_gb_optimizer/global_line/src/local_racing_line_planner.py` | LocalRacinglinePlanner (acados solver) |
| `3d_gb_optimizer/global_line/src/track3D.py` | Track3D (3D 트랙 보간기) |
| `3d_gb_optimizer/global_line/src/point_mass_model.py` | ODE 모델 |
| `3d_gb_optimizer/global_line/src/ggManager.py` | GG 다이어그램 |
| `3d_gb_optimizer/global_line/data/vehicle_params/params_rc_car_10th.yml` | 차량 파라미터 |
| `3d_gb_optimizer/global_line/data/gg_diagrams/rc_car_10th/velocity_frame/` | GG 데이터 |
| `3d_gb_optimizer/global_line/local_racing_line/sim_local_racing_line.py` | **레퍼런스만** (직접 사용 안 함) |

---

## 데이터 파일 (gazebo_wall_2_0402)

| 파일 | 용도 | 크기 |
|------|------|------|
| `gazebo_wall_2_3d_smoothed.csv` | Track3D 입력 | 1799 pts |
| `gazebo_wall_2_3d_rc_car_10th_timeoptimal.csv` | v_ref, n_ref 참조 | 451 pts |
| `global_waypoints.json` | 스택 global waypoints | 860 wpnts, 0.1m 간격 |

---

## 구현 순서

### Phase 1: Pass-through 골격
1. `local_raceline_mux_node.py` 생성 — subscribe → pass-through publish만
2. `3d_base_system.launch` 수정 — arg + remap + node
3. **검증**: `use_local_racing_line:=true`로 실행, state machine 정상 동작 확인

### Phase 2: Solver 초기화 + deviation 모니터링
4. sim_local_racing_line.py 패턴대로 4개 클래스 초기화
5. deviation 계산 + 로그 출력 (ACTIVE 진입은 아직 안 함)
6. **검증**: solver 컴파일 성공, deviation 로그가 합리적인지 확인

### Phase 3: ACTIVE 모드
7. chi 계산 + solver 호출
8. solver → WpntArray 변환 (global 그리드에 보간)
9. 수렴 체크 + PASSTHROUGH 복귀
10. **검증**: 차량 이탈 → ACTIVE → solver 결과 추종 → 수렴 → PASSTHROUGH

### Phase 4: 실차 + 튜닝
11. IMU 연동 (/ekf/imu/data, sim=false일 때)
12. RViz 마커 시각화
13. threshold 튜닝

---

## 주의사항

- **acados 컴파일**: 노드 시작 시 10~30초. `ocp.code_export_directory` 설정으로 재컴파일 방지
- **트랙 wrapping**: s ≈ 0 근처에서 np.unwrap 필요 (기존 solver에서 이미 처리됨)
- **WpntArray 크기**: 전체 860개 유지, solver horizon 구간(~100개)만 덮어쓰기
- **SQP_RTI 첫 호출**: global raceline CSV를 초기 prev_solution으로 세팅하여 warm-start 보장
- **NUC 부하**: PASSTHROUGH에서 ≈0, ACTIVE에서 SQP_RTI N=30 ≈ 5~15ms

---

## 검증 방법

1. **Pass-through**: `rostopic echo` 비교 — scaled_raw vs scaled 동일
2. **Deviation 모니터링**: 주행 중 d_err, v_err 로그
3. **Solver 시간**: calc_raceline 호출 시간 < 50ms 확인
4. **복귀 테스트**: 이탈 → ACTIVE → 복귀 → PASSTHROUGH 전환
5. **Toggle**: false=기존 동일, true=mux 동작
