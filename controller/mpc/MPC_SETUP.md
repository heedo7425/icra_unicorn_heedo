# MPC Tracking Stack — Setup & Changes

이 문서는 `Brojoon-10/ICRA2026_HJ` 기반으로 추가·수정된 내용과 2D simulator에서 재현하는 절차를 기록한다. (2026-04-19 기준)

## 1. 새로 추가된 것

### `controller/mpc/` (패키지 전체 신규)
acados 기반 dynamic-bicycle + Pacejka tracking MPC. `controller_manager` / PP의 drop-in 대체.
- `mpc_node.py` — ROS 노드. 50Hz 루프에서 state 수집 → OCP solve → AckermannDriveStamped publish. RViz 시각화 (`/mpc/prediction`, `/mpc/reference`) 포함.
- `mpcc_ocp.py` — acados OCP 정의. NONLINEAR_LS cost, SQP_RTI, HPIPM QP.
- `vehicle_model.py` — Frenet-frame 동적 자전거 + Pacejka (front/rear) CasADi 모델.
- `reference_builder.py` — `behavior_strategy.local_wpnts` → per-stage param matrix (`kappa, theta, kappa_z, mu, vx_ref, n_ref`).
- `launch/mpc_standalone.launch` — MPC 노드 단독 launch.
- `launch/mpc_sim.launch` — 2D sim용 풀 스택 launch (아래 "실행"에서 설명).
- `config/mpc_srx1.yaml` — 차량별 MPC 튜닝 파라미터. `v_max`, `N_horizon`, 비용 가중치 등.
- `rviz/mpc_sim.rviz` — MPC 시각화 RViz config.
- `scripts/spawn_on_waypoint.py` — sim에서 차를 첫 waypoint로 snap + stuck-감시 respawn.
- `scripts/sign_test*.py` — Frenet 부호 컨벤션 검증 유틸.

### `stack_master/maps/f/` (테스트 맵 신규)
SRX1 실차/sim 공유 실내 트랙. `boundary_*`, `centerline.csv`, `f.pgm/yaml`, `global_waypoints.json`, `ot_sectors.yaml`, `speed_scaling.yaml`, `static_obs_sectors.yaml`.

### `stack_master/config/SIM/global_planner_params.yaml` (신규)
`planner/spliner/src/static_avoidance_node.py` (reactive) 가 launch 시점에 요구. SRX1 config 복사.

## 2. 수정된 파일

### `stack_master/config/SIM/controller.yaml`
PP fallback에서 필요한 키 추가 (base HJ 버전에서 빠져있던 것):
- `AEB_thres`, `lat_correction_mode`, `lat_K_stanley`, `lat_pred_alpha`, `lat_pred_horizon`, `speed_ff_gain`, `trailing_vel_gain`

## 3. 주요 설계 결정 (debug 히스토리 압축)

- **Pacejka slip-angle 부호**: 표준 컨벤션 (`α_f = δ − atan((v_y + l_f·ω)/v_x)`, `α_r = −atan((v_y − l_r·ω)/v_x)`) 사용. 이전 버전은 부호 뒤집혀 있어 차가 회피 반대 방향으로 조향.
- **Frenet `s_dot_scale` smoothing**: `1 − n·κ` 0 근접 시 NaN 방지 위해 `0.5·(raw + sqrt(raw² + 0.01))` 적용.
- **Initial state**: `solver.set(0, "x", x0)` + `lbx/ubx` 동시 설정. 이전엔 lbx/ubx 만 설정해 stage 0가 warm-start 값을 그대로 써 초기 상태 제약 안 걸림.
- **Reference n_ref = 0**: `local_wpnts.x_m/y_m` 이미 회피 경로로 시프트되어 있음. `d_m`은 라벨. MPC 목표는 "wpnt xy 위에 올라가기" 이므로 `n_ref = 0`. 이전엔 `n_ref = d_m` 으로 해서 회피 경로에서 또 d_m 만큼 옆으로 가려고 해 차가 벽으로 돌진.
- **Teleport 감지 + solver cold-start**: `spawn_on_waypoint`의 respawn 감지 (Δxy > 1m), stages 0..N 전부 x0로 시드 + inputs 0.
- **Warm-start rescue**: SQP_RTI 1 iter/tick이라 이전 solution이 infeasible하면 못 빠져나옴. `status ≠ 0`이 5 tick 연속이면 `_cold_start_solver(x0)` 자동 실행.
- **Warmup speed override**: `vx < warmup_vx_min` 시 `speed=0.8, steer=0` 강제. Pacejka Hessian low-speed singular 우회.

## 4. 빌드

도커 컨테이너(`race_stack_nuc`) 내부에서:
```bash
cd /home/$USER/catkin_ws
catkin build f110_msgs    # Wpnt.msg 에 kappa_z_radpm 추가되어 있음
catkin build controller
source devel/setup.bash
```

acados는 `/opt/acados`에 설치되어 있어야 하며, `/etc/profile.d/acados.sh` 로 `LD_LIBRARY_PATH` 자동 설정.

## 5. 실행 (2D sim)

### 기본 주행 (장애물 없음)
```bash
export CAR_NAME=SIM
roslaunch controller mpc_sim.launch map:=f racecar_version:=SIM rviz:=true
```

`mpc_sim.launch`가 한 번에 띄우는 것:
1. `base_system.launch sim:=true` — f1tenth_simulator + frenet + sector servers
2. `dyn_planners_statemachine` 네임스페이스 planner param servers
3. `3d_state_machine_node.py` + `dyn_statemachine` + `state_indicator`
4. Planner 노드들 (recovery_spliner, start_planner, static_avoidance_node, SQP dynamic avoidance, GP opponent predictor)
5. `perception.launch` — abp_detection 기반 scan → /tracking/obstacles
6. `spawn_on_waypoint` — 초기 위치 snap + stuck 감시
7. `mpc_standalone.launch` — MPC 노드
8. RViz

기대 결과: 랩타임 ~40s, avg lateral error 5-7cm, 연속 완주. (맵 `f` 기준)

### 정적 장애물 회피 테스트

**방법 A: 수동 배치 (f1tenth_simulator의 RViz "Publish Point")**
1. RViz의 "Publish Point" 툴로 트랙 위 원하는 지점 클릭 → sim 물리엔진에 obstacle 추가
2. `perception.launch`가 /scan에서 감지해 `/tracking/obstacles` 로 publish
3. state_machine이 OVERTAKE 전이 + `local_wpnts.d_m` 시프트 생성
4. MPC가 시프트된 경로 추종 → 회피 주행

**방법 B: 스크립트로 직접 publish**
`controller/mpc/scripts/static_obs.py` (runtime 생성 예시, 필요시 유틸로 추가 가능):
```python
# 주어진 s, d 에 is_static=True Obstacle 주입
python3 static_obs.py <s_center> [d_center] [size_m]
```

### 확인 토픽
- `/state_machine` (std_msgs/String): `GB_TRACK | TRAILING | OVERTAKE | RECOVERY`
- `/behavior_strategy` (f110_msgs/BehaviorStrategy): `local_wpnts[k].d_m` 시프트 확인
- `/tracking/obstacles`: perception이 올리는 장애물
- `/mpc/prediction`, `/mpc/reference` (MarkerArray): RViz로 시각화
- `/vesc/high_level/ackermann_cmd_mux/input/nav_1`: MPC 최종 drive cmd
- `/lap_data` (f110_msgs/LapData): 랩 완주 시 time / avg_lat / max_lat

## 6. 주의사항

- 트랙 시작 지점 곡률이 크면(예: 맵 f의 s=0, κ≈-0.7) warmup 구간에서 차가 곡선 못 따라 drift할 수 있음. `mpc_srx1.yaml`의 `startup_delay_s` (기본 7s) 로 spawn 후 대기 후 MPC 개시.
- `static_obs_sectors.yaml` 의 `static_obs_section` 플래그가 `false` 면 static avoidance 비활성. 맵 `f` 에서는 `true` 로 설정됨.
- `headtohead.launch` 는 사용하지 않는다. 해당 launch의 controller_manager 가 같은 nav_1 토픽에 publish 해 MPC와 충돌. `mpc_sim.launch` 가 필요한 sub-node들만 선별 launch.
- `smart_static_avoidance_node` (HJ 추가 변형) 은 사용 안 함 — offline optimization 모드라 reactive 테스트에 부적합. `static_avoidance_node.py` (reactive) 를 명시적으로 launch.

## 7. 알려진 이슈 / 후속 작업

- OVERTAKE 전이 시 local_wpnts d_m이 step jump (0 → 0.85) → MPC 수렴 1~2 tick 지연, 해당 구간 max_lat 일시적 70-90cm 도달 가능. 성능보다 안정성 우선 단계에서 허용.
- SIM `v_max = 2.0` 으로 하향된 상태 (raceline vx_ref 도 2.0 이하라 실질적 ceiling). 실차/고속 튜닝은 Phase 2.
- `mu_estimator` (RLS online μ̂) 아직 off — Stage 2 계획.
