# MPCC Planner Trial v1 — 세션 정리

작성일: 2026-04-18
위치: `planner/mpc_planner/`
원본: [heedo7425/icra_unicorn_heedo](https://github.com/heedo7425/icra_unicorn_heedo) `mpc/`

---

## 1. 목적

- 다양한 상황(raceline 주행, 회피, 추월, 복구)에 활용 가능한 **trajectory planner** 가 필요
- 동역학 모델이 중요해지는 3D 맵에서 **dynamic MPC는 infeasible 위험** 때문에 kinematic 기반 MPCC 선정
- 컨트롤러로 쓸 계획은 **아직 없음** — 궤적 형상만 뽑고 속도는 추후 2.5d vel planner가 덮어씀
- 현재 `3d_sampling_based_planner`와 **역할 겹침** → 둘 다 best condition으로 만들어 비교·선택할 목적

## 2. 포팅 전 분석 결과

### 외부 레포 mpc/ 폴더 구조
```
mpc/
├── config/mpcc_params.yaml
├── launch/mpc_controller.launch
└── scripts/
    ├── mpcc_node.py
    └── mpcc_solver.py
```

### 솔버 형태
- CasADi + IPOPT
- Kinematic bicycle, EVO-MPCC 기반
- State: `[x, y, psi, s]`, Control: `[v, delta, p]` (p = progress velocity)
- CasADi B-spline LUT로 트랙 참조 (`cx, cy, dx, dy, psi, dl, dr` 7개)
- 원본은 **컨트롤러**로 설계됨 (ackermann cmd 발행, N=6, dT=0.1, 0.6s horizon)

### 원본 코드의 핵심 판단 포인트
- LUT는 NLP 안에 symbolic하게 삽입되지 **않음** — `_build_nlp()`은 파라미터 `P`만 사용
- LUT는 `_construct_warm_start()`에서만 호출됨, **첫 solve 전 1회만**
- 즉 기존 코드 상태에서 LUT는 사실상 **장식**, s/p는 dead variable

### 3D 맵과의 간극
- 솔버는 완전한 2D (x, y, psi) — pitch/roll/z 모름
- 트랙이 3D 맵이면 입력을 수평 투영하고 출력을 z 리프트해야 함 (마커·Path 시각화용)

---

## 3. 이관 1단계 — 최소 작동 (스켈레톤)

### 생성한 패키지 구조
```
planner/mpc_planner/
├── CMakeLists.txt
├── package.xml
├── config/mpcc_params.yaml
├── launch/mpc_planner.launch
├── node/mpc_planner_node.py
└── src/mpcc_solver.py
```

### 원본 대비 변경
| 항목 | 원본 | v1 초기 |
|------|------|---------|
| ackermann 출력 | `/vesc/.../nav_1` | **삭제** (컨트롤러 아님) |
| `/behavior_strategy` 의존 | 있음 | **삭제** (state_machine 미연동) |
| 로컬 ref 소스 | BehaviorStrategy.local_wpnts | `/global_waypoints` 직접 슬라이스 |
| 발행 토픽 | `/mpc_trajectory` | `/planner/mpc/trajectory` (Path) + `/planner/mpc/markers` (MarkerArray) |
| 3D z 처리 | 없음 | 마커 z를 `g_z`에서 nearest lift (`### HJ` 주석) |

### 1차 빌드/실행
- `catkin build mpc_planner` → 성공
- CasADi 3.7.2 런타임 확인
- 더미 원형 트랙으로 solver 단독 drive test → `ok=True, traj_shape=(7,4)`

---

## 4. 이관 2단계 — 파라미터·참조 슬라이싱 조정

### 초기 증상
- N=6, dT=0.1 그대로 → **예측 궤적 점이 7개밖에 없어 너무 짧게 보임**
- `horizon: 0.6s, 길이 ≈ 1~2m` — 플래너 용도엔 부족

### 조정 1: horizon 확장
- `N: 6 → 20`, `dT: 0.1 → 0.05` → 1.0s / 21점 / 5m/s 기준 25cm 간격
- dT 절반이 되며 smoothness weight 상대적 강화 → 절반 스케일로 보정
  - `w_dv: 19.0 → 9.5`
  - `w_dsteering: 28.0 → 14.0`
  - `w_dprogress: 15.7 → 7.85`
  - `w_progress: 6.0 → 3.0`
  - `w_lag: 1.0 → 2.0` (중심선 추적 강화)

### 조정 2: reference slicing 개선
- **증상**: horizon을 키웠더니 궤적이 꼬불거림
- **원인 확인**: `est_speed = max(|car_vx|, 2.0)`로 `target_s = s_cur + k*dT*est_speed` 계산
  - 차가 5 m/s로 주행 중이어도 reference window는 고작 5m 뒤쪽까지만 깔림
  - 솔버는 v_max=10까지 자유롭게 선택 → reference 끝점에 뒤쪽 스텝 전부 몰리면서 찌그러짐
- **수정**: 각 스텝에서 그 지점의 `g_vx` (raceline 속도)로 `target_s` 전진
  ```python
  local_v = max(float(self.g_vx[idx]), 1.0)
  target_s += local_v * dT
  ```
- 꿀렁거림이 크게 줄어듦

### 조정 3: 차량 파라미터 현실화
- `max_steering: 0.4 → 0.6` (실차 한계 0.6 rad 반영)
- `max_speed: 10 → 12` (raceline 최대 ~10 m/s 여유 확보)
- `vehicle_L: 0.36 → 0.33` (실측)

### 조정 4: 실패 시 warm-start 리셋
- IPOPT `ok=False` 시 다음 solve가 이전 실패 solution을 initial guess로 받으면 연쇄 실패
- 솔버의 `reset_warm_start()` 호출 추가 → 다음 solve가 clean start

---

## 5. 이관 3단계 — 구조적 단순화 (s/p/LUT 제거)

### 결정 배경
- 컨트롤러까지 쓸 계획 없음 → LUT의 진짜 장점(s-자유변수 하의 연속 reference 조회)을 **쓸 일이 없음**
- 현재 구조는 "NLP 안에 LUT가 없고, s/p는 비용·제약 어디에도 의미 있게 연결되지 않음" → **완전 dead weight**
- 다양한 상황용 플래너라면 **reference 소스가 교체 가능해야 함** → LUT 고정은 오히려 방해

### 적용 내용
- State `[x, y, psi, s]` → `[x, y, psi]`
- Control `[v, delta, p]` → `[v, delta]`
- Cost function에서 progress reward 삭제, `w_progress / w_dprogress / max_progress` 파라미터 폐기
- `build_luts, find_nearest_s, filter_s`, 7개 LUT 객체 전부 제거
- `_construct_warm_start`: LUT 호출 대신 슬라이스된 reference로 직접 전파
- nearest-s 계산은 노드에서 numpy로

### 원본 보존
- `src/mpcc_solver_original.py`
- `node/mpc_planner_node_original.py`
- `config/mpcc_params_original.yaml`

### 효과 (같은 맵·같은 주행)
| 지표 | 단순화 전 | 단순화 후 |
|------|----------|-----------|
| NLP 변수 수 | 144 | **103 (-28%)** |
| NLP 제약 수 | 103 | **83 (-19%)** |
| 평균 solve time | 9~16ms | **5~11ms** |
| Warm-start 효과 | 있음 | 뚜렷 (연속 5~7ms) |

---

## 6. 이관 4단계 — Boundary 진단 및 Soft Constraint

### 증상
- 특정 트랙 지점(s ≈ 50.6m)에서 **반복적 IPOPT fail** (1 lap 주기)
- 실패 시 일관된 출력: `v0=12.0, steer=±0.6, iter ≈ 126`
- 진단 로그로 잡힌 데이터:
  ```
  [MPC fail] s=50.6 xy_err=0.05m dpsi=-0.00rad
             ref_v=[6.85..7.30] dl=0.69 dr=0.58
             iter=126 status=Infeasible_Problem_Detected solve=106.5ms
  ```
- 차 위치 완벽, 속도 모던, **그 지점 코리도도 1.27m로 넉넉** → "왜 infeasible?"

### 진단 과정의 실수
- 처음엔 "dl_min=0.398 > inflation 0.34 이므로 코리도 붕괴 없음"이라 **잘못 결론**내림
- **빠뜨린 계산**: 실제 코리도 폭 = `max(dl-inflation, 0) + max(dr-inflation, 0)`
  - 전체 waypoint 중 최악 케이스: `0.058 + 0.057 ≈ 11.5cm`
- s=50.6 현재 위치는 넉넉하지만 **horizon 1초 앞(5~7m) waypoint 중에** dl<0.4 구간이 있으면 거기서 코리도 12cm로 붕괴
  - 그 tube 안에서 7m/s · 20 스텝 + kinematic bicycle 제약 전부 만족하는 해가 **진짜 존재하지 않음** → IPOPT가 정확히 Infeasible 선언

### 적용 수정
1. **근본 원인 제거**: `boundary_inflation: 0.34 → 0.1`
   - 원본 값은 **컨트롤러용** — 추적 오차 + 동역학 여유 합산
   - 플래너는 차 half-width + 소량 여유만 필요 (~0.05~0.1m)
   - 최소 코리도 폭이 12cm → 60cm 수준으로 회복
2. **안전망**: Soft boundary with slack variables
   - 스텝마다 `slack_k ≥ 0` 도입, 제약을 `lo - slack ≤ c ≤ hi + slack`로 완화
   - 비용에 `w_slack * slack_k²` (w_slack=1000) 부과 → 평소엔 0, 진짜 기하 모순 시에만 cm 단위로 사용
   - IPOPT가 Infeasible로 빠지는 경로 자체 제거
3. **진단 강화**: 실패 시 로그에 **horizon window 전체 min(corridor width) + 발생 s** 추가
   - 다음 실패 시 정확한 "앞으로 좁아지는 구간" 위치가 즉시 드러남

### 테스트 결과 (solver 드라이런)
| 케이스 | ok | solve | slack_max | status |
|--------|-----|-------|-----------|--------|
| 정상 50cm 코리도 | True | 34ms | 0.0000 | Solve_Succeeded |
| Warm-start | True | 15ms | 0.0000 | Solve_Succeeded |
| **6cm 코리도 + 10cm off raceline** | **True** | 31ms | 0.0475m | Solve_Succeeded |

세 번째 케이스가 기존엔 Infeasible 실패였음 → **4.7cm slack으로 해 존재**로 전환됨.

---

## 7. 현재 구조 요약

### 파일
- [planner/mpc_planner/src/mpcc_solver.py](../planner/mpc_planner/src/mpcc_solver.py) — 단순화된 kinematic MPC 솔버 + soft boundary
- [planner/mpc_planner/node/mpc_planner_node.py](../planner/mpc_planner/node/mpc_planner_node.py) — Path/MarkerArray 발행, 실패 진단
- [planner/mpc_planner/config/mpcc_params.yaml](../planner/mpc_planner/config/mpcc_params.yaml)
- [planner/mpc_planner/launch/mpc_planner.launch](../planner/mpc_planner/launch/mpc_planner.launch)

### 현재 NLP 요약
- Decision: `X ∈ R^(3×21), U ∈ R^(2×20), SL ∈ R^20` → 총 **103 변수**
- Cost (step당):
  - contouring² + lag² (half-width 0.5 정규화)
  - velocity tracking (`(v-ref_v)/v_bias_max`)²
  - 제어 smoothness (dv², ddelta²)
  - **slack²** (w_slack=1000)
- Constraints:
  - 초기 상태 (3 eq)
  - Kinematic bicycle dynamics (3 eq × 20 step)
  - Soft boundary 2개/step (upper/lower)
  - Box: v∈[0.5,12], delta∈[-0.6,0.6], slack∈[0,1e3]

### 구독·발행
```
Sub:  /global_waypoints (f110_msgs/WpntArray)
      /car_state/pose   (geometry_msgs/PoseStamped)
      /car_state/odom   (nav_msgs/Odometry)

Pub:  /planner/mpc/trajectory (nav_msgs/Path)
      /planner/mpc/markers    (visualization_msgs/MarkerArray)
      [경고] /rosout: [MPC fail] ... (실패 시 상세 진단 1줄)
```

### 튜닝 상수 (현재)
```yaml
N: 20, dT: 0.05            # 1.0s horizon, 21점
vehicle_L: 0.33
max_speed: 12.0, min_speed: 0.5, max_steering: 0.6
w_contour: 3.9, w_lag: 2.0
w_velocity: 3.0, v_bias_max: 1.0
w_dv: 9.5, w_dsteering: 14.0
boundary_inflation: 0.1
w_slack: 1000.0
ipopt_max_iter: 500
```

---

## 8. 남은 이슈

### 이슈 A: 가끔 여전히 fail
- slack 적용 후에도 드물게 `ok=False` 발생
- 다음 주행 로그에서 `slack_max`, `win_corr_min@s=?`, IPOPT status 관찰 필요
- 가설: 수치적 수렴 문제 (`Maximum_Iterations_Exceeded`) 또는 특정 yaw wrap-around 엣지 케이스
- 조치 후보:
  - `ipopt_max_iter` 더 상향 (500 → 1000)
  - yaw wrap-around 처리 검토
  - `fixed_variable_treatment: make_parameter` 옵션이 slack 변수와 상호작용 이슈 가능성

### 이슈 B: 궤적이 차 전방 ~1m를 오버슛
- 차가 raceline 예상 속도보다 빨리 달리면 reference window가 실제보다 뒤쳐짐
- 현재 슬라이서는 `target_s += g_vx[idx] * dT` — **raceline 속도로만 전진**
- 차 실제 속도 `car_vx > g_vx`면 car가 k=1 시점에 도달할 위치는 ref[1]보다 앞
- 솔버는 smoothness로 v를 `ref_v`로 끌어오려 하지만 관성으로 순간적으로 초과 → 궤적 앞쪽 1m가 raceline 보다 앞에 나감
- 조치 후보:
  1. `target_s += max(car_vx, g_vx) * dT` — 둘 중 빠른 쪽으로 reference 페이싱
  2. reference window를 N+M (M=추가 마진) 점으로 확장, 솔버가 앞서 나가도 latch할 ref 존재
  3. `w_velocity / v_bias_max` 더 강화 (v_bias_max 낮추기, w_velocity 올리기) — 솔버 v가 ref_v 엄격 추적하도록
  4. velocity tracking 아예 제거하고 속도는 2.5d vel planner에 전권 위임 (원래 설계 의도)

### 이슈 C: 다중 reference 소스 미지원 (versatility 미완)
- 현재는 `/global_waypoints` 고정 구독
- "다양한 상황" 요구사항 위해 리팩터 필요:
  - `~reference_topic` 파라미터화
  - 또는 state_machine이 선정한 단일 토픽 (예: `/planner/active_reference`) 구독
  - 입력은 Wpnt.msg 포맷이면 무엇이든 OK (회피·추월·복구 경로 전부 이 포맷)
- 3d_sampling_based_planner와 **공평 비교**하려면 동일 reference/obstacle 입력 계약 필요

---

## 9. 다음 단계 우선순위

1. **이슈 B 해결** — 오버슛 제거가 "궤적이 이쁘게 나오는" 목적에 직접적. 1번 또는 3번 조치부터.
2. **이슈 A 관찰** — 다음 세션 주행 로그에서 fail 원인 파악 후 결정.
3. **이슈 C 리팩터** — reference 소스 교체 가능한 구조로. state_machine 통합 준비.
4. **(나중) 공평한 비교**: sampling planner와 동일 입력 계약 (OTWpntArray 출력, 같은 obstacle prediction 소스) → 선택.

---

## 10. 구동 방법 (현재)

### 빌드
```bash
docker exec icra2026 bash -c "source /opt/ros/noetic/setup.bash && \
  source /home/unicorn/catkin_ws/devel/setup.bash && \
  cd /home/unicorn/catkin_ws && catkin build mpc_planner"
```

### 실행 (터미널 3개)
```bash
# T1: localization
roslaunch glim_ros glil_cpu.launch

# T2: base system (global_waypoints, car_state 제공)
roslaunch stack_master 3d_base_system.launch
# 또는
roslaunch stack_master car_race.launch map:=experiment_3d_2

# T3: MPC planner
roslaunch mpc_planner mpc_planner.launch
```

### RViz 토픽
- `/planner/mpc/trajectory` (Path) — 주황 선
- `/planner/mpc/markers` (MarkerArray) — 초록 LINE_STRIP + 파란 SPHERE_LIST

### 실패 로그 포맷
```
[MPC fail] s=<s> xy_err=<m> dpsi=<rad>
           ref_v=[<min>..<max>] dl=<m> dr=<m>
           win_corr_min=<m>@s=<s> iter=<n> status=<IPOPT status> solve=<ms>
```
