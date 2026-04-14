# 3D Dynamic Prediction & Planner 포트 정리

작성일: 2026-04-15
범위: 2D → 3D 포팅 중 동적 장애물 예측 + SQP 동적 회피 경로 생성

---

## 1. 목적 / 범위

ICRA 2026 RoboRacer (3D 트랙) 용으로 기존 2D 스택의 동적 회피 파이프라인을 포팅.
핵심 원칙 (사용자 지시):
- **최대한 원본 구조 그대로** 가져와서 3D 로만 변경.
- **발견되는 버그는 고친다** (원본에 있던 것도).
- 2D 원본 파일은 유지, 3D 버전은 `3d_` 프리픽스 신규 파일로 분리.
- 향후 multi-vehicle 대응 가능성 인지만 해두고 일단 single-vehicle 기준.

## 2. 만든/건드린 파일

### 신규 (3D)
| 파일 | 역할 | 원본 |
|---|---|---|
| `prediction/gp_traj_predictor/src/3d_opp_prediction.py` | 상대 차량 VD/linear/trajectory 예측 | `opp_prediction.py` |
| `prediction/gp_traj_predictor/src/3d_opponent_trajectory.py` | 상대 반주행 궤적 수집 (marker z 3D화) | `opponent_trajectory.py` |
| `planner/sqp_planner/src/3d_sqp_avoidance_node.py` | SLSQP 기반 동적 회피 최적화 | `sqp_avoidance_node.py` |

### 수정
| 파일 | 변경 |
|---|---|
| `stack_master/launch/3d_headtohead.launch` | `enable_prediction`, `enable_planners` 인자로 3D SQP + 3D prediction 그룹 활성화 |

### 기존 공용 라이브러리 (이미 존재, 활용)
- `f110_utils/libs/track_3d_validator/` — Track3DValidator (GridFilter 대체), `frenet_utils` (wrap-around utilities)

---

## 3. 3D 포팅 변경 (surgical)

### 3.1 공통 패턴
- **GridFilter (2D PNG erosion) 제거** → `Track3DValidator` 사용. s-기반 wall ray-cast + d-bound 2-stage 검증.
- **Frenet wrap-around 유틸**: `signed_s_dist`, `circular_s_dist`, `unwrap_s`, `in_s_range`, `s_forward_add` (`from track_3d_validator import ...`). 시작/끝점 주변 "뜀" 현상을 막기 위해 거리/비교/grad 계산시 전부 경유.
- **3D 출력**: `wpnt.z_m = converter.spline_z(s)` (CubicSpline 기반 트랙 표면 높이 보간).
- **Local curvature 속도**: `v = v_ref · √|1 − d·κ_ref|` (recovery/static 과 일관).
- **d_m 보정** (필요 시): `wpnt.d_right = ref.d_right + d_m`, `wpnt.d_left = ref.d_left − d_m` — SQP 출력엔 현재 미적용 (downstream이 안 쓰고 있음). 필요하면 추가.

### 3.2 3d_sqp_avoidance_node.py 전용 변경
- smart_static 경로 제거 (3D에선 미사용).
- `FrenetConverter(x, y, z)` 3D 초기화 (원본이 이미 준비됨, 확인만).
- Track3DValidator 로 스무딩 후 경로 검증. 실패 시 `past_avoidance_d` 리셋 + 빈 `OTWpntArray` 발행.
- CCMA 스무딩 유지.
- SLSQP 옵션 완화: `ftol=1e-3 (← 1e-1), maxiter=50 (← 20)`, warm start 유지.

### 3.3 3d_opp_prediction.py
- smart_static_active_cb / smart_static_wpnts_cb / smart_converter / smart_wpnts_gb_frenet 전부 제거.
- C++ `glob2frenet`/`frenet2glob` 서비스는 그대로 사용 (3D-aware).
- 3 예측 방식 (VD-blended / linear-to-center / trajectory-follow) 원본 유지.
- 디버그 로그 1개 정리.

### 3.4 3d_opponent_trajectory.py
- 단일 변경: 마커 `pose.position.z = spline_z(proj_opp.s)`.
- 나머지 2D 원본 그대로.

---

## 4. 버그 / 이슈 / 해결

### 4.1 포팅 중 발견한 원본 SQP 버그 (수정 완료)

#### ① `scaled_max_idx` / `max_idx_updated` off-by-one
- **원인**: 원본이 `data.wpnts[-1].id = N-1` 를 모듈로로 사용. 배열 크기는 N. `(N-1) % (N-1) = 0` 이라 마지막 웨이포인트가 0번으로 aliased.
- **증거**: [3d_global_planner_node.py:1300](../planner/2.5d_gb_optimizer/src/3d_global_planner_node.py#L1300) 에서 `global_wpnt.id = i` — id 는 enumerate 로 0..N-1.
- **해결**: `len(data.wpnts) = N` 으로 대체. [3d_sqp_avoidance_node.py:162-173](../planner/sqp_planner/src/3d_sqp_avoidance_node.py#L162-L173).

#### ② `psi_constraint` 부호 손실
- **원인**: `0.02 − abs((d[1] − d[0]) − abs(desired_dd))` — 안쪽 `abs(desired_dd)` 때문에 `e_psi < 0` (우측 헤딩) 인데도 `d[1] − d[0] > 0` (좌측 이동) 을 강제. 실제 헤딩과 반대 방향으로 경로 시작.
- **해결**: 안쪽 `abs` 제거. `d[1] − d[0] ≈ desired_dd` (부호 보존). [3d_sqp_avoidance_node.py:478-484](../planner/sqp_planner/src/3d_sqp_avoidance_node.py#L478-L484).

#### ③ `side_consistency_constraint` infeasibility
- **원인**: 모든 `d` 에 side 제약 (`d ≥ 0` or `−d ≥ 0`). `start_on_raceline_constraint` 가 `d[0] = current_d` 를 강제하므로, 차가 raceline 반대편에 있으면 첫 점에서 제약 위배 → SLSQP 실패.
- **해결**: side 제약을 장애물 범위 indices (`obs_downsampled_indices`) 에만 적용. 진입/복귀 구간은 자유. [3d_sqp_avoidance_node.py:515-526](../planner/sqp_planner/src/3d_sqp_avoidance_node.py#L515-L526).

#### ④ `turning_radius_constraint` 의 κ_ref 누락 (dead code 연결)
- **원인**: 원본이 `kappa = y'' / Δs² ≈ d''(s)` 만 κ 로 취급, 레퍼런스 곡률 κ_ref 누락. Frenet 오프셋 경로의 실제 곡률은 `κ_path ≈ κ_ref + d''(s)`. 코너에서 κ_ref 가 마찰 예산을 이미 쓰고 있는데 `|d''|` 까지 한계까지 허용 → 총 곡률 한계 초과 → 미끄러짐.
- **결정적 증거**: `self.global_traj_kappas` 는 `sqp_solver` 에서 계산만 되고 **어디서도 읽히지 않음**. 원저자가 여기 연결할 예정이었다가 누락한 것으로 판단.
- **해결**: `kappa_total = global_traj_kappas + d''/Δs²` 로 연결. [3d_sqp_avoidance_node.py:501-515](../planner/sqp_planner/src/3d_sqp_avoidance_node.py#L501-L515).
- **한계**: `κ_path ≈ κ_ref + d''(s)` 는 1차 근사. 완전식은 `κ = [κ_ref(1 − d·κ_ref) + d'']/[(1 − d·κ_ref)² + d'²]^(3/2)`. 오버테이크 상황 (d<1m, κ_ref<1 rad/m) 에선 `d·κ_ref ≪ 1` 이라 근사로 충분.

### 4.2 포팅 중 발견한 원본 SQP 의 shape/wrap 버그 (수정 완료)

#### ⑤ `np.abs(self.scaled_wpnts − s_start).argmin()` shape 버그
- **원인**: `self.scaled_wpnts` 는 (N, 2) 이고 `s_start` 는 스칼라. 뺄셈 결과가 (N, 2) 로 d 열까지 섞임.
- **해결**: `self.scaled_wpnts[:, 0]` (s 열만) + `circular_s_dist` (wrap 안전).

#### ⑥ 장애물 전/후 판단 mod 버그
- **원인**: `(obs.s_start − cur_s) % max_s` 로 전방 판단 → 뒤쪽 장애물이 반대쪽 앞으로 오인.
- **해결**: `signed_s_dist(cur_s, obs.s_start, max_s)` 로 signed shortest distance. `0 ≤ s_forward < lookahead` 만 채택.

### 4.3 안정성 / 크래시 가드 (신규)

#### ⑦ `/opponent_trajectory` 미수신 상태의 동적 장애물 처리
- **문제**: `self.opponent_wpnts_sm = None` 인데 non-static 장애물 들어오면 `circular_s_dist(None, ...)` 크래시.
- **해결**: 루프 내 필터에서 해당 케이스 skip + `logwarn_throttle` 경고.

#### ⑧ `/global_waypoints_updated` 미수신 크래시
- **문제**: `self.max_idx_updated = None` 일 때 `% None` 크래시.
- **해결**: `wait_for_message` 에 추가 + 루프 상단 ready 가드.

#### ⑨ `/car_state/odom_frenet` 미수신 크래시
- **문제**: `self.cur_yaw` 미설정 상태에서 `psi_constraint` 호출 → AttributeError.
- **해결**: `wait_for_message` 추가.

#### ⑩ SLSQP 수렴 품질 개선
- **문제**: 원본 `ftol=1e-1, maxiter=20` 은 너무 느슨해서 최종 해가 제약을 아슬하게만 만족 → 실주행에서 제약 위반 발생 가능. 또 iter=20 은 복잡한 지형에서 수렴 전에 종료.
- **해결**: `ftol=1e-3, maxiter=50` 으로 완화. warm start 는 이전 해의 `d` 배열을 초기값으로 (원본 로직 유지).

### 4.4 상태머신 연동 확인

- **토픽 경로**: SQP → `/planner/avoidance/otwpnts` → state machine `avoidance_cb` → `self.avoidance_wpnts` → OVERTAKE 상태에서 `behavior_strategy.local_wpnts` 로 중계 → controller.
- **게이트**: state machine 이 `/ot_section_check` 를 publish. SQP 는 subscribe 후 True 일 때만 돌림. 오버테이크 존 밖에선 SQP inactive.
- **빈 메시지 안전성**: [3d_state_machine_node.py:465](../state_machine/src/3d_state_machine_node.py#L465) `if len(data.wpnts) != 0:` 가드로 invalid spline 시 무시.
- **RECOVERY/STATIC vs DYNAMIC 충돌 없음**: state machine 내부 우선순위로 분리, 동시에 local_wpnts 점유 안 함.

---

## 5. Launch 구성

[stack_master/launch/3d_headtohead.launch](../stack_master/launch/3d_headtohead.launch)

```xml
<arg name="enable_planners"   default="False" />  <!-- recovery + static + SQP -->
<arg name="enable_prediction" default="False" />  <!-- GP + opponent trajectory -->
<arg name="dynamic_avoidance_mode" default="SQP" />
```

실행:
```bash
# 풀 스택 (prediction + planners)
roslaunch stack_master 3d_headtohead.launch enable_planners:=true enable_prediction:=true

# 회피만
roslaunch stack_master 3d_headtohead.launch enable_planners:=true

# 예측만 (분석용)
roslaunch stack_master 3d_headtohead.launch enable_prediction:=true
```

---

## 6. 판단 보류 / 미확정

- **ETH spliner / lane_change_planner 포트**: 현재 범위 제외. SQP 로 충분한지 실주행 후 판단.
- **d_m 사후 보정**: SQP 출력 `wpnt.d_right/d_left` 미설정 (0). downstream 이 안 써서 보류, 필요 시 recovery 패턴 적용.
- **`spline_bound_mindist` 동적 조정**: 속도 기반으로 올리면 고속에서 여유 확보 가능. 현재 상수 (0.2m), rqt 에서만 조정.

---

## 7. 앞으로 발전 방향

1. **Multi-vehicle**: 현재는 `/opponent_prediction/obstacles` 가 단일 ObstacleArray. 복수 상대 차량 대응은 obstacle 그룹핑, GP 모델 개별화, SQP constraint 다중화 필요.
2. **κ_path 완전식**: 4.1-④ 의 1차 근사 대신 완전식 적용. `d·κ_ref`, `d'` 이 커질 때 정확도 개선. 현재 코너+오버테이크 동시 상황에선 충분.
3. **Predictor 튜닝**: 3가지 예측 모드 블렌딩 게인이 원본 그대로. 실주행 데이터로 tuning 필요.
4. **SQP latency 모니터링**: `measure:=true` 에서 `/planner/pspliner_sqp/latency` 관찰. avoidance_resolution 을 속도 기반 적응화 검토.
5. **Warm-start 추가 개선**: 현재는 이전 해 그대로 재사용. avoidance_resolution 이 rqt 로 바뀌면 shape mismatch → 초기 apex 로 fallback. shape 변경 시 interp 로 warm start 이어가는 것도 가능.
6. **Side consistency 재검토**: 4.1-③ 에서 장애물 구간에만 적용으로 바꿨지만, `consecutive_points_constraint` 와 중복 느낌 있음. 실주행에서 하나만 남겨도 되는지 검증.
7. **κ_ref 정확도**: `global_traj_kappas` 가 xy 기반 `np.diff` 2회로 구해짐 — noise 에 민감. scaled_wpnts 의 `kappa_radpm` 필드를 바로 보간해 쓰는 편이 안정적일 수도 있음 (검증 필요).

---

## 8. 검증 기록

| 항목 | 결과 |
|---|---|
| `catkin build sqp_planner gp_traj_predictor stack_master` | ✅ pass |
| `py_compile` 3D 파일 3개 | ✅ pass |
| 동적 import (spec_from_file_location) 3개 | ✅ pass |
| launch XML syntax | ✅ (빌드에 포함) |
| 실주행 검증 | ⏳ 대기 |

## 9. 관련 문서

- [TODO_0412.md](TODO_0412.md) — recovery/static 3D 포트 플랜
- [TODO_0413.md](TODO_0413.md) — 동적 회피 + 예측 MVP 플랜
- [TODO_HJ.md](TODO_HJ.md) — 전체 3D 포팅 진행 상황
