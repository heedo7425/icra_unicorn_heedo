# `controller/mpc_only`

글로벌 raceline 직접 추종 MPC. `controller/mpc` 와 병렬로 존재하며 같은 차량 모델·OCP solver 를 공유하지만 **상위 pipeline 을 완전히 배제**.

## `controller/mpc` 와의 차이

| 항목 | `controller/mpc` | `controller/mpc_only` |
|---|---|---|
| 추종 대상 | `/behavior_strategy.local_wpnts` (state_machine 출력) | `/global_waypoints_scaled` (velocity_scaler 출력) |
| 의존 노드 | state_machine, planner_spline, recovery_spliner, behavior_controller | velocity_scaler, global_republisher 만 |
| 장애물 회피 | state_machine 이 local_wpnts 를 시프트 (OVERTAKE 상태) | **없음** (Phase 2 에서 OCP 제약으로 추가 예정) |
| TRAILING/RECOVERY 감속 | 자동 (state_machine) | 없음; MPC 가 raceline vx_ref 그대로 추종 |
| Rosparam 네임스페이스 | `/mpc/*` | `/mpc_only/*` |
| Pub 토픽 | `/mpc/prediction`, `/mpc/reference`, `/mpc/solve_ms` | `/mpc_only/*` 동일 계열 |
| Drive 토픽 | `/vesc/high_level/ackermann_cmd_mux/input/nav_1` | 동일 (동시 실행 금지) |
| Codegen dir | `/tmp/mpc_c_generated` | `/tmp/mpc_only_c_generated` |

## 실행

```bash
CAR_NAME=SIM roslaunch controller mpc_only_sim.launch map:=f rviz:=true
```

## 동작 흐름

1. 첫 수신: `/global_waypoints_scaled` 에서 전체 raceline 을 `(M, 10)` ndarray 로 캐시. 이후 업데이트 시 교체.
2. 매 tick (50 Hz):
   - 현 차량 `(x, y)` 에 가장 가까운 raceline wpnt 인덱스 `idx_near` 계산.
   - `[idx_near, idx_near + window_size)` (기본 200개, wrap-around) 을 슬라이스해서 `wpnts` 로 사용.
   - `_current_state()` 로 `x0 = (s=0, n, dpsi, vx, vy, ω, δ)` 생성.
   - `build_preview(wpnts, ...)` → per-stage `(kappa, θ, κ_z, μ, vx_ref, n_ref=0)`.
   - `solve_once()` → `u0 = (δ_rate, a_x)`.
   - δ, speed 통합해서 AckermannDriveStamped publish.

## 검증 체크리스트

- `rostopic hz /global_waypoints_scaled` ≥ 0.5 Hz
- `rostopic hz /vesc/high_level/ackermann_cmd_mux/input/nav_1` ≈ 50 Hz
- `rosnode list | grep -E 'state_machine|planner_spline|dyn_statemachine'` → 결과 없음
- RViz: `/mpc_only/prediction` (녹색) horizon, `/mpc_only/reference` (주황) raceline window 표시
- 10랩 평균: lap time / avg_lat / max_lat 이 state_machine-based `controller/mpc` 와 동등 또는 우월

## Phase 2 (계획)

- `/tracking/obstacles` 구독 → 각 obstacle `(s_center, d_center, size)` 를 OCP `con_h_expr` 에 soft constraint 로 추가:
  - `(s_k - s_obs)^2 / r_s^2 + (n_k - d_obs)^2 / r_d^2 ≥ 1` 타원형 회피 영역
- `planner_spline` / state_machine 없이 MPC 가 직접 회피 궤적 생성.
