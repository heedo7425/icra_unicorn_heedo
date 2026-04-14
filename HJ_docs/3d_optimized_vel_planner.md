# 3D Optimized Velocity Planner

## 개요

3D 경로 고정 상태에서 **속도 프로파일만 재최적화**하는 ROS 노드.  
기존 `gen_global_racing_line.py`는 경로(n, chi)와 속도를 동시에 최적화해서 ~1분 걸리는데, 이 노드는 **경로를 고정**하고 속도만 최적화해서 **~3초**에 끝난다.

**목적**: 타이어 파라미터/GGV 튜닝 시, 같은 경로 위에서 새 GGV로 속도 프로파일이 어떻게 변하는지 빠르게 확인.

파일 위치: `stack_master/scripts/3d_optimized_vel_planner.py`

---

## 구현 내용

### 핵심 아이디어: Reduced-state NLP

기존 NLP (`gen_global_racing_line.py`)는 **state = [V, n, chi, ax, ay]** (5개), **control = [jx, jy]** (2개)로 경로 전체를 최적화한다. 우리는 **n과 chi를 고정 파라미터로** 취급해서 NLP를 축소한다.

| | 원본 | 이 노드 |
|---|---|---|
| state | V, n, chi, ax, ay | **V, ax** |
| control | jx, jy | **jx** |
| ay | decision variable | **algebraic** (대수적으로 결정) |
| 변수 수 (~400 pts) | ~3,155 | **~1,193** |
| 수렴 iter | ~수백 | **~35** |
| 시간 | ~60초 | **~3초** |

### 수학적 근거: 결과가 동일한가?

원본 NLP의 dynamics 식 `dchi/ds = ay/(V·s_dot) - Ω_z`에서,
**chi(s)가 고정되면** ay는 다음과 같이 **대수적으로** 결정된다:
```
ay = V · s_dot · (dchi/ds + Ω_z)
```

즉 원본 최적해 `(V*, n*, chi*, ax*, ay*)`에서 위 관계가 **이미 성립**하므로, n/chi를 고정한 새 NLP의 해는 `(V*, ax*)`와 **같다** (같은 비용, 같은 제약, 같은 최적점).

나머지는 전부 원본과 동일:
- `Track3D.calc_apparent_accelerations(V, n, chi, ax, ay, s, ...)` 그대로 호출 (n, chi만 상수로)
- GGV diamond 제약 동일
- Cost function `min lap time + jerk regularization` 동일
- RK4 4차 적분 동일

### 경로 고정 (loop closure)

Track3D의 `resample(step_size)`는 고정 step을 쓰므로 `(N-1)·step ≠ L_track`인 경우가 많다. 원본 optimizer는 이 차이로 마지막 0.2m가량 최적화되지 않고 periodic boundary `V[0] == V[N-1]`가 실제 loop closure와 엄밀히 일치하지 않는다.

이 노드는 **트랙 길이에 정확히 맞는 step을 자동 계산**한다:
```python
L_track = track.s[-1] + track.ds
N = round(L_track / desired_step)    # ≈ 398
actual_step = L_track / N             # ≈ 0.19983
track.resample(actual_step)
```

→ NLP 그리드가 `[0, actual_step, ..., (N-1)·actual_step]`로 **전체 트랙을 정확히** 덮고, periodic boundary가 **s=0 ↔ s=L_track** (실제 loop closure)과 일치.

### Dynamics (reduced)

```
s_dot  = V·cos(chi) / (1 - n·Ω_z(s))              ← n, chi는 고정 함수
ay     = V·s_dot·(dchi/ds + Ω_z(s))                ← 대수적으로 결정
dV/ds  = ax / s_dot
dax/ds = jx / s_dot                                 ← jx만 control
```

GGV diamond 제약은 `Track3D.calc_apparent_accelerations`로 apparent (ax_tilde, ay_tilde, g_tilde)를 계산해서 그대로 적용.

### Warm start

`<raceline>.csv`의 `v_opt`, `ax_opt`를 IPOPT 초기 guess로 넣는다. cold start (V=3.0 균일)로도 같은 해에 수렴하지만 warm start가 iter 수를 크게 줄인다.  
**이 NLP는 거의 convex 구조**라서 초기값에 따른 local optimum 편향 위험 없음.

### Solver tolerance (V_min 디제너러시 대응)

V_min을 양수로 설정하면 타이트한 코너에서 `V=V_min`과 `ay=ay_max`가 동시에 active 되는 지점이 생겨 dual multiplier가 진동한다 (primal은 수렴하지만 inf_du가 0.08~2.5에서 왔다갔다). 그래서 tolerance를 다음과 같이 **완화**했다:

- `tol`, `constr_viol_tol`, `compl_inf_tol`: **tight 1e-4** 유지  
  → 제약 위반은 물리적으로 불가능한 trajectory 의미
- `dual_inf_tol`, `acceptable_dual_inf_tol`: **1e-1 / 5.0** 으로 완화  
  → dual noise는 degenerate active set의 자연스러운 현상

object 오차는 `1e-4 × 20s ≈ 2ms` 수준으로 무시 가능. max_iter는 100으로 제한 (보통 30~80회로 수렴).

### Linear solver (HSL ma27 + fallback)

import 시 `ma27` probe를 돌려서:
- **HSL 있으면 → ma27 사용** (2~3배 빠름)
- **없으면 → MUMPS fallback** (그대로 동작)

HSL 설치: `planner/3d_gb_optimizer/fast_ggv_gen/solver/README.md` 참고.

---

## ROS 토픽 구조

```
subscribe: /global_waypoints  (WpntArray, latched)
            ↓
            [첫 메시지 하나만 받고 unregister]
            ↓
            메시지를 템플릿으로 사용 (x, y, z, psi, kappa, ...)
            ↓
            NLP 결과(V, ax)를 각 waypoint의 s_m 기준으로 보간해서 덮어쓰기
            ↓
publish:   /global_waypoints  (same topic, latched)
```

**같은 토픽에 재퍼블리시**하므로 기존 소비자(vel_planner, controller 등)는 추가 수정 없이 새 속도 프로파일을 사용한다. latched message 특성상 우리가 발행한 게 덮어쓴다.

NLP는 **노드 시작 시 1회**만 돌고, 토픽 수신은 메시지 템플릿 취득용. 한 번 퍼블리시 후 spin만 한다.

### NLP 길이 vs 토픽 길이 차이

로그에 `msg L=47.51m, NLP L=49.88m` 처럼 **두 길이가 다르게** 찍힌다. 이건 의도된 동작:

- **NLP L = centerline arc length** (`<map>_3d_smoothed.csv`의 s_m)
- **msg L = racing-line arc length** (`global_waypoints.json`이 재파라미터화한 값)

두 값은 같은 루프의 서로 다른 parameterization. Racing line은 코너를 가로지르며 짧아지므로 센터라인보다 보통 몇 % 짧다. Racing line이 **x, y 공간에서는 동일**하지만, arc-length 좌표계에서는 서로 다른 값을 갖는 것뿐.

### 출력 보간 (periodic wrap)

NLP가 푸는 그리드 길이(centerline)와 토픽의 waypoint s_m(racing-line)이 다르므로 스케일 보정 후 보간:

1. `V_opt` 끝에 `V_opt[0]` 추가 → 순환 배열
2. 토픽의 `s_m`을 NLP 그리드 범위로 스케일링 (`s_query = s_msg * L_nlp / L_msg`)
3. `np.interp` 선형 보간

→ 시작점과 끝점이 부드럽게 연결된다.

---

## 사용법

### 인자 구조

| 인자 | 필수 | 기본값 | 설명 |
|------|:---:|-------|------|
| `--map` | ✅ | — | 맵 폴더명 (예: `eng_0415_v4`). track csv 자동 `<map>_3d_smoothed.csv` |
| `--racecar` | ❌ | — | **shortcut**: 지정 시 raceline/gg_dir/vehicle_yml을 이 이름 기반으로 일괄 설정 |
| `--raceline` | ❌ | `rc_car_10th` (또는 racecar) | racing line variant. 파일명 `<map>_3d_<raceline>_timeoptimal.csv` |
| `--vehicle_yml` | ❌ | `params_rc_car_10th.yml` (또는 `params_<racecar>.yml`) | 차량 yml 파일명 |
| `--gg_dir` | ❌ | `rc_car_10th` (또는 racecar) | GG 폴더명 |
| `--V_min` | ❌ | `1.0` | 최소 속도 하한 (m/s) |
| `--step_size_opt` | ❌ | `0.2` | NLP 그리드 간격 (m). 트랙 길이에 맞게 자동 조정 |
| `--gg_margin` | ❌ | `0.0` | GGV 여유 |

### 인자 우선순위

개별 인자 (`--raceline`, `--gg_dir`, `--vehicle_yml`) > `--racecar` shortcut > 기본값 (`rc_car_10th`).

### 사용 예시

```bash
# 1. 최소 명령 — 기본 rc_car_10th 설정으로
rosrun stack_master 3d_optimized_vel_planner.py --map eng_0415_v4

# 2. racecar shortcut — 세 인자 한 번에
rosrun stack_master 3d_optimized_vel_planner.py --map eng_0415_v4 --racecar rc_car_10th_v7
# → raceline=rc_car_10th_v7, gg_dir=rc_car_10th_v7, vehicle_yml=params_rc_car_10th_v7.yml

# 3. racecar + 일부 override (비교 실험)
rosrun stack_master 3d_optimized_vel_planner.py --map eng_0415_v4 --racecar rc_car_10th_v7 --gg_dir rc_car_10th
# → raceline은 v7, GGV만 기본으로 → "v7 경로를 기본 GGV로 재평가"

# 4. 모든 인자 개별 지정 (명시적)
rosrun stack_master 3d_optimized_vel_planner.py --map eng_0415_v4 \
    --raceline rc_car_10th_v7 --gg_dir rc_car_10th_v7 --vehicle_yml params_rc_car_10th_v7.yml

# 5. V_min 조절
rosrun stack_master 3d_optimized_vel_planner.py --map eng_0415_v4 --racecar rc_car_10th_v7 --V_min 1.5
```

### 자동 해결되는 경로

| 인자 | 실제 경로 |
|------|----------|
| `--map eng_0415_v4` | `stack_master/maps/eng_0415_v4/` (map folder) |
| track (자동) | `<map>/<map>_3d_smoothed.csv` |
| `--raceline X` | `<map>/<map>_3d_X_timeoptimal.csv` |
| `--vehicle_yml Y.yml` | `planner/3d_gb_optimizer/global_line/data/vehicle_params/Y.yml` |
| `--gg_dir Z` | `planner/3d_gb_optimizer/global_line/data/gg_diagrams/Z/velocity_frame/` |

---

## 실행 전제 조건

### ROS 환경
- `roscore`가 이미 떠 있을 것
- `/global_waypoints` 토픽이 latched로 발행되어 있을 것 (예: `stack_master` base_system.launch)
- 컨테이너: `icra2026` Docker 안에서 실행 (`rosrun` 가능한 환경)
- `catkin build` 완료 (package 인식 필요)

### 파일 의존성 (CSV / YAML / GGV)

노드 시작 시 다음 파일들을 **읽는다** (없으면 `FileNotFoundError`로 즉시 중단):

| 경로 | 역할 |
|------|------|
| `stack_master/maps/<map>/<map>_3d_smoothed.csv` | centerline geometry (s, x, y, z, theta, mu, phi, omega_x/y/z, w_tr_left/right) → Track3D |
| `stack_master/maps/<map>/<map>_3d_<raceline>_timeoptimal.csv` | 고정 경로 (s_opt, n_opt, chi_opt, v_opt, ax_opt 등) |
| `planner/3d_gb_optimizer/global_line/data/vehicle_params/<vehicle_yml>` | 차량 파라미터 (m, h, T, delta_max 등) |
| `planner/3d_gb_optimizer/global_line/data/gg_diagrams/<gg_dir>/velocity_frame/*.npy` | GGV diamond (v_list, g_list, alpha_list, rho, gg_exponent, ax_min, ax_max, ay_max) |

→ 위 파일들이 실제 디스크에 존재해야 한다. 일반적으로:
- smoothed.csv, timeoptimal.csv → `3d_mapping.launch` 또는 `3d_global_line.launch`로 미리 생성
- vehicle yml → 손으로 편집/저장
- GG .npy → `fast_ggv_gen/run_on_container.sh` 또는 원본 `gg_diagram_generation/`으로 미리 생성

### 선택 의존성
- HSL ma27 — 있으면 속도 2~3배. 없으면 MUMPS 자동 fallback (정상 동작). 설치 가이드: `fast_ggv_gen/solver/README.md`

---

## 동작 예시 로그

```
[velopt] linear_solver: ma27 (HSL)
[velopt] map=eng_0415_v4
[velopt] track    : .../eng_0415_v4_3d_smoothed.csv
[velopt] raceline : .../eng_0415_v4_3d_rc_car_10th_v7_timeoptimal.csv
[velopt] vehicle  : .../params_rc_car_10th_v7.yml
[velopt] gg       : .../gg_diagrams/rc_car_10th_v7/velocity_frame
[velopt] L_track=49.8800m, desired_step=0.2000, actual_step=0.199829 (N=250)
[velopt] fixed n  : [-0.573, 1.301] m
[velopt] fixed chi: [-0.441, 0.287] rad
[velopt] solving NLP ...
[velopt] NLP: 749 vars, 1250 constraints, 250 points
[velopt] solver built in 0.17s, solving...
[velopt] IPOPT: 2.52s, success=True, laptime=13.25s
[velopt] V range [2.74, 7.51] m/s  success=True
[velopt] waiting for /global_waypoints template message ...
[velopt] published /global_waypoints (msg L=47.51m, NLP L=49.88m, V[0]=2.74, V[-1]=2.74)
```

총 소요: **~3~4초** (build 0.2s + NLP ~2.5s + message I/O).  
HSL 미설치 시 MUMPS로 2~3배 느림 (~7~10초).

---

## 제약 & 주의사항

### 1. 경로와 GGV는 반드시 **매칭**해야 한다
Racing line은 특정 GGV의 그립 한계에 맞춰 설계된다. 다른 GGV로 풀면 infeasible해서 IPOPT가 수렴 못하고 max_iter(100)에서 종료된다. `--racecar` shortcut으로 통일하는 것을 권장.

### 2. 경로는 절대 안 바뀐다
n, chi는 입력 csv에서 고정. 경로 자체를 바꾸고 싶으면 원본 `gen_global_racing_line.py` 사용.

### 3. 파일 생성 없음
토픽에만 퍼블리시. 저장하려면 따로 작업 필요.

### 4. 한 번만 풀고 spin
노드 시작 시 1회. 파라미터 변경하려면 재실행. rqt 기반 실시간 튜닝은 추후 확장.

---

## 향후 계획 (rqt 실시간 튜닝 연동)

현재 구조는 다음 워크플로우의 기반:

```
[rqt 슬라이더]
  타이어 파라미터 (lambda_mu, P_max, ...) 조절
       │
       ├── fast_ggv_gen으로 GGV 재생성 (~0.5초, HSL)
       │
       ├── 3d_optimized_vel_planner로 속도 재계산 (~2.5초)
       │    (노드 재실행 or topic-triggered re-solve)
       │
       └── /global_waypoints에 새 속도 덮어쓰기
           → 실차/시뮬에서 즉시 반영
```

파라미터 변경 → 시뮬 반영까지 **< 5초** 루프 가능 (HSL 활성 시).

---

## 관련 파일

- `stack_master/scripts/3d_optimized_vel_planner.py` (이 노드)
- `planner/3d_gb_optimizer/global_line/global_racing_line/gen_global_racing_line.py` (원본, 참고용, **수정하지 않음**)
- `planner/3d_gb_optimizer/global_line/src/track3D.py` (Track3D, library로 import)
- `planner/3d_gb_optimizer/global_line/src/ggManager.py` (GGManager, library로 import)
- `planner/3d_gb_optimizer/fast_ggv_gen/` (GGV 고속 생성기, 같이 쓰면 튜닝 루프 완성)
- `planner/3d_gb_optimizer/fast_ggv_gen/solver/README.md` (HSL ma27 설치 가이드)
- `HJ_docs/fast_ggv_gen.md` (fast GGV 문서)
