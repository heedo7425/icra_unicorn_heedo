# 3D 포트 버그 카탈로그 — Recovery / Static / Prediction

작성일: 2026-04-17
대상 파일:
- `planner/recovery_spliner/src/3d_recovery_spliner_node.py`
- `planner/spliner/src/3d_static_avoidance_node.py`
- `planner/sqp_planner/src/3d_sqp_avoidance_node.py`
- `prediction/gp_traj_predictor/src/3d_opp_prediction.py`
- `prediction/gp_traj_predictor/src/3d_opponent_trajectory.py`
- `prediction/gp_traj_predictor/src/gaussian_process_opp_traj.py`
- `prediction/gp_traj_predictor/src/predictor_opponent_trajectory.py`
- `f110_utils/libs/track_3d_validator/src/track_3d_validator/track_3d_validator.py`
- `f110_utils/libs/f110_msgs/msg/OppWpnt.msg`

목적: 2D → 3D 포트 중 드러난 버그들을 파일별/증상별로 정리. 각 버그에 대해 **2D 원본에도 있었던 것인지**, **증상**, **원인**, **수정 방법**, **커밋 해시** 를 기록. Wrapping 관련 공통 패턴은 별도 섹션에 모음.

---

## 1. 공통 원칙 / 설계 노트

### 1.1 2D → 3D 포트의 근본 차이

| 관점 | 2D 원본 | 3D 변경 | 드러난 버그 |
|---|---|---|---|
| Frenet 좌표 | XY → (s, d) 2D nearest projection | 3D 트랙은 XY overlap (다리/교차) 가능 | nearest 가 다른 층 s 로 flip |
| z 좌표 | 항상 0 (평면) | 경사면 · 고가 구간 존재 | 마커가 표면 안 따라감 |
| 속도 분포 | 고정 방향 곡선 | 층별 raceline 속도 다름 | vx 가 튀거나 엉뚱한 wpnt 참조 |
| wrap-around | 짧은 트랙이면 드물게 문제 | 상대 수 랩 돌면 경계 자주 히트 | argmin, diff, mod 에서 틀림 |

### 1.2 관찰 중 확인된 일반 원칙
- **"알고 있는 정보를 굳이 재추정하지 말 것"**: 이미 아는 s 값이 있으면 그대로 써야지 get_frenet 재투영으로 덮어쓰는 건 정보 손실 + 오류 기회.
- **"sentinel 값은 downstream 이 체크하지 않으면 함정"**: placeholder (e.g. `proj_vs_mps = 100`) 은 semantics 문서화가 없고 소비자가 blind 로 쓰면 터짐.
- **"wrap-around 는 단일 함수로 추상화"**: `frenet_utils` 의 `circular_s_dist` / `signed_s_dist` 로 통일, 각 호출부에서 `%` 를 따로 처리 안 함.

---

## 2. Recovery Spliner (`3d_recovery_spliner_node.py`)

### 2.1 `get_frenet(x, y)` 2D nearest projection 사용

- **원본 2D 에도 있었나?** ✅ 있었음 (`recovery_spliner_node.py` 의 원본 패턴 그대로 포트됨). 2D 맵에서는 XY overlap 없으니 증상 안 나타남.
- **증상** (실측 로그):
  ```
  s : 17.07 → 17.17 → 50.37 → 17.37    (한 샘플만 s 가 ~33m 점프)
  z : 0.14  → 0.14  → 0.60  → 0.14     (층별 elevation flip)
  vx: 4.01  → 4.09  → 6.41  → 4.24     (엉뚱한 wpnt 의 vx_mps 참조)
  ```
  80Hz 루프 중 가끔 한 실린더만 **공중에 튀거나 2배 높이로 보임**.
- **원인**: `self.converter.get_frenet(samples_xy[:,0], samples_xy[:,1])` 가 2D 최근접 투영. 3D 트랙에서 다리/고가 같은 **XY 오버랩 구간** 에서는 한 샘플의 투영이 다른 층의 s 로 flip 가능. 그 s 로 `gb_wpnt_i` 잡고 `spline_z(s)`, `ref.vx_mps` 참조하니 **s/z/vx 가 동시에 잘못된 층 값** 으로 세팅.
- **수정**: BPoly spline 이 `(cur_x, cur_y)` 에서 시작하고 `cur_s` 는 3D-aware C++ `frenet_conversion` 에서 오므로 **재투영 불필요**:
  - `s = cur_s + cumulative arc-length along XY samples`
  - `d = signed projection onto raceline tangent-normal at that known s`
  - 2D 최근접 투영 완전 제거.
- **커밋**: [`e7d5157`](https://github.com/Brojoon-10/ICRA2026_HJ/commit/e7d5157)

---

## 3. Static Avoidance (`3d_static_avoidance_node.py`)

### 3.1 `get_frenet(x, y)` 2D nearest projection 사용

- **원본 2D 에도 있었나?** ✅ 있었음 (recovery 와 동일 패턴).
- **증상**: recovery 와 동일 원리. 3D 오버랩 구간에서 회피 경로의 한 샘플이 layer flip.
- **원인 / 수정**: recovery 와 동일 방식. arc-length 누적 s + geometric normal d.
- **커밋**: [`c372849`](https://github.com/Brojoon-10/ICRA2026_HJ/commit/c372849)

### 3.2 Side 선택 로직이 장애물 위치 무시 (`_decide_obstacle_strategy_gb_aware`)

- **원본 2D 에도 있었나?** ✅ 있었음 (동일 로직 복사).
- **증상**: 공간 많은 반대편을 두고 좁은 쪽으로 회피 방향 선택 → `Track3DValidator` 필터에 걸려서 회피 실패. 실주행에서 "양옆으로 공간 많은데 벽으로 가네" 현상.
- **원인**: 원본은 `gb_wp.d_left > threshold` 만 체크해서 raceline ↔ 좌측벽 거리가 크면 무조건 "left 로 회피". **장애물의 실제 위치 무시**. 장애물이 좌측벽에 붙어 있으면 raceline ↔ 좌측벽 거리는 커도 실제 통과 gap 은 0에 가까움.
- **수정**: 양쪽 실제 gap 계산 후 더 넓은 쪽 선택:
  ```python
  obs_d_left = obs_d + obs_radius
  obs_d_right = obs_d - obs_radius
  left_gap  = gb_wp.d_left - obs_d_left     # 장애물 left edge ↔ 좌측벽
  right_gap = gb_wp.d_right + obs_d_right   # 장애물 right edge ↔ 우측벽
  # left_ok / right_ok 체크 후 넓은 쪽 선택, 양쪽 다 애매하면 max 선택
  ```
- **커밋**: [`94769d4`](https://github.com/Brojoon-10/ICRA2026_HJ/commit/94769d4)

### 3.3 Savitzky-Golay 스무딩 시작단 taper 누락

- **원본 2D 에도 있었나?** ✅ 있었음 (smart 계열 원본부터).
- **증상**: 생성된 회피 스플라인의 **첫 샘플 근처에 작은 kink** (나머지는 부드럽게 나옴).
- **원인**: savgol (window=51) 이 전체 샘플을 스무딩해서 좌우 끝이 shift 되는데, 코드는 **끝단에만** taper 로 `end_pt` 로 수렴시키고 **시작단은 `samples[0] = start_pt` 하드 복원만** 함. 결과적으로 `samples[0]` (복원된 car 위치) 과 `samples[1..k]` (savgol-shifted) 사이 **불연속** → 시각적 꼬임.
- **수정**: 끝단과 대칭으로 시작단 taper 추가:
  ```python
  for bi in range(blend_len):
      idx = bi
      w = bi / blend_len                              # 0 at idx=0 (pure start_pt)
      samples[idx] = start_pt * (1 - w) + samples[idx] * w
  samples[0] = start_pt                              # 최종 핀
  ```
- **커밋**: [`f38dc7d`](https://github.com/Brojoon-10/ICRA2026_HJ/commit/f38dc7d)

---

## 4. SQP Avoidance (`3d_sqp_avoidance_node.py`)

### 4.1 CCMA 스무딩 후 `get_frenet` 재투영

- **원본 2D 에도 있었나?** ✅ 있었음.
- **증상**: recovery/static 과 유사한 layer flip 가능.
- **원인**: SLSQP 결과 `result.x` (d 배열) 을 `s_array = linspace(start, end, N)` 와 함께 `get_cartesian` 으로 XY 생성 → CCMA 스무딩 → 그 결과를 다시 `get_frenet` 으로 (s, d) 재추출. CCMA 는 XY 만 살짝 스무딩하고 **s 축 위치를 바꾸지 않음**. 그런데도 재투영을 돌림.
- **수정**: `s_array` 재사용 (= `evasion_s`), `d` 는 geometric normal projection 으로 재계산:
  ```python
  # CCMA 후 XY 만 바뀌었으므로:
  evasion_s = np.mod(s_array, scaled_max_s)  # 그대로
  # d = signed normal projection at each known s (spline_x/y 와 도함수 활용)
  ```
- **커밋**: [`d58abac`](https://github.com/Brojoon-10/ICRA2026_HJ/commit/d58abac)

### 4.2 원본 SQP 의 기타 누적 버그들 (이전 세션에 수정 완료)

SQP 3D 포트 초기에 발견된 원본 버그들 (2D 원본에도 다 있던 것):

| 버그 | 내용 | 커밋 |
|---|---|---|
| `scaled_max_idx` off-by-one | `data.wpnts[-1].id` (= N-1) 을 모듈로로 써서 마지막 wpnt 가 0 으로 aliased | `0452f5d` 포함 |
| `psi_constraint` abs 부호 날림 | `abs(desired_dd)` 로 감싸서 우측 헤딩인데도 `d[1]-d[0]` 양수 강제 | `0452f5d` 포함 |
| `side_consistency_constraint` infeasibility | 전체 `d` 에 적용 → `d[0] = current_d` pinning 과 충돌, solver 실패 | `0452f5d` 포함 |
| `turning_radius_constraint` κ_ref 누락 | `kappa = d''(s)` 만 제한, κ_ref 무시. `global_traj_kappas` 는 계산만 되고 dead code 였음. 완전식 `κ_path ≈ κ_ref + d''(s)` 로 연결 | `0452f5d` 포함 |

---

## 5. Prediction (GP Trajectory) — 3 노드 공통

### 5.1 Sentinel `proj_vs_mps = 100` → 마커 높이 폭발

- **원본 2D 에도 있었나?** ✅ 있었음 (`gaussian_process_opp_traj.py:412` 의 주석 "make trajectory with velocity 100 for the first half lap").
- **증상**: `/opponent_traj_markerarray` 의 **일부 실린더가 10m 씩 폭발**. 관측 안 된 구간 (미학습 상태) 에서만 나타나고 관측되면 정상. 달리다가도 아직 안 지나간 구간은 계속 폭발.
- **원인**:
  1. GP 노드가 첫 반 바퀴 동안 전체 lap 의 `proj_vs_mps` 를 `100` 으로 placeholder 세팅.
  2. **Downstream 에서 `== 100` 체크하는 코드 0곳** (stack 전체 grep 결과). 원저자 의도만 있고 consumer 부재.
  3. 마커 공식 `scale.z = proj_vs_mps / max_velocity` 가 센티널을 수학적으로 흡수 → `100 / 10 ≈ 10m` 실린더.
  4. 더 심각하게, **`3d_opp_prediction` 은 실제로 `proj_vs_mps` 를 속도로 써서 미래 위치 예측** ([3d_opp_prediction.py:331](../prediction/gp_traj_predictor/src/3d_opp_prediction.py#L331), [:492](../prediction/gp_traj_predictor/src/3d_opp_prediction.py#L492)) — 100 m/s 라고 취급하면 1초 뒤 100m 앞에 있다고 예측 → 완전 넌센스.
- **수정**: Sentinel 완전 제거 + 명시적 플래그 도입.
  - `OppWpnt.msg` 에 `bool is_observed` 필드 추가.
  - `proj_vs_mps` 는 **항상 usable 한 best-estimate 속도**: 관측 구간은 GP posterior, 미관측 구간은 `raceline vx_mps * 0.9` fallback (상대가 raceline 보다 약간 느리다고 가정).
  - 관측되면 `get_opponnent_wpnts` 가 `is_observed = True` 로 flip.
  - 3 개 viz (`gaussian_process_opp_traj`, `predictor_opponent_trajectory`, `3d_opp_prediction`) 모두 `is_observed` 기반 분기 (주황 = 미관측, 노랑 = 관측).
- **커밋**: [`ed9c19b`](https://github.com/Brojoon-10/ICRA2026_HJ/commit/ed9c19b)

### 5.2 Wrap-unsafe `argmin` 4곳 (`3d_opp_prediction.py`)

- **원본 2D 에도 있었나?** ✅ 있었음 (`opp_prediction.py:330, 335, 345, 426, 617` — 5곳 동일 패턴).
- **증상**: 상대가 wrap 경계를 지나갈 때 **상대 속도 / d 조회가 1~수 m 어긋남**. 3D 에선 `spline_z` 도 같이 어긋나서 마커가 튐.
- **원인**: `np.abs(s_arr - s).argmin()` 은 1D 거리라 wrap 무시. 예: `s = 0.1`, array 가 `[0, 0.1, ..., 99.5, 99.9]` 일 때 정상이지만, `s = 99.9` 검색하면 end 근처 맞추는데 **`s = 0.05` 라면 array 가 `[0.0, 0.1, ...]` 끝나도 99.9 쪽과 비교 안 함** (wrap 거리는 0.15 인데 raw 거리로는 99.85). 결국 경계에서 엉뚱한 인덱스.
- **수정**: 4 곳 전부 `circular_s_dist` 기반으로 교체.
  ```python
  opponent_approx_indx = int(np.argmin(
      circular_s_dist(approx_s_points_global_array, current_opponent_s, self.max_s_opponent)))
  ```
- **커밋**: [`d7dd820`](https://github.com/Brojoon-10/ICRA2026_HJ/commit/d7dd820)

### 5.3 Ego ↔ Opp 거리 체크 중복 / wrap 부분만 처리

- **원본 2D 에도 있었나?** ✅ 있었음 (2D 에서도 같은 OR 조건 사용).
- **증상**: `beginn` 플래그가 가끔 안 켜지거나 잘못 켜짐 → 예측 시작 위치 (`beginn_s`) 오차.
- **원인**: 기존 코드:
  ```python
  if not beginn and ((opp_s - ego_s) % max_s < save_dist
                     or abs(opp_s - ego_s) < save_dist):
  ```
  첫 번째 `%` 항은 wrap-safe, 두 번째 `abs` 는 wrap 무시. `or` 로 묶어서 wrap 한 쪽만 걸면 `True` 되지만 **의미가 명확하지 않고**, ego 와 opp 가 정반대(트랙 지름 거리) 에 있을 때도 두 번째 조건에 걸릴 수 있음.
- **수정**: `signed_s_dist` 로 단일 표현:
  ```python
  fwd_dist = signed_s_dist(current_ego_s, current_opponent_s, self.max_s_updated)
  if not beginn and abs(fwd_dist) < self.save_distance_front:
      ...
  ```
- **커밋**: [`d7dd820`](https://github.com/Brojoon-10/ICRA2026_HJ/commit/d7dd820)

### 5.4 Marker z 하드코딩 → 3D 트랙에서 floating / burying

- **원본 2D 에도 있었나?** ✅ 있었음 (2D 에서는 z=0 기준이라 문제 없음).
- **증상**: 경사면/고가 구간에서 예측 마커 실린더, begin/end 스피어가 **공중에 뜨거나 지면 아래 파묻힘**.
- **원인 (4군데)**:
  1. `visualize_opponent_wpnts`: `marker.pose.position.z = marker_height / 2` — world z=0 기준, 트랙 표면 무시.
  2. VD-blended / linear / trajectory-follow 예측 브랜치 (3 곳): `marker.pose.position.z = 0.1` 고정 하드코딩.
  3. `/opponent_predict/beginn`, `/opponent_predict/end` 스피어: `pose.position.z` 자체를 설정 안 함 → `Marker()` 기본값 0 → 월드 원점 높이에 둥둥 떠있음.
- **수정**:
  - `visualize_opponent_wpnts`: `OppWpnt` 메시지에 z 필드 없으므로 `spline_z(s_m)` 로 보완, `+ marker_height/2` 로 베이스 tangent.
  - 예측 브랜치: `Frenet2GlobArr.z` (C++ 서비스가 3D-aware) 응답 + `scale.z/2` 리프트.
  - begin/end 스피어: 동일하게 `surface_z + scale.z/2` (스피어 바닥이 표면에 닿음).
- **커밋**: [`d7dd820`](https://github.com/Brojoon-10/ICRA2026_HJ/commit/d7dd820)

### 5.5 `/opponent_traj_markerarray` 노드 간 높이 불일치

- **원본 2D 에도 있었나?** ✅ 있었음 (`opp_prediction.py:147` 도 `/ 10.0` 하드코딩).
- **증상**: 같은 토픽에 publish 하는 3 노드가 **높이 공식 다름** → 섞여서 크기 비교 불가.
- **원인**:
  - `gaussian_process_opp_traj.py`: `proj_vs_mps / self.max_velocity` (동적)
  - `predictor_opponent_trajectory.py`: 동일
  - `3d_opp_prediction.py`: `proj_vs_mps / 10.0` (하드코딩)
- **수정**: `3d_opp_prediction.py` 도 `max(wpnts_updated.vx_mps)` 사용하도록 통일. 3 노드 모두 같은 스케일.
- **커밋**: [`d7dd820`](https://github.com/Brojoon-10/ICRA2026_HJ/commit/d7dd820)

### 5.6 Sentinel 시 visualization "주황 + 작은 고정 높이" 1차 패치 (deprecated)

이전 단계에서 적용했던 중간 단계 수정 기록 (이후 5.1 로 재설계되어 **현재 코드에는 없음**):
- 임시 조치: `proj_vs_mps >= 50` sentinel 감지 → 주황 + height = 0.2m 고정.
- 문제: `proj_vs_mps` 필드에 sentinel 기능 섞이는 oversload, 값의 의미가 "속도" 에서 "속도 or placeholder" 로 모호해짐.
- 재설계 (5.1): 의미 분리 → `proj_vs_mps` 는 항상 속도, `is_observed` 가 관측 상태.

---

## 6. Opponent Trajectory (`3d_opponent_trajectory.py`)

### 6.1 Marker z 2D-only
- **상태**: 3D 포트 시점에 이미 `spline_z(proj_opp.s)` 로 수정됨 ([3d_opponent_trajectory.py:471](../prediction/gp_traj_predictor/src/3d_opponent_trajectory.py#L471)).
- 다른 이슈 없음. Audit 결과 2D nearest / wrap bug 해당 사항 없음.

---

## 7. Track3DValidator 공용 라이브러리

### 7.1 Bound-to-s 매핑이 `get_approx_s` 투영 기반 (구 설계)

- **원본 2D 에도 있었나?** 해당 없음 (Track3DValidator 는 3D 전용 신규 컴포넌트).
- **증상**: Stage 2 raycast locality 필터에 쓰이는 `left_seg_s_mid` / `right_seg_s_mid` 가 틀릴 수 있어서 **실제 근처에 있는 벽 세그먼트를 locality 필터가 탈락시킴** → false-negative collision miss.
- **원인**: `get_approx_s(bound_xy)` 로 wall point 마다 2D projection 을 raceline 에 꽂아서 s 계산. 헤어핀/apex/XY overlap 에서 projection 이 깨짐.
- **수정**: **bound 개수 == wpnt 개수일 때 인덱스 직접 할당** (`f110_utils/libs/f110_msgs/OppWpnt` 가 아니라 `wpnt.s_m`). `global_waypoints.json` 은 각 wpnt 에서 d-방향 projection 으로 bound 를 생성하므로 index 대 index 매핑 정확 (검증: 오차 0 m).
- **Fallback**: 개수 불일치 시 projection + `logwarn` 경고.
- **커밋**: [`ab2fea3`](https://github.com/Brojoon-10/ICRA2026_HJ/commit/ab2fea3)

---

## 8. 메시지 스키마 확장

### 8.1 `OppWpnt.msg` 에 `bool is_observed` 필드 추가

- **이유**: 5.1 의 sentinel 100 제거와 연계. 매직 숫자 없이 명시적 플래그로 관측 여부 표현.
- **영향**: `f110_msgs` 재빌드 필요. `OppWpnt()` 기본 생성 시 `is_observed = False`.
- **커밋**: [`ed9c19b`](https://github.com/Brojoon-10/ICRA2026_HJ/commit/ed9c19b)

---

## 9. Wrapping (Frenet s 경계) 공통 대응 패턴

### 9.1 공용 유틸 위치
`f110_utils/libs/track_3d_validator/src/track_3d_validator/frenet_utils.py`

```python
from track_3d_validator import (
    circular_s_dist,  # 무부호 최단거리 (0 ~ max_s/2)
    signed_s_dist,    # 부호 있는 최단거리 (-max_s/2 ~ max_s/2)
    unwrap_s,         # s 배열 단조증가로 펴기
    in_s_range,       # [s_start, s_end] 안에 있는지 (wrap-aware)
    s_forward_add,    # s + ds modulo max_s
)
```

### 9.2 사용 원칙 (이번 수정 사이클에서 정립)

| 용도 | Before (틀린 방식) | After (올바른 방식) |
|---|---|---|
| 전후 거리 판정 | `(opp_s - ego_s) % max_s < thr` OR `abs(opp_s - ego_s) < thr` | `signed_s_dist(ego_s, opp_s, max_s)` |
| 최근접 wpnt 인덱스 | `np.abs(s_arr - s).argmin()` | `int(np.argmin(circular_s_dist(s_arr, s, max_s)))` |
| s 배열 `np.diff` | 그대로 diff (wrap 지점 -max_s 점프 생성) | `unwrap_s(s_arr, max_s)` 후 diff |
| s 범위 안 체크 | `start <= s <= end` (wrap 무시) | `in_s_range(s, start, end, max_s)` |
| s 증분 | `new_s = (s + ds) % max_s` 인라인 | `s_forward_add(s, ds, max_s)` |

### 9.3 적용된 파일
- `3d_sqp_avoidance_node.py`: SQP 솔버 내부 거리 계산 (이전 세션)
- `3d_opp_prediction.py`: argmin 4곳, 전후 거리 1곳 (`d7dd820`)
- `3d_recovery_spliner_node.py`: 직접 arc-length 계산하므로 wrap 유틸 사용 안 함 (arc-length 가 이미 안전)
- `3d_static_avoidance_node.py`: 동일

---

## 10. 2D 원본에도 잠재한 버그 (포트 범위 밖이지만 인지)

3D 포트 파일만 수정했고, 2D 원본에도 아래 버그들이 그대로 남아있음:

| 파일 | 남은 버그 |
|---|---|
| `opp_prediction.py` | Wrap-unsafe argmin 5곳, sentinel `proj_vs_mps = 100` 그대로, `persistent=False` |
| `recovery_spliner_node.py` | `get_frenet` 2D 재투영 (2D 맵에선 문제 없음) |
| `spliner_node.py` (static 원본) | `_more_space` side 판단 버그, savgol 시작단 taper 누락 |
| `sqp_avoidance_node.py` (원본) | scaled_max_idx off-by-one, psi abs, side_consistency, kappa_ref 누락 |
| `gaussian_process_opp_traj.py` | sentinel 100 originate → 실제 GP 출력 파이프라인에 여전히 주입 |

2D 시스템은 현재 `sim:=true` 에서만 쓰이고 3D 트랙과 안 섞이니 당장 급하진 않지만, **같은 레이싱 스택이라 장기적으로 porting back 고려 필요**.

---

## 11. 커밋 타임라인 (관련 부분)

시간 순 (최근 → 과거):

```
d7dd820  3d_opp_prediction: wrap-safe argmin + surface-sitting markers
ed9c19b  OppWpnt: replace sentinel proj_vs_mps=100 with raceline fallback + is_observed
f38dc7d  3d_static_avoidance: blend start after savgol to kill first-sample kink
d58abac  3d_sqp_avoidance: drop 2D get_frenet re-projection after CCMA
94769d4  3d_static_avoidance: fix side choice — consider actual obstacle gaps
c372849  3d_static_avoidance: replace 2D get_frenet with arc-length s + geometric d
e7d5157  3d_recovery_spliner: replace 2D get_frenet with arc-length s + geometric d
ab2fea3  Track3DValidator: index-based bound s-mapping when 1:1 matched
0452f5d  3D dynamic avoidance + GP prediction port  (SQP 원본 버그 4종 포함)
```

---

## 12. 다음에 볼 것

- SQP 동작 검증 (실주행 + rviz 마커 + latency)
- Recovery 튐 재현 여부 (arc-length s 수정 이후)
- Prediction 예측 정확도 (wrap-safe argmin 이후)
- 2D 원본 동일 버그들 백포트 여부 결정
- `persistent=True` 적용 여부 결정 (이전 세션에서 보류)

---

## 13. 공용 체크리스트 (향후 3D 파일 포트 시)

새 파일 3D 포트 시 반드시 확인:

- [ ] `get_frenet(x, y)` / `get_approx_s(x, y)` 호출하는가? 알려진 s 가 있다면 **투영 안 하고 s 직접 사용**.
- [ ] s 배열에서 `argmin` 할 때 wrap 가능성 있는가? → `circular_s_dist`
- [ ] 전후 거리 체크 `% max_s` 만으로 처리하는가? → `signed_s_dist`
- [ ] `np.diff(s_arr)` 가 wrap 경계 포함 가능한가? → `unwrap_s` 먼저
- [ ] Marker `pose.position.z` 를 세팅하는가? → `spline_z(s)` 또는 `Frenet2GlobArr.z`
- [ ] 그 마커 생성 시 `scale.z / 2` 리프트 해서 베이스를 표면에 붙이는가?
- [ ] `proj_vs_mps`, `d`, `s` 등 필드를 **"정상값 + sentinel"** 로 혼용하는가? → 별도 bool 플래그로 분리
- [ ] Service call 을 고주파 반복하는가? → `persistent=True`
- [ ] `rospy.ServiceException` 을 shutdown race 에서 catch 하는가?
