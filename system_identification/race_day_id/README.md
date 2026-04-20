# race_day_id — 대회장 Pacejka 타이어 식별 파이프라인

> ICRA2026 RoboRacer 용, 대회장 도착 직후 **한 번의 커맨드**로 타이어·노면의
> Pacejka Magic Formula 계수 `(B, C, D, E)_{front, rear}` 와 마찰계수 μ 를
> 추정하고, 결과를 MPC 가 읽는 yaml 에 즉시 반영 + MPC 재기동까지 해주는
> race-day 서브패키지.

---

## 1. 왜 이게 필요한가

`controller/mpc/vehicle_model.py:50-173` 에서 acados OCP 는 Frenet-frame
dynamic bicycle + Pacejka 타이어 모델을 사용한다. Pacejka 의 lateral force 는

```
F_y = μ · N · D · sin( C · atan( B·α − E·(B·α − atan(B·α)) ) )
```

이고, `(B, C, D, E, μ)` 가 바닥 재질 × 타이어 온도 × 노면 상태 마다 다르다.
미리 측정할 수 없는 대회장에서 잘못된 값이 주입되면 MPC 의 예측 모델이 실제
차량 거동과 어긋나 첫 랩부터 이탈·스핀 위험이 커진다.

이 패키지는 **타이어 값을 모르는 채로 도착한 팀** 이 피트에서 5~10분 내에
이 값을 직접 측정해 MPC 에 자동 주입할 수 있게 해준다.

---

## 2. 파이프라인 개요

```
      [서비스 트리거]
             │
      ┌──────▼──────┐
      │ SAFETY_CHECK │  /car_state/odom, /vesc/sensors/imu/raw 살아 있는지
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │   WARMUP    │  (Phase 5 예정: 타이어 웜업 저속 주행)
      └──────┬──────┘
             │
  ┌──────────▼──────────┐
  │ MANEUVER_SELECT/RUN │  ramp → circle → slalom → free_lap (큐)
  └──────────┬──────────┘
             │    각 maneuver 동안 Recorder 가 70 Hz 로
             │    [t, vx, vy, ω, δ, ax] CSV 로 수집
      ┌──────▼──────┐
      │   FIT       │  on_track_sys_id/solve_pacejka 호출
      └──────┬──────┘        (LSQ, scipy.optimize.least_squares)
             │
      ┌──────▼──────┐
      │ QUALITY_GATE│  RMSE · 커버리지 · bound · μ 범위 · Df sanity 5 중 체크
      └──┬────────┬─┘
     accept     reject
        │          │
        │   ┌──────▼──────┐
        │   │   REJECT    │ 다음 maneuver 로 재시도, 전부 실패면 ABORT
        │   └─────────────┘
        │
  ┌─────▼─────┐
  │ YAML_WRITE│  stack_master/config/{car}/{car}_pacejka.yaml
  │           │  원자적(write→fsync→rename) + 타임스탬프 백업
  │           │  /tire_front/*, /tire_rear/*, /mu rosparam 도 동시 갱신
  └─────┬─────┘
        │
  ┌─────▼──────┐
  │MPC_RESPAWN │  1) /mpc/reload_params 서비스 호출 시도
  │            │  2) 실패 시 rosnode kill + roslaunch 자동 재기동
  └─────┬──────┘
        │
  ┌─────▼─────┐
  │   DONE    │  새 타이어 파라미터로 MPC 가 주행 재개
  └───────────┘
```

### 핵심 원리 3가지

**(1) Bicycle steady-state 방정식 + Pacejka LSQ fit**
`on_track_sys_id/src/helpers/solve_pacejka.py:80-93` 의 solver 를 그대로
재사용한다. 계산 과정:
```
α_f = −atan((v_y + ω·l_f)/v_x) + δ
α_r = −atan((v_y − ω·l_r)/v_x)
F_yf = m·l_r·v_x·ω / ((l_r+l_f)·cos δ)   (yaw/lateral 평형에서)
F_yr = m·l_f·v_x·ω / (l_r+l_f)
F_zf = m·(−a_x·h + g·l_r) / l_wb           (가중이동 포함)
F_zr = m·(+a_x·h + g·l_f) / l_wb
```
이렇게 측정한 `(α, F_z, F_y)` 를 `F_y = F_z·D·sin(C·atan(Bα − E(Bα − atan Bα)))`
에 비선형 LSQ (scipy.optimize.least_squares) 로 적합 → `(B, C, D, E)` 반환.
μ 는 `D_r / (m·g·l_f/l_wb)` 로부터 스케일 추출.

**(2) 다중 maneuver 로 α 공간 커버리지 확보**
- ramp-steer: 직선 일정 속도 + δ 선형 증가 → α_f 스윕
- steady-state 원: δ 고정, 속도 계단 증가 → α, F_y 정상상태
- slalom/figure-8: 양방향 α 커버 + transient dynamics
- free lap: 레이스라인 1~N 랩, 자연 스러운 α 분포

**(3) Quality gate**
LSQ 는 반드시 수렴하지만 "잘 적합됐는지" 는 별개. 5 중 gate:
- RMSE (잔차/평균 F_y) ≤ 임계
- α bin 커버리지 ≥ 최소 개수 (데이터가 α 공간을 충분히 탐색했나)
- solver bound 도달 여부 (bound 에 붙었으면 모델 부족)
- μ 가 물리적으로 가능한 범위 [0.3, 1.5]
- Df 가 기존 yaml 대비 ±40% 이내 (극단적 변화 의심)

---

## 3. 디렉토리 구조

```
race_day_id/
├── CMakeLists.txt, package.xml, setup.py
├── README.md                    ← 이 문서
├── launch/
│   ├── race_day_id.launch       실차 full pipeline
│   ├── race_day_id_sim.launch   f110-sim wiring + synthetic-GT 검증
│   └── mpc_respawn.launch       MPC subgraph 재기동 전용 (fallback 경로)
├── config/
│   ├── race_day_id_params.yaml  gate 임계값, 안전 임계, 재기동 대기
│   └── maneuver_profiles.yaml   maneuver 별 v/δ/duration 등 프로파일
├── src/race_day_id/
│   ├── orchestrator_node.py     13-state FSM, /race_day_id/start 서비스
│   ├── maneuvers/
│   │   ├── base_maneuver.py     start/step/stop 컨트랙트
│   │   ├── ramp_steer.py        δ linear ramp (id_controller Exp5 호환)
│   │   ├── steady_circle.py     δ 고정 + v 계단, ay 포화 감지 abort
│   │   ├── slalom.py            sinusoidal chirp + figure-8, lateral abort
│   │   └── free_lap.py          upenn_trainer lap-wrap 감지 재사용
│   ├── recorder.py              70 Hz ring-buffer, CSV flush
│   ├── fit_pipeline.py          Butterworth → solve_pacejka → evaluate_fit
│   ├── synthetic_gt.py          자기정합 steady-state 데이터 생성 (sim 검증용)
│   ├── yaml_writer.py           ruamel round-trip atomic write + backup
│   ├── mpc_respawn.py           service reload → kill+respawn fallback
│   └── safety_monitor.py        |ω|, |α_r|, 외부 abort latch
├── scripts/
│   └── inject_synthetic_gt.py   오프라인 solver 복구 검증 (exit 0=OK)
└── data/
    ├── backups/                 yaml 타임스탬프 백업 저장
    └── recordings/              maneuver 별 CSV 저장
```

---

## 4. 파라미터 레퍼런스

### `config/race_day_id_params.yaml`

| 키 | 기본값 | 의미 |
|---|---|---|
| `maneuver_queue` | `[ramp_steer, steady_circle, slalom, free_lap]` | 오케스트레이터가 순차 실행할 maneuver 순서. 원소 빼면 해당 maneuver 스킵 |
| `rmse_front_max` | `0.15` | 전륜 Pacejka 잔차 RMSE / mean(\|F_yf\|) 상한. 이하면 accept |
| `rmse_rear_max` | `0.15` | 후륜 동일 |
| `coverage_front_bins_min` | `5` | \|α_f\| 를 `coverage_front_bin_rad` 폭으로 bin 했을 때 채워진 bin 개수 최소 요구 |
| `coverage_rear_bins_min` | `4` | 후륜 동일 (범위 좁음) |
| `coverage_front_bin_rad` | `0.05` | 전륜 bin 폭 (rad) |
| `coverage_rear_bin_rad` | `0.02` | 후륜 bin 폭 (rad) |
| `mu_min` / `mu_max` | `0.3 / 1.5` | 추정 μ 물리 허용 범위 |
| `df_sanity_frac` | `0.40` | 새 Df 가 기존 yaml Df 대비 허용되는 상대 변화. 초과 시 의심 → REJECT |
| `allow_bound_violation` | `false` | solver bound [1,0.1,0.1,0]~[20,20,20,5] 경계에 붙은 파라미터 허용 여부 |
| `recorder_hz` | `70.0` | Recorder 샘플링 주파수 (id_analyser 와 동일) |
| `recorder_buffer_s` | `120.0` | ring-buffer 길이 (초). 이 이상 녹음하면 앞쪽부터 드롭 |
| `abort_omega_rad` | `8.0` | SafetyMonitor: \|ω\| 이상이면 즉시 abort |
| `abort_alpha_r_rad` | `0.35` | SafetyMonitor: \|α_r\| 이상이면 abort (스핀 직전) |
| `abort_pad_radius_m` | `20.0` | 차량이 시작 위치 반경을 벗어나면 abort (Phase 5에서 확장) |
| `respawn_wait_s` | `5.0` | kill+respawn 후 MPC 노드 재등장을 기다리는 시간 |
| `respawn_retry` | `1` | 재기동 실패 시 추가 시도 횟수 |
| `dry_run` | `false` | **true 면 yaml 에 쓰지 않고 MPC 재기동도 안 함.** 데이터 수집 + fit + gate 까지만 진행. **실차 초도 테스트 시 반드시 true 로 시작** |
| `synthetic_gt` | `false` | true 면 실제 측정 대신 합성 데이터로 solver 를 돌림. sim wiring 검증용 |

### `config/maneuver_profiles.yaml`

#### `ramp_steer` (직선 일정 속도 + 조향각 선형 증가)
| 키 | 기본값 | 의미 |
|---|---|---|
| `enabled` | `true` | false 면 큐에서 자동 스킵 |
| `v_target` | `3.0` (m/s) | 램프 구간 일정 속도 |
| `start_angle` | `0.10` (rad) | 초기 조향각 |
| `end_angle` | `0.40` (rad) | 최종 조향각 |
| `angle_time` | `20.0` (s) | 램프 소요 시간 |
| `drive_topic` | `/vesc/high_level/ackermann_cmd_mux/input/nav_1` | AckermannDriveStamped 발행 토픽 |

필요 공간: **약 15 × 4 m** 직선

#### `steady_circle` (일정 반경 원, 속도 계단 증가)
| 키 | 기본값 | 의미 |
|---|---|---|
| `delta_fixed` | `0.25` (rad) | 전체 maneuver 동안 고정 조향각 |
| `v_levels` | `[1.0, 1.5, 2.0, 2.5, 3.0]` | 계단 속도 목록 |
| `dwell_s` | `3.0` (s) | 각 속도 레벨 유지 시간 |
| `radius_min_m` | `2.0` | 예상 회전 반경. ay 포화 판정 시 `v²/R` 비교 기준 |
| `ay_sat_ratio` | `0.3` | 레벨 종료 시 \|ay\|_max 가 `ay_sat_ratio · v²/R` 미만이면 abort (타이어 이미 포화라 이 이상의 데이터는 무의미) |

필요 공간: **약 5 × 5 m** (반경 2 m)

#### `slalom` (좌우 사인 조향 + 주파수 스윕)
| 키 | 기본값 | 의미 |
|---|---|---|
| `amplitude_rad` | `0.20` | 사인 조향 진폭 |
| `freq_start_hz` | `0.5` | 시작 주파수 |
| `freq_end_hz` | `2.0` | 종료 주파수 (선형 chirp) |
| `sweep_s` | `10.0` | 스윕 시간 |
| `v_fixed` | `2.5` (m/s) | 일정 전진 속도 |
| `figure_eight` | `false` | true 면 chirp 대신 ±`amplitude_rad` piecewise-constant (8 자 궤적) |
| `lateral_abort_m` | `2.5` | 시작점 대비 횡 방향 이탈 허용 (frenet n 가 있으면 n, 없으면 euclidean) |

필요 공간: **slalom 20 × 3 m / figure-8 6 × 4 m**

#### `free_lap` (패시브 수집, 기존 MPC/FTG 가 주행)
| 키 | 기본값 | 의미 |
|---|---|---|
| `min_vx` | `1.0` | 이 속도 이상일 때만 lap 카운트 |
| `max_laps` | `1` | 완료에 필요한 랩 수 |
| `require_gb_track` | `true` | state_machine 이 GB_TRACK 을 퍼블리시할 때만 수집 (Phase 5 에서 gate 활성화) |

필요 공간: **레이스라인 자체** (전용 공간 불필요)

---

## 5. 런타임 토픽 / 서비스

### 서비스
| 이름 | 타입 | 설명 |
|---|---|---|
| `/race_day_id/start` | `std_srvs/Trigger` | 파이프라인 시작 (IDLE 상태일 때만 accept) |

### 토픽
| 이름 | 방향 | 타입 | 설명 |
|---|---|---|---|
| `/race_day_id/state` | Pub (latched) | `std_msgs/String` | 현재 FSM 상태 문자열 |
| `/race_day_id/metrics` | Pub (latched) | `std_msgs/String` | 이벤트별 JSON (maneuver_start/_done, quality_gate, yaml_written, mpc_respawn, safety_latch) |
| `/race_day_id/abort` | Sub | `std_msgs/Empty` | 외부에서 즉시 abort 트리거 |
| `/car_state/odom` | Sub | `nav_msgs/Odometry` | vx, vy, ω 입력 |
| `/car_state/odom_frenet` | Sub | `nav_msgs/Odometry` | s (pose.x), n (pose.y) — lap 감지, lateral abort |
| `/vesc/sensors/imu/raw` | Sub | `sensor_msgs/Imu` | ax, ay 입력 |
| `/vesc/high_level/ackermann_cmd_mux/input/nav_1` | Pub | `ackermann_msgs/AckermannDriveStamped` | maneuver 가 명령 내리는 drive 토픽 |

### 쓰여지는 파일
| 경로 | 설명 |
|---|---|
| `stack_master/config/{car}/{car}_pacejka.yaml` | **C_Pf, C_Pr, mu 키만 덮어쓰기.** I_z, m, C_acc 등 다른 키는 round-trip 으로 완전 보존 |
| `race_day_id/data/backups/{car}_pacejka.yaml.{ISO8601}.bak` | 덮어쓰기 직전 원본 스냅샷 |
| `race_day_id/data/recordings/{car}_{ISO8601}.csv` | 수집된 원시 데이터 |

---

## 6. 실차에서 굴리는 순서 — Step by step

> ⚠️ **첫 실차 실행은 반드시 `dry_run:=true` 로.** yaml 안 건드리고 MPC 재기동도
> 안 하는 모드라, maneuver 실행/데이터 수집/fit 까지만 리허설 가능.

### (0) 사전 준비 — 차량 & 트랙

체크리스트:
- [ ] VESC 전원 ON, `/vesc/sensors/imu/raw` 정상 발행 (`rostopic hz`)
- [ ] 로컬라이제이션 (GLIL 또는 대체) 기동, `/car_state/odom` 발행
- [ ] 차량 geometry rosparam 확인: `/vehicle/m`, `/vehicle/l_f`, `/vehicle/l_r`, `/vehicle/l_wb`, `/vehicle/I_z`, `/vehicle/h_cg`
- [ ] `stack_master/config/{CAR}/{CAR}_pacejka.yaml` 존재. 없으면 SIM 것 복사 후 model_name 교체
- [ ] `CAR_NAME` 환경변수 또는 `/racecar_version` rosparam 설정 (SRX1 등)

**공간 확보**:
- ramp 용 직선 ≥ 15 m
- 원 용 반경 ≥ 2 m 의 5 m × 5 m 정사각 공간
- slalom 용 20 × 3 m 또는 figure-8 용 6 × 4 m
- 주변 2 m 이내 장애물 없음

### (1) 시스템 기동 (Docker 컨테이너 내부)

```bash
docker exec -it icra2026 bash
source /opt/ros/noetic/setup.bash
source /home/hmcl/catkin_ws/devel/setup.bash
```

터미널 1 — 로컬라이제이션:
```bash
roslaunch glim_ros glil_cpu.launch
```

터미널 2 — 메인 파이프라인 (state_estimation, vesc drivers 포함):
```bash
roslaunch stack_master car_race.launch map:=<대회맵>
```
(or `base_system.launch` if 2D, or 당일 사용 launch)

### (2) 리허설: dry-run 으로 maneuver 만 테스트

터미널 3 — race_day_id:
```bash
roslaunch race_day_id race_day_id.launch racecar_version:=SRX1 dry_run:=true
```

터미널 4 — 모니터:
```bash
rostopic echo /race_day_id/state
# 별도 창
rostopic echo /race_day_id/metrics
```

트리거:
```bash
rosservice call /race_day_id/start "{}"
```

**관찰 포인트**:
- state 문자열이 IDLE → SAFETY_CHECK → WARMUP → MANEUVER_SELECT → MANEUVER_RUN → ... 순서로 진행되는가
- 각 maneuver 중 차량이 예상대로 움직이는가 (ramp 에서 조향각 서서히 증가, circle 에서 원 그림, slalom 에서 좌우)
- `|ω|` 가 `abort_omega_rad=8.0` 을 넘지 않는가
- maneuver_done 이벤트 후 FIT, QUALITY_GATE 까지 도달
- 마지막 DONE 또는 ABORT 상태 확인

**이상 징후 시 즉시 abort**:
```bash
rostopic pub -1 /race_day_id/abort std_msgs/Empty
```

### (3) 수집된 데이터 오프라인 검증

```bash
ls /home/hmcl/icra2026_ws/ICRA2026_HJ/system_identification/race_day_id/data/recordings/
# 가장 최근 CSV 확인
tail -5 <마지막 csv>
```

`/race_day_id/metrics` latched 메시지에서 quality_gate 이벤트의 `accept`, `rmse_front`, `rmse_rear`, `coverage_front_bins`, `mu` 필드를 확인. accept=false 였으면 어떤 항목에서 fail 했는지 metrics 의 값을 임계값과 비교해 원인 파악:
- rmse 가 높으면 → 노이즈/드리프트 의심. maneuver 속도 낮추거나 `recorder_hz` 확인
- coverage 부족 → `angle_time` 늘리거나 `v_levels` 많이
- mu 가 범위 밖 → geometry rosparam 잘못
- bound_violation → solver 초기값 `C_Pf_model` 이 현실과 너무 다름 → 기존 yaml 업데이트

### (4) 본 실행: yaml 자동 반영 + MPC 재기동

dry-run 통과 후 실제 반영:

```bash
# 이전 race_day_id 인스턴스 종료 후
roslaunch race_day_id race_day_id.launch racecar_version:=SRX1 dry_run:=false
rosservice call /race_day_id/start "{}"
```

DONE 도달 시 자동으로:
1. `stack_master/config/SRX1/SRX1_pacejka.yaml` 업데이트
2. 기존 MPC 노드 kill
3. `mpc_respawn.launch` 로 MPC 재기동 (새 yaml 로드)
4. `/race_day_id/metrics` 에 `yaml_written`, `mpc_respawn` 이벤트 발행

확인:
```bash
# yaml 반영 확인
cat /home/hmcl/icra2026_ws/ICRA2026_HJ/stack_master/config/SRX1/SRX1_pacejka.yaml
# rosparam 반영 확인
rosparam get /tire_front/B
rosparam get /tire_front/D
rosparam get /mu
# MPC 재기동 확인
rosnode list | grep mpc
rostopic hz /mpc/trajectory     # or /mpc/prediction
```

### (5) 롤백 (문제 발생 시)

백업에서 복원:
```bash
ls system_identification/race_day_id/data/backups/
# 원하는 시점 백업 선택
cp system_identification/race_day_id/data/backups/SRX1_pacejka.yaml.20260420T153000.bak \
   stack_master/config/SRX1/SRX1_pacejka.yaml
# MPC 재기동
rosnode kill /mpc_controller
roslaunch controller mpc_standalone.launch racecar_version:=SRX1
```

### (6) 개별 maneuver 스킵

특정 maneuver 만 돌리고 싶으면 `race_day_id_params.yaml` 의 `maneuver_queue` 수정:
```yaml
maneuver_queue: [ramp_steer]                    # 램프만
maneuver_queue: [steady_circle, slalom]         # 둘만
```
또는 `maneuver_profiles.yaml` 에서 해당 maneuver 의 `enabled: false`.

---

## 7. 시뮬 선검증

f110-simulator 는 마찰이 scalar 라 Pacejka 계수 실제 복원은 불가. 그래도
**배선 (orchestrator 전이, 서비스, yaml write, rosparam 갱신)** 과
**solver 정확도 (synthetic-GT 모드)** 는 검증 가능.

### (a) solver 오프라인 검증
```bash
cd system_identification/race_day_id
python3 scripts/inject_synthetic_gt.py --tol 0.10 --noise 0.01
```
Exit 0 이면 solver 가 주어진 GT 파라미터를 1% 노이즈 하에 10% 이내로 복원함을
확인. 이 검증은 ROS 없이도 돌아감.

### (b) sim 풀 파이프라인
```bash
# f110 sim 기동 (별도 터미널)
roslaunch stack_master base_system.launch sim:=true map:=f

# sim 용 race_day_id (synthetic_gt 모드)
roslaunch race_day_id race_day_id_sim.launch
rosservice call /race_day_id/start "{}"
```
synthetic_gt=true 이므로 실제 maneuver 는 명령 발행해도 수집된 데이터를 사용하지
않고 합성 데이터로 fit. **MPC 재기동 경로 및 yaml write 경로 테스트용.**

---

## 8. 안전/한계

- **SafetyMonitor 는 latched**: 한 번 트립되면 orchestrator 는 ABORT 로 들어가고
  수동 재기동 필요. 다시 `roslaunch race_day_id race_day_id.launch` 할 것.
- **ay 포화 감지 (circle)** 는 `ay_sat_ratio·v²/R` 미달 시만 트립. 반경 R 이
  실제보다 많이 작으면 false-positive 가능 → `radius_min_m` 을 보수적으로.
- **Pacejka E 계수는 ill-conditioned** (synthetic 검증에서도 노이즈 1% 에 E 오차
  ~5%). 다른 계수보다 편차 큼이 정상.
- **μ 는 `D_r / (m·g·l_f/l_wb)` 로 유도 — D 에 강하게 묶여 있음**. 독립적
  추정이 필요하면 `ekf_mpc` 의 EKF 기반 online μ 를 병행.
- **현재 scope 외**: 종방향 타이어 곡선 (driving/braking), 종-횡 coupled
  friction circle — `id_analyser/analyse_tires.py` 후속 워크플로에 위임.

---

## 9. 내부 구조 요약 (유지보수용)

### FSM 상태 (`orchestrator_node.py:STATES`)
```
IDLE → SAFETY_CHECK → WARMUP → MANEUVER_SELECT → MANEUVER_RUN
     → RECORD_FLUSH → FIT → QUALITY_GATE
     → { YAML_WRITE → MPC_RESPAWN → DONE  |  REJECT → NEXT_MANEUVER  |  ABORT }
```

### 의존성 (모두 import, 수정 금지)
| 대상 | 용도 |
|---|---|
| `system_identification/on_track_sys_id/src/helpers/solve_pacejka.py` | LSQ fit + bicycle 방정식 |
| `system_identification/id_controller/` (Exp5) | ramp-steer 원리 참조 |
| `system_identification/id_analyser/analyse_tires.py` | Butterworth LPF 헬퍼 |
| `controller/upenn_mpc/scripts/upenn_trainer.py:218-233` | lap-wrap 감지 로직 |

### 추가 patch 지점 (optional)
- `controller/mpc/vehicle_model.py` 에 `/mpc/reload_params` service 훅 추가하면
  kill+respawn 없이 즉시 새 값 반영 가능. 현재는 service 없으면 자동 fallback.

---

## 10. 트러블슈팅

| 증상 | 확인 / 조치 |
|---|---|
| `/race_day_id/start` 호출했는데 state 가 계속 IDLE | `auto_start:=true` 안 준 상태, `rosservice list` 로 서비스 존재 확인 |
| SAFETY_CHECK 에서 진행 안 됨 | `/car_state/odom` 퍼블리시 확인, `rostopic hz /car_state/odom` |
| ramp 중 차량이 왼쪽/오른쪽으로만 가는데 반대 못 돔 | start_angle 부호 확인. `-0.4 → +0.4` 로 바꾸면 양방향 |
| circle 즉시 abort (ay_sat_abort) | `radius_min_m` 를 실제 반경보다 작게 설정한 경우. 늘리거나 `ay_sat_ratio` 낮추기 |
| quality_gate 항상 REJECT: coverage 부족 | `v_levels`/`angle_time` 확장해 α 공간 탐색량 증가 |
| yaml 이 업데이트 안 됨 | `dry_run:=true` 로 돌린 건 아닌지, `data/backups/` 에 .bak 생성됐는지 |
| MPC 재기동 후 `/tire_front/B` 가 새 값 아님 | `mpc_respawn.launch` 의 `rosparam load` 가 yaml 을 다시 읽도록 되어 있음 — 이 경로가 작동하는지 log 확인 (`~/.ros/race_day_id/mpc_respawn_*.log`) |
| synthetic_gt 검증에서 모든 계수 오차 0 이지만 실차에서 fit 실패 | 기대 동작. 실차는 노이즈·모델 오차 있어 gate 가 더 까다롭게 동작. 데이터 길이/커버리지 확인 |

---

## 11. 검증 상태 (2026-04-20 기준)

| 단계 | 상태 | 방법 |
|---|---|---|
| 패키지 빌드 | ✅ catkin build race_day_id 통과 | Docker icra2026 컨테이너 |
| Python import | ✅ 모든 모듈 resolvable | `python3 -c "import race_day_id..."` |
| solver 복구 (0% 노이즈) | ✅ 전 계수 오차 0.0 | `inject_synthetic_gt.py --noise 0.0` |
| solver 복구 (1% 노이즈) | ✅ 전 계수 <1% (E 만 ~5%) | `inject_synthetic_gt.py --noise 0.01` |
| yaml writer 원자성 + 보존 | ✅ I_z/C_acc 등 타 키 보존 확인 | 임시 디렉토리 writer 테스트 |
| MPC 재기동 경로 | ✅ 구현, ⚠ 실환경 검증 필요 | roscore + sim 환경에서 확인 예정 |
| 실차 종단 | ⏳ 하드웨어 테스트 대기 | 이 README §6 절차 |

---

## 12. 참고 문헌

- Pacejka, H. B., *Tire and Vehicle Dynamics* (3rd ed.), Butterworth-Heinemann, 2012 — Magic Formula 원전
- Althoff, M. et al., "CommonRoad: Composable Benchmarks for Motion Planning on Roads", IV 2017 — bicycle model 관례
- Nagy, T. et al., "Ensemble Gaussian Processes for Adaptive Autonomous Driving on Multi-friction Surfaces", IFAC WC 2023 — `controller/upenn_mpc` 의 residual 모델 (보완적 접근)

---

*generated as part of ICRA2026 race-day toolchain. 관련 plan:
`~/.claude/plans/jiggly-gliding-lobster.md`.*
