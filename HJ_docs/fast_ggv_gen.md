# Fast GGV Generator

## 개요

3D GB Optimizer의 GGV(가감속 다이어그램) 생성 파이프라인을 **원본 대비 ~6배 고속화**했다.  
타이어 파라미터 튜닝 → GGV 반영 → 시각화까지의 피드백 루프를 단축하여, 향후 rqt 기반 실시간 튜닝 파이프라인의 기반을 마련했다.

| 모드 | V | g | alpha | NLP 호출 수 | 시간 | 용도 |
|------|---|---|-------|-----------|------|------|
| 원본 (`gg_diagram_generation/`) | 15 | 9 | 125 | 16,875 | **~2분** | 최종 생성 |
| `--fast` (현재) | 5 | 3 | 20 | 300 | **~2초** | 튜닝 방향 확인 |
| `--fast` (이전) | 7 | 5 | 40 | 1,400 | **~5초** | 튜닝 방향 확인 |
| `--full` | 15 | 9 | 125 | 16,875 | **~22초** | 정밀 생성 |

- 범위는 fast/full 동일: V=[1.5, 12.0] m/s, g=[1.0, 20.0] m/s², alpha=[-π/2, π/2]
- fast는 범위 내 등간격(linspace)으로 격자만 성김, 나머지는 보간으로 채움

---

## 무엇을 바꿨나

### 1. Parametric NLP — solver를 한 번만 빌드

**원본 문제**: 매 (V, g, alpha) 조합마다 CasADi `nlpsol()`을 새로 생성 → 16,875번의 symbolic compilation.

**개선**: V, g_force, alpha를 CasADi **parameter (`p`)**로 선언하여, solver를 **1번만 빌드**하고 parameter 값만 바꿔서 재호출.

```python
# 원본: alpha 루프 안에서 매번 solver 생성
for alpha in alpha_list:
    nlp = {"x": x_n, "f": f, "g": vertcat(*g)}  # 매번 새 NLP
    solver = nlpsol("solver", "ipopt", nlp, opts)  # 매번 빌드 (~ms 단위 오버헤드)
    solver(x0=x0_n, ...)

# 개선: solver 1번 빌드, parameter만 변경
p = vertcat(p_V, p_g, p_alpha)
nlp = {"x": x_n, "f": f, "g": vertcat(*g_con), "p": p}
solver = nlpsol("solver", "ipopt", nlp, opts)  # 1번만
for alpha in alpha_list:
    solver(x0=x0_n, ..., p=vertcat(V, g_force, alpha))  # 재호출만
```

**왜 결과가 동일한가**: NLP의 수학적 구조(목적함수, 제약조건, 변수)가 완전히 동일하다. V, g, alpha가 symbolic graph에서 "상수 노드"였던 것이 "파라미터 노드"로 바뀌었을 뿐, IPOPT가 푸는 최적화 문제 자체는 변하지 않는다.

### 2. Multiprocessing fork 병렬화

**원본**: `joblib.Parallel`로 V별 병렬화 (pickle 기반).

**개선**: `multiprocessing.Pool` + `fork` start method 사용. fork는 부모 프로세스 메모리를 그대로 상속하므로, CasADi solver 객체를 pickle 없이 자식 프로세스에서 바로 사용 가능.

```python
import multiprocessing as mp
mp.set_start_method('fork', force=True)
pool = mp.Pool(processes=min(num_cores, V_N))
processed_list = pool.map(calc_rho_for_V, V_list)
```

**왜 결과가 동일한가**: 각 V에서의 계산은 완전히 독립적이다. 원본도 V별 병렬화를 하고 있었으며, 병렬화 단위와 방식만 달라졌을 뿐 각 NLP 문제의 입출력은 동일하다.

### 3. calc_max_slip_map도 parametric solver화

slip map 계산(200개 하중별 최대 slip 탐색)도 동일한 parametric solver 패턴을 적용. 하중 N을 parameter로 선언하여 solver를 2번(Fx용, Fy용)만 빌드.

### 4. 해상도 조절 (`--fast` / `--full`)

| 파라미터 | `--fast` (현재) | `--full` |
|---------|---------------|---------|
| V 격자 | 5 | 15 |
| g 격자 | 3 | 9 |
| alpha 격자 | 20 | 125 |
| **총 NLP** | **300** | **16,875** |

`--fast` 출력은 보간으로 채워지므로 diamond fitting과 시각화에는 충분하다. 최종 최적화기 투입 시에는 `--full`로 생성.

### fast vs full 디테일 차이 확인

g=9.81 (평지) 조건에서의 diamond plot을 비교한 결과, fast와 full의 **시각적 차이는 거의 없었다.** 이는 diamond fitting이 rho의 전체적인 envelope을 4개 파라미터(gg_exponent, ax_min, ax_max, ay_max)로 요약하기 때문이며, 입력 rho의 격자가 조밀하든 성기든 envelope의 형태는 크게 변하지 않는다.

**차이가 나타나는 경우:**
- **g_tilde가 격자 사이에 있을 때**: fast(5개 g격자)는 보간 오차가 full(9개)보다 클 수 있다. 특히 3D 맵의 dip/bump에서 g_tilde가 극단값(1~20 m/s²)을 가질 때 영향이 있을 수 있다.
- **고속 영역의 V 보간**: fast(7개 V격자)는 V=8~12 m/s 구간에서 보간 해상도가 낮아 파워 리밋 경계가 부드럽지 않을 수 있다.
- **alpha 방향 해상도**: fast(40개)는 GG envelope의 코너(순수 가속 ↔ 선회 전환 영역)에서 rho가 약간 과대평가될 수 있다.

**TODO**: fast와 full의 실제 diamond 파라미터(ax_min, ax_max, ay_max, gg_exponent) 수치 차이를 정량적으로 비교 확인 필요. 특히 극단 g_tilde 조건(dip/bump)에서의 보간 오차가 최적화기 결과에 미치는 영향을 검증할 것.

**결론**: 튜닝 방향 확인(파라미터 A vs B 어느 쪽이 나은가)에는 fast로 충분할 것으로 예상되나, 위 검증이 선행되어야 한다. 최종 최적화기 투입용 GGV는 반드시 `--full`로 생성할 것.

---

## 출력 경로 (기존 데이터 보호)

```
기존 (읽기만):  global_line/data/vehicle_params/params_rc_car_10th.yml
              global_line/data/gg_diagrams/rc_car_10th/  ← 절대 건드리지 않음

fast_ggv_gen (쓰기): fast_ggv_gen/output/rc_car_10th/
                     ├── vehicle_frame/ (v_list, g_list, alpha_list, rho, diamond .npy)
                     ├── velocity_frame/ (동일 구조)
                     ├── gg_diamond.png
                     └── gg_polar.png
```

---

## 사용법

### 컨테이너 내부에서 (직접 실행)
```bash
cd /home/unicorn/catkin_ws/src/race_stack/planner/3d_gb_optimizer/fast_ggv_gen

# fast (~2초)
./run_on_container.sh rc_car_10th

# full (~22초)
./run_on_container.sh rc_car_10th --full
```

### 호스트에서 (docker exec 자동 호출)
```bash
cd planner/3d_gb_optimizer/fast_ggv_gen

# fast (~2초)
./run.sh rc_car_10th

# full (~22초)
./run.sh rc_car_10th --full
```

---

## 향후 계획: rqt 실시간 튜닝 파이프라인

현재 구조는 다음과 같은 실시간 튜닝 워크플로우의 기반이 된다:

```
[rqt 슬라이더 GUI]
  lambda_mu_x/y, P_max, p_Dx_1 등 조절
       │
       ├── rosparam 업데이트 (YAML I/O 불필요)
       │
       ├── solver 재빌드 (0.01초) + GGV solve (~5초)
       │
       ├── diamond fitting + GG envelope 시각화
       │
       └── 파라미터 확정 → --full로 최종 GGV 생성
```

- solver 빌드가 0.01초이므로, 파라미터 변경 시 solver 재생성 비용은 무시할 수 있다
- rosparam에서 직접 `tire_params` dict를 구성하면 YAML 파일 수정 없이 파라미터 반영 가능
- HSL(ma27) 선형 solver 설치 시 full 해상도도 ~7~10초 내로 단축 예상 (신청 완료, 대기 중)

### TODO: FWBW script 연동

fast/full GGV의 실질적 차이를 확인하려면 diamond plot만으로는 부족하다. 생성된 GGV를 **실제 경로 위에 FWBW(Forward-Backward) 속도 프로파일로 입혀서** 비교해야 해상도 차이가 체감된다. fast GGV → FWBW → 속도 프로파일 시각화까지의 파이프라인을 연결하는 스크립트가 필요하다.

### TODO: rqt 기반 통합

현재 run.sh 기반 CLI 워크플로우를 rqt GUI로 통합 시도:
- rqt dynamic_reconfigure 또는 custom rqt plugin으로 타이어 파라미터 슬라이더 구성
- 슬라이더 변경 → rosparam 업데이트 → fast GGV 재생성 → GG envelope + 속도 프로파일 실시간 갱신
- 최종 확정 시 --full로 전환하여 정밀 GGV 생성

---

## 파일 구조

```
planner/3d_gb_optimizer/fast_ggv_gen/
├── fast_gen_gg_diagrams.py      # 핵심: parametric NLP + fork 병렬
├── calc_max_slip_map.py         # parametric slip map 계산
├── gen_diamond_representation.py  # diamond fitting (원본과 동일 로직)
├── plot_gg_diagrams.py          # 시각화 + PNG 저장
├── run_on_container.sh          # 컨테이너 내부 실행 스크립트
├── run.sh                       # 호스트 실행 (docker exec wrapper)
└── output/                      # 생성 결과 (기존 데이터와 분리)
```
