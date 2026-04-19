# `controller/rls_mpc`

2D f1tenth sim 에서 **peak-tracking RLS** 기반 μ 추정 실험용 **독립 패키지**. 이 패키지는 **이력 보존용** — 더 정확한 추정은 `controller/ekf_mpc` 참조.

`controller/mpc`, `controller/mpc_only`, `controller/ekf_mpc` 와 완전 격리 (별도 네임스페이스 `/rls_mpc/*`, 별도 codegen dir `/tmp/rls_mpc_c_generated`).

---

## 1. 요약

μ 를 scalar 로 추정하는 가장 단순한 접근. `ay_peak/g` 를 직접 관측으로 사용.

**결과**: 시뮬레이션 상 모든 patch 에서 유사한 ay_peak (2-3 m/s²) 를 측정해 **patch 구분 실패**. ay-only + saturation-가정 RLS 의 본질적 한계.

**후속**: `controller/ekf_mpc` 에서 Pacejka 기반 EKF + longitudinal channel + s-memory prior 로 개선됨 — patch 구분 가능.

---

## 2. 파일 구조

```
controller/rls_mpc/
├── rls_mpc_node.py                  # MPC 노드 (런타임 μ 수신)
├── config/
│   ├── rls_mpc_srx1.yaml
│   └── mu_patches_f.yaml            # s-range 기반 μ 패치
├── launch/rls_mpc_sim.launch
├── rviz/rls_mpc.rviz
└── scripts/
    ├── mu_patch_publisher.py        # /mu_ground_truth + rviz 마커
    ├── mu_estimator_rls.py          # peak-tracking RLS
    ├── mu_estimator_gp.py           # GP stub (MA fallback)
    ├── mu_applier.py                # cmd scaling (가상 마찰)
    ├── mu_hud.py                    # rviz gt/est 비교
    └── mu_toggle_gui.py             # tkinter 대시보드
```

---

## 3. 실행

```bash
CAR_NAME=SIM roslaunch controller rls_mpc_sim.launch \
    map:=f mu_source:=<static|ground_truth|rls|gp>
```

`mu_source`:
- `static`: `mu_default=0.85` 고정
- `ground_truth`: `topic_tools/relay` 로 `/mu_ground_truth → /rls_mpc/mu_estimate` 직결 (sanity check)
- `rls`: peak-ay 기반 RLS
- `gp`: MA fallback stub

---

## 4. 결과 (참고)

| Patch | gt | RLS est 평균 | \|err\| |
|---|---|---|---|
| icy_long | 0.40 | ~0.80 | 0.40 |
| grippy_top | 1.20 | ~1.09 | 0.14 |
| grippy_corner | 1.20 | ~0.89 | 0.32 |
| icy_hairpin | 0.40 | ~0.42 | 0.10 |
| normal | 0.85 | ~0.82 | 0.19 |

**한계**: 직선 구간 (icy_long) 에서 ay 낮아 관측 불가 → prior 유지. 단일 스칼라 센서만으로는 multi-surface 구분 본질적 어려움.

---

## 5. 주요 토픽

| 토픽 | 역할 |
|---|---|
| `/rls_mpc/mu_estimate` | RLS 출력 |
| `/rls_mpc/mu_used` | MPC 가 실제 OCP 에 주입한 μ |
| `/rls_mpc/rls_ay_peak` | 최근 peak \|ay\| |
| `/mu_ground_truth` | patch yaml 기반 gt |
| `/mu_hud/markers`, `/mu_patches/markers` | rviz |

---

## 6. 다음 단계

`controller/ekf_mpc` 참조 — Pacejka-based EKF + longitudinal channel + s-memory prior 로 patch 식별 성공.
