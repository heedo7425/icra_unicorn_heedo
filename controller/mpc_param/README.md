# `controller/mpc_param`

자동 MPC 튜닝 사이드카. sim 옆에서 데몬으로 떠 있다가 lap 끝마다 로그 →
진단 → 룰 평가 → yaml 패치 → 노드 hot-reload.

> Scope: 처음엔 `upenn_mpc` 만 타겟. 안정화 후 `upenn_mpcc / upenn_mpcc_rm /
> ekf_mpc / rls_mpc / wuee_mpcc` 로 확장. 모든 산출물은 이 폴더 안에서만 생성.

## 디렉토리

```
mpc_param/
├── README.md             ← this file
├── docs/
│   └── inventory.md      ← Phase 0: yaml key → 사용처 → hot/cold 분류
├── rules/
│   └── upenn_mpc.yaml    ← 증상 → action 룰 (사용자 편집 가능)
├── sectors/
│   └── <map>.yaml        ← 트랙별 sector 분할 override (없으면 곡률 자동)
├── scripts/              ← CLI / one-shot 도구
├── daemon/               ← 데몬 본체 (subscriber + metric + rule + patcher)
└── logs/
    ├── tune_trail.csv    ← lap 단위 메트릭 + 적용 diff append-only
    └── runs/             ← 런 단위 상세 dump (선택)
```

## 워크플로

```
sim 띄움 (한 번)            controller 노드 (yaml ROS param)
        │                         │
        ▼                         ▼
   tuner_daemon  ── reload svc ──▶ 노드는 계속 주행
        │
        ├─ lap 이벤트 (/lap_data) 구독
        ├─ 토픽 in-memory 윈도우 (per-lap)
        ├─ 메트릭 계산 (sector 분할)
        ├─ rules/upenn_mpc.yaml 평가
        ├─ yaml 패치 (controller/upenn_mpc/config/upenn_mpc_*.yaml)
        ├─ rosparam load + reload svc
        └─ git auto-commit
```

자동 모드가 디폴트. Ctrl-C 로 종료.

## 안정장치

- Step bound: 한 lap 당 한 키 ≤ ±20%, 누적 ≤ ±30%
- Clamp: 룰 yaml 의 `clamp: [min, max]` 강제
- Rollback: primary metric `lap_time + λ * infeasible_count` 가 직전 대비 +X% 악화 → 직전 yaml 자동 복귀
- Cooldown: 같은 키 연속 3 lap 변경 후 1 lap 평가 강제
- Convergence: 연속 2 lap 추천이 LOW only + lap_time 변화 < 1% → FROZEN
- Crash guard: solver infeasible 폭증/sim 끊김 → 직전 yaml 복귀

## Docker / 빌드 주의

CLAUDE.md 규칙: race_stack 빌드/실행은 컨테이너 `icra2026` 안에서.
mpc_param 도구는 컨테이너 안에서 돌리는 것이 정석. 호스트에서 yaml 편집은 OK.
