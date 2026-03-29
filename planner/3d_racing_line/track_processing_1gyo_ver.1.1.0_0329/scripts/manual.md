# 파이프라인 매뉴얼

## 빠른 시작

```bash
./scripts/run_pipeline.sh --track localization_bridge_2
```

## 사전 준비

### 폴더 구조

```
bag/<트랙이름>/<트랙이름>.bag          # rosbag (odometry pose)
pcd/<트랙이름>/wall.pcd               # 벽 포인트 클라우드
data/vehicle_params/params_<차량>.yml  # 차량 파라미터
data/gg_diagrams/<gg차량>/velocity_frame/  # GG 다이어그램
```

### 의존성

- Python 3.8+
- numpy, scipy, pandas, matplotlib, open3d, casadi, pyyaml, rosbag

## 파이프라인 단계

| 단계 | 설명 | 출력 |
|------|------|------|
| 1 | bag + wall.pcd → boundary CSV 생성 | `data/raw_track_data/<트랙>_bounds_3d.csv` |
| 2 | 루프 닫기 (마지막 행 = 첫 행) | 같은 파일 |
| 3 | 3D 트랙 생성 (v2, 바운더리 수직 탄젠트) | `data/3d_track_data/<트랙>_3d.csv` |
| 4 | 트랙 스무딩 (CasADi/IPOPT NLP) | `data/smoothed_track_data/<트랙>_3d_smoothed.csv` |
| 5 | 글로벌 레이싱라인 최적화 | `data/global_racing_lines/<트랙>_3d_<차량>_timeoptimal.csv` |
| 6 | 시각화 (figure 3개) | matplotlib 창 |

## 스크립트 파라미터

`run_pipeline.sh --track <이름>`

### 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--track` | `localization_bridge_2` | 트랙 이름. bag/pcd/출력 경로가 이 이름 기준으로 결정됨 |

### 내부 설정 (스크립트 안에서 직접 수정)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `VEHICLE` | `rc_car_10th` | 차량 파라미터 파일 이름 |
| `GG_VEHICLE` | `dallaraAV21` | GG 다이어그램 출처 (rc_car_10th는 gg 없음) |

### 바운더리 검출 파라미터 (1단계)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `--min-dist` | `0.25` | 센터라인 포인트 간격 [m]. 작을수록 촘촘하지만 탄젠트 노이즈 증가 |
| `--smooth-window` | `1` | 원형 스무딩 윈도우. 1 권장, 3이면 커브에서 수축 발생 |
| `--no-plot` | 설정됨 | CSV 생성 중 인터랙티브 플롯 생략 |

### 스무딩 가중치 (4단계)

| 가중치 | 값 | 설명 |
|--------|-----|------|
| `w_c` | `1e5` | 센터라인 측정값 추종 |
| `w_l` | `1e5` | 좌측 바운더리 측정값 추종 |
| `w_r` | `1e5` | 우측 바운더리 측정값 추종 |
| `w_theta` | `1e2` | 스무딩: 헤딩 각도 |
| `w_mu` | `1e2` | 스무딩: 피치 |
| `w_phi` | `1e1` | 스무딩: 뱅킹/롤 |
| `w_nl` | `1e-2` | 스무딩: 좌측 바운더리 |
| `w_nr` | `1e-2` | 스무딩: 우측 바운더리 |

## 새 트랙 추가 방법

1. 파일 배치:
   ```
   bag/<새트랙>/<새트랙>.bag
   pcd/<새트랙>/wall.pcd
   ```

2. 실행:
   ```bash
   ./scripts/run_pipeline.sh --track <새트랙>
   ```

## 새 트랙 추가 후 결과 확인

파이프라인 완료 시 자동으로 시각화 창이 뜸 (figure 3개: 트랙맵, 프로파일, 파이프라인 비교).
