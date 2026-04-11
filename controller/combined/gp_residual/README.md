# GP Residual — Steering Correction

PP steering의 kinematic residual(모델-실차 차이)을 학습하여 보정.

## 구조
```
gp_residual/
├── bag/                  — rosbag 파일
├── data/                 — 추출된 CSV
├── models/               — 학습된 모델 (.pkl)
└── scripts/
    ├── gp_extract.py     — bag에서 토픽 추출 → CSV
    ├── gp_train.py       — CSV → GP 모델 학습
    ├── gp_eval.py        — 모델 검증 + 시각화
    └── gp_pipeline.sh    — 원클릭: bag → CSV → 학습 → 모델
```

## 사용법
```bash
cd scripts
./gp_pipeline.sh ../bag/*.bag
```

## 적용
```
rosparam set L1_controller/gp_model_path .../gp_residual/models/gp_residual_model.pkl
```
