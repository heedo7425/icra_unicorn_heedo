# TODO - HJ (3D Pipeline Migration)

## 즉시 (시스템이 돌아가게 하는 것)

- [ ] **1. car_race.launch `/car_state/odom` 이중 발행 해결**
  - 현상: carstate_node(line 188)와 odom_relay(line 35-36)가 동시에 `/car_state/odom` publish
  - carstate_node 분석 결과 (car_race.launch 인자 기준):
    - odom_topic=/glim_ros/odom, debug=False
    - line 149에서 carstate_odom_msg = self.ekf_odom → TF pose 세팅(line 143-145)이 덮어써짐
    - 실질적으로 `/glim_ros/odom` 원본에서 twist.linear.y만 pose 미분+MA로 교체
    - 추가로 /car_state/pose(TF기반), /car_state/pitch(IMU), /car_state/odom_diff 발행
  - 방안: 둘 중 하나를 선택해야 함 (GLIL vy 정확도 확인 필요)
- [ ] **2. d_right 음수 수정** — convert_to_global_waypoints.py에서 abs(w_tr_right_m) 처리 후 JSON 재생성
- [ ] **3. psi_rad normalize** — [-pi, pi] 범위로 wrap

## 단기 (안정적 주행)

- [ ] **4. Controller Python FrenetConverter 점검** — 2D converter가 실제로 뭘 하는지 확인, odom_frenet 토픽만으로 충분한지 판단
- [ ] **5. 경사면 속도 보상** — pitch 기반 속도 커맨드 보정 (오르막 +, 내리막 -)

## 중기 (정밀도 향상)

- [ ] **6. Glob2Frenet.srv에 z 추가** — frenet_conversion_server에서 3D 서비스 지원
- [ ] **7. CalcFrenetPoint slope-aware arc-length** — 경사면에서 실제 호 길이 반영
- [ ] **8. 시각화 마커 z 반영** — sector_server, ot_sector_server, lap_analyser 등 z=0 하드코딩 수정

## 참고: 현재 3D 파이프라인 상태

| 단계 | 상태 | 비고 |
|------|------|------|
| Localization (GLIL) | OK | x,y,z,roll,pitch,yaw 출력 |
| Waypoint (Wpnt.msg) | OK | z_m 필드 있음 |
| Frenet C++ (closest point) | OK | 3D Euclidean distance |
| Frenet C++ (s,d 계산) | 2D | XY 평면 투영만 |
| Frenet Conversion Server | 2D | srv에 z 없음 |
| Sector Servers | 무관 | 인덱스 기반 |
| Velocity Scaler | 무관 | 인덱스 기반 |
| State Machine | 무관 | s,d만 사용 |
| Controller (L1/PP) | 2D | Python FrenetConverter z없음, pitch 미사용 |
