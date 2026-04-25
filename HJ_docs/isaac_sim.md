# Isaac Sim 연동을 위해 race_stack 에 필요한 것

icra2026 컨테이너(ROS1 Noetic) 가 issac_icra 컨테이너(Isaac Sim 4.5, Humble 기반)
와 통신하려면 아래 3가지가 race_stack 쪽에 있어야 합니다. Isaac Humble ↔ Noetic
직접 브리지가 불가해서 **rosbridge_suite 의 WebSocket(JSON)** 으로 우회하는 설계.

두 컨테이너 모두 `--network=host` 전제.

---

## 1. apt 패키지: `ros-noetic-rosbridge-suite`

WebSocket 서버(`:9090`) 제공. Isaac 쪽 `roslibpy` 가 여기로 붙어 토픽을 pub/sub.

- 적용 위치: `.docker_utils/Dockerfile.nuc` 에 추가되어 있음
- 현재 실행 중인 `icra2026` 컨테이너엔 수동 apt 로 설치돼 있어 지금은 동작하지만,
  컨테이너 재생성 시 사라짐. 이미지 빌드로 영구 반영 필요:
  ```bash
  cd /home/hmcl/icra2026_ws/ICRA2026_HJ/.docker_utils
  docker build -f Dockerfile.nuc -t race_stack_nuc:latest ..
  ```

---

## 2. 전용 런치: `controller/upenn_mpc/launch/upenn_mpc_isaac.launch`

`rosbridge_websocket.launch port:=9090` 을 include 하여 WS 서버를 띄우고,
Isaac 과 토픽 계약을 맞춘 채로 MPC 파이프라인을 기동.

핵심 포함:
- `rosbridge_server/rosbridge_websocket.launch` (port 9090)
- `base_system.launch sim:=false` (Gazebo 없이 TF/map 만)
- `upenn_mpc_node` + `reference_builder`
- (옵션) RViz

실행:
```bash
docker exec icra2026 bash -c \
  "source /home/hmcl/catkin_ws/devel/setup.bash && \
   CAR_NAME=SIM roslaunch controller upenn_mpc_isaac.launch map:=f"
```

---

## 3. 보조 스크립트: `controller/upenn_mpc/scripts/isaac_odom_tf.py`

Isaac 이 내보내는 odom 토픽(`/car_state/odom`) 을 받아 `map → base_link`
TF 로 broadcast 하는 어댑터. 실차에선 SE(GLIL/cartographer) 가 TF 를
대신 쏘지만, Isaac 모드에선 GT odom 을 바로 TF 에 얹어 rviz/프레네 변환에
사용.

---

## 토픽 계약 (참고)

rosbridge 를 경유하지만 **토픽 이름은 실차와 동일** → 알고리즘 코드 무변경.

| 방향 | 토픽 | 타입 | 실측 |
|---|---|---|---|
| race_stack → Isaac | `/vesc/low_level/ackermann_cmd_mux/output` | `ackermann_msgs/AckermannDriveStamped` | 50 Hz |
| Isaac → race_stack | `/car_state/odom` | `nav_msgs/Odometry` | 17 Hz |
| Isaac → race_stack | `/livox/imu` | `sensor_msgs/Imu` | 8 Hz |
| Isaac → race_stack | `/livox/lidar` | `sensor_msgs/PointCloud2` | 1 Hz (JSON 병목) |

---

## 실차 전환 시

위 3 가지는 **Isaac 전용** 이라 실차에선 전부 불필요:
1. rosbridge_suite 는 기동 안 함
2. `upenn_mpc_isaac.launch` 대신 `upenn_mpc_real_nostate.launch` 등 실차용 런치
3. `isaac_odom_tf.py` 는 미사용 (SE 가 TF 발행)

알고리즘 레이어(`upenn_mpc_node.py` 등)는 동일 토픽 계약으로 그대로 동작.

상세: `controller/upenn_mpc/ISAAC_SIM.md`
