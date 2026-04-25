# Isaac Sim integration — race_stack side

ICRA 2026 race stack 이 **Isaac Sim 4.5** 시뮬레이터와 연동되는 구조의 race_stack
쪽 문서. Isaac Sim 쪽 소스는 별도 레포:
[heedo7425/isaac_2026icra](https://github.com/heedo7425/isaac_2026icra).

---

## 전체 파이프라인

```
┌────────────────────────────────────┐            ┌────────────────────────────────────┐
│  icra2026 컨테이너 (Ubuntu 20.04)   │            │  issac_icra 컨테이너 (Ubuntu 22.04) │
│  ROS 1 Noetic                      │            │  Isaac Sim 4.5.0                   │
├────────────────────────────────────┤            ├────────────────────────────────────┤
│  • roscore                         │            │  run_mpc_sim.py                    │
│  • rosbridge_server (WS :9090)     │◄──── WS ──►│    ├─ SRX8-GTE Articulation        │
│  • upenn_mpc_node (이 레포)         │   (JSON)   │    ├─ Pacejka per-step 주입         │
│  • reference_builder               │            │    ├─ Livox Mid-360 raycaster       │
│  • RViz + GLIL (옵션)               │            │    └─ kd_0420_v1 3D 트랙            │
│                                    │            │  roslibpy client (ros2_io.py)      │
└────────────────────────────────────┘            └────────────────────────────────────┘
                                  │  호스트 네트워크 공유
```

두 컨테이너 간 통신은 **rosbridge_suite** 의 WebSocket (JSON) 을 사용합니다.
Isaac Sim 이 Ubuntu 22.04 + Humble 베이스라 공식 `ros1_bridge` (Noetic+Humble)
이미지가 없고 source build 가 매우 번거로워 rosbridge 로 우회했습니다.

---

## 토픽 계약

race_stack 은 **기존 토픽 이름을 그대로 재사용** 합니다. Gazebo 플러그인
대신 Isaac Sim 이 같은 이름으로 publish/subscribe 합니다.

| 방향 | 토픽 | 메시지 타입 | 실측 주파수 |
|---|---|---|---|
| race_stack → Isaac | `/vesc/low_level/ackermann_cmd_mux/output` | `ackermann_msgs/AckermannDriveStamped` | 50 Hz (MPC) |
| Isaac → race_stack | `/car_state/odom` | `nav_msgs/Odometry` | 17 Hz |
| Isaac → race_stack | `/livox/imu` | `sensor_msgs/Imu` | 8 Hz |
| Isaac → race_stack | `/livox/lidar` | `sensor_msgs/PointCloud2` | 1 Hz |

> LiDAR 가 실스펙 (10 Hz) 보다 현저히 낮은 것은 rosbridge 의 JSON + base64
> 디시리얼라이즈 CPU 병목 때문입니다. MPC 자체는 GT odom 만 쓰면 되므로
> 현재 설정으로 검증 가능합니다.

---

## 실행 순서

### Prereq
- 호스트에 `issac_icra` 컨테이너 실행 중 (`isaac_2026icra` 레포 참고)
- 호스트에 `icra2026` 컨테이너 실행 중 (이 race_stack 이미지)
- 두 컨테이너 모두 `--network=host`

### Step 1 — race_stack 쪽 MPC + rosbridge 기동

```bash
docker exec icra2026 bash -c \
  "source /home/hmcl/catkin_ws/devel/setup.bash && \
   CAR_NAME=SIM roslaunch controller upenn_mpc_isaac.launch map:=f"
```

이 런치는:
1. `rosbridge_websocket.launch port:=9090` 을 include → Isaac Sim 이 roslibpy
   로 붙을 WebSocket 서버 기동
2. `/shared_vehicle/vehicle_srx1.yaml` 을 GLOBAL rosparam 으로 로드
3. `stack_master/launch/base_system.launch` 를 `sim:=false` 로 include
   (Gazebo 기동 없이 TF / map / pcd 만 설정)
4. `upenn_mpc_node` 실행 (GP 비활성)
5. `reference_builder` 실행
6. RViz 기동

### Step 2 — Isaac Sim 기동 (별도 셸)

```bash
docker exec -e DISPLAY=$DISPLAY -d issac_icra bash -c \
  "cd /workspace/usd && /isaac-sim/python.sh \
   /workspace/scripts/run_mpc_sim.py --speed 1.5 > /workspace/logs/sim.log 2>&1"
```

차량이 spawn 되고 roslibpy 가 `ws://localhost:9090` 로 자동 연결합니다.

### Step 3 — 동작 검증

race_stack 쪽에서:
```bash
# Isaac 에서 오는 odom
rostopic hz /car_state/odom     # 15-20 Hz
# MPC 가 publish 중
rostopic hz /vesc/low_level/ackermann_cmd_mux/output   # ~50 Hz
# 차량 위치 실시간
rostopic echo -n 1 /car_state/odom
```

---

## 추가된 파일 (race_stack 측)

| 파일 | 역할 |
|---|---|
| `.docker_utils/Dockerfile.nuc` | `ros-noetic-rosbridge-suite` apt 설치 추가 |
| `controller/upenn_mpc/launch/upenn_mpc_isaac.launch` | Isaac Sim 전용 런치 (rosbridge + MPC) |

기존 알고리즘 코드 (`upenn_mpc_node.py`, `reference_builder.py`,
`frenet_conversion_node.py` 등) 는 **무변경** 입니다. Gazebo 시절과 동일한
토픽 계약이라 제어/추정 로직이 그대로 작동합니다.

---

## 이미지 리빌드 (영구 반영)

현재 실행 중인 `icra2026` 컨테이너엔 `rosbridge_suite` 가 수동 apt 로 들어가
있어 당장 사용에는 지장 없지만, 컨테이너 재생성 시 사라집니다. 이미지로
영구화 하려면:

```bash
cd /home/hmcl/icra2026_ws/ICRA2026_HJ/.docker_utils
docker build -f Dockerfile.nuc -t race_stack_nuc:latest ..
# → 이후 컨테이너 재생성 시 rosbridge_suite 자동 포함
```

---

## 실차 전환

Isaac Sim 은 실차와 동일한 토픽 이름으로 odom/imu/lidar 를 내보내고
Ackermann cmd 를 받으므로, 실차 배포 시:

1. rosbridge_suite + Isaac Sim 정지
2. 실차의 ROS1 센서 드라이버 (VESC, Livox, IMU, GLIL SLAM) 를 기동
3. `upenn_mpc_isaac.launch` 대신 기존 실차용 launch 사용

MPC/GP 레이어는 코드 변경 없이 그대로 동작합니다.

---

## 알려진 제약

- **LiDAR 주파수 1 Hz 상한**: rosbridge JSON + base64 디시리얼라이즈 비용.
  장애물 회피용으로는 한계. SLAM 은 Isaac GT odom 으로 대체 가능.
- **IMU 8 Hz**: 실차 200 Hz 보다 훨씬 느림. IMU 의존도 낮은 컨트롤러에는
  영향 작음.
- **Isaac Sim Pacejka 없음**: `ros2_io.py` + `pacejka_step.py` 가 Isaac 쪽
  sim 루프 안에서 매 step `set_external_force_and_torque` 로 주입.
- **SRX8-GTE 1/8 스케일 USD**: HMCLab `srx8_cfg` 기반 (box 샤시 + 실린더
  휠). 비주얼 상세 모델링은 미포함.

---

## 관련 레포

- Isaac Sim 쪽: https://github.com/heedo7425/isaac_2026icra
- race_stack (이 레포): https://github.com/heedo7425/icra_unicorn_heedo
- 공용 차량 파라미터: `/shared_vehicle/vehicle_srx1.yaml`
