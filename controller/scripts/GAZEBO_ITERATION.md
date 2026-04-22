# Gazebo 빠른 iteration 워크플로

> "Gazebo 는 켜둔 채 MPC 만 교체" — 2D sim 급 튜닝 cycle.

## Step 1 — Gazebo plant 한 번만 띄움 (유지)

**Headless (권장)** — GUI 렌더 없음, rviz 로만 viz:
```bash
docker exec -it roboracer-gazebo bash -lc "\
    source /opt/ros/noetic/setup.bash && \
    source /home/hmcl/workspace/devel/setup.bash && \
    roslaunch roboracer_gazebo kd_0420_v1_headless.launch pacejka:=true"
```

**GUI 필요 시**:
```bash
roslaunch roboracer_gazebo kd_0420_v1.launch
```

Gazebo 가 차량 스폰 완료하면 **그대로 둠**. 아래 cycle 은 Gazebo 재시작 없음.

## Step 2 — MPC 교체 (튜닝 cycle)

```bash
# config yaml 편집 후
./controller/scripts/reload_mpc.sh upenn_mpc_controller
# → icra2026 컨테이너 안의 upenn_mpc_controller 만 kill + restart
# → Gazebo 에선 센서/차량 유지, MPC 만 새로 cmd 보내기 시작
```

## Step 3 — rosparam live reload (yaml 편집 없이)

yaml 값만 바꿔보고 싶을 때:
```bash
docker exec icra2026 bash -c "source /opt/ros/noetic/setup.bash && \
    rosparam set /upenn_mpc/w_d 18.0 && \
    rosparam set /upenn_mpc/friction_margin 0.82"
./controller/scripts/reload_mpc.sh upenn_mpc_controller
# → 새 파라미터로 OCP 재빌드
```

## 왜 빠른가

| 기존 (full relaunch) | 이 워크플로 |
|---|---|
| Ctrl+C 전체 종료 30초 | — |
| Gazebo 재시작 30-60초 | — (유지) |
| 차량 스폰/안정화 15초 | — (유지) |
| MPC acados rebuild 3-5초 | 3-5초 |
| 랩 완주 대기 30-90초 | 30-90초 |
| **1 iter 2-3분** | **1 iter 40-100초** |

## 트러블슈팅

- **MPC 가 안 뜸**: `docker exec icra2026 tail -30 /tmp/upenn_mpc_controller.log`
- **acados 재빌드 강제**: `CLEAN_CODEGEN=1 ./reload_mpc.sh ...`
- **다른 MPC 노드**: 첫 인자 바꿈 — `./reload_mpc.sh ekf_mpc_controller ekf_mpc ekf_mpc_node.py`
- **Gazebo 도 리셋**: Step 1 의 roslaunch 종료 후 재실행 (드물게 필요)

## real_time_factor 가속 (선택)

`roboracer_gazebo/world/kd_0420_v1.world` 의 physics:
```xml
<physics type='ode'>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>      <!-- 2.0 로 올리면 2x 가속 -->
  <real_time_update_rate>500</real_time_update_rate>
</physics>
```
CPU 여유 있으면 `real_time_factor=2.0` 로 올려 실측. 타이밍 문제 없을 시 유지.
