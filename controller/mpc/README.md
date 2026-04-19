# MPC Controller (ICRA2026)

UNICORN Racing Team 의 ICRA 2026 RoboRacer 용 적응형 MPC 컨트롤러.

## 개요
- `controller_manager.py` 의 PP/MAP 대체용 **독립 ROS 노드** (parallel 구조).
- Dynamic bicycle + Pacejka tire model 을 acados 로 OCP 화.
- Online μ̂ (RLS) + saturation watchdog + (추후) warmup excitation.

## 토픽 계약 (PP 와 동일)
- Sub: `/car_state/odom`, `/car_state/pose`, `/car_state/odom_frenet`, `/behavior_strategy`
- Pub: `/vesc/high_level/ackermann_cmd_mux/input/nav_1` (AckermannDriveStamped)

## 실행
```bash
roslaunch controller mpc_standalone.launch map:=experiment_3d_2
```

## 상태
- **Day 1 (현재)**: skeleton stub — (0,0) publish 하며 토픽 접속만 검증
- Day 3-6: tracking MPC baseline (acados, Pacejka)
- Day 7-10: watchdog + RLS μ̂ + warmup excitation
- Day 11-14: 실차 dry run

자세한 계획: `~/.claude/plans/partitioned-marinating-goblet.md`
