# CLAUDE.md - Project Context

## 프로젝트 개요
UNICORN Racing Team의 자율주행 레이싱 스택 (ROS1, Noetic).
현재 2D 평면 레이싱에서 **3D 맵 주행**으로 확장 중 (ICRA 2026 RoboRacer).

## 언어
- 한국어로 설명하고 대화할 것

## 작업 원칙
- 2D 시스템에서 사용되던 큰 파이프라인(base_system.launch, headtohead.launch)의 실제 작동 코드를 타고 가면서 정확한 파이프라인을 인지하고, 3차원 확장에 insight 제공
- 3d_base_system.launch를 만들어서 기존 파이프라인과 유사하게 3D로 확장하되, 기존 코드와 충돌 나지 않게 작업
- 판단에는 항상 근거와 증거(파일 경로, 라인 번호, 토픽명 등)를 제시할 것
- 노드 분석 시 반드시 실제 launch 파일의 인자(arg/param)를 추적하여, 해당 인자 조합에서의 실제 실행 경로를 파악할 것. 코드만 보고 "이 기능이 있다"가 아니라 "이 launch 설정에서 이 코드 경로가 실행된다"를 확인
- 3D 확장 관련 수정에는 `### HJ :` 주석 사용
  - 예: `### HJ : add z for 3D closest-point search`
- 작업 중 발생하는 이슈, 해결 방법, 해결된 사항을 `TODO_HJ.md`에 즉시 업데이트할 것
  - 새 이슈 발견 시 TODO 항목 추가
  - 해결 완료 시 체크박스 체크 + 해결 방법 간단히 기록
  - 판단 보류 사항은 별도 섹션에 근거와 함께 기록

## 핵심 경로
- 맵 데이터: `stack_master/maps/<map_name>/`
- 현재 3D 맵: `stack_master/maps/experiment_3d_2/`
- Frenet 변환 C++: `f110_utils/libs/frenet_conversion/src/frenet_conversion.cc`
- Frenet 변환 Python (2D only): `f110_utils/libs/frenet_conversion/src/frenet_converter/frenet_converter.py`
- 웨이포인트 메시지: `f110_utils/libs/f110_msgs/msg/Wpnt.msg`
- 컨트롤러: `controller/controller_manager.py`, `controller/combined/src/Controller.py`
- 상태머신: `state_machine/src/state_machine_node.py`
- 런치 (2D 원본): `stack_master/launch/base_system.launch`, `stack_master/launch/headtohead.launch`
- 런치 (3D 작업용): `stack_master/launch/car_race.launch` (임시), 향후 `3d_base_system.launch`

## Docker 환경
- 컨테이너 이름: `icra2026`
- 호스트 → 컨테이너 경로 매핑: `/home/unicorn/icra_2026_ws/UNICORN-ICCAS_2025` → `/home/unicorn/catkin_ws/src/race_stack`
- **빌드, import 테스트, 코드 실행은 반드시 Docker 컨테이너 안에서 할 것**
```bash
# 컨테이너 내부에서 명령 실행
docker exec icra2026 bash -c "source /opt/ros/noetic/setup.bash && source /home/unicorn/catkin_ws/devel/setup.bash && <명령>"

# 빌드 예시
docker exec icra2026 bash -c "source /opt/ros/noetic/setup.bash && source /home/unicorn/catkin_ws/devel/setup.bash && cd /home/unicorn/catkin_ws && catkin build"
```
- 로컬 파일 수정은 호스트에서, 빌드/테스트는 Docker에서
- **로컬 git에 영향주는 docker 명령 (fetch, checkout 등) 절대 금지**

## 빌드
```bash
cd /home/unicorn/icra_2026_ws && catkin build
```

## 실행
```bash
# GLIL localization (별도 터미널)
roslaunch glim_ros glil_cpu.launch

# 메인 시스템 (현재 임시)
roslaunch stack_master car_race.launch map:=experiment_3d_2

# 컨트롤러 + 상태머신
roslaunch stack_master headtohead.launch
```

## 미확인 사항 (판단 보류)
- carstate_node.py: GLIL 환경에서 필요 여부 미확정
  - vy를 pose 미분+moving average로 자체 계산, /car_state/pitch 발행 등의 기능이 있음
  - GLIL base_odom이 vy, pitch를 충분히 제공하는지 확인 필요

## 주의사항
- f1tenth_simulator는 2D only → sim:=true 사용 불가 (3D 맵에서)
- base_system.launch는 2D 맵 전용 → 3D에서는 별도 런치 사용
- TODO 관리: `TODO_HJ.md` 참조
