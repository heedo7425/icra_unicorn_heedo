# 디버그 노트: `glim`이 `devel/_setup_util.py`를 망가뜨려 catkin 워크스페이스 전체 빌드 실패

## 증상

새 셸에서 `source devel/setup.bash` 후 `catkin build`를 돌리면 다음과 같은 에러가 다수 패키지에서 발생:

```
ModuleNotFoundError: No module named 'genmsg'
ModuleNotFoundError: No module named 'catkin'
```

영향 받은 패키지: `blink1`, `f110_msgs`, `cartographer_ros_msgs`, `global_line_3d` 등. 사실 이 패키지들 자체에는 문제가 없으며, 의존 패키지들이 줄줄이 abandoned 처리되며 연쇄 실패가 발생함.

## 빠른 재현 확인

```bash
bash -c 'source /opt/ros/noetic/setup.bash && echo BEFORE=$PYTHONPATH \
  && source ~/catkin_ws/devel/setup.bash && echo AFTER=$PYTHONPATH'
```

정상 상태:
```
BEFORE=/opt/ros/noetic/lib/python3/dist-packages
AFTER=/home/unicorn/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages
```

깨진 상태 (이 버그):
```
BEFORE=/opt/ros/noetic/lib/python3/dist-packages
AFTER=/home/unicorn/catkin_ws/devel/lib/python3/dist-packages
```
ROS 경로가 **체이닝되지 않고 통째로 사라짐**.

## 근본 원인

`devel/_setup_util.py`의 271번 줄이 ROS 경로 없이 devel만 포함한 채로 박제되어 있음:

```python
CMAKE_PREFIX_PATH = r'/home/unicorn/catkin_ws/devel'.split(';')
```

정상이라면:

```python
CMAKE_PREFIX_PATH = r'/home/unicorn/catkin_ws/devel;/opt/ros/noetic'.split(';')
```

`_setup_util.py`가 source될 때, 이 스크립트는 이전 워크스페이스(=ROS)의 PYTHONPATH 항목을 **롤백(제거)**하고, 위 리스트에 있는 경로만 다시 **prepend**한다. 그런데 ROS 경로가 리스트에 없으므로 롤백만 일어나고 재추가는 없음 → `genmsg`, `catkin` 등 ROS Python 모듈이 `sys.path`에서 사라지게 됨.

## `_setup_util.py`가 박제되는 메커니즘

`devel/_setup_util.py`는 catkin 패키지의 cmake configure가 실행될 때마다 `/opt/ros/noetic/share/catkin/cmake/catkin_generate_environment.cmake:40`의 `configure_file()`로 재생성됨. 치환되는 값 `@CMAKE_PREFIX_PATH_AS_IS@`는 `all.cmake:55`에서 캡처됨:

```cmake
set(CMAKE_PREFIX_PATH_AS_IS ${CMAKE_PREFIX_PATH})
```

즉, **마지막으로 cmake configure가 성공한 패키지의 `CMAKE_PREFIX_PATH` 값**이 `_setup_util.py`에 박제되는 구조.

## 범인 식별 (mtime 포렌식)

```
devel/_setup_util.py                          → 2026-04-07 10:45:02   ← 망가진 파일
logs/global_line_3d/build.cmake.002.log       → 2026-04-07 10:45:02   ← 같은 초!
```

각 패키지의 cmake 로그를 교차 확인:

```
grep -r "Using CMAKE_PREFIX_PATH" ~/catkin_ws/logs/
```

거의 모든 패키지가 `/opt/ros/noetic`을 보여줌 — 단 두 패키지만 예외:

```
logs/glim/build.cmake.log
  -- Using CMAKE_PREFIX_PATH: .../GLIL_unicorn_racing/glim/../../kiss_matcher_install
logs/global_line_3d/build.cmake.001.log
  -- Using CMAKE_PREFIX_PATH: /home/unicorn/catkin_ws/devel
logs/global_line_3d/build.cmake.002.log
  -- Using CMAKE_PREFIX_PATH: /home/unicorn/catkin_ws/devel
```

### 부패가 일어난 시간 순서

```
10:41:58  global_line_3d build.cmake.000  → /opt/ros/noetic        (정상)
10:42:03  glim          build.cmake.000   → kiss_matcher_install   (이상 발생)
10:44:26  global_line_3d build.cmake.001  → /home/.../devel        (이미 깨짐)
10:45:02  global_line_3d build.cmake.002  → /home/.../devel        ← _setup_util.py 덮어씀
```

## `glim`이 근본 원인인 이유

`glim`의 [package.xml](../../state_estimation/GLIL_unicorn_racing/glim/package.xml)은 다음과 같이 선언함:

```xml
<buildtool_depend condition="$ROS_VERSION == 1">catkin</buildtool_depend>
<export>
  <build_type condition="$ROS_VERSION == 1">catkin</build_type>
</export>
```

따라서 catkin tools는 glim을 **catkin 패키지**로 인식한다. 그런데 정작 [CMakeLists.txt](../../state_estimation/GLIL_unicorn_racing/glim/CMakeLists.txt)에서는 **`find_package(catkin)`도 `catkin_package()`도 호출하지 않음**. 대신 다음을 수행:

```cmake
list(PREPEND CMAKE_PREFIX_PATH "${KISS_MATCHER_PREFIX}")
```

`package.xml`(catkin 선언)과 `CMakeLists.txt`(순수 cmake 동작)의 **불일치**가 catkin tools의 환경 캐시를 흐트러뜨림. glim 빌드 후, catkin tools가 다음 패키지(`global_line_3d`)에 전달하는 per-package env 캐시에서 `/opt/ros/noetic`이 빠지게 됨. 그 결과 `global_line_3d`가 `CMAKE_PREFIX_PATH`가 깨진 상태로 cmake configure를 실행하고, 성공적으로 끝나면서 `devel/_setup_util.py`를 잘못된 값으로 덮어씀.

이 시점부터 매번 `source devel/setup.bash`가 ROS의 `PYTHONPATH`를 wipe하고, 다음 `catkin build`는 `genmsg`/`catkin` Python 모듈이 필요한 모든 패키지에서 실패한다.

## 해결 방법 (권장 — glim 최소 패치)

[glim/CMakeLists.txt](../../state_estimation/GLIL_unicorn_racing/glim/CMakeLists.txt)를 열고 `project(...)` 바로 뒤에 **두 줄만** 추가:

```cmake
project(glim VERSION 1.1.0 LANGUAGES C CXX)

find_package(catkin REQUIRED)
catkin_package()

add_compile_options(-std=c++17)
...
```

이렇게 하면:
- `find_package(catkin)`이 `all.cmake`를 include하면서 `CMAKE_PREFIX_PATH_AS_IS`에 ROS 경로가 정상 캡처됨
- `catkin_package()`가 `devel/_setup_util.py`를 올바른 체인 값으로 재생성함
- `package.xml`의 `build_type=catkin` 선언과 `CMakeLists.txt`가 일치하게 됨
- glim 자체의 빌드 로직(GTSAM, kiss_matcher, TBB 등)은 **전혀 건드리지 않음**

## 대안 — glim을 plain cmake 패키지로 선언

[glim/package.xml](../../state_estimation/GLIL_unicorn_racing/glim/package.xml)을 다음과 같이 수정:

```xml
<buildtool_depend>cmake</buildtool_depend>
<export>
  <build_type>cmake</build_type>
</export>
```

이렇게 하면 catkin tools가 glim을 "plain cmake package"로 다루고, 빌드 도중 `devel/_setup_util.py`를 일절 건드리지 않음.

## 패치 적용 후 복구 절차

```bash
cd ~/catkin_ws
catkin clean -y
unset PYTHONPATH
source /opt/ros/noetic/setup.bash
echo $CMAKE_PREFIX_PATH      # /opt/ros/noetic 포함 확인
catkin build
```

빌드 후 검증:

```bash
grep "^            CMAKE_PREFIX_PATH = r" devel/_setup_util.py
# 기대 결과:
#     CMAKE_PREFIX_PATH = r'/home/unicorn/catkin_ws/devel;/opt/ros/noetic'.split(';')
```

그리고:

```bash
bash -c 'source /opt/ros/noetic/setup.bash && source devel/setup.bash && python3 -c "import genmsg, catkin; print(\"OK\")"'
```

## 핵심 파일

- [devel/_setup_util.py:271](../../../devel/_setup_util.py#L271) — 박제되는 줄
- `/opt/ros/noetic/share/catkin/cmake/all.cmake:55` — `CMAKE_PREFIX_PATH_AS_IS`가 캡처되는 위치
- `/opt/ros/noetic/share/catkin/cmake/catkin_generate_environment.cmake:40` — 템플릿이 configure되는 위치
- [glim/CMakeLists.txt](../../state_estimation/GLIL_unicorn_racing/glim/CMakeLists.txt) — 패치 대상 파일
- [glim/package.xml](../../state_estimation/GLIL_unicorn_racing/glim/package.xml) — 불일치가 있는 선언 파일
