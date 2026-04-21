# pcdtostl — SLAM `.pcd` → Isaac Sim `.stl`

SLAM 으로 얻은 point cloud (GLIM/GLIL 출력 `.pcd`) 를 Isaac Sim 에 로드 가능한 mesh (`.stl`) 로 변환. 완성된 `.stl` 은 Blender/Omniverse 쪽에서 추가로 `.usd` 변환해 `issac_icra` 컨테이너의 `/workspace/usd/` 로 넣으면 됨.

## 사용

```bash
source /opt/ros/noetic/setup.bash && source <catkin>/devel/setup.bash
roslaunch stack_master pcd_to_stl.launch \
    pcd:=$(rospack find stack_master)/maps/kd_0420_v1/kd_0420_v1.pcd \
    stl:=$(rospack find stack_master)/maps/kd_0420_v1/kd_0420_v1.stl
```

## 파라미터

| 인자 | 기본 | 설명 |
|---|---|---|
| `pcd` | 필수 | 입력 `.pcd` 경로 |
| `stl` | 필수 | 출력 `.stl` 경로 |
| `method` | `poisson` | `poisson` (smooth closed) 또는 `ball_pivot` (open track) |
| `voxel` | `0.05` | [m] voxel down-sample, 0 = skip |
| `normal_radius` | `0.15` | [m] 노말 추정 반경 |
| `poisson_depth` | `9` | octree depth — 크면 디테일↑ 메모리↑ |
| `density_quantile` | `0.02` | 저밀도 정점 제거 quantile |
| `ball_radii` | `0.1,0.2,0.4` | ball-pivot 반경 리스트 |

## 의존성

`python3-open3d` 필요:
```bash
pip3 install open3d
```

## 파이프라인

```
GLIM .pcd ──(pcd_to_stl.launch)──► .stl ──(Blender/usdcat)──► .usd
                                                              │
                        /workspace/usd/ in issac_icra ◄───────┘
```
