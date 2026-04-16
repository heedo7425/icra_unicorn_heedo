# 3d_sampling_based_planner

ROS1 (Noetic) wrapper around [TUMRT/sampling_based_3D_local_planning](https://github.com/TUMRT/sampling_based_3D_local_planning)
for the UNICORN ICRA 2026 3D racing stack.

**Status**: Phase 0 — observation mode only. Publishes its chosen trajectory on its own topics;
**does not** inject into `/global_waypoints_scaled` or the control pipeline.

## Layout

```
3d_sampling_based_planner/
├── CMakeLists.txt, package.xml
├── README.md                  ← this file (stack integration notes)
├── README_UPSTREAM.md         ← original upstream README
├── LICENSE                    ← GPLv3 (inherited from upstream)
├── install/
│   ├── install_deps.sh        ← pip3 installer, runs inside icra2026 container
│   └── requirements.txt
├── config/default.yaml        ← node params
├── launch/
│   └── sampling_planner_observe.launch
├── node/
│   └── sampling_planner_node.py
│
├── src/                       ← UPSTREAM: planner core (Track3D, ggManager, LocalSamplingPlanner, …)
├── data/                      ← UPSTREAM: gg-diagrams, vehicle params, tracks (reference/sample data)
├── local_sampling_based/      ← UPSTREAM: offline sim script + experiment yaml
├── global_racing_line/        ← UPSTREAM: offline racing-line generator
├── gg_diagram_generation/     ← UPSTREAM: gg-diagram synthesis tools
└── track_processing/          ← UPSTREAM: track CSV preprocessing
```

All `src/`, `data/`, `local_sampling_based/`, etc. are verbatim from upstream. Only the top-level ROS
glue (`package.xml`, `CMakeLists.txt`, `node/`, `launch/`, `config/`, `install/`) is stack-specific.

### Shared modules with `3d_gb_optimizer`

To cut duplication, `src/ggManager.py` has been **removed from this package** and is resolved from
`../3d_gb_optimizer/global_line/src/ggManager.py` via `sys.path`. Both copies were verified byte-identical
at the time of split (`diff` empty). The sys.path fallback is added in:

- `node/sampling_planner_node.py` (the ROS entry point)
- `local_sampling_based/sim_sampling_based_planner.py`
- `global_racing_line/gen_global_racing_line.py`
- `global_racing_line/plot_global_racing_line.py`
- `gg_diagram_generation/plot_gg_diagrams.py`

If upstream later updates `ggManager.py` and the API drifts, either re-vendor it locally or keep the
shared canonical copy under `3d_gb_optimizer`.

`track3D.py`, `point_mass_model.py`, `local_racing_line_planner.py`, `visualizer.py` are **NOT shared** —
the two packages carry divergent versions (HJ modifications in `3d_gb_optimizer`). Consolidate later
only after API compatibility is verified.

## License

Upstream is **GPLv3** — see [LICENSE](LICENSE). This subpackage inherits GPLv3. Communicate with the
rest of the stack **via ROS topics only** (no static/dynamic linking, no Python imports from outside
this package) so the rest of the stack is not infected by GPL.

## Install (inside the `icra2026` container)

```bash
docker exec icra2026 bash \
  /home/unicorn/catkin_ws/src/race_stack/planner/3d_sampling_based_planner/install/install_deps.sh
```

Then build:

```bash
docker exec icra2026 bash -c "source /opt/ros/noetic/setup.bash && \
  source /home/unicorn/catkin_ws/devel/setup.bash && \
  cd /home/unicorn/catkin_ws && catkin build 3d_sampling_based_planner"
```

## Run

```bash
# With GLIL/localization + /car_state/odom_frenet already publishing:
roslaunch 3d_sampling_based_planner sampling_planner_observe.launch \
  map_name:=gazebo_wall_2_iy vehicle_name:=rc_car_10th_latest
```

## Topics

### Subscribed

| Topic | Type | Role |
|---|---|---|
| `/car_state/odom_frenet` | `nav_msgs/Odometry` | current `s, n, s_dot` |

### Published (under `~` = `/sampling_planner_node/`)

| Topic | Type | Role |
|---|---|---|
| `~best_trajectory` | `f110_msgs/WpntArray` | chosen candidate as waypoints (x, y, z, v, ...) |
| `~best_path` | `nav_msgs/Path` | same as above, RViz-friendly (3D z filled) |
| `~candidates_marker` | `visualization_msgs/MarkerArray` | LINE_STRIP of chosen trajectory (TODO: all candidates) |
| `~status` | `std_msgs/String` | INIT_OK / WAITING_ODOM / OK / NO_FEASIBLE / EXCEPTION:<name> |
| `~timing_ms` | `std_msgs/Float32` | solve time per tick |

## Inputs fed from the stack

| Asset | Path resolution in launch |
|---|---|
| 3D smoothed track CSV | `stack_master/maps/<map_name>/<map_name>_3d_smoothed.csv` |
| Reference raceline CSV | `stack_master/maps/<map_name>/<map_name>_3d_<vehicle>_timeoptimal.csv` |
| GG diagrams | `3d_gb_optimizer/global_line/data/gg_diagrams/<vehicle>/velocity_frame/` |
| Vehicle params YAML | `3d_gb_optimizer/global_line/data/vehicle_params/params_<vehicle>.yml` |

## Known limitations (Phase 0)

- No dynamic obstacle handling (empty `prediction`). Static track only.
- Candidate visualization currently shows only the chosen trajectory.
- Raceline dict is built from the CSV columns with best-effort name matching — if CSV schema drifts
  you may see "NO_FEASIBLE". Inspect fallback in `node/sampling_planner_node.py::_load_raceline_dict`.
- Wpnt `z_m` field is filled only if the local `f110_msgs/Wpnt.msg` has the 3D extension
  (`### HJ : add z for 3D`). Otherwise only `best_path` carries z.
