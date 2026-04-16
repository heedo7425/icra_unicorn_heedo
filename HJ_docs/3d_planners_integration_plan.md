# 3D Planners Integration Plan

> **Goal**: Add two new planner packages — `3d_sampling_based_planner` (TUM sampling) and `mpcc_planner` (Heedo MPCC, extended for 3D) — under `planner/`, run them in **observation mode** (publish output topics only, no pipeline injection), and prepare a clean container-side install path.

---

## 0. Scope & Non-Goals

**In scope (Phase 0–2)**
- Two ROS1 packages built in workspace
- Each subscribes to existing stack topics, publishes its own trajectory topic
- Self-contained install scripts that set up Python deps **inside the `icra2026` container**
- Per-package data assets reuse from `3d_gb_optimizer/global_line/data/`
- RViz markers for visual comparison vs current global raceline

**Out of scope (deferred to later phases)**
- Wiring into `state_machine` (no pipeline takeover yet)
- Replacing `controller_manager` outputs
- Dynamic obstacle handling (static evaluation first)
- Cost morphing / state-aware cost (architecture only — implementation in Phase 3)

---

## 1. Reference Repos

| Package | Source | License | Role |
|---|---|---|---|
| `3d_sampling_based_planner` | https://github.com/TUMRT/sampling_based_3D_local_planning | **GPLv3** | Frenet sampling + gg-validation, 3D native |
| `mpcc_planner` | https://github.com/heedo7425/icra_unicorn_heedo (`mpc/`) | None (ask author) | CasADi+IPOPT MPCC, 2D → extend for 3D |

**License note**: GPLv3 contagion risk for sampling planner. Keep it as a **separate, optional package** — never link statically into core stack code. ROS topic boundary is sufficient isolation per FSF interpretation but verify with team before publishing.

---

## 2. Directory Layout

```
planner/
├── 3d_sampling_based_planner/         # NEW (GPLv3 isolated)
│   ├── CMakeLists.txt
│   ├── package.xml                    # depends: rospy, f110_msgs, visualization_msgs
│   ├── README.md
│   ├── LICENSE                        # GPLv3 (inherited)
│   ├── install/
│   │   ├── install_deps.sh            # pip installs inside container
│   │   └── requirements.txt           # numpy, scipy, shapely, pandas, joblib, casadi==3.5.6rc2
│   ├── src/                           # vendored from upstream + modifications
│   │   ├── sampling_based_planner.py  # core (LocalSamplingPlanner)
│   │   ├── track3D.py                 # SHARE w/ 3d_gb_optimizer if API matches
│   │   ├── ggManager.py               # SHARE w/ 3d_gb_optimizer if API matches
│   │   └── trajectory_generator.py    # polynomial samples
│   ├── nodes/
│   │   └── sampling_planner_node.py   # ROS wrapper, observation mode
│   ├── config/
│   │   └── default.yaml               # n_range, K, weights
│   ├── launch/
│   │   └── sampling_planner_observe.launch
│   └── rviz/
│       └── sampling_observe.rviz
│
├── mpcc_planner/                      # NEW
│   ├── CMakeLists.txt
│   ├── package.xml                    # depends: rospy, f110_msgs, ackermann_msgs, nav_msgs
│   ├── README.md
│   ├── LICENSE                        # MIT or copy-from-upstream-after-asking
│   ├── install/
│   │   ├── install_deps.sh            # pip install casadi (IPOPT bundled)
│   │   └── requirements.txt           # casadi>=3.6, numpy, scipy
│   ├── src/
│   │   ├── mpcc_solver.py             # vendored, modified for 3D
│   │   ├── mpcc_solver_3d.py          # NEW: 3D-augmented model
│   │   └── path_lut_3d.py             # 3D B-spline LUT (x, y, z, theta_pitch, theta_roll)
│   ├── nodes/
│   │   └── mpcc_planner_node.py       # ROS wrapper, observation mode
│   ├── config/
│   │   ├── mpcc_2d.yaml               # baseline (Heedo's params)
│   │   └── mpcc_3d.yaml               # 3D-augmented
│   ├── launch/
│   │   └── mpcc_planner_observe.launch
│   └── rviz/
│       └── mpcc_observe.rviz
│
└── 3d_gb_optimizer/                   # EXISTING — code shared via symlink or sys.path
    └── global_line/
        └── data/                      # gg_diagrams, vehicle_params, smoothed_track_data
```

### Code Sharing Strategy

The TUM sampling repo bundles `track3D.py`, `ggManager.py`, `point_mass_model.py` — **identical or near-identical** to `3d_gb_optimizer/global_line/src/`. Two options:

- **Option A (recommended for Phase 0)**: Vendor copies into `3d_sampling_based_planner/src/`. Pros: fully isolated, no sys.path hacks. Cons: code duplication.
- **Option B (Phase 2 cleanup)**: Extract `track3D`, `ggManager` into a shared library (`f110_utils/libs/track3d_utils/`) and import from both packages.

Start with A, refactor to B once both packages are running stably.

---

## 3. Required Inputs & Where They Come From

Both planners consume the same physical "what is the world right now" inputs. Listed once, then per-planner specifics.

### 3.1 Static Assets (loaded at startup)

| Asset | Source | Used by |
|---|---|---|
| 3D smoothed track CSV (`*_3d_smoothed.csv`) | `stack_master/maps/<map>/` | sampling, mpcc-3d |
| Track bounds CSV (`*_bounds_3d*.csv`) | `stack_master/maps/<map>/` | sampling |
| Global raceline CSV (`*_timeoptimal.csv`) | `stack_master/maps/<map>/` | both (reference / contour) |
| `global_waypoints.json` | `stack_master/maps/<map>/` | both (Wpnt grid template) |
| GG diagram (`gg_diagrams/rc_car_10th_latest/`) | `3d_gb_optimizer/global_line/data/` | sampling (and mpcc-3d for friction limits) |
| Vehicle params (`params_rc_car_10th_latest.yml`) | `3d_gb_optimizer/global_line/data/` | both |

Pass these via launch params (`map_name:=gazebo_wall_2_iy`) — node resolves paths from `$(find stack_master)/maps/$(arg map_name)/`.

### 3.2 Runtime Subscriptions

| Topic | Type | Source node | Used as |
|---|---|---|---|
| `/car_state/odom_frenet` | `nav_msgs/Odometry` | `frenet_odom_republisher` | s, n, vs |
| `/car_state/odom` | `nav_msgs/Odometry` | `glim_ros` (relayed) | x, y, z, yaw |
| `/car_state/pose` | `geometry_msgs/PoseStamped` | `extract_pose_from_odom.py` | pose (alt) |
| `/global_waypoints` | `f110_msgs/WpntArray` | `global_planner` (offline-loaded) | reference raceline |
| `/global_waypoints_scaled` | `f110_msgs/WpntArray` | `vel_scaler` | scaled v_ref |
| `/ekf/imu/data` *(real car only)* | `sensor_msgs/Imu` | IMU driver | ax, ay (sim: derive from prev solution) |
| `/dynamic_obstacles` *(later)* | `f110_msgs/ObstacleArray` | perception | obstacle predictions (Phase 3) |

### 3.3 State Variables Mapping

```
solver state          ROS source                              fill rule
─────────────────────────────────────────────────────────────────────────
s                     /car_state/odom_frenet pose.x          direct
n                     /car_state/odom_frenet pose.y          direct
V (=vs)               /car_state/odom_frenet twist.linear.x  direct
chi (heading err)     yaw(/car_state/odom) - theta_track(s)  computed, wrap [-pi,pi]
ax                    /ekf/imu/data lin_acc.x  OR  prev      conditional on real_car arg
ay                    /ekf/imu/data lin_acc.y  OR  prev      conditional on real_car arg
z                     /car_state/odom pose.z                 direct (3D only)
pitch, roll           track3D.theta_*(s)                     interpolator output (3D only)
```

---

## 4. Per-Planner Specs

### 4.1 `3d_sampling_based_planner`

**Algorithm**: For each timer tick, sample K lateral offsets `n_k ∈ [n_min, n_max]` over horizon `s ∈ [s_now, s_now+L]`, generate cubic-polynomial Frenet trajectories, validate against gg-diagram + track bounds, score by cost, publish best.

**Key params (config/default.yaml)**:
```yaml
horizon_m: 8.0          # lookahead distance
n_samples_lateral: 11   # candidate count
n_range_m: [-0.5, 0.5]  # lateral offset range
dt: 0.1                 # internal time step
publish_rate_hz: 20

cost:
  w_raceline_dev: 5.0
  w_smoothness:   2.0
  w_progress:    -1.0   # negative → reward
  w_velocity:     1.0
```

**Subscriptions**: §3.2 set
**Publications**:
- `/sampling_planner/best_trajectory` (`f110_msgs/WpntArray`) — chosen trajectory
- `/sampling_planner/all_candidates` (`visualization_msgs/MarkerArray`) — colored by cost
- `/sampling_planner/status` (`std_msgs/String`) — "OK" / "NO_FEASIBLE" / etc.
- `/sampling_planner/timing_ms` (`std_msgs/Float32`) — solve time

**3D handling**: native via `Track3D` (z, pitch, roll come from interpolators).

### 4.2 `mpcc_planner`

**Baseline** (Phase 1): vendored Heedo MPCC, 2D kinematic bicycle, N=6, dT=0.1, IPOPT.

**3D Extension Plan** (Phase 2, see §6):
- Augment path LUT with `(x(s), y(s), z(s), theta_pitch(s), theta_roll(s))` — built from `Track3D`
- Modify cost: contour error in 3D (project onto track tangent plane, not horizontal)
- Add slope-aware velocity limit: `v_max(s) = √(μ·g·cos(pitch) / κ_eff(s))`
- Friction circle uses gg-diagram entry indexed by current `a_z` (vertical accel from slope)

**Key params (config/mpcc_3d.yaml)**:
```yaml
N: 10
dt: 0.1
vehicle_L: 0.36
max_speed: 15.0
max_steer: 0.4

cost:
  w_contour: 10.0
  w_lag:      5.0
  w_progress: 1.0
  w_v_ref:    2.0
  w_d_steer:  0.5
  w_d_v:      0.3
```

**Subscriptions**: §3.2 set + needs `/global_waypoints` for reference path B-spline build (one-shot at startup).

**Publications**:
- `/mpcc_planner/horizon_trajectory` (`f110_msgs/WpntArray`) — N predicted points
- `/mpcc_planner/horizon_path` (`nav_msgs/Path`) — RViz friendly
- `/mpcc_planner/predicted_v` (`std_msgs/Float32MultiArray`) — v profile
- `/mpcc_planner/status` (`std_msgs/String`)
- `/mpcc_planner/timing_ms` (`std_msgs/Float32`)

---

## 5. Container Setup (Install Scripts)

**Container**: `icra2026` (existing, from CLAUDE.md). Workspace mounted at `/home/unicorn/catkin_ws/src/race_stack`.

### 5.1 `3d_sampling_based_planner/install/install_deps.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
# Run INSIDE the icra2026 container.
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[3d_sampling] Installing Python deps into container Python..."
pip3 install --no-cache-dir -r "${PKG_DIR}/install/requirements.txt"

# Optional: acados is NOT required for sampling-only mode — skip
echo "[3d_sampling] Done."
```

`requirements.txt`:
```
numpy>=1.21
scipy>=1.7
shapely>=1.8
pandas>=1.3
joblib>=1.1
matplotlib>=3.5
```

### 5.2 `mpcc_planner/install/install_deps.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[mpcc] Installing CasADi (bundles IPOPT) into container Python..."
pip3 install --no-cache-dir -r "${PKG_DIR}/install/requirements.txt"

# Sanity check
python3 -c "import casadi; print('CasADi:', casadi.__version__)"
echo "[mpcc] Done."
```

`requirements.txt`:
```
casadi>=3.6.0
numpy>=1.21
scipy>=1.7
```

### 5.3 Invocation

```bash
# Host (one-liners; install scripts run inside container)
docker exec icra2026 bash /home/unicorn/catkin_ws/src/race_stack/planner/3d_sampling_based_planner/install/install_deps.sh
docker exec icra2026 bash /home/unicorn/catkin_ws/src/race_stack/planner/mpcc_planner/install/install_deps.sh
```

**Idempotency**: `pip install` is idempotent. Scripts can be re-run safely.

**Build**:
```bash
docker exec icra2026 bash -c "source /opt/ros/noetic/setup.bash && \
  cd /home/unicorn/catkin_ws && catkin build 3d_sampling_based_planner mpcc_planner"
```

**Env isolation**: NOT using virtualenv to avoid ROS Python path conflicts. Container's system Python is the target. If conflict arises, switch to `--user` install per package.

---

## 6. MPCC 3D Extension — Honest Assessment & Plan

### 6.0 Reality Check

**Q: Is MPCC commonly done in 3D?**
**A: No.** Almost all published MPCC (Liniger 2014/2015 origin, EVO-MPCC, F1TENTH variants) is **planar**. 3D MPCC exists for aerial vehicles (SE(3)), but for ground racing on banked/sloped tracks it's rare. The standard approach when the track is 3D is to **bake 3D physics into an offline raceline + speed profile** (which `3d_gb_optimizer` already does), and let MPCC track that 2D-projected reference. So "MPCC in 3D" usually means "MPCC tracking a 3D-aware reference," not "MPCC with 3D dynamics."

**Q: Is g_tilde (effective gravity along track tangent plane) considered?**
**A: Not in kinematic bicycle.** g_tilde requires force/mass — kinematic bicycle has neither. It's a pure geometry+velocity model:
```
ẋ = v cos(ψ),  ẏ = v sin(ψ),  ψ̇ = v/L · tan(δ)
```
There is no `m`, no `g`, no `μ`. You **cannot inject gravity into the dynamics** — only into **constraints** added externally.

**Q: Does friction increase/decrease appear?**
**A: No, not natively.** Friction enters only if:
- (a) Model becomes dynamic (lateral force balance: `m·v²·κ = F_y ≤ μ·F_z`)
- (b) You add a **constraint** like `v² · κ ≤ μ_eff · g_normal` outside the dynamics

Kinematic bicycle gives you (b) at best — and that's a **velocity ceiling**, not a model of how friction shapes trajectory.

### 6.1 What's Actually Possible (3 honest options)

| Option | Model | g_tilde / friction | Effort | Verdict |
|---|---|---|---|---|
| **A. Kinematic + 3D constraints** | Heedo MPCC unchanged | Velocity ceiling only (constraint, not dynamics) | Low | Hacky but workable for small slopes (≤10°) |
| **B. Dynamic bicycle 2.5D** | Replace with dynamic bicycle (lateral tire force, mass) + 3D-tilted gravity vector | g_tilde in force balance, μ in tire model | Medium-high | Correct physics, but bigger rewrite. CasADi handles it fine. |
| **C. Point-mass on 3D Frenet** | Same as `3d_gb_optimizer/point_mass_model.py` | Native 3D, gg-diagram | High | Becomes what we're trying to escape. Defeats the purpose. |

### 6.2 Recommended: Option A for Phase 2a, Option B if needed

**Option A "Kinematic + 3D Reference + Slope-aware Velocity Cap"** — what we *actually* implement:

1. **Reference is 3D**: Path LUT extended with `(x(s), y(s), z(s), pitch(s), roll(s), kappa(s))` from `Track3D`. MPCC contour/lag computed in **track tangent plane** (project car position into local Frenet frame at nearest s).

2. **Dynamics stay 2D kinematic**: Solver still works in `(x, y, ψ, s)`. Banking does NOT enter dynamics. The car "thinks" it's flat.

3. **Velocity ceiling as soft state constraint** (this is the only place 3D physics enters):
   ```
   At each prediction step k:
     pitch_k = pitch_LUT(s_k)
     roll_k  = roll_LUT(s_k)
     g_n     = g · cos(pitch_k) · cos(roll_k)         # gravity normal to track
     a_lat_max = μ · g_n                              # friction circle (lateral budget)
     v_max(s_k) = √( a_lat_max / |κ(s_k)| )
     constraint: v_k ≤ v_max(s_k)  (soft, with slack)
   ```
   This is a **kinematic-level approximation** of "you slip on a steep tilted curve."

4. **Optional: longitudinal slope effect as feedforward**
   ```
   At each step:
     a_long_gravity = -g · sin(pitch_k)               # downhill = positive accel
   ```
   Add as a **bias term in v_dot prediction**, NOT a force balance. Correct sign helps the planner anticipate downhill speedup / uphill slowdown but is still a hack.

5. **Cost adds for 3D awareness**:
   - `w_z_dev * (z_car - z_ref(s))²` — discourages geometric drift on rolled banks
   - `w_v_slope * (v - v_max_slope)²` — soft preference for slope-respecting speed

### 6.3 What Option A does NOT do (be honest)

- ❌ Weight transfer (front-rear, left-right)
- ❌ Slip angle / tire saturation
- ❌ Coupling between steering and longitudinal force
- ❌ Aerodynamic effects (irrelevant at 15 m/s anyway)
- ❌ Gyroscopic effects on banked turns
- ❌ Real friction-circle interaction (we only cap lateral; longitudinal-lateral coupling ignored)

In short: **Option A is "MPCC pretending the track is flat, with a 3D-shaped speed limit bolted on."** It is not "3D MPCC" in any rigorous sense.

### 6.4 Why this is still useful

- Global raceline (`*_timeoptimal.csv`) already encodes 3D-optimal speed profile (computed offline by acados with full point-mass 3D physics)
- Online MPCC's job: **track that reference** + react to small disturbances
- Slope-aware velocity ceiling prevents MPCC from "improving" on the offline profile in ways that ignore slope physics
- This is exactly how Roborace, Indy Autonomous, and most racing stacks do it: **3D in offline planner, 2D in online tracker**, with velocity caps as the bridge

### 6.5 When to upgrade to Option B (dynamic bicycle)

Promote only if **all** of these are true:
- Phase 2a gives observed slip / corner-cutting on banked sections
- Lap-time gap to offline raceline > 5% on 3D maps
- We have measured tire μ values for the RC car (currently estimated)
- Team has 1-2 weeks to rewrite + retune

If we go to Option B, the dynamics become:
```
ẋ = v cos(ψ + β)
ẏ = v sin(ψ + β)
ψ̇ = v · cos(β) / L · tan(δ)
v̇ = (1/m) [ F_drive - F_drag - m·g·sin(pitch_proj) ]
β = atan( (lr/L) · tan(δ) )               # slip angle approx
F_y_max = μ · m · g · cos(pitch) · cos(roll)
constraint: |m · v² · κ| ≤ F_y_max         # friction circle (lateral)
```
Still a single-track approximation but now `m, g, μ, pitch, roll` actually appear in the dynamics. This is what most people mean by "3D-aware MPC."

### 6.6 Decision

- **Phase 2a (immediate)**: Implement **Option A**. Document explicitly that it is kinematic+caps, not full 3D dynamics.
- **Phase 2b (deferred)**: Option B only if Phase 2a shows clear failure modes. Estimate 1-2 weeks rewrite.
- **Reject**: Option C. If we want point-mass 3D, just use the existing `3d_gb_optimizer` online — that's what it is.

---

## 7. Implementation Phases

### Phase 0 — Skeletons (Day 1)
- [ ] Create both package directories with `package.xml`, `CMakeLists.txt`
- [ ] Vendor source files from upstream repos (clean, attribute, mark license)
- [ ] Write `install/install_deps.sh` and `requirements.txt`
- [ ] Run install scripts in container, verify imports
- [ ] `catkin build` both packages successfully
- **Verify**: `rosrun 3d_sampling_based_planner sampling_planner_node.py --help` runs

### Phase 1 — Observation-mode publishers (Day 2-3)
- [ ] Sampling node: subscribe to topics in §3.2, run sampler, publish `/sampling_planner/*`
- [ ] MPCC 2D node: subscribe, run baseline solver, publish `/mpcc_planner/*`
- [ ] Launch files for each, with map_name arg
- [ ] RViz config to visualize candidates + chosen trajectory
- **Verify**: With car driving global raceline (sim or playback), both nodes produce sensible markers; no exceptions for 5 min run

### Phase 2a — MPCC kinematic + 3D constraints (Day 4-5)
- [ ] Build 3D path LUT from `Track3D` (x,y,z,pitch,roll,kappa over s)
- [ ] Modify contour/lag cost: project car position into local Frenet tangent plane
- [ ] Add soft constraint `v_k ≤ √(μ·g·cos(pitch)·cos(roll) / |κ|)` per step
- [ ] Optional: longitudinal slope feedforward `a_long_g = -g·sin(pitch)`
- [ ] New launch (`mpcc_planner_observe_3d.launch`) with `mpcc_3d.yaml`
- **Verify**: On 3D map (gazebo_wall_2_iy), predicted v at known steep section is lower than 2D version; contour error stays bounded on banked turns
- **Document explicitly**: this is kinematic+caps, NOT 3D dynamics

### Phase 3 — State-aware cost (separate plan)
- Architecture only, not implemented here. See [State-aware cost plan TBD].

### Phase 4 — Pipeline integration (separate plan)
- Mux into `/global_waypoints_scaled` only after Phase 1-3 validated. Out of scope for this doc.

---

## 8. Validation Checklist

For each planner, before declaring Phase 1 done:

- [ ] Solve time < 50 ms (sampling) or < 30 ms (MPCC) on NUC
- [ ] No infeasibility for 90%+ of timer ticks during straight + corner driving
- [ ] Output WpntArray fields populated: `s_m, d_m, x_m, y_m, z_m, vx_mps, psi_rad, kappa_radpm, ax_mps2`
- [ ] RViz markers visible and sane
- [ ] Status topic transitions logged
- [ ] Logged 5-minute rosbag with all `/sampling_planner/*` and `/mpcc_planner/*` topics

---

## 9. Risks & Open Questions

| # | Risk | Mitigation |
|---|---|---|
| 1 | GPLv3 contagion from sampling planner | Isolate as separate package, communicate via topics only, document |
| 2 | CasADi/IPOPT version conflict in container | Pin CasADi 3.6.x; fallback to `--user` install if needed |
| 3 | `track3D.py` API drift between vendored and `3d_gb_optimizer` copies | Phase-2 cleanup: extract to shared lib |
| 4 | sim vs real ax/ay handling | Use `real_car:=true/false` arg; default to prev-solution-derived in sim |
| 5 | MPCC 2D contour cost meaningless on banked corners | 3D extension (§6) addresses |
| 6 | gg-diagram lookup latency in tight inner loop | Precompute LUT, no per-step interpolation |

**Open questions**:
- Heedo repo license? (currently absent — must request before vendoring publicly)
- Do we replicate `Track3D` and `GGManager` or share via `f110_utils/libs/`? (Phase 2 decision)
- Real-car IMU topic name confirmation needed (`/ekf/imu/data` assumed from existing code)

---

## 10. Quick-Start Commands (Once Built)

```bash
# Sampling planner observe
roslaunch 3d_sampling_based_planner sampling_planner_observe.launch \
  map_name:=gazebo_wall_2_iy

# MPCC 2D observe
roslaunch mpcc_planner mpcc_planner_observe.launch \
  map_name:=gazebo_wall_2_iy use_3d:=false

# MPCC 3D observe
roslaunch mpcc_planner mpcc_planner_observe.launch \
  map_name:=gazebo_wall_2_iy use_3d:=true

# Compare both alongside global raceline in RViz
rviz -d $(rospack find 3d_sampling_based_planner)/rviz/sampling_observe.rviz
```

---

**Owner**: HJ
**Created**: 2026-04-16
**Status**: Plan — pending review before Phase 0 execution
