# launch_mode

Current-based throttle override for race-start launch boost.

## What it does
- Watches a dedicated RC channel (`launch_arm`) + autonomous-arm + GO signal.
- When all gates pass and the vehicle is stationary, publishes raw motor
  current at 200 Hz to `/vesc/commands/motor/current`, bypassing the VESC
  speed-PID wind-up that otherwise costs ~100–300 ms of full torque.
- Steering is passed through from whichever controller is active
  (mu_ppc / upenn_mpc / mpcc_dyna) — we never override steer.
- Slip guard cuts current when `(v_wheel - v_body)/v_wheel` exceeds the
  threshold for `slip_confirm_n` ticks.
- Exits on first of: distance, v ratio, time, steer magnitude, upcoming
  curvature. Hand-off pre-loads VESC speed-PID with current `v_meas`.

## Rear-grid feasibility advisory
On a staggered grid (front car ~80 cm ahead of rear car), the rear car
must still launch to win, but launching from rear can rear-end the
front car if lateral clearance is too tight. The node publishes a
continuous advisory the human reads before flipping `launch_arm`.

Topics produced for the human/dashboard:
| Topic | Type | Meaning |
|-------|------|---------|
| `/launch_mode/feasibility` | `std_msgs/String` | `GREEN` / `YELLOW` / `RED` / `UNKNOWN` |
| `/launch_mode/feasible`    | `std_msgs/Bool`   | `true` for `GREEN` or `YELLOW` |
| `/launch_mode/markers`     | `visualization_msgs/MarkerArray` | RViz overlay (latched) |

Add a `MarkerArray` display in RViz on `/launch_mode/markers` with the
fixed frame matching `viz_frame` (default `base_link`). You will see:
* Floating text above the car, coloured by feasibility
  (green / yellow / red / gray-for-UNKNOWN), with the latest
  `front_obs` and `lateral_gap` values inline.
* The front collision-corridor ROI rectangle in the same colour.
* Left/right side ROI rectangles in light blue.

Feasibility colour legend:
| Colour | Condition | Recommendation |
|--------|-----------|----------------|
| GREEN  | front grid, OR rear with no front obstacle | full launch is safe |
| YELLOW | rear, front blocked, lateral gap > `safe_lateral_gap_m` | reduced launch is safer |
| RED    | rear, front blocked, lateral gap ≤ `safe_lateral_gap_m` | risky — human's call |
| UNKNOWN | scan stale | wait / fix sensor |

When the human triggers GO, intent is picked from the *current* scan:

| Situation | Intent |
|-----------|--------|
| `grid_slot=front` | FULL |
| rear, no front obstacle | FULL |
| rear, front blocked | REDUCED (`i × reduced_i_factor`, exit at `reduced_s_launch_m`) |
| rear, scan stale | ABORT (only auto-veto: sensor failure) |

Lateral tightness is only used by the advisory, not as a veto. The
human is always boss. While LAUNCH is active, runtime cutoff
(`front_ttc_th_s`, `front_proximity_min_m`) still aborts on imminent
collision regardless of intent.

`grid_slot` is set per race in `launch_mode.yaml`.

## Files you (the user) edit
All tunables and integration knobs live in `config/`. The node does not
hardcode any topic, channel, or limit.

| File | Purpose | Required edit |
|------|---------|---------------|
| `config/launch_mode.yaml` | every runtime parameter (channels, limits, FSM, slip, topics, lookahead) | tune per car/track |
| `config/vesc_limits_per_car.yaml` | reference of each car's VESC firmware caps | fill from VESC Tool |
| `config/elrs_channel_map.yaml` | reference doc for which RC switch maps to which `/joy` index | verify with `rostopic echo /joy` |
| `config/high_level_mux_snippet.yaml` | snippet to paste into the car's `high_level_mux.yaml` | append entry |

Nothing in the code needs to change to add a new car or remap channels.

## Required external edits

1. **`stack_master/config/<CAR>/devices/high_level_mux.yaml`** — append the
   `launch` subscriber from `config/high_level_mux_snippet.yaml`. Priority
   must be higher than `nav_1`.
2. **VESC Tool** — confirm `motor_current_max`, `abs_max_current`,
   `battery_current_max`. Mirror `motor_current_max` into
   `launch_mode.yaml::i_firmware_cap` and document in
   `vesc_limits_per_car.yaml`.
3. **ELRS transmitter** — assign physical switches/buttons for
   `launch_arm` and `go`. Verify `/joy` indices via `rostopic echo /joy`.
   Set `launch_arm_button` / `go_button` (or `_axis`) in
   `launch_mode.yaml`.

## Calibration
1. Set `i_firmware_cap` to your VESC Tool `motor_current_max`.
2. Sweep `i_launch_max` 30 A → ceiling in 5 A steps. Log `t_to_5m` and
   `slip_max`. Pick value where slip stays below threshold.
3. Tune `s_launch_m` to the distance to the first corner of the track.
4. `kappa_exit` and `steer_exit_rad` are early-exit safety — tune so a
   normal exit triggers ~1 m before the first apex.

## Layout
```
launch_mode/
  config/
    launch_mode.yaml              # all runtime params
    vesc_limits_per_car.yaml      # reference: per-car VESC Tool values
    elrs_channel_map.yaml         # reference: /joy index mapping
    high_level_mux_snippet.yaml   # snippet to paste into car's mux yaml
  launch/launch_mode.launch
  src/
    launch_fsm.py                 # state transitions
    slip_guard.py                 # slip detection + current scaling
    launch_mode_node.py           # ROS node, current publishing, hand-off
```
