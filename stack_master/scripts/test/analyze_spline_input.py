#!/usr/bin/env python3
"""
Analyze do_spline inputs saved from smart_static_avoidance_node.py
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load saved data
with open('/tmp/do_spline_input_fixed.pkl', 'rb') as f:
    data = pickle.load(f)

obs = data['obs']
gb_wpnts = data['gb_wpnts']
use_fixed_path = data['use_fixed_path']
cur_x = data['cur_x']
cur_y = data['cur_y']
cur_s = data['cur_s']
cur_d = data['cur_d']

print("="*80)
print("DO_SPLINE INPUT ANALYSIS")
print("="*80)
print(f"Mode: {'FIXED' if use_fixed_path else 'GB'}")
print(f"Current position: x={cur_x:.2f}, y={cur_y:.2f}, s={cur_s:.2f}, d={cur_d:.2f}")
print()

print("OBSTACLE:")
print(f"  id: {obs.id}")
print(f"  s_center: {obs.s_center:.2f}")
print(f"  d_center: {obs.d_center:.2f}")
print(f"  d_left: {obs.d_left:.2f}")
print(f"  d_right: {obs.d_right:.2f}")
print(f"  size: {obs.size:.2f}")
print()

print(f"WAYPOINTS: {len(gb_wpnts)} points")
print(f"  First wpnt: x={gb_wpnts[0].x_m:.2f}, y={gb_wpnts[0].y_m:.2f}, s={gb_wpnts[0].s_m:.2f}, d={gb_wpnts[0].d_m:.2f}")
print(f"             d_left={gb_wpnts[0].d_left:.2f}, d_right={gb_wpnts[0].d_right:.2f}")
print(f"  Last wpnt:  x={gb_wpnts[-1].x_m:.2f}, y={gb_wpnts[-1].y_m:.2f}, s={gb_wpnts[-1].s_m:.2f}, d={gb_wpnts[-1].d_m:.2f}")
print(f"             d_left={gb_wpnts[-1].d_left:.2f}, d_right={gb_wpnts[-1].d_right:.2f}")
print()

# Extract waypoint data
wpnt_x = np.array([w.x_m for w in gb_wpnts])
wpnt_y = np.array([w.y_m for w in gb_wpnts])
wpnt_s = np.array([w.s_m for w in gb_wpnts])
wpnt_d_left = np.array([w.d_left for w in gb_wpnts])
wpnt_d_right = np.array([w.d_right for w in gb_wpnts])

print("WAYPOINT STATISTICS:")
print(f"  x range: [{wpnt_x.min():.2f}, {wpnt_x.max():.2f}]")
print(f"  y range: [{wpnt_y.min():.2f}, {wpnt_y.max():.2f}]")
print(f"  s range: [{wpnt_s.min():.2f}, {wpnt_s.max():.2f}]")
print(f"  d_left range: [{wpnt_d_left.min():.2f}, {wpnt_d_left.max():.2f}]")
print(f"  d_right range: [{wpnt_d_right.min():.2f}, {wpnt_d_right.max():.2f}]")
print()

# Check for issues
print("POTENTIAL ISSUES:")
if np.any(wpnt_d_left == 0.0):
    print(f"  WARNING: {np.sum(wpnt_d_left == 0.0)} waypoints have d_left=0!")
if np.any(wpnt_d_right == 0.0):
    print(f"  WARNING: {np.sum(wpnt_d_right == 0.0)} waypoints have d_right=0!")
if obs.d_left == 0.0 or obs.d_right == 0.0:
    print(f"  WARNING: Obstacle has zero boundary! d_left={obs.d_left}, d_right={obs.d_right}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Waypoint path
ax = axes[0, 0]
ax.plot(wpnt_x, wpnt_y, 'b.-', linewidth=1, markersize=2, label='Waypoints')
ax.plot(cur_x, cur_y, 'go', markersize=10, label='Current position')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Waypoint Path (XY)')
ax.legend()
ax.grid(True)
ax.axis('equal')

# Plot 2: S coordinates
ax = axes[0, 1]
ax.plot(wpnt_s, 'b.-', linewidth=1, markersize=3)
ax.axhline(obs.s_center, color='r', linestyle='--', label=f'Obs s={obs.s_center:.2f}')
ax.set_xlabel('Waypoint index')
ax.set_ylabel('S coordinate (m)')
ax.set_title('S Coordinates')
ax.legend()
ax.grid(True)

# Plot 3: Boundary distances
ax = axes[1, 0]
ax.plot(wpnt_d_left, 'g.-', linewidth=1, markersize=2, label='d_left')
ax.plot(wpnt_d_right, 'r.-', linewidth=1, markersize=2, label='d_right')
ax.axhline(obs.d_left, color='g', linestyle='--', alpha=0.5, label=f'Obs d_left={obs.d_left:.2f}')
ax.axhline(obs.d_right, color='r', linestyle='--', alpha=0.5, label=f'Obs d_right={obs.d_right:.2f}')
ax.set_xlabel('Waypoint index')
ax.set_ylabel('Distance to boundary (m)')
ax.set_title('Track Boundaries')
ax.legend()
ax.grid(True)

# Plot 4: S spacing histogram
ax = axes[1, 1]
s_diff = np.diff(wpnt_s)
ax.hist(s_diff, bins=50, edgecolor='black')
ax.set_xlabel('S spacing (m)')
ax.set_ylabel('Count')
ax.set_title(f'S Spacing Distribution (mean={s_diff.mean():.4f}m)')
ax.grid(True)

plt.tight_layout()
plt.savefig('/tmp/do_spline_input_analysis.png', dpi=150)
print(f"Saved visualization to /tmp/do_spline_input_analysis.png")
plt.show()
