import json
import matplotlib.pyplot as plt

# Load JSON file
json_path = '/home/hj/unicorn_ws/UNICORN/stack_master/maps/iccas/smart_global_waypoints.json'

with open(json_path, 'r') as f:
    file_content = json.load(f)

# Extract waypoint data
wpnts_data = file_content['global_traj_wpnts_iqp']['wpnts']

# Check if waypoint data exists
if not wpnts_data:
    print("Error: No waypoints found in the data.")
else:
    # Create lists for plotting
    ids = [wp['id'] for wp in wpnts_data]
    x_coords = [wp['x_m'] for wp in wpnts_data]
    y_coords = [wp['y_m'] for wp in wpnts_data]
    s_coords = [wp['s_m'] for wp in wpnts_data]

    print(f"Total waypoints: {len(wpnts_data)}")
    print(f"First waypoint: id={ids[0]}, s={s_coords[0]:.2f}, xy=({x_coords[0]:.2f}, {y_coords[0]:.2f})")
    print(f"Last waypoint: id={ids[-1]}, s={s_coords[-1]:.2f}, xy=({x_coords[-1]:.2f}, {y_coords[-1]:.2f})")

    # Create the plot
    plt.figure(figsize=(14, 14))
    plt.plot(x_coords, y_coords, marker='.', linestyle='-', markersize=3, color='blue', linewidth=1, label='Fixed Path')

    # Annotate points with their ID (every 20th point + first/last)
    for i in range(len(ids)):
        if i == 0 or i == len(ids) - 1 or i % 20 == 0:
            plt.annotate(
                f"id={ids[i]}\ns={s_coords[i]:.1f}",
                (x_coords[i], y_coords[i]),
                textcoords="offset points",
                xytext=(5, 5),
                ha='left',
                fontsize=7,
                color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
            )

    # Mark start point
    plt.plot(x_coords[0], y_coords[0], 'go', markersize=12, label='Start (id=0)', zorder=5)

    # Mark end point
    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=12, label='End (id={})'.format(ids[-1]), zorder=5)

    # Add labels, title, grid
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.title("Fixed Path Waypoints (smart_global_waypoints.json)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.legend()

    # Save the plot
    output_path = '/home/hj/unicorn_ws/UNICORN/stack_master/scripts/fixed_path_waypoints.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved successfully as {output_path}")

    plt.show()