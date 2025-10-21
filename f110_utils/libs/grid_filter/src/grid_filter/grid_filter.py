import rospy
import cv2
import numpy as np
import yaml
from nav_msgs.msg import OccupancyGrid


class GridFilter:
    def __init__(self, map_topic=None, debug=False):
        self.resolution = None  # Map resolution (m/pixel)
        self.origin = None  # Map origin (x, y)
        self.map_data = None  # Store map data
        self.image = None  # Convert OccupancyGrid to OpenCV image
        self.eroded_image = None  # Processed image with erosion
        self.kernel_size = 3  # Default erosion kernel size
        self.debug = debug
        self.map_topic = map_topic  # User-specified map topic

        # Ensure the topic is set before subscribing
        if self.map_topic:
            self.subscribe_to_map(self.map_topic)
        else:
            rospy.logwarn("No map topic provided. Waiting for user input.")

    def subscribe_to_map(self, map_topic):
        """Subscribe to the given map topic."""
        rospy.loginfo(f"Subscribing to map topic: {map_topic}")
        rospy.Subscriber(map_topic, OccupancyGrid, self.map_callback)



    def map_callback(self, msg):
        """Process the received map data."""
        if self.image is None:
            rospy.logwarn("Received map data")

            self.resolution = msg.info.resolution
            self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)

            width, height = msg.info.width, msg.info.height
            image = np.array(msg.data, dtype=np.int8).reshape((height, width))

            # Convert to 0-255 scale (255: obstacle, 0: free space)
            self.image = np.where(image == 100, 0, 255).astype(np.uint8)

            # Flip image to correct ROS coordinate system
            # self.image = cv2.flip(self.image, 0)

            if self.debug:
                # Display the modified map
                cv2.imshow("Original Map with Points", self.image)
                cv2.waitKey(0)

            self.update_image()
            rospy.logwarn("Map image initialized.")

    # ===== HJ ADDED: Load map from file (for modified maps) =====
    def load_from_file(self, png_path, yaml_path):
        """
        Load map from PNG and YAML files directly (without ROS topic).
        Useful for loading modified maps (e.g., with obstacles added).

        Args:
            png_path: Path to PNG image file
            yaml_path: Path to YAML metadata file
        """
        try:
            # Load YAML metadata
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            self.resolution = yaml_data['resolution']
            self.origin = (yaml_data['origin'][0], yaml_data['origin'][1])

            # Load PNG image
            loaded_image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if loaded_image is None:
                rospy.logerr(f"Failed to load image from {png_path}")
                return False

            # Convert to 0-255 scale (255: free, 0: occupied)
            # Assuming PNG is already in correct format (white=free, black=occupied)
            self.image = loaded_image

            if self.debug:
                cv2.imshow("Loaded Map from File", self.image)
                cv2.waitKey(0)

            self.update_image()
            rospy.loginfo(f"Map loaded from file: {png_path}")
            return True

        except Exception as e:
            rospy.logerr(f"Failed to load map from file: {e}")
            import traceback
            traceback.print_exc()
            return False
    # ===== HJ ADDED END =====

    def set_erosion_kernel_size(self, size):
        """Update kernel size and apply erosion."""
        self.kernel_size = size
        self.update_image()

    def update_image(self):
        """Apply erosion to the map."""
        if self.image is None:
            rospy.logwarn("Map image not initialized.")
            return

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        self.eroded_image = cv2.erode(self.image, kernel)

        if self.debug:
            cv2.imshow("Eroded Map", self.eroded_image)
            cv2.waitKey(0)

    def world_to_pixel(self, x, y):
        """Convert world coordinates to pixel coordinates."""
        px = int((x - self.origin[0]) / self.resolution)
        py = int((y - self.origin[1]) / self.resolution)
        return px, py

    def is_point_inside(self, x, y):
        """Check if a world coordinate is inside an obstacle."""
        if self.eroded_image is None:
            # rospy.logwarn("Eroded map not available.")
            return False

        px, py = self.world_to_pixel(x, y)

        if px < 0 or py < 0 or px >= self.eroded_image.shape[1] or py >= self.eroded_image.shape[0]:
            return False

        return self.eroded_image[py, px] == 255
