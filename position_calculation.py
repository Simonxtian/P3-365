import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
import math

class PlankPositionCalculator(Node):
    def __init__(self):
        super().__init__('plank_position_calculator')

        # Subscriptions
        self.rotation_sub = self.create_subscription(
            Float32, 'rotation_pot', self.rotation_callback, 10)
        
        self.ground_length_sub = self.create_subscription(
            Float32, 'ground_length', self.ground_length_callback, 10)

        # Publisher for (x, y) position
        self.position_pub = self.create_publisher(Float32MultiArray, 'plank_position', 10)

        # Store latest values
        self.rotation_angle = None  # Rotation in degrees
        self.ground_length = None  # Ground length in meters

    def rotation_callback(self, msg):
        self.rotation_angle = msg.data  # Store latest rotation value
        self.calculate_and_publish()

    def ground_length_callback(self, msg):
        self.ground_length = msg.data  # Store latest ground length
        self.calculate_and_publish()

    def calculate_and_publish(self):
        if self.rotation_angle is None or self.ground_length is None:
            return  # Wait until both values are received

        # Convert rotation angle to radians
        rotation_radians = math.radians(self.rotation_angle)

        # Compute x and y position
        x = self.ground_length * math.cos(rotation_radians)
        y = self.ground_length * math.sin(rotation_radians)

        # Publish (x, y) as a Float32MultiArray
        position_msg = Float32MultiArray()
        position_msg.data = [x, y]

        self.position_pub.publish(position_msg)

        self.get_logger().info(f'Rotation: {self.rotation_angle}°, Ground Length: {self.ground_length:.2f}m -> X: {x:.2f}, Y: {y:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = PlankPositionCalculator()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
