import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import math

class GroundLengthCalculator(Node):
    def __init__(self):
        super().__init__('ground_length_calculator')

        # Subscribe to pitch and linear topics
        self.pitch_subscription = self.create_subscription(
            Float32, 'pitch_pot', self.pitch_callback, 10)
            
        self.linear_subscription = self.create_subscription(
            Float32, 'linear_pot', self.linear_callback, 10)

        # Publisher for ground length
        self.publisher = self.create_publisher(Float32, 'ground_length', 10)

        # Define the plank length
        self.plank_length = 10.0  # meters
        self.linear_value = None
        self.pitch_angle = None
    
    def linear_callback(self, msg):
        self.linear_value = msg.data
        self.calculate_and_publish()
    
    def pitch_callback(self, msg):
        self.pitch_angle = msg.data  # Get pitch angle in degrees
        self.calculate_and_publish()

    def calculate_and_publish(self):
        if self.linear_value is None or self.pitch_angle is None:
            return  # Wait until both values are received
        
        # Convert angle to radians
        pitch_radians = math.radians(self.pitch_angle)
        
        # Calculate ground length
        ground_length = (self.plank_length + self.linear_value/100.0 - 0.05) * math.cos(pitch_radians)

        # Publish the result
        self.publisher.publish(Float32(data=ground_length))

        self.get_logger().info(f'Linear: {self.linear_value} Pitch: {self.pitch_angle}°, Ground Length: {ground_length:.2f}m')

def main(args=None):
    rclpy.init(args=args)
    node = GroundLengthCalculator()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
