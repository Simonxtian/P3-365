import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32MultiArray

class AreWeMoving(Node):
    def __init__(self):
        super().__init__('are_we_moving')

        # Subscribe to linear topic
        self.subscription = self.create_subscription(
            Float32, 'linear_pot', self.linear_callback, 10)

        # Publisher for direction
        self.publisher = self.create_publisher(Int32MultiArray, 'forwards_backwards', 10)

    def linear_callback(self, msg):
        linear_val = msg.data  # Get linear potmeter value

        direction = Int32MultiArray()

        if 6 >= linear_val >= 4:
            forward = 0
            backward = 0

        elif linear_val > 6:
            forward = 0
            backward = 1

        elif linear_val < 4:
            forward = 1
            backward = 0

        # Add the integer values to the data field of the Int32MultiArray
        direction.data = [forward, backward]

        # Publish the result
        self.publisher.publish(direction)

        self.get_logger().info(f'Forwards: {forward}, Backward: {backward}')

def main(args=None):
    rclpy.init(args=args)
    node = AreWeMoving()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
