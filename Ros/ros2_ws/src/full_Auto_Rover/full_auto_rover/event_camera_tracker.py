#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from interfaces.msg import Point2D
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition

from full_auto_rover.utils import clamp


class EventCameraTracker(Node):
    def __init__(self):
        super().__init__('event_camera_tracker')
        self.declare_parameter('event_center_topic', '/full_auto_rover/event_center')
        self.declare_parameter('aligned_topic', '/full_auto_rover/event_aligned')
        self.declare_parameter('servo_topic', '/servo_controller')
        self.declare_parameter('yaw_servo_id', 1)
        self.declare_parameter('pitch_servo_id', 4)
        self.declare_parameter('initial_yaw', 500)
        self.declare_parameter('initial_pitch', 150)
        self.declare_parameter('min_yaw', 0)
        self.declare_parameter('max_yaw', 1000)
        self.declare_parameter('min_pitch', 100)
        self.declare_parameter('max_pitch', 720)
        self.declare_parameter('center_tolerance_px', 25)
        self.declare_parameter('yaw_gain', 0.08)
        self.declare_parameter('pitch_gain', 0.08)

        self.yaw = int(self.get_parameter('initial_yaw').value)
        self.pitch = int(self.get_parameter('initial_pitch').value)

        servo_topic = self.get_parameter('servo_topic').value
        center_topic = self.get_parameter('event_center_topic').value
        aligned_topic = self.get_parameter('aligned_topic').value

        self.servo_pub = self.create_publisher(ServosPosition, servo_topic, 1)
        self.aligned_pub = self.create_publisher(Bool, aligned_topic, 1)
        self.create_subscription(Point2D, center_topic, self.center_callback, 1)

        self.publish_servo(1.0)
        self.get_logger().info('event_camera_tracker started')

    def center_callback(self, msg):
        target_x = msg.width / 2.0
        target_y = msg.height / 2.0
        error_x = msg.x - target_x
        error_y = msg.y - target_y
        tolerance = float(self.get_parameter('center_tolerance_px').value)

        aligned = abs(error_x) <= tolerance and abs(error_y) <= tolerance
        if not aligned:
            yaw_gain = float(self.get_parameter('yaw_gain').value)
            pitch_gain = float(self.get_parameter('pitch_gain').value)
            self.yaw = int(clamp(self.yaw + error_x * yaw_gain,
                                 int(self.get_parameter('min_yaw').value),
                                 int(self.get_parameter('max_yaw').value)))
            self.pitch = int(clamp(self.pitch + error_y * pitch_gain,
                                   int(self.get_parameter('min_pitch').value),
                                   int(self.get_parameter('max_pitch').value)))
            self.publish_servo(0.05)

        out = Bool()
        out.data = aligned
        self.aligned_pub.publish(out)

    def publish_servo(self, duration):
        yaw_id = int(self.get_parameter('yaw_servo_id').value)
        pitch_id = int(self.get_parameter('pitch_servo_id').value)
        set_servo_position(self.servo_pub, duration, ((yaw_id, self.yaw), (pitch_id, self.pitch)))


def main():
    rclpy.init()
    node = EventCameraTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
