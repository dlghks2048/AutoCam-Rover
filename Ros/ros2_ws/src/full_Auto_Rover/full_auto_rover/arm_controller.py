#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition
from std_msgs.msg import String


class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller')
        self.declare_parameter('arm_command_topic', '/full_auto_rover/arm_command')
        self.declare_parameter('arm_status_topic', '/full_auto_rover/arm_status')
        self.declare_parameter('servo_topic', '/servo_controller')
        self.declare_parameter('move_duration_sec', 0.8)
        self.declare_parameter('home_pose', ['10:300', '5:500', '4:210', '3:40', '2:665', '1:500'])
        self.declare_parameter('camera_ready_pose', ['10:300', '5:500', '4:210', '3:40', '2:665', '1:500'])
        self.declare_parameter('event_focus_pose', ['10:300', '5:500', '4:190', '3:70', '2:690', '1:500'])
        self.declare_parameter('photo_pose', ['10:300', '5:500', '4:180', '3:80', '2:700', '1:500'])

        self.servo_pub = self.create_publisher(
            ServosPosition,
            self.get_parameter('servo_topic').value,
            1,
        )
        self.status_pub = self.create_publisher(
            String,
            self.get_parameter('arm_status_topic').value,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('arm_command_topic').value,
            self.command_callback,
            10,
        )
        self.apply_pose('home_pose')
        self.get_logger().info('arm_controller started')

    def command_callback(self, msg):
        command = msg.data.strip().lower()
        pose_name = self.pose_name_for_command(command)
        if pose_name is None:
            self.publish_status('unknown_command:%s' % command)
            return
        self.apply_pose(pose_name)
        self.publish_status('done:%s' % command)

    def pose_name_for_command(self, command):
        if command in ('home', 'reset'):
            return 'home_pose'
        if command in ('camera_ready', 'ready', 'scan'):
            return 'camera_ready_pose'
        if command in ('event_focus', 'focus', 'track'):
            return 'event_focus_pose'
        if command in ('photo_pose', 'photo', 'capture'):
            return 'photo_pose'
        return None

    def apply_pose(self, pose_name):
        positions = self.parse_pose(self.get_parameter(pose_name).value)
        duration = float(self.get_parameter('move_duration_sec').value)
        set_servo_position(self.servo_pub, duration, positions)

    def parse_pose(self, raw_pose):
        positions = []
        for item in raw_pose:
            servo_id, position = str(item).split(':', 1)
            positions.append((int(servo_id), int(position)))
        return tuple(positions)

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main():
    rclpy.init()
    node = ArmController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
