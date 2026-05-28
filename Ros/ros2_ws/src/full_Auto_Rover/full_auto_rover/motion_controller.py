#!/usr/bin/env python3
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String


class MotionController(Node):
    def __init__(self):
        super().__init__('motion_controller')
        self.declare_parameter('cmd_vel_topic', '/controller/cmd_vel')
        self.declare_parameter('motion_command_topic', '/full_auto_rover/motion_command')
        self.declare_parameter('event_status_topic', '/full_auto_rover/event_status')
        self.declare_parameter('linear_speed', 0.12)
        self.declare_parameter('angular_speed', 0.55)
        self.declare_parameter('command_timeout_sec', 0.7)
        self.declare_parameter('pause_on_event', True)

        self.current_command = 'stop'
        self.paused_by_event = False
        self.last_command_time = time.time()

        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            1,
        )
        self.create_subscription(
            String,
            self.get_parameter('motion_command_topic').value,
            self.command_callback,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('event_status_topic').value,
            self.event_callback,
            10,
        )
        self.create_timer(0.1, self.timer_callback)
        self.stop()
        self.get_logger().info('motion_controller started')

    def command_callback(self, msg):
        command = msg.data.strip().lower()
        self.last_command_time = time.time()

        if command in ('resume', 'continue'):
            self.paused_by_event = False
            self.current_command = 'forward'
            return

        if command in ('manual', 'pause'):
            self.paused_by_event = True
            self.stop()
            return

        self.paused_by_event = False
        self.current_command = command

    def event_callback(self, msg):
        if not bool(self.get_parameter('pause_on_event').value):
            return
        if msg.data:
            self.paused_by_event = True
            self.current_command = 'stop'
            self.stop()

    def timer_callback(self):
        timeout = float(self.get_parameter('command_timeout_sec').value)
        if time.time() - self.last_command_time > timeout:
            self.current_command = 'stop'
        if self.paused_by_event:
            self.stop()
            return
        self.cmd_pub.publish(self.twist_for_command(self.current_command))

    def twist_for_command(self, command):
        linear_speed = float(self.get_parameter('linear_speed').value)
        angular_speed = float(self.get_parameter('angular_speed').value)
        twist = Twist()

        if command in ('forward', 'go'):
            twist.linear.x = linear_speed
        elif command in ('backward', 'back'):
            twist.linear.x = -linear_speed
        elif command in ('left', 'turn_left', 'rotate_left'):
            twist.angular.z = angular_speed
        elif command in ('right', 'turn_right', 'rotate_right'):
            twist.angular.z = -angular_speed
        elif command in ('scan_left',):
            twist.angular.z = angular_speed * 0.45
        elif command in ('scan_right', 'rotate_scan'):
            twist.angular.z = -angular_speed * 0.45
        return twist

    def stop(self):
        self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = MotionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
