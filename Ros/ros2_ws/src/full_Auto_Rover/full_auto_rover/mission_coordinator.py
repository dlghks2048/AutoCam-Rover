#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


class MissionCoordinator(Node):
    def __init__(self):
        super().__init__('mission_coordinator')
        self.declare_parameter('event_status_topic', '/full_auto_rover/event_status')
        self.declare_parameter('aligned_topic', '/full_auto_rover/event_aligned')
        self.declare_parameter('motion_command_topic', '/full_auto_rover/motion_command')
        self.declare_parameter('arm_command_topic', '/full_auto_rover/arm_command')
        self.declare_parameter('capture_trigger_topic', '/full_auto_rover/capture_request')
        self.declare_parameter('resume_after_capture_sec', 2.0)
        self.declare_parameter('auto_resume', False)
        self.declare_parameter('auto_patrol', False)
        self.declare_parameter('patrol_command', 'forward')
        self.declare_parameter('use_arm_pose_commands', False)

        self.last_event = ''
        self.last_capture_time = 0.0

        self.motion_pub = self.create_publisher(
            String,
            self.get_parameter('motion_command_topic').value,
            10,
        )
        self.arm_pub = self.create_publisher(
            String,
            self.get_parameter('arm_command_topic').value,
            10,
        )
        self.capture_pub = self.create_publisher(
            String,
            self.get_parameter('capture_trigger_topic').value,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('event_status_topic').value,
            self.event_callback,
            10,
        )
        self.create_subscription(
            Bool,
            self.get_parameter('aligned_topic').value,
            self.aligned_callback,
            10,
        )
        self.create_timer(0.5, self.timer_callback)
        self.publish(self.motion_pub, 'stop')
        self.publish(self.arm_pub, 'camera_ready')
        self.get_logger().info('mission_coordinator started')

    def event_callback(self, msg):
        if not msg.data or msg.data == self.last_event:
            return
        self.last_event = msg.data
        self.publish(self.motion_pub, 'stop')
        if bool(self.get_parameter('use_arm_pose_commands').value):
            self.publish(self.arm_pub, 'event_focus')
        self.get_logger().info('event detected: %s' % msg.data)

    def aligned_callback(self, msg):
        if not msg.data:
            return
        self.last_capture_time = time.time()
        if bool(self.get_parameter('use_arm_pose_commands').value):
            self.publish(self.arm_pub, 'photo_pose')
        self.publish(self.capture_pub, self.last_event or 'event_aligned')

    def timer_callback(self):
        if bool(self.get_parameter('auto_patrol').value) and not self.last_event:
            self.publish(self.motion_pub, self.get_parameter('patrol_command').value)

        if not bool(self.get_parameter('auto_resume').value):
            return
        if self.last_capture_time <= 0:
            return
        delay = float(self.get_parameter('resume_after_capture_sec').value)
        if time.time() - self.last_capture_time >= delay:
            self.last_capture_time = 0.0
            self.last_event = ''
            if bool(self.get_parameter('use_arm_pose_commands').value):
                self.publish(self.arm_pub, 'camera_ready')
            self.publish(self.motion_pub, 'resume')

    def publish(self, publisher, value):
        msg = String()
        msg.data = value
        publisher.publish(msg)


def main():
    rclpy.init()
    node = MissionCoordinator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
