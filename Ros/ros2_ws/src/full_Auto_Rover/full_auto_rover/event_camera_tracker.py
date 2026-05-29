#!/usr/bin/env python3
import time

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
        self.declare_parameter('control_period_sec', 0.1)
        self.declare_parameter('max_yaw_step', 8)
        self.declare_parameter('max_pitch_step', 5)
        self.declare_parameter('align_stable_count', 5)
        self.declare_parameter('target_timeout_sec', 2.0)
        self.declare_parameter('scan_enabled', True)
        self.declare_parameter('scan_interval_sec', 1.2)
        self.declare_parameter('scan_yaw_positions', [250, 375, 500, 625, 750])
        self.declare_parameter('scan_pitch', 150)
        self.declare_parameter('scan_step', 10)

        self.yaw = int(self.get_parameter('initial_yaw').value)
        self.pitch = int(self.get_parameter('initial_pitch').value)
        self.target = None
        self.last_target_time = 0.0
        self.aligned_count = 0
        self.last_aligned = False
        self.scan_index = 0
        self.scan_direction = 1
        self.last_scan_time = 0.0

        servo_topic = self.get_parameter('servo_topic').value
        center_topic = self.get_parameter('event_center_topic').value
        aligned_topic = self.get_parameter('aligned_topic').value

        self.servo_pub = self.create_publisher(ServosPosition, servo_topic, 1)
        self.aligned_pub = self.create_publisher(Bool, aligned_topic, 1)
        self.create_subscription(Point2D, center_topic, self.center_callback, 1)
        self.create_timer(float(self.get_parameter('control_period_sec').value), self.control_callback)

        self.publish_servo(1.0)
        self.get_logger().info('event_camera_tracker started')

    def center_callback(self, msg):
        self.target = msg
        self.last_target_time = time.time()
        self.last_aligned = False

    def control_callback(self):
        if self.has_recent_target():
            self.track_target()
        else:
            self.aligned_count = 0
            self.last_aligned = False
            self.scan()

    def has_recent_target(self):
        if self.target is None:
            return False
        timeout = float(self.get_parameter('target_timeout_sec').value)
        return time.time() - self.last_target_time <= timeout

    def track_target(self):
        msg = self.target
        target_x = msg.width / 2.0
        target_y = msg.height / 2.0
        error_x = msg.x - target_x
        error_y = msg.y - target_y
        tolerance = float(self.get_parameter('center_tolerance_px').value)

        is_centered = abs(error_x) <= tolerance and abs(error_y) <= tolerance
        if is_centered:
            self.aligned_count += 1
            if self.aligned_count >= int(self.get_parameter('align_stable_count').value):
                self.publish_aligned(True)
            return

        self.aligned_count = 0
        self.publish_aligned(False)

        yaw_gain = float(self.get_parameter('yaw_gain').value)
        pitch_gain = float(self.get_parameter('pitch_gain').value)
        yaw_delta = self.limited_step(error_x * yaw_gain, int(self.get_parameter('max_yaw_step').value))
        pitch_delta = self.limited_step(error_y * pitch_gain, int(self.get_parameter('max_pitch_step').value))

        self.yaw = int(clamp(self.yaw + yaw_delta,
                             int(self.get_parameter('min_yaw').value),
                             int(self.get_parameter('max_yaw').value)))
        self.pitch = int(clamp(self.pitch + pitch_delta,
                               int(self.get_parameter('min_pitch').value),
                               int(self.get_parameter('max_pitch').value)))
        self.publish_servo(float(self.get_parameter('control_period_sec').value))

    def scan(self):
        if not bool(self.get_parameter('scan_enabled').value):
            return

        now = time.time()
        if now - self.last_scan_time < float(self.get_parameter('scan_interval_sec').value):
            return
        self.last_scan_time = now

        scan_positions = [int(value) for value in self.get_parameter('scan_yaw_positions').value]
        if not scan_positions:
            return

        target_yaw = scan_positions[self.scan_index]
        target_pitch = int(self.get_parameter('scan_pitch').value)
        step = int(self.get_parameter('scan_step').value)
        moved = False

        next_yaw = self.move_toward(self.yaw, target_yaw, step)
        if next_yaw != self.yaw:
            self.yaw = next_yaw
            moved = True

        next_pitch = self.move_toward(self.pitch, target_pitch, step)
        if next_pitch != self.pitch:
            self.pitch = next_pitch
            moved = True

        if moved:
            self.publish_servo(float(self.get_parameter('scan_interval_sec').value))
            return

        if self.scan_index == len(scan_positions) - 1:
            self.scan_direction = -1
        elif self.scan_index == 0:
            self.scan_direction = 1
        self.scan_index += self.scan_direction

    def publish_aligned(self, aligned):
        if aligned == self.last_aligned and not aligned:
            return
        self.last_aligned = aligned
        out = Bool()
        out.data = aligned
        self.aligned_pub.publish(out)
        if aligned:
            self.get_logger().info('event centered; capture allowed')

    def limited_step(self, value, limit):
        return int(clamp(value, -limit, limit))

    def move_toward(self, current, target, step):
        if abs(target - current) <= step:
            return target
        return current + step if target > current else current - step

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
