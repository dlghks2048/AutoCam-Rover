#!/usr/bin/env python3
import threading
import time

import rclpy
from interfaces.msg import Point2D
from kinematics import transform
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import SetRobotPose
from rclpy.node import Node
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition
from std_msgs.msg import Bool

from full_auto_rover.utils import clamp


class EventCameraTracker(Node):
    def __init__(self):
        super().__init__('event_camera_tracker')
        self.declare_parameter('event_center_topic', '/full_auto_rover/event_center')
        self.declare_parameter('aligned_topic', '/full_auto_rover/event_aligned')
        self.declare_parameter('servo_topic', '/servo_controller')
        self.declare_parameter('kinematics_service', '/kinematics/set_pose_target')
        self.declare_parameter('initial_yaw', 500)
        self.declare_parameter('initial_z', 0.41)
        self.declare_parameter('min_yaw', 200)
        self.declare_parameter('max_yaw', 800)
        self.declare_parameter('min_z', 0.36)
        self.declare_parameter('max_z', 0.46)
        self.declare_parameter('x_offset', 0.0)
        self.declare_parameter('y_offset', 0.0)
        self.declare_parameter('target_pitch', 0.0)
        self.declare_parameter('pitch_range', [-180.0, 180.0])
        self.declare_parameter('pitch_resolution', 1.0)
        self.declare_parameter('center_tolerance_px', 25)
        self.declare_parameter('yaw_gain', 0.04)
        self.declare_parameter('z_gain', 0.00005)
        self.declare_parameter('control_period_sec', 0.02)
        self.declare_parameter('max_yaw_step', 12)
        self.declare_parameter('max_z_step', 0.006)
        self.declare_parameter('align_stable_count', 2)
        self.declare_parameter('target_timeout_sec', 8.0)
        self.declare_parameter('scan_enabled', False)
        self.declare_parameter('scan_interval_sec', 0.02)
        self.declare_parameter('scan_yaw_positions', [250, 375, 500, 625, 750])
        self.declare_parameter('scan_z', 0.41)
        self.declare_parameter('scan_step', 8)
        self.declare_parameter('publish_servo_commands', True)

        self.yaw = int(self.get_parameter('initial_yaw').value)
        self.z = float(self.get_parameter('initial_z').value)
        self.x = transform.link3 + transform.tool_link + float(self.get_parameter('x_offset').value)
        self.y = float(self.get_parameter('y_offset').value)
        self.target = None
        self.last_target_time = 0.0
        self.aligned_count = 0
        self.last_aligned = False
        self.scan_index = 0
        self.scan_direction = 1
        self.last_scan_time = 0.0

        self.servo_pub = self.create_publisher(
            ServosPosition,
            self.get_parameter('servo_topic').value,
            1,
        )
        self.aligned_pub = self.create_publisher(
            Bool,
            self.get_parameter('aligned_topic').value,
            1,
        )
        self.kinematics_client = self.create_client(
            SetRobotPose,
            self.get_parameter('kinematics_service').value,
        )
        self.wait_for_kinematics()

        self.create_subscription(
            Point2D,
            self.get_parameter('event_center_topic').value,
            self.center_callback,
            1,
        )
        self.worker = threading.Thread(target=self.control_loop, daemon=True)
        self.worker.start()
        self.get_logger().info('event_camera_tracker started')

    def wait_for_kinematics(self):
        while rclpy.ok() and not self.kinematics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for kinematics service')

    def center_callback(self, msg):
        self.target = msg
        self.last_target_time = time.time()
        self.last_aligned = False

    def control_loop(self):
        while rclpy.ok():
            start_time = time.time()
            if self.has_recent_target():
                self.track_target()
            else:
                self.aligned_count = 0
                self.last_aligned = False
                self.scan()

            period = float(self.get_parameter('control_period_sec').value)
            elapsed = time.time() - start_time
            time.sleep(max(0.001, period - elapsed))

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
        self.update_pose(error_x, error_y)
        self.publish_pose(float(self.get_parameter('control_period_sec').value))

    def update_pose(self, error_x, error_y):
        yaw_delta = self.limited_step(
            error_x * float(self.get_parameter('yaw_gain').value),
            int(self.get_parameter('max_yaw_step').value),
        )
        z_delta = self.limited_float_step(
            error_y * float(self.get_parameter('z_gain').value),
            float(self.get_parameter('max_z_step').value),
        )
        self.yaw = int(clamp(
            self.yaw + yaw_delta,
            int(self.get_parameter('min_yaw').value),
            int(self.get_parameter('max_yaw').value),
        ))
        self.z = float(clamp(
            self.z + z_delta,
            float(self.get_parameter('min_z').value),
            float(self.get_parameter('max_z').value),
        ))

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
        target_z = float(self.get_parameter('scan_z').value)
        moved = False

        next_yaw = self.move_toward(self.yaw, target_yaw, int(self.get_parameter('scan_step').value))
        if next_yaw != self.yaw:
            self.yaw = next_yaw
            moved = True

        next_z = self.move_float_toward(self.z, target_z, float(self.get_parameter('max_z_step').value))
        if next_z != self.z:
            self.z = next_z
            moved = True

        if moved:
            self.publish_pose(float(self.get_parameter('scan_interval_sec').value))
            return

        if self.scan_index == len(scan_positions) - 1:
            self.scan_direction = -1
        elif self.scan_index == 0:
            self.scan_direction = 1
        self.scan_index += self.scan_direction

    def publish_pose(self, duration):
        if not bool(self.get_parameter('publish_servo_commands').value):
            return

        request = set_pose_target(
            [self.x, self.y, self.z],
            float(self.get_parameter('target_pitch').value),
            [float(value) for value in self.get_parameter('pitch_range').value],
            float(self.get_parameter('pitch_resolution').value),
        )
        future = self.kinematics_client.call_async(request)
        deadline = time.time() + 0.2
        while rclpy.ok() and not future.done() and time.time() < deadline:
            time.sleep(0.005)
        if not future.done() or future.result() is None:
            self.get_logger().warn('kinematics request timed out')
            return

        result = future.result()
        if not result.success or not result.pulse:
            self.get_logger().warn('kinematics has no valid solution')
            return

        servo_data = result.pulse
        set_servo_position(
            self.servo_pub,
            duration,
            ((10, 500), (5, 500), (4, servo_data[3]), (3, servo_data[2]), (2, servo_data[1]), (1, int(self.yaw))),
        )

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

    def limited_float_step(self, value, limit):
        return float(clamp(value, -limit, limit))

    def move_toward(self, current, target, step):
        if abs(target - current) <= step:
            return target
        return current + step if target > current else current - step

    def move_float_toward(self, current, target, step):
        if abs(target - current) <= step:
            return target
        return current + step if target > current else current - step


def main():
    rclpy.init()
    node = EventCameraTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
