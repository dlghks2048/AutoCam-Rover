#!/usr/bin/env python3
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool, Trigger


class CameraObstacleAvoidance(Node):
    """Map-free classroom driving using only the depth camera."""

    def __init__(self):
        super().__init__('camera_obstacle_avoidance')
        self.declare_parameter('depth_topic', '/depth_cam/depth/image_raw')
        self.declare_parameter('cmd_vel_topic', '/controller/cmd_vel')
        self.declare_parameter('start_on_launch', True)
        self.declare_parameter('linear_speed', 0.11)
        self.declare_parameter('turn_speed', 0.45)
        self.declare_parameter('reverse_speed', -0.06)
        self.declare_parameter('obstacle_distance_m', 0.65)
        self.declare_parameter('hard_stop_distance_m', 0.38)
        self.declare_parameter('min_valid_depth_m', 0.12)
        self.declare_parameter('max_valid_depth_m', 4.0)
        self.declare_parameter('blocked_ratio_threshold', 0.22)
        self.declare_parameter('clear_path_depth_m', 1.15)
        self.declare_parameter('stale_depth_timeout_sec', 0.7)
        self.declare_parameter('roi_top_ratio', 0.42)
        self.declare_parameter('roi_bottom_ratio', 0.92)
        self.declare_parameter('roi_left_ratio', 0.08)
        self.declare_parameter('roi_right_ratio', 0.92)
        self.declare_parameter('log_interval_sec', 1.0)

        self.is_running = bool(self.get_parameter('start_on_launch').value)
        self.last_depth_time = 0.0
        self.last_log_time = 0.0

        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            1,
        )
        self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self.depth_callback,
            1,
        )
        self.create_service(SetBool, '~/set_running', self.set_running_callback)
        self.create_service(Trigger, '~/stop', self.stop_callback)
        self.create_timer(0.1, self.safety_timer)

        self.stop_robot()
        self.get_logger().info('camera_obstacle_avoidance started')

    def set_running_callback(self, request, response):
        self.is_running = bool(request.data)
        if not self.is_running:
            self.stop_robot()
        response.success = True
        response.message = 'running' if self.is_running else 'stopped'
        return response

    def stop_callback(self, request, response):
        self.is_running = False
        self.stop_robot()
        response.success = True
        response.message = 'stopped'
        return response

    def depth_callback(self, msg):
        if not self.is_running:
            return

        depth = self.depth_image_to_meters(msg)
        if depth is None:
            self.stop_robot()
            return

        roi = self.crop_roi(depth)
        twist, metrics = self.decide_command(roi)
        self.last_depth_time = time.time()
        self.cmd_pub.publish(twist)
        self.log_metrics(metrics)

    def depth_image_to_meters(self, msg):
        encoding = msg.encoding.upper()
        if encoding in ('16UC1', 'MONO16'):
            dtype = np.uint16
            scale = 0.001
        elif encoding == '32FC1':
            dtype = np.float32
            scale = 1.0
        else:
            self.get_logger().warning('Unsupported depth encoding: %s' % msg.encoding)
            return None

        item_size = np.dtype(dtype).itemsize
        row_values = int(msg.step / item_size)
        try:
            depth = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_values)
        except ValueError as exc:
            self.get_logger().warning('Invalid depth image shape: %s' % exc)
            return None

        depth = depth[:, :msg.width].astype(np.float32) * scale
        depth[~np.isfinite(depth)] = np.nan
        depth[depth <= 0.0] = np.nan
        return depth

    def crop_roi(self, depth):
        h, w = depth.shape[:2]
        y1 = int(self.ratio_param('roi_top_ratio') * h)
        y2 = int(self.ratio_param('roi_bottom_ratio') * h)
        x1 = int(self.ratio_param('roi_left_ratio') * w)
        x2 = int(self.ratio_param('roi_right_ratio') * w)
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        return depth[y1:y2, x1:x2]

    def ratio_param(self, name):
        return max(0.0, min(1.0, float(self.get_parameter(name).value)))

    def decide_command(self, roi):
        min_valid = float(self.get_parameter('min_valid_depth_m').value)
        max_valid = float(self.get_parameter('max_valid_depth_m').value)
        obstacle_distance = float(self.get_parameter('obstacle_distance_m').value)
        hard_stop_distance = float(self.get_parameter('hard_stop_distance_m').value)
        blocked_threshold = float(self.get_parameter('blocked_ratio_threshold').value)

        valid = roi[(roi >= min_valid) & (roi <= max_valid)]
        if valid.size < 80:
            return self.turn_in_place(1.0), self.metrics('search', math.nan, math.nan, math.nan, 0.0)

        left_roi, center_roi, right_roi = np.array_split(roi, 3, axis=1)
        left = self.sector_clearance(left_roi, min_valid, max_valid)
        center = self.sector_clearance(center_roi, min_valid, max_valid)
        right = self.sector_clearance(right_roi, min_valid, max_valid)
        blocked_ratio = float(np.count_nonzero(valid < obstacle_distance) / valid.size)

        front_blocked = center < obstacle_distance or blocked_ratio > blocked_threshold
        emergency = center < hard_stop_distance

        if emergency:
            direction = self.best_turn_direction(left, right)
            return self.reverse_and_turn(direction), self.metrics('emergency', left, center, right, blocked_ratio)
        if front_blocked:
            direction = self.best_turn_direction(left, right)
            return self.turn_in_place(direction), self.metrics('avoid', left, center, right, blocked_ratio)
        return self.forward_with_wall_bias(left, right), self.metrics('forward', left, center, right, blocked_ratio)

    def sector_clearance(self, sector, min_valid, max_valid):
        values = sector[(sector >= min_valid) & (sector <= max_valid)]
        if values.size == 0:
            return 0.0
        return float(np.percentile(values, 25))

    def best_turn_direction(self, left_clearance, right_clearance):
        return 1.0 if left_clearance >= right_clearance else -1.0

    def forward_with_wall_bias(self, left_clearance, right_clearance):
        twist = Twist()
        twist.linear.x = float(self.get_parameter('linear_speed').value)
        clear_path = max(0.1, float(self.get_parameter('clear_path_depth_m').value))
        turn_speed = float(self.get_parameter('turn_speed').value)
        bias = (left_clearance - right_clearance) / clear_path
        twist.angular.z = max(-0.35, min(0.35, bias)) * turn_speed
        return twist

    def turn_in_place(self, direction):
        twist = Twist()
        twist.angular.z = float(direction) * float(self.get_parameter('turn_speed').value)
        return twist

    def reverse_and_turn(self, direction):
        twist = self.turn_in_place(direction)
        twist.linear.x = float(self.get_parameter('reverse_speed').value)
        return twist

    def safety_timer(self):
        if not self.is_running or self.last_depth_time == 0.0:
            return
        timeout = float(self.get_parameter('stale_depth_timeout_sec').value)
        if time.time() - self.last_depth_time > timeout:
            self.stop_robot()

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def metrics(self, state, left, center, right, blocked_ratio):
        return {
            'state': state,
            'left': left,
            'center': center,
            'right': right,
            'blocked_ratio': blocked_ratio,
        }

    def log_metrics(self, metrics):
        now = time.time()
        interval = float(self.get_parameter('log_interval_sec').value)
        if now - self.last_log_time < interval:
            return
        self.last_log_time = now
        self.get_logger().info(
            'state=%s left=%.2f center=%.2f right=%.2f blocked=%.2f'
            % (
                metrics['state'],
                metrics['left'],
                metrics['center'],
                metrics['right'],
                metrics['blocked_ratio'],
            )
        )


def main():
    rclpy.init()
    node = CameraObstacleAvoidance()
    try:
        rclpy.spin(node)
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
