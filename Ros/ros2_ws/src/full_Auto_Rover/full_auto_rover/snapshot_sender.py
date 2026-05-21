#!/usr/bin/env python3
import os
import time
import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class SnapshotSender(Node):
    def __init__(self):
        super().__init__('snapshot_sender')
        self.declare_parameter('image_topic', '/depth_cam/rgb/image_raw')
        self.declare_parameter('aligned_topic', '/full_auto_rover/event_aligned')
        self.declare_parameter('save_dir', '/tmp/full_auto_rover')
        self.declare_parameter('cooldown_sec', 3.0)

        self.latest_image = None
        self.last_save_time = 0.0
        os.makedirs(self.get_parameter('save_dir').value, exist_ok=True)

        self.create_subscription(Image, self.get_parameter('image_topic').value, self.image_callback, 1)
        self.create_subscription(Bool, self.get_parameter('aligned_topic').value, self.aligned_callback, 1)
        self.get_logger().info('snapshot_sender started')

    def image_callback(self, msg):
        self.latest_image = msg

    def aligned_callback(self, msg):
        if not msg.data or self.latest_image is None:
            return
        now = time.time()
        cooldown = float(self.get_parameter('cooldown_sec').value)
        if now - self.last_save_time < cooldown:
            return
        self.last_save_time = now
        self.save_snapshot(self.latest_image)

    def save_snapshot(self, msg):
        image = np.ndarray(shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        filename = 'event_%Y%m%d_%H%M%S.jpg'
        path = os.path.join(self.get_parameter('save_dir').value, time.strftime(filename))
        cv2.imwrite(path, bgr)
        self.get_logger().info('saved snapshot: %s' % path)


def main():
    rclpy.init()
    node = SnapshotSender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
