#!/usr/bin/env python3
import json
import os
import threading
from datetime import datetime

import requests
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class SnapshotUploaderNode(Node):
    def __init__(self):
        super().__init__('snapshot_uploader_node')
        self.declare_parameter('server_url', 'http://192.168.0.10:5000')
        self.declare_parameter('status_topic', '/final_compilation/scan_status')
        self.declare_parameter('odom_topic', '/odom_raw')
        self.declare_parameter('request_timeout_sec', 5.0)
        self.declare_parameter('enabled', True)

        self.pose_lock = threading.Lock()
        self.location_x = 0.0
        self.location_y = 0.0
        self.last_odom_stamp = None

        self.session = requests.Session()
        self.create_subscription(
            String,
            self.get_parameter('status_topic').value,
            self.status_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self.odom_callback,
            10,
        )
        self.get_logger().info(
            'snapshot uploader started: %s/upload'
            % self.get_parameter('server_url').value.rstrip('/')
        )

    def odom_callback(self, msg):
        with self.pose_lock:
            self.location_x = float(msg.pose.pose.position.x)
            self.location_y = float(msg.pose.pose.position.y)
            self.last_odom_stamp = self.get_clock().now().to_msg()

    def status_callback(self, msg):
        if not bool(self.get_parameter('enabled').value):
            return
        if not msg.data.startswith('snapshot:'):
            return
        try:
            payload = json.loads(msg.data[len('snapshot:'):])
        except json.JSONDecodeError as exc:
            self.get_logger().warning('invalid snapshot status json: %s' % exc)
            return
        threading.Thread(target=self.upload_snapshot, args=(payload,), daemon=True).start()

    def upload_snapshot(self, payload):
        path = payload.get('path', '')
        event_type = payload.get('event_type') or 'motion'
        if not path or not os.path.isfile(path):
            self.get_logger().warning('snapshot file not found: %s' % path)
            return

        with self.pose_lock:
            location_x = self.location_x
            location_y = self.location_y

        base_url = self.get_parameter('server_url').value.rstrip('/')
        upload_url = '%s/upload' % base_url
        timeout = float(self.get_parameter('request_timeout_sec').value)
        captured_at = payload.get('captured_at') or datetime.now().isoformat(timespec='milliseconds')
        metadata = {
            'source_node': self.get_name(),
            'snapshot_path': path,
            'scan_yaw': payload.get('scan_yaw'),
            'camera_yaw': payload.get('camera_yaw'),
            'camera_z': payload.get('camera_z'),
        }

        try:
            with open(path, 'rb') as image_file:
                files = {
                    'image': (
                        os.path.basename(path),
                        image_file,
                        'image/jpeg',
                    ),
                }
                data = {
                    'captured_at': captured_at,
                    'event_type': event_type,
                    'location_x': str(location_x),
                    'location_y': str(location_y),
                    'metadata_json': json.dumps(metadata, ensure_ascii=False),
                }
                response = self.session.post(upload_url, files=files, data=data, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                self.get_logger().info(
                    'uploaded snapshot #%s event=%s location=(%.3f, %.3f)'
                    % (result.get('event_id', '?'), event_type, location_x, location_y)
                )
            else:
                self.get_logger().warning(
                    'upload failed: status=%d body=%s'
                    % (response.status_code, response.text[:200])
                )
        except requests.RequestException as exc:
            self.get_logger().warning('upload request failed: %s' % exc)
        except OSError as exc:
            self.get_logger().warning('snapshot read failed: %s' % exc)

    def destroy_node(self):
        self.session.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = SnapshotUploaderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
