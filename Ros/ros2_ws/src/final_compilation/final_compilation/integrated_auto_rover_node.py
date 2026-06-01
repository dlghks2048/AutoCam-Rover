#!/usr/bin/env python3
import os
import sys
import threading
import time
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from kinematics import transform
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import SetRobotPose
from rclpy.node import Node
from sensor_msgs.msg import Image
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition
from std_msgs.msg import String

from final_compilation.utils import box_area, box_center, clamp, iou


def normalized_class_name(name):
    return str(name).strip().lower().replace(' ', '_')


class DangerScanNode(Node):
    def __init__(self):
        super().__init__('integrated_auto_rover_node')
        self.declare_parameter('image_topic', '/depth_cam/rgb/image_raw')
        self.declare_parameter('depth_topic', '/depth_cam/depth/image_raw')
        self.declare_parameter('servo_topic', '/servo_controller')
        self.declare_parameter('cmd_vel_topic', '/controller/cmd_vel')
        self.declare_parameter('kinematics_service', '/kinematics/set_pose_target')
        self.declare_parameter('status_topic', '/full_auto_rover/scan_status')
        self.declare_parameter('save_dir', '/home/ubuntu/ros2_ws/src/full_Auto_Rover/saved_images')

        self.declare_parameter('yolov5_dir', '/home/ubuntu/third_party_ros2/yolov5')
        self.declare_parameter('model_path', '/home/ubuntu/third_party_ros2/yolov5/yolov5n.pt')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('image_size', 320)
        self.declare_parameter('conf_threshold', 0.35)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('max_detections', 10)

        self.declare_parameter('scan_yaw_positions', [200, 286, 371, 457, 543, 629, 714, 800])
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
        self.declare_parameter('move_duration_sec', 0.45)
        self.declare_parameter('settle_sec', 0.45)
        self.declare_parameter('loop_pause_sec', 0.1)

        self.declare_parameter('center_tolerance_px', 40)
        self.declare_parameter('max_refine_steps', 3)
        self.declare_parameter('yaw_gain', 0.04)
        self.declare_parameter('z_gain', 0.00005)
        self.declare_parameter('max_yaw_step', 45)
        self.declare_parameter('max_z_step', 0.012)
        self.declare_parameter('event_pause_sec', 0.5)
        self.declare_parameter('detect_interval_sec', 0.35)
        self.declare_parameter('event_cooldown_sec', 1.5)
        self.declare_parameter('resume_delay_sec', 0.3)
        self.declare_parameter('publish_gripper_servo', False)
        self.declare_parameter('publish_wrist_servo', False)
        self.declare_parameter('gripper_servo_position', 500)
        self.declare_parameter('wrist_servo_position', 500)

        self.declare_parameter('start_navigation_on_launch', True)
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
        self.declare_parameter('navigation_log_interval_sec', 1.0)

        self.declare_parameter('min_score', 0.45)
        self.declare_parameter('near_iou_threshold', 0.05)
        self.declare_parameter('near_center_distance_px', 170)
        self.declare_parameter('person_object_margin_px', 45)
        self.declare_parameter('obstacle_center_band_ratio', 0.45)
        self.declare_parameter('obstacle_min_area_ratio', 0.08)
        self.declare_parameter('fallen_person_aspect_ratio', 1.35)
        self.declare_parameter('weapon_classes', ['knife', 'scissors', 'baseball_bat'])
        self.declare_parameter('obstacle_classes', [
            'chair', 'bench', 'couch', 'dining_table', 'backpack', 'handbag',
            'suitcase', 'bottle', 'potted_plant', 'traffic_cone'
        ])
        self.declare_parameter('fire_classes', ['fire', 'flame'])
        self.declare_parameter('smoke_classes', ['smoke'])

        self.image_lock = threading.Lock()
        self.latest_image = None
        self.latest_stamp = 0.0
        self.model = None
        self.device = None
        self.stride = 32
        self.names = {}
        self.running = True
        self.navigation_enabled = bool(self.get_parameter('start_navigation_on_launch').value)
        self.navigation_paused = False
        self.last_depth_time = 0.0
        self.last_navigation_log_time = 0.0
        self.last_event_time = 0.0
        self.yaw = 500
        self.z = float(self.get_parameter('initial_z').value)
        self.x = transform.link3 + transform.tool_link + float(self.get_parameter('x_offset').value)
        self.y = float(self.get_parameter('y_offset').value)

        os.makedirs(self.get_parameter('save_dir').value, exist_ok=True)
        self.servo_pub = self.create_publisher(ServosPosition, self.get_parameter('servo_topic').value, 1)
        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 1)
        self.status_pub = self.create_publisher(String, self.get_parameter('status_topic').value, 10)
        self.kinematics_client = self.create_client(SetRobotPose, self.get_parameter('kinematics_service').value)
        self.create_subscription(Image, self.get_parameter('image_topic').value, self.image_callback, 1)
        self.create_subscription(Image, self.get_parameter('depth_topic').value, self.depth_callback, 1)
        self.create_timer(0.1, self.navigation_safety_timer)

        self.wait_for_kinematics()
        self.load_model()
        self.worker = threading.Thread(target=self.event_monitor_loop, daemon=True)
        self.worker.start()
        self.get_logger().info('danger_scan_node started')

    def wait_for_kinematics(self):
        while rclpy.ok() and not self.kinematics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for kinematics service')

    def load_model(self):
        yolov5_dir = self.get_parameter('yolov5_dir').value
        model_path = self.get_parameter('model_path').value
        if yolov5_dir not in sys.path:
            sys.path.insert(0, yolov5_dir)

        from models.common import DetectMultiBackend
        from utils.general import check_img_size
        from utils.torch_utils import select_device

        self.device = select_device(self.get_parameter('device').value)
        self.model = DetectMultiBackend(model_path, device=self.device, dnn=False, data=None, fp16=False)
        self.stride = int(self.model.stride)
        self.image_size = check_img_size(int(self.get_parameter('image_size').value), s=self.stride)
        self.names = self.model.names
        self.get_logger().info('loaded YOLO model: %s' % model_path)

    def image_callback(self, msg):
        try:
            image = self.ros_image_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn('image decode failed: %s' % exc)
            return
        with self.image_lock:
            self.latest_image = image
            self.latest_stamp = time.time()

    def ros_image_to_bgr(self, msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        encoding = msg.encoding.lower()
        if encoding in ('rgb8', 'bgr8'):
            image = data.reshape((msg.height, msg.width, 3))
            if encoding == 'rgb8':
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image.copy()
        if encoding in ('mono8', '8uc1'):
            image = data.reshape((msg.height, msg.width))
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if encoding in ('mjpeg', 'jpeg'):
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('failed to decode mjpeg image')
            return image
        if len(data) == msg.height * msg.width * 3:
            image = data.reshape((msg.height, msg.width, 3))
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        raise ValueError('unsupported image encoding: %s' % msg.encoding)

    def latest_frame(self):
        with self.image_lock:
            if self.latest_image is None:
                return None
            return self.latest_image.copy()

    def scan_loop(self):
        while rclpy.ok() and self.latest_frame() is None:
            self.publish_status('waiting_for_image')
            time.sleep(0.2)

        while rclpy.ok() and self.running:
            scan_positions = [int(value) for value in self.get_parameter('scan_yaw_positions').value]
            if not scan_positions:
                self.get_logger().warn('scan_yaw_positions is empty')
                time.sleep(1.0)
                continue

            for scan_yaw in scan_positions:
                if not rclpy.ok() or not self.running:
                    break
                self.scan_once(scan_yaw)
                time.sleep(float(self.get_parameter('loop_pause_sec').value))

    def event_monitor_loop(self):
        while rclpy.ok() and self.latest_frame() is None:
            self.publish_status('waiting_for_image')
            time.sleep(0.2)

        while rclpy.ok() and self.running:
            image = self.latest_frame()
            if image is None:
                self.publish_status('no_image')
                time.sleep(0.2)
                continue

            detections = self.detect(image)
            event = self.find_event(detections, image.shape[1], image.shape[0])
            now = time.time()
            cooldown = float(self.get_parameter('event_cooldown_sec').value)
            if event is None or now - self.last_event_time < cooldown:
                time.sleep(float(self.get_parameter('detect_interval_sec').value))
                continue

            self.last_event_time = now
            self.handle_event(event, image)
            time.sleep(float(self.get_parameter('detect_interval_sec').value))

    def handle_event(self, event, image):
        label, box = event
        self.navigation_paused = True
        self.stop_base()
        self.publish_status('event:%s:paused' % label)
        self.get_logger().info('event found while driving: %s' % label)

        try:
            current_yaw = int(clamp(self.yaw, int(self.get_parameter('min_yaw').value), int(self.get_parameter('max_yaw').value)))
            current_z = float(clamp(self.z, float(self.get_parameter('min_z').value), float(self.get_parameter('max_z').value)))
            centered_image, centered_event = self.refine_event_view(current_yaw, current_z, image, event)
            self.save_snapshot(centered_image, centered_event[0], self.yaw)
            time.sleep(float(self.get_parameter('event_pause_sec').value))
        finally:
            self.move_camera(500, float(self.get_parameter('initial_z').value), float(self.get_parameter('move_duration_sec').value))
            time.sleep(float(self.get_parameter('resume_delay_sec').value))
            self.navigation_paused = False
            self.publish_status('navigation_resumed')

    def scan_once(self, scan_yaw):
        scan_z = float(self.get_parameter('initial_z').value)
        self.stop_base()
        self.move_camera(scan_yaw, scan_z, float(self.get_parameter('move_duration_sec').value))
        time.sleep(float(self.get_parameter('settle_sec').value))

        image = self.latest_frame()
        if image is None:
            self.publish_status('no_image')
            return

        detections = self.detect(image)
        event = self.find_event(detections, image.shape[1], image.shape[0])
        if event is None:
            self.publish_status('clear:yaw:%d' % scan_yaw)
            return

        label, box = event
        self.publish_status('event:%s:yaw:%d' % (label, scan_yaw))
        self.get_logger().info('event found at yaw %d: %s' % (scan_yaw, label))
        centered_image, centered_event = self.refine_event_view(scan_yaw, scan_z, image, event)
        self.save_snapshot(centered_image, centered_event[0], scan_yaw)
        time.sleep(float(self.get_parameter('event_pause_sec').value))
        self.move_camera(scan_yaw, scan_z, float(self.get_parameter('move_duration_sec').value))
        time.sleep(float(self.get_parameter('settle_sec').value))

    def refine_event_view(self, original_yaw, original_z, image, event):
        current_event = event
        current_image = image
        self.yaw = original_yaw
        self.z = original_z

        for _ in range(int(self.get_parameter('max_refine_steps').value)):
            center = box_center(current_event[1])
            if center is None:
                break
            width = current_image.shape[1]
            height = current_image.shape[0]
            error_x = center[0] - width / 2.0
            error_y = center[1] - height / 2.0
            tolerance = float(self.get_parameter('center_tolerance_px').value)
            if abs(error_x) <= tolerance and abs(error_y) <= tolerance:
                break

            yaw_step = clamp(
                error_x * float(self.get_parameter('yaw_gain').value),
                -float(self.get_parameter('max_yaw_step').value),
                float(self.get_parameter('max_yaw_step').value),
            )
            z_step = clamp(
                error_y * float(self.get_parameter('z_gain').value),
                -float(self.get_parameter('max_z_step').value),
                float(self.get_parameter('max_z_step').value),
            )
            self.yaw = int(clamp(
                self.yaw + yaw_step,
                int(self.get_parameter('min_yaw').value),
                int(self.get_parameter('max_yaw').value),
            ))
            self.z = float(clamp(
                self.z + z_step,
                float(self.get_parameter('min_z').value),
                float(self.get_parameter('max_z').value),
            ))
            self.move_camera(self.yaw, self.z, float(self.get_parameter('move_duration_sec').value))
            time.sleep(float(self.get_parameter('settle_sec').value))

            next_image = self.latest_frame()
            if next_image is None:
                break
            next_detections = self.detect(next_image)
            next_event = self.find_event(next_detections, next_image.shape[1], next_image.shape[0])
            if next_event is None:
                break
            current_image = next_image
            current_event = next_event

        return current_image, current_event

    def move_camera(self, yaw, z, duration):
        self.yaw = int(clamp(yaw, int(self.get_parameter('min_yaw').value), int(self.get_parameter('max_yaw').value)))
        self.z = float(clamp(z, float(self.get_parameter('min_z').value), float(self.get_parameter('max_z').value)))
        request = set_pose_target(
            [self.x, self.y, self.z],
            float(self.get_parameter('target_pitch').value),
            [float(value) for value in self.get_parameter('pitch_range').value],
            float(self.get_parameter('pitch_resolution').value),
        )
        future = self.kinematics_client.call_async(request)
        deadline = time.time() + 0.25
        while rclpy.ok() and not future.done() and time.time() < deadline:
            time.sleep(0.005)

        if future.done() and future.result() is not None and future.result().pulse:
            servo_data = future.result().pulse
            positions = []
            if bool(self.get_parameter('publish_gripper_servo').value):
                positions.append((10, int(self.get_parameter('gripper_servo_position').value)))
            if bool(self.get_parameter('publish_wrist_servo').value):
                positions.append((5, int(self.get_parameter('wrist_servo_position').value)))
            positions.extend([
                (4, servo_data[3]),
                (3, servo_data[2]),
                (2, servo_data[1]),
                (1, self.yaw),
            ])
        else:
            positions = [(1, self.yaw)]
            self.get_logger().warn('kinematics unavailable; publishing yaw only')
        set_servo_position(self.servo_pub, duration, positions)

    def detect(self, image):
        import torch
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes

        resized = letterbox(image, self.image_size, stride=self.stride, auto=True)[0]
        tensor = resized.transpose((2, 0, 1))[::-1]
        tensor = np.ascontiguousarray(tensor)
        tensor = torch.from_numpy(tensor).to(self.device).float()
        tensor /= 255.0
        if len(tensor.shape) == 3:
            tensor = tensor[None]

        prediction = self.model(tensor, augment=False, visualize=False)
        prediction = non_max_suppression(
            prediction,
            float(self.get_parameter('conf_threshold').value),
            float(self.get_parameter('iou_threshold').value),
            None,
            False,
            max_det=int(self.get_parameter('max_detections').value),
        )

        detections = []
        for det in prediction:
            if len(det) == 0:
                continue
            det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in det:
                class_id = int(cls)
                class_name = self.names[class_id] if isinstance(self.names, list) else self.names.get(class_id, str(class_id))
                box = [int(value.item()) for value in xyxy]
                detections.append({'class_name': class_name, 'score': float(conf.item()), 'box': box})
        return detections

    def find_event(self, detections, image_width, image_height):
        min_score = float(self.get_parameter('min_score').value)
        valid = [det for det in detections if det['score'] >= min_score and self.valid_box(det['box'])]
        persons = self.by_class(valid, ['person'])
        weapons = self.by_class(valid, self.get_parameter('weapon_classes').value)
        obstacles = self.by_class(valid, self.get_parameter('obstacle_classes').value)
        fires = self.by_class(valid, self.get_parameter('fire_classes').value)
        smokes = self.by_class(valid, self.get_parameter('smoke_classes').value)

        event = self.find_pair_event('fire_near_person', persons, fires)
        if event is not None:
            return event
        event = self.find_pair_event('smoke_near_person', persons, smokes)
        if event is not None:
            return event
        event = self.find_pair_event('attack_risk', persons, weapons)
        if event is not None:
            return event
        for person in persons:
            if self.is_fallen_person_candidate(person['box']):
                return 'fallen_person_candidate', person['box']
        obstacle = self.find_obstacle_ahead(obstacles, image_width, image_height)
        if obstacle is not None:
            return 'obstacle_ahead:%s' % normalized_class_name(obstacle['class_name']), obstacle['box']
        if fires:
            return 'fire', fires[0]['box']
        if smokes:
            return 'smoke', smokes[0]['box']
        if weapons:
            return 'weapon_detected:%s' % normalized_class_name(weapons[0]['class_name']), weapons[0]['box']
        return None

    def by_class(self, detections, class_names):
        wanted = set(normalized_class_name(name) for name in class_names)
        return [det for det in detections if normalized_class_name(det['class_name']) in wanted]

    def find_pair_event(self, event_type, persons, targets):
        for person in persons:
            for target in targets:
                if self.is_near(person['box'], target['box']):
                    x1 = min(person['box'][0], target['box'][0])
                    y1 = min(person['box'][1], target['box'][1])
                    x2 = max(person['box'][2], target['box'][2])
                    y2 = max(person['box'][3], target['box'][3])
                    return '%s:%s' % (event_type, normalized_class_name(target['class_name'])), [x1, y1, x2, y2]
        return None

    def find_obstacle_ahead(self, obstacles, image_width, image_height):
        center_band_ratio = float(self.get_parameter('obstacle_center_band_ratio').value)
        min_area_ratio = float(self.get_parameter('obstacle_min_area_ratio').value)
        left = image_width * (0.5 - center_band_ratio / 2.0)
        right = image_width * (0.5 + center_band_ratio / 2.0)
        best = None
        best_area = 0
        for obstacle in obstacles:
            center = box_center(obstacle['box'])
            if center is None:
                continue
            area = box_area(obstacle['box'])
            area_ratio = area / float(image_width * image_height)
            if left <= center[0] <= right and area_ratio >= min_area_ratio and area > best_area:
                best = obstacle
                best_area = area
        return best

    def is_near(self, box_a, box_b):
        if iou(box_a, box_b) >= float(self.get_parameter('near_iou_threshold').value):
            return True
        if self.expanded_intersects(box_a, box_b):
            return True
        center_a = box_center(box_a)
        center_b = box_center(box_b)
        if center_a is None or center_b is None:
            return False
        distance = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5
        return distance <= float(self.get_parameter('near_center_distance_px').value)

    def expanded_intersects(self, box_a, box_b):
        margin = int(self.get_parameter('person_object_margin_px').value)
        ax1, ay1, ax2, ay2 = box_a[:4]
        bx1, by1, bx2, by2 = box_b[:4]
        ax1, ay1, ax2, ay2 = ax1 - margin, ay1 - margin, ax2 + margin, ay2 + margin
        return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)

    def is_fallen_person_candidate(self, box):
        x1, y1, x2, y2 = box[:4]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        return (width / height) >= float(self.get_parameter('fallen_person_aspect_ratio').value)

    def valid_box(self, box):
        if len(box) < 4:
            return False
        x1, y1, x2, y2 = box[:4]
        return x2 > x1 and y2 > y1

    def save_snapshot(self, image, label, scan_yaw):
        safe_label = normalized_class_name(label).replace(':', '_')
        timestamp = time.strftime('event_%Y%m%d_%H%M%S')
        filename = '%s_%s_yaw%d.jpg' % (timestamp, safe_label, scan_yaw)
        path = os.path.join(self.get_parameter('save_dir').value, filename)
        cv2.imwrite(path, image)
        self.publish_status('saved:%s' % path)
        self.get_logger().info('saved event snapshot: %s' % path)

    def stop_base(self):
        self.cmd_pub.publish(Twist())

    def depth_callback(self, msg):
        if not self.navigation_enabled or self.navigation_paused:
            return

        depth = self.depth_image_to_meters(msg)
        if depth is None:
            self.stop_base()
            return

        roi = self.crop_navigation_roi(depth)
        twist, metrics = self.decide_navigation_command(roi)
        self.last_depth_time = time.time()
        self.cmd_pub.publish(twist)
        self.log_navigation_metrics(metrics)

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

    def crop_navigation_roi(self, depth):
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

    def decide_navigation_command(self, roi):
        min_valid = float(self.get_parameter('min_valid_depth_m').value)
        max_valid = float(self.get_parameter('max_valid_depth_m').value)
        obstacle_distance = float(self.get_parameter('obstacle_distance_m').value)
        hard_stop_distance = float(self.get_parameter('hard_stop_distance_m').value)
        blocked_threshold = float(self.get_parameter('blocked_ratio_threshold').value)

        valid = roi[(roi >= min_valid) & (roi <= max_valid)]
        if valid.size < 80:
            return self.turn_in_place(1.0), self.navigation_metrics('search', math.nan, math.nan, math.nan, 0.0)

        left_roi, center_roi, right_roi = np.array_split(roi, 3, axis=1)
        left = self.sector_clearance(left_roi, min_valid, max_valid)
        center = self.sector_clearance(center_roi, min_valid, max_valid)
        right = self.sector_clearance(right_roi, min_valid, max_valid)
        blocked_ratio = float(np.count_nonzero(valid < obstacle_distance) / valid.size)

        front_blocked = center < obstacle_distance or blocked_ratio > blocked_threshold
        emergency = center < hard_stop_distance
        if emergency:
            direction = self.best_turn_direction(left, right)
            return self.reverse_and_turn(direction), self.navigation_metrics('emergency', left, center, right, blocked_ratio)
        if front_blocked:
            direction = self.best_turn_direction(left, right)
            return self.turn_in_place(direction), self.navigation_metrics('avoid', left, center, right, blocked_ratio)
        return self.forward_with_wall_bias(left, right), self.navigation_metrics('forward', left, center, right, blocked_ratio)

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

    def navigation_safety_timer(self):
        if not self.navigation_enabled or self.navigation_paused or self.last_depth_time == 0.0:
            return
        timeout = float(self.get_parameter('stale_depth_timeout_sec').value)
        if time.time() - self.last_depth_time > timeout:
            self.stop_base()

    def navigation_metrics(self, state, left, center, right, blocked_ratio):
        return {
            'state': state,
            'left': left,
            'center': center,
            'right': right,
            'blocked_ratio': blocked_ratio,
        }

    def log_navigation_metrics(self, metrics):
        now = time.time()
        interval = float(self.get_parameter('navigation_log_interval_sec').value)
        if now - self.last_navigation_log_time < interval:
            return
        self.last_navigation_log_time = now
        self.get_logger().info(
            'nav=%s left=%.2f center=%.2f right=%.2f blocked=%.2f'
            % (
                metrics['state'],
                metrics['left'],
                metrics['center'],
                metrics['right'],
                metrics['blocked_ratio'],
            )
        )

    def shutdown(self):
        self.running = False
        for _ in range(5):
            self.stop_base()
            time.sleep(0.03)
        self.publish_status('stopped')

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main():
    rclpy.init()
    node = DangerScanNode()
    try:
        rclpy.spin(node)
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
