#!/usr/bin/env python3
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from interfaces.msg import ObjectInfo, ObjectsInfo
from rclpy.node import Node
from sensor_msgs.msg import Image


class YoloObjectDetector(Node):
    def __init__(self):
        super().__init__('yolo_object_detector')
        self.declare_parameter('image_topic', '/depth_cam/rgb/image_raw')
        self.declare_parameter('object_topic', '/yolov5/object_detect')
        self.declare_parameter('yolov5_dir', '/home/ubuntu/third_party_ros2/yolov5')
        self.declare_parameter('model_path', '/home/ubuntu/third_party_ros2/yolov5/yolov5n.pt')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('image_size', 640)
        self.declare_parameter('conf_threshold', 0.35)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('max_detections', 20)
        self.declare_parameter('process_every_n_frames', 3)

        self.frame_count = 0
        self.last_report_time = 0.0
        self.model = None
        self.device = None
        self.stride = 32
        self.names = {}

        self.object_pub = self.create_publisher(
            ObjectsInfo,
            self.get_parameter('object_topic').value,
            1,
        )
        self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.image_callback,
            1,
        )
        self.load_model()
        self.get_logger().info('yolo_object_detector started')

    def load_model(self):
        yolov5_dir = self.get_parameter('yolov5_dir').value
        model_path = self.get_parameter('model_path').value
        if yolov5_dir not in sys.path:
            sys.path.insert(0, yolov5_dir)

        from models.common import DetectMultiBackend
        from utils.general import check_img_size
        from utils.torch_utils import select_device

        requested_device = self.get_parameter('device').value
        self.device = select_device(requested_device)
        self.model = DetectMultiBackend(model_path, device=self.device, dnn=False, data=None, fp16=False)
        self.stride = int(self.model.stride)
        image_size = int(self.get_parameter('image_size').value)
        self.image_size = check_img_size(image_size, s=self.stride)
        self.names = self.model.names
        self.get_logger().info('loaded YOLO model: %s' % model_path)

    def image_callback(self, msg):
        self.frame_count += 1
        every_n = max(1, int(self.get_parameter('process_every_n_frames').value))
        if self.frame_count % every_n != 0:
            return

        try:
            image = self.ros_image_to_bgr(msg)
            detections = self.detect(image)
            self.object_pub.publish(self.to_message(detections))
            self.report_detections(detections)
        except Exception as exc:
            self.get_logger().error('YOLO detection failed: %s' % exc)

    def ros_image_to_bgr(self, msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        encoding = msg.encoding.lower()

        if encoding in ('rgb8', 'bgr8'):
            image = data.reshape((msg.height, msg.width, 3))
            if encoding == 'rgb8':
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image

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
                x1, y1, x2, y2 = [int(value.item()) for value in xyxy]
                detections.append((class_name, float(conf.item()), [x1, y1, x2, y2]))
        return detections

    def to_message(self, detections):
        msg = ObjectsInfo()
        for class_name, score, box in detections:
            obj = ObjectInfo()
            obj.class_name = class_name
            obj.score = score
            obj.box = box
            msg.objects.append(obj)
        return msg

    def report_detections(self, detections):
        now = time.time()
        if now - self.last_report_time < 2.0:
            return
        self.last_report_time = now
        if not detections:
            self.get_logger().info('YOLO detected no objects')
            return
        summary = ', '.join('%s %.2f' % (name, score) for name, score, _ in detections[:5])
        self.get_logger().info('YOLO detected: %s' % summary)


def main():
    rclpy.init()
    node = YoloObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
