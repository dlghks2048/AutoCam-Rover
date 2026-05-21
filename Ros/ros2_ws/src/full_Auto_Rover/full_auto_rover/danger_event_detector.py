#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from interfaces.msg import ObjectsInfo, Point2D

from full_auto_rover.utils import box_center, iou


class DangerEventDetector(Node):
    def __init__(self):
        super().__init__('danger_event_detector')
        self.declare_parameter('object_topic', '/yolov5/object_detect')
        self.declare_parameter('event_center_topic', '/full_auto_rover/event_center')
        self.declare_parameter('event_status_topic', '/full_auto_rover/event_status')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('min_score', 0.45)
        self.declare_parameter('near_iou_threshold', 0.05)

        object_topic = self.get_parameter('object_topic').value
        center_topic = self.get_parameter('event_center_topic').value
        status_topic = self.get_parameter('event_status_topic').value

        self.center_pub = self.create_publisher(Point2D, center_topic, 1)
        self.status_pub = self.create_publisher(String, status_topic, 1)
        self.create_subscription(ObjectsInfo, object_topic, self.objects_callback, 1)

        self.get_logger().info('danger_event_detector started')

    def objects_callback(self, msg):
        min_score = float(self.get_parameter('min_score').value)
        detections = [obj for obj in msg.objects if obj.score >= min_score and len(obj.box) >= 4]
        event = self.find_event(detections)
        if event is None:
            return

        label, center = event
        out = Point2D()
        out.width = int(self.get_parameter('image_width').value)
        out.height = int(self.get_parameter('image_height').value)
        out.x = center[0]
        out.y = center[1]
        self.center_pub.publish(out)

        status = String()
        status.data = label
        self.status_pub.publish(status)

    def find_event(self, detections):
        # First useful rule: person close to another detected object.
        persons = [obj for obj in detections if obj.class_name == 'person']
        others = [obj for obj in detections if obj.class_name != 'person']
        threshold = float(self.get_parameter('near_iou_threshold').value)

        for person in persons:
            for other in others:
                if iou(person.box, other.box) >= threshold:
                    return 'person_object_overlap:%s' % other.class_name, self.combined_center(person.box, other.box)

        if persons:
            center = box_center(persons[0].box)
            if center is not None:
                return 'person_detected', center
        return None

    def combined_center(self, box_a, box_b):
        ax, ay = box_center(box_a)
        bx, by = box_center(box_b)
        return int((ax + bx) / 2), int((ay + by) / 2)


def main():
    rclpy.init()
    node = DangerEventDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
