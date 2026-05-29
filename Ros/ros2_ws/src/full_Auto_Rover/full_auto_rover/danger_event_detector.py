#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from interfaces.msg import ObjectsInfo, Point2D

from full_auto_rover.utils import box_center, iou


def normalized_class_name(name):
    return name.strip().lower().replace(' ', '_')


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

        object_topic = self.get_parameter('object_topic').value
        center_topic = self.get_parameter('event_center_topic').value
        status_topic = self.get_parameter('event_status_topic').value

        self.center_pub = self.create_publisher(Point2D, center_topic, 1)
        self.status_pub = self.create_publisher(String, status_topic, 1)
        self.create_subscription(ObjectsInfo, object_topic, self.objects_callback, 1)

        self.get_logger().info('danger_event_detector started')

    def objects_callback(self, msg):
        min_score = float(self.get_parameter('min_score').value)
        detections = [obj for obj in msg.objects if obj.score >= min_score and self.valid_box(obj.box)]
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
        persons = self.by_class(detections, ['person'])
        weapons = self.by_class(detections, self.get_parameter('weapon_classes').value)
        obstacles = self.by_class(detections, self.get_parameter('obstacle_classes').value)
        fires = self.by_class(detections, self.get_parameter('fire_classes').value)
        smokes = self.by_class(detections, self.get_parameter('smoke_classes').value)

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
            if self.is_fallen_person_candidate(person.box):
                return 'fallen_person_candidate', box_center(person.box)

        obstacle = self.find_obstacle_ahead(obstacles)
        if obstacle is not None:
            return 'obstacle_ahead:%s' % normalized_class_name(obstacle.class_name), box_center(obstacle.box)

        if fires:
            return 'fire', box_center(fires[0].box)

        if smokes:
            return 'smoke', box_center(smokes[0].box)

        if weapons:
            return 'weapon_detected:%s' % normalized_class_name(weapons[0].class_name), box_center(weapons[0].box)

        if persons:
            center = self.person_focus_center(persons[0].box)
            if center is not None:
                return 'person_detected', center
        return None

    def by_class(self, detections, class_names):
        wanted = set(normalized_class_name(name) for name in class_names)
        return [obj for obj in detections if normalized_class_name(obj.class_name) in wanted]

    def find_pair_event(self, event_type, persons, targets):
        for person in persons:
            for target in targets:
                if self.is_near(person.box, target.box):
                    target_name = normalized_class_name(target.class_name)
                    return '%s:%s' % (event_type, target_name), self.combined_box_center(person.box, target.box)
        return None

    def find_obstacle_ahead(self, obstacles):
        image_width = float(self.get_parameter('image_width').value)
        image_height = float(self.get_parameter('image_height').value)
        center_band_ratio = float(self.get_parameter('obstacle_center_band_ratio').value)
        min_area_ratio = float(self.get_parameter('obstacle_min_area_ratio').value)
        left = image_width * (0.5 - center_band_ratio / 2.0)
        right = image_width * (0.5 + center_band_ratio / 2.0)

        best = None
        best_area = 0
        for obstacle in obstacles:
            center = box_center(obstacle.box)
            if center is None:
                continue
            area = self.box_area(obstacle.box)
            area_ratio = area / (image_width * image_height)
            if left <= center[0] <= right and area_ratio >= min_area_ratio and area > best_area:
                best = obstacle
                best_area = area
        return best

    def is_near(self, box_a, box_b):
        threshold = float(self.get_parameter('near_iou_threshold').value)
        if iou(box_a, box_b) >= threshold:
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

    def person_focus_center(self, box):
        x1, y1, x2, y2 = box[:4]
        center_x = int((x1 + x2) / 2)
        height = max(1, y2 - y1)
        if y1 < 10:
            center_y = int(y1 + height / 4)
        else:
            center_y = int(y1 + height / 3)
        return center_x, center_y

    def valid_box(self, box):
        if len(box) < 4:
            return False
        x1, y1, x2, y2 = box[:4]
        return x2 > x1 and y2 > y1

    def box_area(self, box):
        x1, y1, x2, y2 = box[:4]
        return max(0, x2 - x1) * max(0, y2 - y1)

    def combined_center(self, box_a, box_b):
        ax, ay = box_center(box_a)
        bx, by = box_center(box_b)
        return int((ax + bx) / 2), int((ay + by) / 2)

    def combined_box_center(self, box_a, box_b):
        x1 = min(box_a[0], box_b[0])
        y1 = min(box_a[1], box_b[1])
        x2 = max(box_a[2], box_b[2])
        y2 = max(box_a[3], box_b[3])
        return box_center([x1, y1, x2, y2])


def main():
    rclpy.init()
    node = DangerEventDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
