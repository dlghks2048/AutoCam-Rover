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
    """Depth camera만 사용해서 맵 없이 강의실 안을 천천히 주행하는 노드.

    이 노드는 SLAM 지도, Nav2 목적지, 라인 트레이싱을 사용하지 않습니다.
    카메라가 보는 깊이 영상에서 전방/왼쪽/오른쪽의 여유 거리를 계산하고,
    그 결과에 따라 /controller/cmd_vel 토픽으로 Twist 속도 명령을 보냅니다.
    """

    def __init__(self):
        super().__init__('camera_obstacle_avoidance')

        # depth_topic:
        #   깊이 카메라의 거리 영상 토픽입니다.
        #   각 픽셀 값은 카메라에서 해당 물체까지의 거리입니다.
        #   이 로봇에서는 보통 /depth_cam/depth/image_raw 를 사용합니다.
        self.declare_parameter('depth_topic', '/depth_cam/depth/image_raw')

        # cmd_vel_topic:
        #   로봇 바퀴 컨트롤러로 속도 명령을 보내는 토픽입니다.
        #   Twist.linear.x 는 전진/후진, Twist.angular.z 는 좌/우 회전입니다.
        self.declare_parameter('cmd_vel_topic', '/controller/cmd_vel')

        # start_on_launch:
        #   true이면 launch 실행 직후 자율주행을 시작합니다.
        #   false로 바꾸면 노드는 켜지지만 set_running 서비스 호출 전까지 움직이지 않습니다.
        self.declare_parameter('start_on_launch', True)

        # 주행 속도 관련 파라미터입니다.
        # 강의실 테스트용이라 기본 전진 속도는 일부러 낮게 잡았습니다.
        self.declare_parameter('linear_speed', 0.11)
        self.declare_parameter('turn_speed', 0.45)
        self.declare_parameter('reverse_speed', -0.06)

        # obstacle_distance_m:
        #   전방 중심 영역의 여유 거리가 이 값보다 작으면 장애물로 보고 회피합니다.
        # hard_stop_distance_m:
        #   이 값보다 가까우면 너무 위험하다고 보고 살짝 후진하면서 회전합니다.
        self.declare_parameter('obstacle_distance_m', 0.65)
        self.declare_parameter('hard_stop_distance_m', 0.38)

        # 카메라 depth 값 중 사용할 유효 거리 범위입니다.
        # 너무 작은 값은 센서 노이즈일 수 있고, 너무 먼 값은 회피 판단에 덜 중요합니다.
        self.declare_parameter('min_valid_depth_m', 0.12)
        self.declare_parameter('max_valid_depth_m', 4.0)

        # 화면 안에서 obstacle_distance_m보다 가까운 픽셀 비율이 이 값보다 크면
        # 전방이 넓게 막힌 상황으로 판단합니다.
        self.declare_parameter('blocked_ratio_threshold', 0.22)

        # 좌우 거리 차이를 회전 보정값으로 바꿀 때 사용하는 기준 거리입니다.
        # 값이 작을수록 좌우 차이에 더 민감하게 회전합니다.
        self.declare_parameter('clear_path_depth_m', 1.15)

        # depth 영상이 일정 시간 들어오지 않으면 안전을 위해 정지합니다.
        self.declare_parameter('stale_depth_timeout_sec', 0.7)

        # ROI(Region Of Interest): 깊이 영상 중 주행 판단에 사용할 영역입니다.
        # 화면 위쪽은 천장/먼 벽이 섞일 수 있고, 맨 아래/가장자리는 로봇 몸체나 왜곡이
        # 들어갈 수 있어서 중앙 하단 위주로 잘라 사용합니다.
        self.declare_parameter('roi_top_ratio', 0.42)
        self.declare_parameter('roi_bottom_ratio', 0.92)
        self.declare_parameter('roi_left_ratio', 0.08)
        self.declare_parameter('roi_right_ratio', 0.92)

        # 로그를 너무 자주 찍으면 터미널이 지저분해지므로 출력 간격을 둡니다.
        self.declare_parameter('log_interval_sec', 1.0)

        # 현재 자율주행이 활성화되어 있는지 저장합니다.
        # set_running 서비스로 실행 중에도 true/false를 바꿀 수 있습니다.
        self.is_running = bool(self.get_parameter('start_on_launch').value)

        # 마지막 depth 수신 시각입니다.
        # safety_timer에서 영상이 끊겼는지 확인하는 데 사용합니다.
        self.last_depth_time = 0.0
        self.last_log_time = 0.0

        # 로봇 이동 명령을 내보내는 publisher입니다.
        # 이 토픽으로 Twist 메시지를 보내면 바퀴 컨트롤러가 움직입니다.
        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            1,
        )

        # 깊이 카메라 영상을 구독합니다.
        # 새 depth 이미지가 들어올 때마다 depth_callback이 호출되고,
        # 그 안에서 장애물 판단과 속도 명령 발행이 이루어집니다.
        self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self.depth_callback,
            1,
        )

        # 외부에서 자율주행을 켜고 끄기 위한 서비스입니다.
        # 예:
        #   ros2 service call /camera_obstacle_avoidance/set_running std_srvs/srv/SetBool "{data: false}"
        self.create_service(SetBool, '~/set_running', self.set_running_callback)

        # 긴급 정지용 서비스입니다.
        # 호출하면 is_running을 false로 만들고 정지 명령을 보냅니다.
        self.create_service(Trigger, '~/stop', self.stop_callback)

        # 0.1초마다 depth 영상이 끊겼는지 확인합니다.
        # 영상이 끊긴 상태에서 마지막 속도 명령이 유지되면 위험하기 때문에 필요합니다.
        self.create_timer(0.1, self.safety_timer)

        # 노드가 시작될 때 일단 정지 명령을 보내서 이전 속도 명령이 남아있을 가능성을 줄입니다.
        self.stop_robot()
        self.get_logger().info('camera_obstacle_avoidance started')

    def set_running_callback(self, request, response):
        # request.data == true  -> 자율주행 시작
        # request.data == false -> 자율주행 일시정지 및 정지 명령 발행
        self.is_running = bool(request.data)
        if not self.is_running:
            self.stop_robot()
        response.success = True
        response.message = 'running' if self.is_running else 'stopped'
        return response

    def stop_callback(self, request, response):
        # set_running false와 비슷하지만, 긴급 정지 용도로 더 짧게 호출할 수 있게 만든 서비스입니다.
        self.is_running = False
        self.stop_robot()
        response.success = True
        response.message = 'stopped'
        return response

    def depth_callback(self, msg):
        # 자율주행이 비활성화되어 있으면 depth 영상이 들어와도 아무 명령을 만들지 않습니다.
        if not self.is_running:
            return

        # ROS Image 메시지의 raw data를 numpy 배열로 변환하고, 단위를 meter로 통일합니다.
        depth = self.depth_image_to_meters(msg)
        if depth is None:
            # 지원하지 않는 encoding이거나 이미지 shape이 이상하면 안전하게 정지합니다.
            self.stop_robot()
            return

        # 전체 화면을 다 쓰지 않고, 주행 판단에 필요한 중앙 하단 영역만 사용합니다.
        roi = self.crop_roi(depth)

        # ROI 안의 깊이값을 보고 전진/회피/긴급회피 중 하나를 결정합니다.
        twist, metrics = self.decide_command(roi)

        # 정상적으로 depth를 처리한 시각을 기록합니다.
        # safety_timer가 이 값을 보고 카메라 끊김을 감지합니다.
        self.last_depth_time = time.time()

        # 계산한 속도 명령을 실제 바퀴 컨트롤러로 보냅니다.
        self.cmd_pub.publish(twist)
        self.log_metrics(metrics)

    def depth_image_to_meters(self, msg):
        # depth camera는 보통 두 가지 형식 중 하나로 거리를 보냅니다.
        # 16UC1: unsigned 16-bit integer, 보통 mm 단위라 0.001을 곱해 meter로 바꿉니다.
        # 32FC1: 32-bit float, 이미 meter 단위인 경우가 많습니다.
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

        # msg.step은 한 줄(row)이 몇 byte인지 나타냅니다.
        # 일부 카메라는 padding 때문에 step이 width * pixel_size보다 클 수 있어서
        # row_values로 reshape한 뒤 실제 width만 잘라냅니다.
        item_size = np.dtype(dtype).itemsize
        row_values = int(msg.step / item_size)
        try:
            depth = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_values)
        except ValueError as exc:
            self.get_logger().warning('Invalid depth image shape: %s' % exc)
            return None

        # 단위를 meter로 통일하고, 0/inf/nan처럼 거리 판단에 쓸 수 없는 값은 nan으로 처리합니다.
        depth = depth[:, :msg.width].astype(np.float32) * scale
        depth[~np.isfinite(depth)] = np.nan
        depth[depth <= 0.0] = np.nan
        return depth

    def crop_roi(self, depth):
        # depth.shape은 (height, width)입니다.
        # ratio 파라미터는 0.0~1.0 사이 비율이고, 실제 픽셀 좌표로 변환해서 잘라냅니다.
        # 예: roi_top_ratio=0.42이면 이미지 높이의 42% 지점부터 사용합니다.
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
        # 잘못된 파라미터 값이 들어와도 ROI가 이미지 밖으로 나가지 않도록 0.0~1.0으로 제한합니다.
        return max(0.0, min(1.0, float(self.get_parameter(name).value)))

    def decide_command(self, roi):
        # 매 프레임마다 최신 파라미터 값을 읽습니다.
        # 그래서 ros2 param set으로 실행 중 튜닝해도 바로 반영됩니다.
        min_valid = float(self.get_parameter('min_valid_depth_m').value)
        max_valid = float(self.get_parameter('max_valid_depth_m').value)
        obstacle_distance = float(self.get_parameter('obstacle_distance_m').value)
        hard_stop_distance = float(self.get_parameter('hard_stop_distance_m').value)
        blocked_threshold = float(self.get_parameter('blocked_ratio_threshold').value)

        # 유효 거리 범위 안에 있는 픽셀만 뽑습니다.
        # 유효 픽셀이 너무 적으면 카메라가 막혔거나 depth가 제대로 안 들어오는 상황일 수 있습니다.
        valid = roi[(roi >= min_valid) & (roi <= max_valid)]
        if valid.size < 80:
            # 확실한 거리 정보를 얻지 못하면 전진하지 않고 제자리 회전하며 탐색합니다.
            return self.turn_in_place(1.0), self.metrics('search', math.nan, math.nan, math.nan, 0.0)

        # ROI를 가로 방향으로 3등분합니다.
        # 왼쪽/중앙/오른쪽의 여유 거리를 따로 계산해서 어느 쪽이 더 안전한지 판단합니다.
        left_roi, center_roi, right_roi = np.array_split(roi, 3, axis=1)
        left = self.sector_clearance(left_roi, min_valid, max_valid)
        center = self.sector_clearance(center_roi, min_valid, max_valid)
        right = self.sector_clearance(right_roi, min_valid, max_valid)

        # ROI 전체에서 가까운 픽셀 비율입니다.
        # 중심 한 점만 보는 것보다 의자/벽처럼 넓게 막힌 상황을 더 잘 잡기 위한 값입니다.
        blocked_ratio = float(np.count_nonzero(valid < obstacle_distance) / valid.size)

        # front_blocked:
        #   중앙 전방이 가까운 물체로 막혔거나, 화면 전체에서 가까운 물체 비율이 높은 경우입니다.
        # emergency:
        #   중앙 전방이 매우 가까워 바로 회피해야 하는 경우입니다.
        front_blocked = center < obstacle_distance or blocked_ratio > blocked_threshold
        emergency = center < hard_stop_distance

        if emergency:
            # 너무 가까우면 그냥 회전만 하지 않고, 살짝 후진하면서 더 넓은 쪽으로 돕니다.
            direction = self.best_turn_direction(left, right)
            return self.reverse_and_turn(direction), self.metrics('emergency', left, center, right, blocked_ratio)
        if front_blocked:
            # 전방이 막혔지만 아직 긴급 거리는 아니면 제자리에서 더 넓은 쪽으로 회전합니다.
            direction = self.best_turn_direction(left, right)
            return self.turn_in_place(direction), self.metrics('avoid', left, center, right, blocked_ratio)

        # 전방이 충분히 열려 있으면 전진합니다.
        # 다만 좌우 벽과의 거리 차이를 이용해 약간씩 방향을 보정합니다.
        return self.forward_with_wall_bias(left, right), self.metrics('forward', left, center, right, blocked_ratio)

    def sector_clearance(self, sector, min_valid, max_valid):
        # 해당 영역 안의 유효 depth 값만 사용합니다.
        values = sector[(sector >= min_valid) & (sector <= max_valid)]
        if values.size == 0:
            return 0.0

        # 평균이 아니라 25 percentile을 사용합니다.
        # 평균은 멀리 있는 배경 픽셀 때문에 실제 가까운 장애물을 놓칠 수 있습니다.
        # 25 percentile은 "가까운 편에 속하는 거리"라 회피 판단에 더 보수적입니다.
        return float(np.percentile(values, 25))

    def best_turn_direction(self, left_clearance, right_clearance):
        # 왼쪽 여유 거리가 오른쪽보다 크거나 같으면 왼쪽으로 회전합니다.
        # angular.z는 양수일 때 왼쪽, 음수일 때 오른쪽 회전입니다.
        return 1.0 if left_clearance >= right_clearance else -1.0

    def forward_with_wall_bias(self, left_clearance, right_clearance):
        twist = Twist()
        twist.linear.x = float(self.get_parameter('linear_speed').value)

        # 좌우 여유 거리 차이로 회전 보정값을 만듭니다.
        # 예:
        #   left > right  -> 오른쪽 벽/장애물이 더 가까움 -> 왼쪽으로 살짝 회전
        #   right > left  -> 왼쪽 벽/장애물이 더 가까움 -> 오른쪽으로 살짝 회전
        clear_path = max(0.1, float(self.get_parameter('clear_path_depth_m').value))
        turn_speed = float(self.get_parameter('turn_speed').value)
        bias = (left_clearance - right_clearance) / clear_path

        # 회전 보정이 너무 커지지 않도록 -0.35~0.35로 제한한 뒤 turn_speed를 곱합니다.
        twist.angular.z = max(-0.35, min(0.35, bias)) * turn_speed
        return twist

    def turn_in_place(self, direction):
        # direction = 1.0  -> 왼쪽 회전
        # direction = -1.0 -> 오른쪽 회전
        twist = Twist()
        twist.angular.z = float(direction) * float(self.get_parameter('turn_speed').value)
        return twist

    def reverse_and_turn(self, direction):
        # 긴급 상황에서 사용하는 명령입니다.
        # 회전만 하면 가까운 벽이나 의자에 계속 붙어 있을 수 있어서,
        # 아주 느리게 후진하면서 더 넓은 방향으로 빠져나오게 합니다.
        twist = self.turn_in_place(direction)
        twist.linear.x = float(self.get_parameter('reverse_speed').value)
        return twist

    def safety_timer(self):
        # depth_callback은 카메라 영상이 들어올 때만 호출됩니다.
        # 카메라가 끊기면 callback 자체가 안 불리기 때문에 타이머로 별도 감시합니다.
        if not self.is_running or self.last_depth_time == 0.0:
            return
        timeout = float(self.get_parameter('stale_depth_timeout_sec').value)
        if time.time() - self.last_depth_time > timeout:
            self.stop_robot()

    def stop_robot(self):
        # 모든 속도 성분이 0인 Twist를 보내면 정지 명령입니다.
        self.cmd_pub.publish(Twist())

    def metrics(self, state, left, center, right, blocked_ratio):
        # 로그 출력용 상태값을 dict로 묶습니다.
        # 나중에 대시보드와 연결할 때 이 값을 서버로 보내면
        # "전진/회피/긴급회피/탐색" 상태를 화면에 표시할 수 있습니다.
        return {
            'state': state,
            'left': left,
            'center': center,
            'right': right,
            'blocked_ratio': blocked_ratio,
        }

    def log_metrics(self, metrics):
        # 매 프레임마다 로그를 찍으면 너무 많으므로 log_interval_sec 간격으로만 출력합니다.
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
    # ROS2 Python 노드를 초기화하고 spin으로 callback 처리를 시작합니다.
    rclpy.init()
    node = CameraObstacleAvoidance()
    try:
        rclpy.spin(node)
    finally:
        # Ctrl+C 또는 launch 종료 시 마지막으로 정지 명령을 보내고 노드를 정리합니다.
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
