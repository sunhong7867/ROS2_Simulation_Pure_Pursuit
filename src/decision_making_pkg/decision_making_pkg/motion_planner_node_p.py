import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
)

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand
from .lib import decision_making_func_lib as DMFL

#---------------Variable Setting---------------
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"

#----------------------------------------------

# 모션 플랜 발행 주기 (초) - 소수점 필요 (int형은 반영되지 않음)
TIMER = 0.1

class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # 토픽 이름 설정
        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter('sub_lane_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_traffic_light_topic = self.declare_parameter('sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
        self.sub_lidar_obstacle_topic = self.declare_parameter('sub_lidar_obstacle_topic', SUB_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        
        self.timer_period = self.declare_parameter('timer', TIMER).value

        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 변수 초기화
        self.detection_data = None
        self.path_data = None
        self.traffic_light_data = None
        self.lidar_data = None

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0

        # 부드러운 조향용 게인과 필터 계수
        self.steering_gain    = 0.08    # target_slope → steering 값으로 변환하는 비례 상수
        self.max_steering     = 7       # 조향의 절대 최대값
        self.smoothing_alpha  = 0.5    # 로우패스 필터 계수 (0~1, 1에 가까울수록 관성 커짐)
        
        # 속도 제어 튜닝 계수 추가
        self.speed_adj_gain = 10 # 조향값에 따라 속도를 줄이는 계수

        # 교차로/횡단보도 통과 후 저속 주행
        self.intersection_drive_active = False
        self.intersection_drive_start_time = None
        self.INTERSECTION_DRIVE_DURATION = 6.0 # 저속 주행 시간 (초)
        self.INTERSECTION_DRIVE_SPEED = 160 # 교차로 통과 후 속도 (raw)

        # 서브스크라이버 설정
        self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile)
        self.traffic_light_sub = self.create_subscription(String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile)
        self.lidar_sub = self.create_subscription(Bool, self.sub_lidar_obstacle_topic, self.lidar_callback, self.qos_profile)

        # 퍼블리셔 설정
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)

        # 타이머 설정
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))
                
    def traffic_light_callback(self, msg: String):
        self.traffic_light_data = msg

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg
        
    def timer_callback(self):
        # 횡단보도 감지 로직
        crosswalk_in_zone = False
        if self.detection_data is not None and self.detection_data.detections:
            for det in self.detection_data.detections:
                if det.class_name == 'crosswalk':
                    # 횡단보도의 하단 Y 좌표를 기준으로 특정 영역(예: 460~480)에 있을 때 감지
                    bottom_y = det.bbox.center.position.y + det.bbox.size.y / 2
                    if 460 <= bottom_y <= 480:
                        crosswalk_in_zone = True
                        break

        # 교차로 통과 타이머 로직
        if crosswalk_in_zone and not self.intersection_drive_active:
            # 횡단보도 진입 시 타이머 시작
            self.intersection_drive_active = True
            self.intersection_drive_start_time = self.get_clock().now()
        
        if self.intersection_drive_active:
            elapsed = (self.get_clock().now() - self.intersection_drive_start_time).nanoseconds / 1e9
            if elapsed > self.INTERSECTION_DRIVE_DURATION:
                # 지정된 시간이 지나면 저속 주행 모드 해제
                self.intersection_drive_active = False
                self.intersection_drive_start_time = None
        
        if self.path_data is None or len(self.path_data) < 3:
            # 경로 준비 안 됨 → 정지 및 조향 0 유지
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0

            cmd = MotionCommand()
            cmd.steering = self.steering_command
            cmd.left_speed = self.left_speed_command
            cmd.right_speed = self.right_speed_command
            self.publisher.publish(cmd)
            return

        if self.lidar_data is not None and self.lidar_data.data is True:
            # 라이다가 장애물을 감지한 경우
            self.steering_command = 0 
            self.left_speed_command = 0 
            self.right_speed_command = 0 

        elif self.traffic_light_data is not None and self.traffic_light_data.data == 'Red':
            # 빨간색 신호등을 감지한 경우
            if self.detection_data is not None:
                for detection in self.detection_data.detections:
                    if detection.class_name=='traffic_light':
                        y_max = int(detection.bbox.center.position.y + detection.bbox.size.y / 2) # bbox의 우측하단 꼭짓점 y좌표

                        if y_max < 150:
                            # 신호등 위치에 따른 정지명령 결정
                            self.steering_command = 0 
                            self.left_speed_command = 0 
                            self.right_speed_command = 0
                            break
                # 신호등이 150 이상에 있거나 감지되지 않은 경우, 기본 로직 실행
                else:
                    self._apply_control_logic()
        
        elif self.intersection_drive_active:
            # 교차로 통과 후 저속 주행
            self.steering_command = self._calculate_steering_command()
            self.left_speed_command = self.INTERSECTION_DRIVE_SPEED
            self.right_speed_command = self.INTERSECTION_DRIVE_SPEED

        else:
            self._apply_control_logic()

        self.get_logger().info(f"steering: {self.steering_command}, " 
                               f"left_speed: {self.left_speed_command}, " 
                               f"right_speed: {self.right_speed_command}")

        # 모션 명령 메시지 생성 및 퍼블리시
        motion_command_msg = MotionCommand()
        motion_command_msg.steering = self.steering_command
        motion_command_msg.left_speed = self.left_speed_command
        motion_command_msg.right_speed = self.right_speed_command
        self.publisher.publish(motion_command_msg)


    def _apply_control_logic(self):
        base_speed = 200
        if self.path_data is None:
            raw_steer = 0
        else:
            # 1) 경로의 시작과 끝 사이 기울기 계산
            target_slope = None
            target_slope = DMFL.calculate_slope_between_points(
                self.path_data[-10], self.path_data[-1]
            )
            for det in self.detection_data.detections:
                if det.class_name == 'crosswalk':
                    target_slope = DMFL.calculate_slope_between_points(
                        self.path_data[28], self.path_data[-1]
                    )
                    base_speed = 160
            # 2) 프로포셔널 제어: 기울기에 비례한 조향값
            raw_steer = int(self.steering_gain * target_slope)
            # 3) 최대값으로 클램프
            raw_steer = max(-self.max_steering, min(self.max_steering, raw_steer))
        
        # 4) 로우패스 필터로 부드럽게 섞기
        self.steering_command = int(
            self.smoothing_alpha * self.steering_command +
            (1 - self.smoothing_alpha) * raw_steer
        )

        # 5) 커브가 클수록 속도 낮추기 (새로운 변수 사용)
        speed_adj  = int(abs(self.steering_command) * self.speed_adj_gain)
        speed_cmd  = max(50, base_speed - speed_adj)      # 최소 속도 50 보장
        self.left_speed_command  = speed_cmd
        self.right_speed_command = speed_cmd

    def _calculate_steering_command(self):
        if self.path_data is None:
            return 0
        
        target_slope = DMFL.calculate_slope_between_points(
            self.path_data[-10], self.path_data[-1]
        )
        raw_steer = int(self.steering_gain * target_slope)
        raw_steer = max(-self.max_steering, min(self.max_steering, raw_steer))
        
        return int(
            self.smoothing_alpha * self.steering_command +
            (1 - self.smoothing_alpha) * raw_steer
        )


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()