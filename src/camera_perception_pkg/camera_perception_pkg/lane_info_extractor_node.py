import cv2
import rclpy
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray, BoundingBox2D, Detection
from .lib import camera_perception_func_lib as CPFL

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"  # 추가: ROI 이미지 퍼블리시 토픽

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1하여 입력
SHOW_IMAGE = True
#----------------------------------------------

# 교차로 감지 y-범위(현재 차선에 따라 다르게)
LANE2_Y_RANGE = (460, 490)  # 현재 lane2 주행 중일 때
LANE1_Y_RANGE = (400, 440)  # 현재 lane1 주행 중일 때

class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        self.switch_arm_time = None       # 조건 최초 만족 시각
        self.switch_hold_sec = 0.5        # 0.5초 유지 시 스위치

        self.current_lane = "lane2"
        self.last_switch_time = 0  # 최초 전환 시각 (epoch time)
        self.cooldown_sec = 30      # 쿨타임 (초)
        
        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value

        self.cv_bridge = CvBridge()

        # QoS settings
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.subscriber = self.create_subscription(DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)

        # ROI 이미지 퍼블리셔 추가
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            self.switch_arm_time = None
            return

        # Step 1: mask 또는 bbox 중 crosswalk의 가장 위쪽 y 좌표 찾기
        crosswalk_bottom_y = None

        for det in detection_msg.detections:
            if det.class_name == 'crosswalk':
                # 마스크가 있는 경우
                if det.mask.data:
                    ys = [p.y for p in det.mask.data]
                    bottom = max(ys)
                # 바운딩 박스만 있는 경우
                elif det.bbox:
                    bottom = det.bbox.center.position.y + det.bbox.size.y / 2
                else:
                    continue

                if crosswalk_bottom_y is None or bottom < crosswalk_bottom_y:
                    crosswalk_bottom_y = bottom

        # ---- 0.5초 지연 스위치 (debounce/hold) ----
        current_time = time.time()
        y_min, y_max = (LANE2_Y_RANGE if self.current_lane == "lane2" else LANE1_Y_RANGE)
        in_zone = (crosswalk_bottom_y is not None and y_min <= crosswalk_bottom_y <= y_max)


        if in_zone:
            # 최초 진입 시각 기록
            if self.switch_arm_time is None:
                self.switch_arm_time = current_time

            held = (current_time - self.switch_arm_time) >= self.switch_hold_sec
            time_since_last = current_time - self.last_switch_time

            if held and time_since_last >= self.cooldown_sec:
                # 전환
                self.current_lane = "lane1" if self.current_lane == "lane2" else "lane2"
                self.last_switch_time = current_time
                self.switch_arm_time = None   # 소모
                self.get_logger().info(f"[DELAYED SWITCH] -> {self.current_lane} (held {self.switch_hold_sec}s)")
        else:
            # 영역 벗어나면 무효화
            self.switch_arm_time = None

        # Step 2: 조건 판단
        current_time = time.time()  # 현재 시간 (초 단위 float)
        y_min, y_max = (LANE2_Y_RANGE if self.current_lane == "lane2" else LANE1_Y_RANGE)
        if crosswalk_bottom_y is not None and y_min <= crosswalk_bottom_y <= y_max:
            time_since_last = current_time - self.last_switch_time

            if time_since_last >= self.cooldown_sec:
                # 전환 가능
                if self.current_lane == "lane2":
                    self.current_lane = "lane1"
                else:
                    self.current_lane = "lane2"

                self.last_switch_time = current_time  # 전환 시각 업데이트
                self.get_logger().info(f"Switched to {self.current_lane} (Δt={time_since_last:.2f}s)")
            else:
                # 쿨타임 안 지나서 무시
                self.get_logger().info(f"Cooldown active ({time_since_last:.2f}s < {self.cooldown_sec}s) — no switch")

        # Step 3: 에지 이미지 선택
        lane2_edge_image = CPFL.draw_edges(detection_msg, cls_name='lane2', color=255)
        lane1_edge_image = CPFL.draw_edges(detection_msg, cls_name='lane1', color=255)

        # Step 4: 선택된 라인으로 버드뷰 변환 및 ROI 적용
        if self.current_lane == "lane1":
            selected_edge_image = lane1_edge_image
        else:
            selected_edge_image = lane2_edge_image

        (h, w) = (selected_edge_image.shape[0], selected_edge_image.shape[1])
        dst_mat = [[round(w * 0.3), round(h * 0.0)], [round(w * 0.7), round(h * 0.0)], [round(w * 0.7), h], [round(w * 0.3), h]]
        src_mat = [[238, 316],[402, 313], [501, 476], [155, 476]]

        bird_image = CPFL.bird_convert(selected_edge_image, srcmat=src_mat, dstmat=dst_mat)
        roi_image = CPFL.roi_rectangle_below(bird_image, cutting_idx=300)

        if self.show_image:
            #cv2.imshow('edge_image', selected_edge_image)
            #cv2.imshow('bird_img', bird_image)
            #cv2.imshow('roi_img', roi_image)
            cv2.waitKey(1)

        # roi_image를 uint8 형식으로 변환
        roi_image = cv2.convertScaleAbs(roi_image)  # 64FC1 -> uint8로 변환

        # roi_image를 ROS Image 메시지로 변환
        try:
            roi_image_msg = self.cv_bridge.cv2_to_imgmsg(roi_image, encoding="mono8")
            # ROI 이미지를 퍼블리시
            self.roi_image_publisher.publish(roi_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert and publish ROI image: {e}")
        
        grad = CPFL.dominant_gradient(roi_image, theta_limit=70)
                
        target_points = []
        for target_point_y in range(5, 155, 50):  # 예시로 5에서 155까지 50씩 증가
            target_point_x = CPFL.get_lane_center(roi_image, detection_height=target_point_y, 
                                                detection_thickness=10, road_gradient=grad, lane_width=300)
            
            target_point = TargetPoint()
            target_point.target_x = round(target_point_x)
            target_point.target_y = round(target_point_y)
            target_points.append(target_point)

        lane = LaneInfo()
        lane.slope = grad
        lane.target_points = target_points

        self.publisher.publish(lane)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
  
if __name__ == '__main__':
    main()
