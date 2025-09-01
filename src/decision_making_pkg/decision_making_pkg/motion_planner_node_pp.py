import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
)

from nav_msgs.msg import Odometry
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand
from .lib import decision_making_func_lib as DMFL  # 사용 시 필요, 미사용이면 제거 가능

import math
import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


#--------------- Variable Setting ---------------
# 토픽 이름
SUB_PATH_TOPIC_NAME       = "path_planning_result"
SUB_ODOM_TOPIC_NAME       = "odom"
SUB_DETECTION_TOPIC_NAME  = "detections"
PUB_TOPIC_NAME            = "topic_control_signal"

# 제어 주기
TIMER = 0.1

# 픽셀-미터 변환 및 차량 파라미터
LANE_WIDTH_PX = 300      # 인식 이미지에서의 차선 폭 (px)
LANE_WIDTH_M  = 3.5      # 실제 차선 폭 (m)
PX_PER_M      = LANE_WIDTH_PX / LANE_WIDTH_M
CAR_CX = 320.0           # 버드뷰 차량 중심 x(px)
CAR_CY = 179.0           # 버드뷰 앞바퀴 기준선 y(px)
WB_M   = 2.86            # 휠베이스 (m)
MAX_STEER_RAD = 0.8727   # 최대 조향(휠) 각도 (rad)

# 속도 PID
KP_V = 0.8
KI_V = 0.25
KD_V = 0.05
I_CLAMP    = 0.5
TD_D       = 0.05
D_FILTER_N = 15.0

# Pure Pursuit 룩어헤드
LD_MIN     = 1
LD_MAX     = 8
K_LD_A     = 2.7       # Ld = A + B*v
K_LD_B     = 0.82

# 동적 게인 파라미터 (보조 로직에서 사용)
LD_CURV_GAIN_BASE  = 2.8
LD_CURV_GAIN_SCALE = 4
LD_CURV_GAIN_MAX   = 6.0
KAPPA_THRESHOLD_FOR_ACCEL = 0.03
KAPPA_SPEED_GAIN = 7

# 커브 바이어스
KAPPA_BIAS_ON   = 0.02   # 커브로 인식하는 임계값
KAPPA_BIAS_MAX  = 0.5   # 강커브 스케일 상한
BIAS_MAX_M      = -0.2  # 최대 바이어스 크기 (m 단위)
BIAS_LP_ALPHA   = 0.6    # 편향 저역통과 필터 계수
BIAS_LP_BETA    = 0.4    # 편향 저역통과 보정계수

# 속도/가감속
RAW_MAX        = 255.0
MAX_SPEED_MPS  = 3
AY_MAX         = 8

# 제어 출력 스케일/필터
MAX_STEERING_ABS  = 7
SMOOTHING_ALPHA   = 0.43
MAX_STEERING_STEP = 2
OFFSET_PX         = -38

# 속도 목표 (raw)
DEFAULT_TARGET_SPEED = 200
CROSSWALK_SPEED      = 160
MIN_SPEED_RAW        = 50 
#------------------------------------------------


class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # 파라미터 선언
        self.sub_path_topic      = self.declare_parameter('sub_path_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_odom_topic      = self.declare_parameter('sub_odom_topic', SUB_ODOM_TOPIC_NAME).value
        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.pub_topic           = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.timer_period        = self.declare_parameter('timer', TIMER).value

        # QoS
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 상태 변수
        self.path_data = None
        self.detection_data = None
        self.v_mps = 0.0

        self.is_lane_changing = False
        self.prev_is_lane_changing = False
        self.post_lane_change_slowdown = False
        self.is_in_crosswalk_mode = False

        # 교차로/횡단보도 통과 후 저속 주행
        self.intersection_drive_active = False
        self.intersection_drive_start_time = None
        self.INTERSECTION_DRIVE_DURATION = 5
        self.INTERSECTION_DRIVE_SPEED = CROSSWALK_SPEED

        # 속도 PID 상태
        self.speed_err_integral = 0.0
        self.d_f                = 0.0
        self.meas_prev          = None

        # 출력 명령
        self.steering_command      = 0
        self.left_speed_command    = 0
        self.right_speed_command   = 0
        self.current_target_speed  = DEFAULT_TARGET_SPEED

        # ROS I/O
        self.path_sub = self.create_subscription(
            PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.sub_odom_topic, self.odom_callback, self.qos_profile
        )
        self.det_sub = self.create_subscription(
            DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile
        )

        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    #--------------- Callbacks ---------------
    def path_callback(self, msg: PathPlanningResult):
        x = np.asarray(msg.x_points, dtype=float)
        y = np.asarray(msg.y_points, dtype=float)

        if x.size < 5:
            self.path_data = list(zip(x, y))
            return

        if _HAVE_SCIPY and x.size >= 9:
            x_s = savgol_filter(x, window_length=9, polyorder=2, mode='interp')
            y_s = savgol_filter(y, window_length=9, polyorder=2, mode='interp')
        else:
            x_s = _moving_average_same_len(x, 7)
            y_s = _moving_average_same_len(y, 7)

        self.path_data = list(zip(x_s.tolist(), y_s.tolist()))

        # 차선 변경 상태
        self.is_lane_changing = getattr(msg, 'is_lane_changing', False)

        # 차선 변경 종료 감지 → 감속 모드 시작
        if self.prev_is_lane_changing and not self.is_lane_changing:
            self.post_lane_change_slowdown = True
        self.prev_is_lane_changing = self.is_lane_changing


    def odom_callback(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.v_mps = math.hypot(vx, vy)

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    #--------------- Main Timer ---------------
    def timer_callback(self):
        
        # 경로 준비 안 됨 → 정지
        if self.path_data is None or len(self.path_data) < 3:
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0
            self._publish()
            return

        # 1) 횡단보도/교차로 위치 기반 감지
        crosswalk_in_zone = False
        if (self.detection_data is not None and
            hasattr(self.detection_data, 'detections') and
            self.detection_data.detections is not None):
            for det in self.detection_data.detections:
                if det.class_name == 'crosswalk':
                    bottom_y = det.bbox.center.position.y + det.bbox.size.y / 2
                    if 460 <= bottom_y <= 480:
                        crosswalk_in_zone = True
                        break
        self.is_in_crosswalk_mode = crosswalk_in_zone

        # 2) 교차로 타이머 운용
        if self.is_in_crosswalk_mode and not self.intersection_drive_active:
            self.intersection_drive_active = True
            self.intersection_drive_start_time = self.get_clock().now()
        if self.intersection_drive_active:
            elapsed = (self.get_clock().now() - self.intersection_drive_start_time).nanoseconds / 1e9
            if elapsed > self.INTERSECTION_DRIVE_DURATION:
                self.intersection_drive_active = False
                self.intersection_drive_start_time = None

        # 3) 목표 속도 결정
        if self.intersection_drive_active:
            self.current_target_speed = self.INTERSECTION_DRIVE_SPEED
        elif self.is_in_crosswalk_mode or self.post_lane_change_slowdown:
            self.current_target_speed = CROSSWALK_SPEED
        else:
            self.current_target_speed = DEFAULT_TARGET_SPEED

        # 4) 제어
        self._apply_control_logic()
        self._publish()

    #--------------- Control Core ---------------
    def _apply_control_logic(self):
        # 차선변경 중에는 스무딩 완화
        current_smoothing_alpha = 0.1 if self.is_lane_changing else SMOOTHING_ALPHA

        if self.path_data is None or len(self.path_data) < 2:
            self.steering_command = 0
            return

        # 픽셀 → 차량좌표계 (x: 전방[m], y: 좌+ / 우- [m])
        path_m = []
        for (x_pix, y_pix) in self.path_data:
            xm = (CAR_CY - y_pix) / PX_PER_M
            ym = (x_pix - (CAR_CX + OFFSET_PX)) / PX_PER_M
            if xm > 0.0:
                path_m.append((xm, ym))
        if not path_m:
            self.steering_command = 0
            return

        # 곡률 추정(최근접 3~6점 사용) — 속도상한 계산 등에만 활용
        kappa = self._estimate_curvature_from_points(
            path_m[:max(3, min(6, len(path_m)))]
        )
        self.get_logger().info(f"[DEBUG] Curvature kappa={kappa:.4f}, speed={self.v_mps:.2f} m/s")

        # 차선변경 직후 감속 모드 해제 조건(경로가 충분히 완만해지면 해제)
        if self.post_lane_change_slowdown:
            kappa_tmp = self._estimate_curvature_from_points(
                path_m[:max(3, min(6, len(path_m)))]
            )
            if abs(kappa_tmp) < KAPPA_THRESHOLD_FOR_ACCEL:
                self.post_lane_change_slowdown = False

        # --- 룩어헤드 계산 (속도 기반) ---
        Ld_base = max(LD_MIN, min(LD_MAX, K_LD_A + K_LD_B * max(self.v_mps, 0.0)))

        # 곡률 기반 동적 게인으로 Ld를 짧/길게 조절(조향 바이어스가 아니라 추종 안정성용)
        dynamic_ld_gain = LD_CURV_GAIN_BASE + LD_CURV_GAIN_SCALE * abs(kappa)
        dynamic_ld_gain = min(dynamic_ld_gain, LD_CURV_GAIN_MAX)
        Ld = Ld_base / (1.0 + dynamic_ld_gain * abs(kappa))
        Ld = max(LD_MIN, min(LD_MAX, Ld))

        # --- 2점 블렌딩: (근거리 Ld1 / 원거리 Ld2) ---
        Ld1 = max(2.0, 0.5 * Ld)
        Ld2 = Ld
        t1_xm, t1_ym = min(path_m, key=lambda p: abs(math.hypot(p[0], p[1]) - Ld1))
        t2_xm, t2_ym = min(path_m, key=lambda p: abs(math.hypot(p[0], p[1]) - Ld2))
        w_near = 0.3  # 곡률/기울기 의존 제거(고정)
        target_xm = w_near * t1_xm + (1.0 - w_near) * t2_xm
        target_ym = w_near * t1_ym + (1.0 - w_near) * t2_ym

        #----------- 커브 바깥쪽 편향 -----------
        alpha = math.atan2(target_ym, target_xm)

        bias_m = 0.0
        if alpha > 0 and abs(kappa) > KAPPA_BIAS_ON:
            s = (abs(kappa) - KAPPA_BIAS_ON) / max(1e-6, (KAPPA_BIAS_MAX - KAPPA_BIAS_ON))
            s = max(0.0, min(1.0, s))
            bias_m = BIAS_MAX_M * s
        else:
            bias_m = 0.0

        # 저역통과 필터
        if not hasattr(self, "_bias_lp"):
            self._bias_lp = 0.0
        self._bias_lp = BIAS_LP_ALPHA * self._bias_lp + BIAS_LP_BETA * bias_m

        target_ym += self._bias_lp
        alpha = math.atan2(target_ym, target_xm)  # 적용 후 재계산

        # 최종 조향각 계산 (Pure Pursuit)
        alpha = math.atan2(target_ym, target_xm)  # 편향 반영 후 재계산
        delta_rad = math.atan2(2.0 * WB_M * math.sin(alpha), Ld)
        steer_ratio = delta_rad / MAX_STEER_RAD
        raw_steer_float = steer_ratio * MAX_STEERING_ABS
        raw_steer = int(max(-MAX_STEERING_ABS, min(MAX_STEERING_ABS, raw_steer_float)))

        # 조향 스무딩(IIR) + 레이트 제한
        prev = self.steering_command
        smooth = int(current_smoothing_alpha * prev + (1.0 - current_smoothing_alpha) * raw_steer)
        step = max(-MAX_STEERING_STEP, min(MAX_STEERING_STEP, smooth - prev))
        self.steering_command = prev + step

        # --- 속도 제어: 곡률 기반 속도 상한 + PID ---
        kappa_abs = abs(kappa)
        kappa_eff = KAPPA_SPEED_GAIN * kappa_abs
        kappa_eff = max(kappa_eff, 1e-3)  # 0 나눗셈/과대속도 방지

        v_cap_curv = math.sqrt(AY_MAX / kappa_eff)   # [m/s]
        v_ref_base = (self.current_target_speed / RAW_MAX) * MAX_SPEED_MPS
        v_ref_mps = min(v_ref_base, v_cap_curv)

        dt = max(self.timer_period, 1e-3)
        v_error = v_ref_mps - self.v_mps

        # PID
        self.speed_err_integral = max(-I_CLAMP, min(I_CLAMP, self.speed_err_integral + v_error * dt))
        if self.meas_prev is None:
            self.meas_prev = self.v_mps
        d_meas = (self.v_mps - self.meas_prev) / dt
        self.meas_prev = self.v_mps

        alpha_d = TD_D / (TD_D + D_FILTER_N * dt)
        self.d_f = alpha_d * self.d_f + (1.0 - alpha_d) * d_meas

        acc_cmd = KP_V * v_error + KI_V * self.speed_err_integral - KD_V * self.d_f
        commanded_speed_mps = max(0.0, min(MAX_SPEED_MPS, self.v_mps + acc_cmd))
        commanded_speed_mps = min(commanded_speed_mps, v_ref_mps)

        raw_speed = int(commanded_speed_mps / MAX_SPEED_MPS * RAW_MAX)
        raw_speed = max(MIN_SPEED_RAW, raw_speed)

        self.left_speed_command  = raw_speed
        self.right_speed_command = raw_speed


    #--------------- Helpers ---------------
    @staticmethod
    def _estimate_curvature_from_points(pts_m):
        if len(pts_m) < 3:
            return 0.0
        (x1, y1) = pts_m[0]
        (x2, y2) = pts_m[len(pts_m)//2]
        (x3, y3) = pts_m[-1]

        a = math.hypot(x2 - x1, y2 - y1)
        b = math.hypot(x3 - x2, y3 - y2)
        c = math.hypot(x1 - x3, y1 - y3)
        s = 0.5 * (a + b + c)

        area_sq = max(0.0, s * (s - a) * (s - b) * (s - c))
        area = math.sqrt(area_sq) if area_sq > 0.0 else 0.0
        if area < 1e-6:
            return 0.0

        R = (a * b * c) / (4.0 * area)
        if R < 1e-3:
            return 1.0
        kappa = 1.0 / R
        return max(0.0, min(1.0, kappa))

    #--------------- Publish ---------------
    def _publish(self):
        msg = MotionCommand()
        msg.steering   = self.steering_command
        msg.left_speed = self.left_speed_command
        msg.right_speed= self.right_speed_command
        self.publisher.publish(msg)


def _moving_average_same_len(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr
    pad = w // 2
    arr_pad = np.pad(arr, (pad, pad), mode='edge')
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(arr_pad, kernel, mode='valid')


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nShutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
