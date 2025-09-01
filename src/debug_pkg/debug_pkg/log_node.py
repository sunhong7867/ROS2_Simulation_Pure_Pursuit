#!/usr/bin/env python3
import os, csv, math, atexit, json, time, cv2
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from interfaces_pkg.msg import MotionCommand
from interfaces_pkg.msg import DetectionArray
from scipy.signal import savgol_filter

import numpy as np
# pandas/matplotlib 은 종료 시점에만 import (비침투성 ↑)

try:
    from scipy.signal import savgol_filter
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    def savgol_filter(x, *args, **kwargs):
        return x
    
# ---------- 유틸 ----------
def rotate_xy_deg(xy: np.ndarray, deg: float, center=None) -> np.ndarray:
    """xy: (N,2), deg만큼 CCW 회전. center 기준 회전 가능."""
    if xy.size == 0:
        return xy
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=float)
    if center is None:
        return (R @ xy.T).T
    ctr = np.asarray(center, dtype=float)
    return ((R @ (xy - ctr).T).T + ctr)

def unwrap(a: np.ndarray) -> np.ndarray:
    return np.unwrap(a)

def rms(x: np.ndarray) -> float:
    if x.size == 0 or np.all(np.isnan(x)): return 0.0
    return float(np.sqrt(np.nanmean(x**2)))

# ---------- 메인 노드 ----------
class TrajectoryLogger(Node):
    def __init__(self):
        super().__init__('trajectory_logger')

        # ---- params ----
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('cmd_topic',  'topic_control_signal')
        self.declare_parameter('save_root',  os.path.expanduser('~/ros2_trajectory_logs'))
        self.declare_parameter('run_name',   '')
        self.declare_parameter('plot_rotate_deg', 90.0)   # 오버레이 회전(반시계+)
        self.declare_parameter('kappa_threshold', 0.08)   # 커브 구간 기준 [1/m]

        # 비침투성(Non-intrusive) I/O 파라미터
        self.declare_parameter('flush_every_n', 50)       # N개마다 flush
        self.declare_parameter('flush_every_sec', 0.5)    # 또는 S초마다 flush

        # ★ 추가: 속도 스케일/기준선 파라미터 (Motion Planner와 독립적으로 사용)
        self.declare_parameter('raw_max', 255.0)          # RAW_MAX
        self.declare_parameter('max_speed_mps', 3.0)      # MAX_SPEED_MPS
        self.declare_parameter('target_speed_raw', 200)   # baseline raw 속도(예: 200)

        self.odom_topic  = self.get_parameter('odom_topic').value
        self.cmd_topic   = self.get_parameter('cmd_topic').value
        self.save_root   = self.get_parameter('save_root').value
        self.run_name    = self.get_parameter('run_name').value
        self.plot_rotate_deg = float(self.get_parameter('plot_rotate_deg').value)
        self.kappa_thr   = float(self.get_parameter('kappa_threshold').value)

        self.flush_every_n   = int(self.get_parameter('flush_every_n').value)
        self.flush_every_sec = float(self.get_parameter('flush_every_sec').value)

        # ★ 추가 파라미터 실제 값
        self.raw_max          = float(self.get_parameter('raw_max').value)
        self.max_speed_mps    = float(self.get_parameter('max_speed_mps').value)
        self.target_speed_raw = float(self.get_parameter('target_speed_raw').value)

        # ---- output dir & files ----
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.t0 = self.get_clock().now().nanoseconds * 1e-9

        if not self.run_name:
            self.run_name = f'run_{ts}'
        self.out_dir = os.path.join(self.save_root, self.run_name)
        os.makedirs(self.out_dir, exist_ok=True)

        self.odom_csv = os.path.join(self.out_dir, f'odom_{ts}.csv')
        self.cmd_csv  = os.path.join(self.out_dir, f'control_cmd_{ts}.csv')

        # ---- CSV writers ----
        self.odom_f = open(self.odom_csv, 'w', newline='')
        self.odom_w = csv.writer(self.odom_f)
        self.odom_w.writerow(['t','x','y','yaw_rad','vx','vy','v_yaw'])

        self.cmd_f = open(self.cmd_csv, 'w', newline='')
        self.cmd_w = csv.writer(self.cmd_f)
        self.cmd_w.writerow(['t','steering','left_speed','right_speed'])

        # 비침투성: 매 콜백 flush 금지 → 버퍼링 카운터/타이머
        self._write_count = 0
        self._last_flush = time.monotonic()

        # 로거 메시지도 최소화 (콘솔 I/O도 잠재적 지연 요소)
        self.get_logger().info(f'[logger] writing to: {self.out_dir}')
        self.t0 = None

        # 오버레이 시작점 기억
        self._start_xy = None

        # ---- QoS (비침투성 프로파일) ----
        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.create_subscription(Odometry,      self.odom_topic, self.odom_cb, qos_be)
        self.create_subscription(MotionCommand, self.cmd_topic,  self.cmd_cb,  qos_be)

        # 종료 시 자동 분석/그래프
        atexit.register(self._finalize_and_plot)

    # ---- utils ----
    def _now_rel(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None: self.t0 = now
        return now - self.t0

    def _maybe_flush(self):
        self._write_count += 1
        now = time.monotonic()
        if (self._write_count >= self.flush_every_n) or (now - self._last_flush >= self.flush_every_sec):
            try:
                self.odom_f.flush()
                self.cmd_f.flush()
            except Exception:
                pass
            self._write_count = 0
            self._last_flush = now

    # ---- callbacks ----
    def odom_cb(self, msg: Odometry):
        t   = self._now_rel()
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        q   = msg.pose.pose.orientation
        siny = 2.0*(q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        yaw  = math.atan2(siny, cosy)
        vx   = msg.twist.twist.linear.x
        vy   = msg.twist.twist.linear.y
        v_yaw= msg.twist.twist.angular.z

        if self._start_xy is None:
            self._start_xy = (float(x), float(y))

        self.odom_w.writerow([t, f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}', f'{vx:.6f}', f'{vy:.6f}', f'{v_yaw:.6f}'])
        self._maybe_flush()

    def cmd_cb(self, msg: MotionCommand):
        t = self._now_rel()
        self.cmd_w.writerow([t, int(msg.steering), int(msg.left_speed), int(msg.right_speed)])
        self._maybe_flush()

    # ---------- 분석 유틸 ----------
    def _compute_curvature(self, xs, ys, s_cut=0.0):
        if xs.size < 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        dx = np.diff(xs)
        dy = np.diff(ys)
        ds = np.hypot(dx, dy)
        psi = np.arctan2(dy, dx)
        dpsi = np.diff(np.unwrap(psi))
        ds_mid = ds[1:]
        valid = ds_mid > 0.05
        kappa = np.full_like(dpsi, np.nan)
        kappa[valid] = dpsi[valid] / ds_mid[valid]
        s = np.concatenate([[0.0], np.cumsum(ds)])
        s_mid = s[1:-1]
        mask = s_mid >= s_cut
        return kappa[mask], s_mid[mask], dpsi[mask], ds_mid[mask]

    def _compute_speed(self, vx, vy):
        return np.hypot(vx, vy)

    # ---- finalize: close & plot ----
    def _finalize_and_plot(self):
        try:
            self.odom_f.flush(); self.cmd_f.flush()
        except: pass
        try: self.odom_f.close()
        except: pass
        try: self.cmd_f.close()
        except: pass

        if not (os.path.exists(self.odom_csv) and os.path.exists(self.cmd_csv)):
            print('[graph] CSV not found, skip plotting'); return

        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            odom = pd.read_csv(self.odom_csv)
            cmds = pd.read_csv(self.cmd_csv)

            # --- 기본 배열 ---
            t  = odom['t'].to_numpy(dtype=float)
            xs = odom['x'].to_numpy(dtype=float)
            ys = odom['y'].to_numpy(dtype=float)
            vx = odom['vx'].to_numpy(dtype=float)
            vy = odom['vy'].to_numpy(dtype=float)
            v  = self._compute_speed(vx, vy)
            lap_time = float(t[-1]) if t.size else 0.0

            # ===== 1) Overlay: Trajectory (회전 + 시작점) =====
            if xs.size >= 1:
                xy = np.column_stack([xs, ys])
                if self._start_xy is not None:
                    ctr = np.array(self._start_xy)
                    th  = math.radians(self.plot_rotate_deg)
                    c, s = math.cos(th), math.sin(th)
                    R = np.array([[c, -s], [s, c]], dtype=float)
                    xy_plot = ((R @ (xy - ctr).T).T + ctr)
                    start_xy = np.array(self._start_xy)
                else:
                    xy_plot = xy
                    start_xy = xy_plot[0]

                plt.figure(figsize=(10,8), dpi=150)
                # 초록색 실선으로 실제 궤적 표시
                plt.plot(xy_plot[:,0], xy_plot[:,1], 'g-', lw=0.05, label='Actual Trajectory')
                plt.scatter([start_xy[0]], [start_xy[1]], c='red', s=30, zorder=5, label='Start')
                plt.axis('equal')
                plt.xlim(-50, 0)
                plt.ylim(-40, 0)
                plt.title(f'Overlay: Vehicle Trajectory')
                plt.xlabel('X [m]'); plt.ylabel('Y [m]')
                plt.grid(True, alpha=0.4)
                plt.savefig(os.path.join(self.out_dir, 'overlay_trajectory.png')); plt.close()

            # ===== 2) 속도 & 평균 (t ≥ 10s 평균 + raw=200 기준선 m/s) =====
            avg_speed = float(np.nanmean(v)) if v.size else 0.0
            avg_speed_after_10 = avg_speed
            baseline_mps = (self.target_speed_raw / max(1.0, self.raw_max)) * self.max_speed_mps

            if t.size and v.size:
                # t >= 10s 마스크
                mask_10 = t >= 10.0
                if np.any(mask_10):
                    avg_speed_after_10 = float(np.nanmean(v[mask_10]))
                else:
                    avg_speed_after_10 = float(np.nanmean(v))  # fallback

                plt.figure(figsize=(10,6), dpi=150)
                plt.plot(t, v, label='Speed [m/s]', lw=0.2)                         # 기본 속도 곡선
                plt.axhline(avg_speed_after_10, ls='--', color='red',
                            label=f'Avg (t≥10s) = {avg_speed_after_10:.3f} m/s')  # 빨간색 평균선
                plt.axhline(baseline_mps, ls=':', color='black',
                            label=f'Baseline(raw={int(self.target_speed_raw)}) = {baseline_mps:.3f} m/s')  # 검정 기준선
                
                # y축 범위를 0.0부터 4.0까지 고정
                plt.ylim(0.0, 4.0)
                
                plt.title('Vehicle Speed')
                plt.xlabel('Time [s]'); plt.ylabel('Speed [m/s]')
                plt.grid(True, alpha=0.4); plt.legend()
                plt.savefig(os.path.join(self.out_dir, 'speed_with_avg.png')); plt.close()

            # ===== 3) 누적 주행 거리 =====
            # 1) 기본: 속도 적분(사다리꼴) 기반 누적거리
            if t.size >= 2 and v.size >= 2:
                dt        = np.diff(t)                                  # [N-1]
                v_mid     = 0.5 * (v[1:] + v[:-1])                      # [N-1]
                dist_inc  = v_mid * dt                                  # [N-1]
                dist_cum  = np.concatenate([[0.0], np.cumsum(dist_inc)])# [N]
                total_dist = float(dist_cum[-1])

            # 2) 보조: 좌표 기반(속도 없을 때만 사용; 잡음 민감)
            elif xs.size >= 2:
                seg       = np.hypot(np.diff(xs), np.diff(ys))
                dist_cum  = np.concatenate([[0.0], np.cumsum(seg)])
                total_dist = float(dist_cum[-1])

            else:
                dist_cum   = np.array([0.0])
                total_dist = 0.0

            # 3) 플롯
            if t.size:
                plt.figure(figsize=(10,6), dpi=150)
                # dist_cum 길이 = t 길이(N) 이므로 그대로 사용
                plt.plot(t, dist_cum, label='Cumulative Distance')
                plt.axhline(total_dist, ls='--', linewidth=1.0, color='gray',
                            label=f'Total = {total_dist:.2f} m')
                plt.axvline(lap_time,   ls='--', linewidth=1.0, color='gray',
                            label=f'Time = {lap_time:.2f} s')
                plt.title('Cumulative Distance')
                plt.xlabel('Time [s]'); plt.ylabel('Distance [m]')
                plt.grid(True, alpha=0.4)
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(self.out_dir, 'distance_cumulative.png')); plt.close()


            # ===== 4) 조향 히스토그램 (+ 평균선) =====
            steering = cmds['steering'].to_numpy(dtype=float) if len(cmds) else np.array([])
            total_steer_var = 0.0
            if steering.size >= 2:
                total_steer_var = np.sum(np.abs(np.diff(steering)))
            if steering.size:
                s_mean = float(np.mean(steering))
                plt.figure(figsize=(10,6), dpi=150)

                max_steer_abs = 7
                bins = np.arange(-max_steer_abs - 0.5, max_steer_abs + 1.5, 1)

                plt.hist(steering, bins=bins, alpha=0.9, edgecolor='black', rwidth=0.8)
                plt.axvline(s_mean, ls='--', color='red', label=f'Mean = {s_mean:.2f} deg') # 평균값 점선

                # x축 눈금(ticks)을 1도 단위로 설정
                plt.xticks(np.arange(-max_steer_abs, max_steer_abs + 1, 1))
                
                # 범례에 조향 총변동값 추가
                plt.plot([], [], 'w', label=f'Total Variation = {total_steer_var:.2f} deg')
                # Y축 범위를 0에서 250까지로 고정
                plt.ylim(0, 330)
                
                plt.title('Steering Angle Histogram')
                plt.xlabel('Steering [deg]'); plt.ylabel('Count')
                plt.grid(True, alpha=0.3); plt.legend()
                plt.savefig(os.path.join(self.out_dir, 'steering_hist.png')); plt.close()

            # ===== 5) 조향 변화율 =====
            steer_rate_std = 0.0
            if steering.size >= 2 and 't' in cmds:
                t_cmd = cmds['t'].to_numpy(dtype=float)
                dsteer = np.diff(steering)
                dt     = np.maximum(np.diff(t_cmd), 1e-6)
                steer_rate = dsteer / dt
                steer_rate_std = float(np.std(steer_rate)) if steer_rate.size else 0.0
                rms_steer_rate = rms(steer_rate) # RMS 값 계산
                plt.figure(figsize=(10,6), dpi=150)
                plt.plot(t_cmd [1:], steer_rate, label='Δsteer/Δt', lw=0.8)
                plt.axhline(0.0, ls=':', color='gray')
                # RMS 값을 빨간색 점선으로 추가
                plt.axhline(rms_steer_rate, ls='--', color='red', label=f'RMS = {rms_steer_rate:.2f} deg/s')
                # 제목에서 RMS 값 제거
                plt.title('Steering Rate')
                plt.xlabel('Time [s]'); plt.ylabel('deg/s')
                plt.grid(True, alpha=0.4)
                plt.legend(loc='upper left')
                plt.ylim(-25, 45)

                plt.savefig(os.path.join(self.out_dir, 'steer_rate.png')); plt.close()

            # ===== 6) 곡률 관련 =====
            # 동역학 기반 곡률: kappa = omega / v  (물리 한계로 clip)
            # 시간축은 odom의 t 자체를 사용
            t_k = t

            # yaw rate
            if 'v_yaw' in odom:
                omega = odom['v_yaw'].to_numpy(dtype=float)
            else:
                # v_yaw 없으면 yaw 를 미분해서 얻음
                yaw = odom['yaw_rad'].to_numpy(dtype=float) if 'yaw_rad' in odom else np.zeros_like(t_k)
                dt_arr = np.maximum(np.diff(t_k, prepend=t_k[0]), 1e-3)
                omega = np.gradient(yaw, dt_arr)

            # 속도 (너무 작은 값은 0 나눗셈 방지)
            v_safe = np.maximum(v, 0.2)
            kappa = omega / v_safe

            # 물리 한계로 clip
            WB_M = 2.86
            MAX_STEER_RAD = 0.8727
            kappa_lim = math.tan(MAX_STEER_RAD) / WB_M          # ≈ 0.42 [1/m]
            kappa = np.clip(kappa, -1.2 * kappa_lim, 1.2 * kappa_lim)

            # ---- 통계량(요약에 사용) ----
            abs_kappa = np.abs(kappa)
            mean_abs_kappa = float(np.nanmean(abs_kappa)) if abs_kappa.size else 0.0
            max_abs_kappa  = float(np.nanmax(abs_kappa))  if abs_kappa.size else 0.0
            std_abs_kappa  = float(np.nanstd(abs_kappa))  if abs_kappa.size else 0.0

            # ∫ |kappa| ds  = ∫ |omega| dt  (ds = v dt, kappa=omega/v)
            dt_arr = np.maximum(np.diff(t_k, prepend=t_k[0]), 1e-3)
            int_abs_kappa = float(np.nansum(np.abs(omega) * dt_arr))

            # ∑ |Δψ| = ∫ |omega| dt  (동일)
            sum_abs_dpsi = float(np.nansum(np.abs(omega) * dt_arr))

            # 10초 이후 평균 (표시에 사용)
            mask10 = t_k >= 10.0
            mean_abs_kappa_after_10 = float(np.nanmean(abs_kappa[mask10])) if np.any(mask10) else 0.0

            # ---- RAW: κ (부호 포함) ----
            plt.figure(figsize=(10,6), dpi=150)
            plt.plot(t_k, kappa, linewidth=0.2, label='κ')
            plt.axhline(0.0, ls=':', color='gray')

            # RMS 값 추가
            rms_kappa = rms(kappa[mask10]) if np.any(mask10) else 0.0
            plt.axhline(rms_kappa, ls='--', color='red', label=f'RMS (t≥10s) = {rms_kappa:.3f}', zorder=3)
            plt.axhline(-rms_kappa, ls='--', color='red', zorder=3)
            
            plt.title('Curvature κ')
            plt.xlabel('Time [s]'); plt.ylabel('κ [1/m]')
            plt.grid(True, alpha=0.4)
            plt.xlim(0, None)
            plt.ylim(-0.15, 0.2)
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(self.out_dir, 'curvature.png'))
            plt.close()

            # ---- RAW: |κ| (선만, zorder로 평균선 위) ----
            plt.figure(figsize=(10,6), dpi=150)
            plt.plot(t_k, abs_kappa, linewidth=0.2, label='|κ|', zorder=2)
            plt.axhline(mean_abs_kappa_after_10, ls='--', color='red',
                        label=f'Mean (t≥10s) = {mean_abs_kappa_after_10:.3f}', zorder=3)
            plt.title('Absolute Curvature |κ|')
            plt.xlabel('Time [s]'); plt.ylabel('|κ| [1/m]')
            plt.grid(True, alpha=0.4)
            plt.xlim(0, None)
            plt.ylim(0, 0.17)
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(self.out_dir, 'curvature_abs.png'))
            plt.close()

            # ===== 7) 가로가속도 (Lateral Acceleration) =====
            if kappa.size and v.size:
                try:
                    v_s = savgol_filter(v, 15, 2) if HAVE_SCIPY and v.size >= 15 else v
                except Exception:
                    v_s = v
                a_y = (v_s**2) * kappa

                mask = t_k >= 10.0
                a_y_sel = a_y[mask]
                rms_ay = rms(a_y_sel) if a_y_sel.size else 0.0

                plt.figure(figsize=(10,6), dpi=150)
                plt.plot(t_k, a_y, label='Lateral Acceleration', lw=0.2)
                plt.axhline(0, ls=':', color='gray', zorder=1)
                
                # RMS 값을 선으로 표시하고, 범례에 추가
                plt.axhline(rms_ay, ls='--', color='red', label=f'RMS = {rms_ay:.2f} m/s²', lw=1)
                plt.axhline(-rms_ay, ls='--', color='red', lw=1)

                plt.title('Lateral Acceleration')
                plt.xlabel('Time [s]'); plt.ylabel('Lateral Acceleration [m/s²]')
                plt.grid(True, alpha=0.4)
                plt.xlim(0, None)
                plt.ylim(-1.5, 2)
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(self.out_dir, 'lateral_accel.png')); plt.close()

            # ===== 7-2) 가로가속도 절대값 (Absolute Lateral Acceleration) =====
            if kappa.size and v.size:
                try:
                    v_s = savgol_filter(v, 15, 2) if HAVE_SCIPY and v.size >= 15 else v
                except Exception:
                    v_s = v
                abs_a_y = np.abs((v_s**2) * kappa)

                mask = t_k >= 10.0
                abs_a_y_sel = abs_a_y[mask]
                mean_abs_ay = float(np.nanmean(abs_a_y_sel)) if abs_a_y_sel.size else 0.0

                plt.figure(figsize=(10,6), dpi=150)
                plt.plot(t_k, abs_a_y, label='|a_y|', zorder=2, lw=0.2)
                
                # 평균값을 선으로 표시하고, 범례에 추가
                plt.axhline(mean_abs_ay, ls='--', color='red', label=f'Mean = {mean_abs_ay:.2f} m/s²', lw=0.2)

                plt.title('Absolute Lateral Acceleration')
                plt.xlabel('Time [s]'); plt.ylabel('Lateral Acceleration [m/s²]')
                plt.grid(True, alpha=0.4)
                plt.xlim(0, None)
                plt.ylim(0, 2)
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(self.out_dir, 'lateral_accel_abs.png')); plt.close()

            # ===== 8) 종방향 저크 (Longitudinal Jerk) =====
            if v.size >= 5 and t.size >= 5:
                # (a) 원 데이터의 대표 주기 사용 (10~50 Hz로 클램프)
                dt_raw = np.diff(t)
                dt_raw = dt_raw[np.isfinite(dt_raw) & (dt_raw > 1e-4)]
                dt_med = float(np.median(dt_raw)) if dt_raw.size else 0.05
                dt_target = float(np.clip(dt_med, 0.02, 0.10))

                t_uni = np.arange(t[0], t[-1] + 1e-9, dt_target)
                v_uni = np.interp(t_uni, t, v)

                # (b) 0.3 s 윈도 Savitzky–Golay 스무딩
                def _sg(x, dt, poly=2, span_s=0.30):
                    if not HAVE_SCIPY: return x
                    win = max(5, int(round(span_s / max(dt, 1e-3))))
                    win += (win + 1) % 2  # 홀수 보정
                    if x.size >= win:
                        try: return savgol_filter(x, win, poly)
                        except Exception: return x
                    return x

                v_f   = _sg(v_uni, dt_target, poly=2, span_s=0.30)
                a_uni = np.gradient(v_f, dt_target)
                a_f   = _sg(a_uni, dt_target, poly=2, span_s=0.30)
                jerk_uni = np.gradient(a_f, dt_target)

                # (c) 아주 저속 구간은 저크 신뢰도 낮으므로 제외 (선택)
                speed_mask = np.interp(t_uni, t, v) >= 0.5  # 0.5 m/s 이상만
                mask10     = (t_uni >= 10.0) & np.isfinite(jerk_uni) & speed_mask

                # (d) 통계 (raw vs smooth 비교용)
                jerk_raw   = np.gradient(np.gradient(v_uni, dt_target), dt_target)
                rms_jerk_raw  = float(np.sqrt(np.nanmean(jerk_raw[mask10]**2))) if np.any(mask10) else 0.0
                rms_jerk_smooth = float(np.sqrt(np.nanmean(jerk_uni[mask10]**2))) if np.any(mask10) else 0.0

                # 플롯
                plt.figure(figsize=(10,6), dpi=150)
                plt.plot(t_uni, jerk_uni, label='Longitudinal Jerk (smoothed)', linewidth=0.2)
                plt.axhline(0.0, ls=':', color='gray')
                if rms_jerk_smooth > 1e-6:
                    plt.axhline( rms_jerk_smooth, ls='-',  color='red',   linewidth=0.2,
                                label=f'RMS (t≥10s) = {rms_jerk_smooth:.2f} m/s³')
                    plt.axhline(-rms_jerk_smooth, ls='-',  color='red',   linewidth=0.2)
                plt.title('Longitudinal Jerk')
                plt.xlabel('Time [s]'); plt.ylabel('Jerk [m/s³]')
                plt.grid(True, alpha=0.4); plt.xlim(0, None); plt.legend(loc='lower right')
                plt.ylim(-15, 15)  # 보기 좋게 축도 현실적으로
                plt.savefig(os.path.join(self.out_dir, 'longitudinal_jerk.png')); plt.close()

                # 요약에 저장 (기존 키는 '스무딩' 값으로, raw도 같이 기록)
                rms_jerk = rms_jerk_smooth



            # ===== 9) 조향 포화율 (Steering Saturation) =====
            # 이 분석은 Summary에만 포함
            if steering.size:
                max_steer_abs = 7
                saturated = np.abs(steering) >= max_steer_abs
                if saturated.size > 0:
                    sat_ratio = np.sum(saturated) / len(steering)
                else:
                    sat_ratio = 0
            
            # ===== 10) 조향 총변동 (Total Steering Variation) =====
            if steering.size >= 2:
                total_steer_var = np.sum(np.abs(np.diff(steering)))

            # ===== 11) Summary =====
            summary = {
                'lap_time_s'                : round(lap_time, 3),
                'speed_mean_mps'            : round(float(np.mean(v)) if v.size else 0.0, 6),
                'speed_mean_after_10s_mps'  : round(float(avg_speed_after_10), 6),
                'speed_median_mps'          : round(float(np.median(v)) if v.size else 0.0, 6),
                'speed_std_mps'             : round(float(np.std(v)) if v.size else 0.0, 6),
                'baseline_raw'              : int(self.target_speed_raw),
                'baseline_mps'              : round(float(baseline_mps), 6),

                'path_efficiency_int_abs_kappa': round(int_abs_kappa, 6),
                'curvature_mean_abs_1pm'    : round(mean_abs_kappa, 6),
                'curvature_max_abs_1pm'     : round(max_abs_kappa, 6),
                'curvature_std_abs_1pm'     : round(std_abs_kappa, 6),
                'heading_change_sum_abs_rad': round(sum_abs_dpsi, 6),

                'steer_rate_std_degps'      : round(steer_rate_std, 6),
                'lateral_accel_rms_mps2'    : round(float(rms_ay), 6) if 'rms_ay' in locals() else 0.0,
                'longitudinal_jerk_rms_mps3': round(float(rms_jerk), 6) if 'rms_jerk' in locals() else 0.0,
                'steer_saturation_ratio'    : round(float(sat_ratio), 6) if 'sat_ratio' in locals() else 0.0,
                'total_steer_variation_deg' : round(float(total_steer_var), 6) if 'total_steer_var' in locals() else 0.0,

                'files': [
                    'overlay_trajectory.png',
                    'speed_with_avg.png',
                    'distance_cumulative.png',
                    'steering_hist.png',
                    'steer_rate.png',
                    'curvature.png',
                    'curvature_abs.png',
                    'lateral_accel.png',
                    'lateral_accel_abs.png',
                    'longitudinal_jerk.png',
                ]
            }
            with open(os.path.join(self.out_dir, 'summary.txt'), 'w') as f:
                f.write(json.dumps(summary, indent=2) + '\n')

            print(f'[graph] done. outputs saved in: {self.out_dir}')
        except Exception as e:
            print('[graph] error:', e)
            print(f'[graph] error traceback: {e.__traceback__.tb_lineno}')
            
    def destroy_node(self):
        try:
            self.odom_f.flush(); self.cmd_f.flush()
        except: pass
        try: self.odom_f.close()
        except: pass
        try: self.cmd_f.close()
        except: pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
