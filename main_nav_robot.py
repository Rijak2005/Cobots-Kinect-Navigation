from __future__ import annotations

import os
import time
import csv
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List

import cv2
import numpy as np

from grid_core import KinectGridSystem, map_camera_point_to_color_xy, grid_xy_to_camera
from robot_tracker import ArucoRobotTrackerAuto, RobotPose2D
from cmd_vel_controller import AxisStepController, AxisGains, AxisOut
from lite3_udp_commander import Lite3UdpCommander, Lite3UdpConfig


WINDOW_NAME = "Kinect v2 - Robot Mission (HOME <-> GRID)"
DISPLAY_SCALE = 0.7

ARUCO_STRICTNESS = 0.30
ROBOT_ARUCO_ID = 871
ROBOT_ALT_IDS = [0]
PREFERRED_ARUCO_DICT = "DICT_4X4_1000"
HEADING_OFFSET_DEG = 0.0
MIN_MARKER_SIZE_PX = 20.0

GRID_SPACING_M = 0.60

# Target approach is BEFORE target in +X direction (tx - A)
# Home approach is AFTER home in +X direction (hx + A)
APPROACH_BEFORE_X_M = 0.30
HOME_APPROACH_X = 1.10

# Back out a bit more than the approach point
BACK_EXTRA_M = 0.20

# Pose / body height knobs (deadzone for body height is |v| > 20000)
# Must be <= -20001 (or >= +20001) to do anything. Negative lowers.
TARGET_LOWER_AXIS = -24500  # <-- KNOB (start conservative)
LOWER_RAMP_S = 2.0
LOW_HOLD_S = 5.0
POSE_SEND_HZ = 30.0  # send body-height commands at this rate during pose states

FIT_EVERY_N_FRAMES = 10
FITS_TO_LOCK = 25
MAX_SAMPLES = 8000
RANSAC_ITERS = 140
INLIER_THRESH_M = 0.015
ROI_X_FRAC = (0.10, 0.90)
ROI_Y_FRAC = (0.20, 0.95)

CONTROL_HZ = 12.0
PAUSE_AT_HOME_S = 1.0

LOST_TIMEOUT_S = 0.80

DEFAULT_LAT_SIGN = -1

POINT_COLOR = (0, 255, 0)
TARGET_COLOR = (0, 165, 255)
APPROACH_COLOR = (255, 0, 255)
HOME_COLOR = (255, 255, 0)
ROBOT_COLOR = (0, 255, 255)
HUD_COLOR = (255, 0, 0)
CLICK_COLOR = (255, 0, 255)

AXIS_COLOR_X = (0, 255, 255)
AXIS_COLOR_Y = (255, 255, 0)


class MissionState(str, Enum):
    NEED_HOME = "NEED_HOME"

    GO_TARGET_APPROACH = "GO_TARGET_APPROACH"
    FACE_TARGET = "FACE_TARGET"
    GO_TARGET_FINAL = "GO_TARGET_FINAL"

    TARGET_LOWER = "TARGET_LOWER"
    TARGET_HOLD_LOW = "TARGET_HOLD_LOW"
    TARGET_RAISE = "TARGET_RAISE"

    BACK_TO_TARGET_APPROACH = "BACK_TO_TARGET_APPROACH"

    GO_HOME_APPROACH = "GO_HOME_APPROACH"
    GO_HOME_FINAL = "GO_HOME_FINAL"
    WAIT_HOME = "WAIT_HOME"

    DONE = "DONE"
    PAUSED = "PAUSED"


@dataclass
class Mission:
    home_xy: Optional[Tuple[float, float]] = None
    targets_xy: List[Tuple[float, float]] = field(default_factory=list)
    target_index: int = 0
    state: MissionState = MissionState.NEED_HOME

    # Safe default (prevents accidental HOME bounce)
    next_after_home_approach: MissionState = MissionState.GO_TARGET_APPROACH

    state_enter_time: float = 0.0
    paused_resume_state: MissionState = MissionState.NEED_HOME

    have_completed_first_target: bool = False

    def current_target(self) -> Optional[Tuple[float, float]]:
        if not self.targets_xy:
            return None
        if not (0 <= self.target_index < len(self.targets_xy)):
            return None
        return self.targets_xy[self.target_index]

    def current_approach(self) -> Optional[Tuple[float, float]]:
        t = self.current_target()
        if t is None:
            return None
        return (t[0] - APPROACH_BEFORE_X_M, t[1])

    def current_backout_goal(self) -> Optional[Tuple[float, float]]:
        t = self.current_target()
        if t is None:
            return None
        return (t[0] - (APPROACH_BEFORE_X_M + BACK_EXTRA_M), t[1])

    def home_approach(self) -> Optional[Tuple[float, float]]:
        if self.home_xy is None:
            return None
        return (self.home_xy[0] + HOME_APPROACH_X, self.home_xy[1])


def bgra_to_bgr(bgra: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


def draw_hud(img: np.ndarray, lines: list[str]) -> None:
    y = 26
    for s in lines:
        cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD_COLOR, 2, cv2.LINE_AA)
        y += 26


def draw_marker_cross(img: np.ndarray, x: float, y: float, color=CLICK_COLOR) -> None:
    xi, yi = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 0 <= xi < w and 0 <= yi < h:
        cv2.drawMarker(img, (xi, yi), color, markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)


def _put_text_outlined(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.9,
    thickness: int = 2,
    color=(255, 255, 255),
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_grid_points(img: np.ndarray, ksys: KinectGridSystem, spacing_m: float) -> None:
    frame = ksys.grid_frame
    if frame is None:
        return
    for yy in (-spacing_m, 0.0, spacing_m):
        for xx in (-spacing_m, 0.0, spacing_m):
            p_cam = grid_xy_to_camera(xx, yy, frame)
            uv = map_camera_point_to_color_xy(ksys.kinect, p_cam)
            if uv is None:
                continue
            u, v = int(uv[0]), int(uv[1])
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 6, POINT_COLOR, -1, cv2.LINE_AA)


def draw_target_order_numbers(img: np.ndarray, ksys: KinectGridSystem, targets: List[Tuple[float, float]]) -> None:
    frame = ksys.grid_frame
    if frame is None or not targets:
        return

    for i, (tx, ty) in enumerate(targets):
        p_cam = grid_xy_to_camera(tx, ty, frame)
        uv = map_camera_point_to_color_xy(ksys.kinect, p_cam)
        if uv is None:
            continue
        u, v = int(uv[0]), int(uv[1])
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            _put_text_outlined(img, str(i + 1), (u + 12, v - 12), font_scale=0.95, thickness=2)


def draw_goal_point(img, ksys, goal_xy, label, color, radius=10) -> Optional[Tuple[int, int]]:
    if goal_xy is None:
        return None
    frame = ksys.grid_frame
    if frame is None:
        return None
    p_cam = grid_xy_to_camera(goal_xy[0], goal_xy[1], frame)
    uv = map_camera_point_to_color_xy(ksys.kinect, p_cam)
    if uv is None:
        return None
    u, v = int(uv[0]), int(uv[1])
    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
        cv2.circle(img, (u, v), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (u, v), radius + 6, color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (u + 12, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return (u, v)
    return None


def draw_robot(img, robot: RobotPose2D, ksys: KinectGridSystem) -> Optional[Tuple[int, int]]:
    frame = ksys.grid_frame
    if frame is None:
        return None
    p_cam = grid_xy_to_camera(robot.x, robot.y, frame)
    uv = map_camera_point_to_color_xy(ksys.kinect, p_cam)
    if uv is None:
        return None
    u, v = int(uv[0]), int(uv[1])
    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
        cv2.circle(img, (u, v), 8, ROBOT_COLOR, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"Robot: x={robot.x:+.2f} y={robot.y:+.2f} conf={robot.confidence:.2f}",
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            ROBOT_COLOR,
            2,
            cv2.LINE_AA,
        )
        return (u, v)
    return None


def draw_axes_on_floor(img: np.ndarray, ksys: KinectGridSystem, axis_len_m: float = 0.60) -> None:
    frame = ksys.grid_frame
    if frame is None:
        return

    pts_grid = [(0.0, 0.0), (axis_len_m, 0.0), (0.0, axis_len_m)]
    uvs: list[Optional[Tuple[int, int]]] = []

    for gx, gy in pts_grid:
        p_cam = grid_xy_to_camera(gx, gy, frame)
        uvf = map_camera_point_to_color_xy(ksys.kinect, p_cam)
        if uvf is None:
            uvs.append(None)
            continue
        u, v = int(uvf[0]), int(uvf[1])
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            uvs.append((u, v))
        else:
            uvs.append(None)

    o, px, py = uvs
    if o is None:
        return

    cv2.circle(img, o, 6, (255, 255, 255), -1, cv2.LINE_AA)

    if px is not None:
        cv2.arrowedLine(img, o, px, AXIS_COLOR_X, 3, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(img, "+X", (px[0] + 6, px[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, AXIS_COLOR_X, 2, cv2.LINE_AA)

    if py is not None:
        cv2.arrowedLine(img, o, py, AXIS_COLOR_Y, 3, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(img, "+Y", (py[0] + 6, py[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, AXIS_COLOR_Y, 2, cv2.LINE_AA)


def build_target_order_lines(targets: List[Tuple[float, float]]) -> List[str]:
    return [f"{i+1}: ({x:+.2f}, {y:+.2f})" for i, (x, y) in enumerate(targets)]


def make_targets_columnwise_right_to_left_top_to_bottom() -> List[Tuple[float, float]]:
    """
    FIXED to the "other axis" you want:

    Column-wise (not row-wise):
      - Iterate columns RIGHT -> LEFT   (x = +s, 0, -s)
      - Within each column: TOP -> BOTTOM (y = -s, 0, +s)

    This yields numbering like:
      right column: 1,2,3 (top->bottom)
      middle column: 4,5,6
      left column: 7,8,9
    """
    s = GRID_SPACING_M
    xs = [+s, 0.0, -s]      # right -> middle -> left
    ys = [-s, 0.0, +s]      # top -> middle -> bottom
    return [(x, y) for x in xs for y in ys]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _deadzone_safe_lower_value(v: int) -> int:
    v = int(v)
    v = max(-32767, min(32767, v))
    if v == 0:
        return 0
    if abs(v) <= 20000:
        v = -20500 if v < 0 else 20500
    return v


def _ramp_value(t: float, t0: float, ramp_s: float, start: int, end: int) -> int:
    if ramp_s <= 1e-6:
        return end
    a = (t - t0) / ramp_s
    a = float(np.clip(a, 0.0, 1.0))
    return int(round(start + (end - start) * a))


def main() -> int:
    robot_ip = os.environ.get("LITE3_ROBOT_IP", "192.168.2.1").strip()
    robot_port = int(os.environ.get("LITE3_ROBOT_PORT", "43893").strip())
    local_port = int(os.environ.get("LITE3_LOCAL_PORT", "12345").strip())

    commander = Lite3UdpCommander(
        Lite3UdpConfig(
            robot_ip=robot_ip,
            robot_port=robot_port,
            local_port=local_port,
            send_hz=50.0,
            verbose=True,
            yaw_positive_is_right=True,
        )
    )
    commander.start()
    commander.set_motion_mode("move")

    ksys = KinectGridSystem(plane_smooth_alpha=0.15)
    display_w = int(ksys.color_w * DISPLAY_SCALE)
    display_h = int(ksys.color_h * DISPLAY_SCALE)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    tracker = ArucoRobotTrackerAuto(
        kinect_sys=ksys,
        strictness=ARUCO_STRICTNESS,
        robot_id=ROBOT_ARUCO_ID,
        alt_ids=ROBOT_ALT_IDS,
        preferred_dict=PREFERRED_ARUCO_DICT,
        heading_offset_deg=HEADING_OFFSET_DEG,
        min_marker_size_px=MIN_MARKER_SIZE_PX,
    )

    gains = AxisGains(
        axis_fwd_fast=8400,
        axis_fwd_near=7000,
        axis_lat_mag=16000,
        axis_yaw_mag=11000,
        axis_yaw_mag_target=11000,
        axis_tol_m=0.03,
        arrive_radius_m=0.07,
        stable_reach_ticks=4,
        slow_y_m=0.25,
        slow_x_m=0.25,
        turn_tol_deg=8.0,
        latch_turn=True,
        bump_over_deadzone=True,
        target_yaw_start_deg=15.0,
        yaw_pulse_period=3,
        yaw_pulse_on=1,
        reenter_y_hysteresis=2.0,
    )
    ctrl = AxisStepController(gains)
    ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)

    clicked_color_xy: Optional[Tuple[float, float]] = None
    status_msg = "1) Wait plane lock, 2) Left-click GRID CENTER tape."
    targets_order_lines: List[str] = []

    mission = Mission(home_xy=None, targets_xy=[], target_index=0, state=MissionState.NEED_HOME, state_enter_time=time.monotonic())
    paused_manual = False
    paused_reason_lost = False
    lost_since: Optional[float] = None
    frame_idx = 0

    pose_lower = _deadzone_safe_lower_value(TARGET_LOWER_AXIS)
    pose_last_send_t = 0.0
    pose_send_dt = 1.0 / max(5.0, float(POSE_SEND_HZ))

    _ensure_dir("logs")
    log_name = time.strftime("logs/mission_log_%Y%m%d_%H%M%S.csv")
    log_f = open(log_name, "w", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    log_w.writerow([
        "t_mono", "state", "paused_manual", "paused_reason_lost", "marker_seen", "det_id",
        "robot_x", "robot_y", "robot_heading_deg",
        "goal_x", "goal_y", "ex", "ey", "dist",
        "out_phase", "out_reached",
        "sent_ax", "sent_ay", "sent_az",
        "pose_body_height_cmd",
        "note",
    ])
    log_f.flush()

    last_out: Optional[AxisOut] = None
    last_sent = (0, 0, 0)
    last_pose_cmd = 0

    def set_state(s: MissionState) -> None:
        mission.state = s
        mission.state_enter_time = time.monotonic()

    def log_tick(
        now_mono: float,
        state: MissionState,
        marker_seen: bool,
        robot: Optional[RobotPose2D],
        goal: Optional[Tuple[float, float]],
        out: Optional[AxisOut],
        sent: Tuple[int, int, int],
        pose_cmd: int,
        note: str = "",
    ) -> None:
        det_id = tracker.last_detected_id

        if robot is None:
            rx = ry = rh = ""
        else:
            rx = f"{robot.x:.6f}"
            ry = f"{robot.y:.6f}"
            rh = f"{np.degrees(robot.heading):.3f}"

        if goal is None or robot is None:
            gx = gy = ex = ey = dist = ""
        else:
            gx = f"{goal[0]:.6f}"
            gy = f"{goal[1]:.6f}"
            ex_v = goal[0] - robot.x
            ey_v = goal[1] - robot.y
            ex = f"{ex_v:.6f}"
            ey = f"{ey_v:.6f}"
            dist = f"{(ex_v**2 + ey_v**2) ** 0.5:.6f}"

        if out is None:
            out_phase = ""
            out_reached = ""
        else:
            out_phase = out.phase
            out_reached = int(bool(out.reached))

        sent_ax, sent_ay, sent_az = sent

        log_w.writerow([
            f"{now_mono:.6f}",
            state.value,
            int(paused_manual),
            int(paused_reason_lost),
            int(marker_seen),
            det_id if det_id is not None else "",
            rx, ry, rh,
            gx, gy,
            ex, ey, dist,
            out_phase, out_reached,
            sent_ax, sent_ay, sent_az,
            pose_cmd,
            note,
        ])
        log_f.flush()

    def on_mouse(event, x, y, flags, userdata):
        nonlocal clicked_color_xy, status_msg, targets_order_lines
        if event == cv2.EVENT_LBUTTONDOWN:
            cx = float(np.clip(x / DISPLAY_SCALE, 0, ksys.color_w - 1))
            cy = float(np.clip(y / DISPLAY_SCALE, 0, ksys.color_h - 1))
            clicked_color_xy = (cx, cy)
            status_msg = f"Clicked GRID CENTER at ({cx:.1f}, {cy:.1f}). Setting origin..."
        if event == cv2.EVENT_RBUTTONDOWN:
            clicked_color_xy = None
            ksys.grid_frame = None

            mission.home_xy = None
            mission.targets_xy = []
            targets_order_lines = []
            mission.target_index = 0
            mission.have_completed_first_target = False
            mission.next_after_home_approach = MissionState.GO_TARGET_APPROACH

            set_state(MissionState.NEED_HOME)
            ctrl.reset()
            ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
            status_msg = "Cleared origin + HOME. Left-click GRID CENTER again."

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    dt = 1.0 / max(1e-6, CONTROL_HZ)
    next_control = time.monotonic()

    try:
        while True:
            if not ksys.update_frames():
                time.sleep(0.002)
                continue

            ksys.try_update_plane(
                fit_every_n_frames=FIT_EVERY_N_FRAMES,
                frame_idx=frame_idx,
                fits_to_lock=FITS_TO_LOCK,
                max_samples=MAX_SAMPLES,
                ransac_iters=RANSAC_ITERS,
                inlier_thresh_m=INLIER_THRESH_M,
                roi_x_frac=ROI_X_FRAC,
                roi_y_frac=ROI_Y_FRAC,
            )

            bgra = ksys.get_color_bgr()
            if bgra is None:
                time.sleep(0.002)
                continue

            bgr = bgra_to_bgr(bgra)

            bgr_detect = np.ascontiguousarray(bgr.copy())
            robot: Optional[RobotPose2D] = tracker.detect_and_estimate(bgr_detect)

            if clicked_color_xy is not None and ksys.grid_frame is None and ksys.plane is not None:
                ok = ksys.set_grid_center_from_color_click(clicked_color_xy, search_radius=160)
                status_msg = "Grid origin set ✅. Put robot at HOME and press 'h'." if ok else "Could not set origin yet. Click again or wait."

            if clicked_color_xy is not None:
                draw_marker_cross(bgr, clicked_color_xy[0], clicked_color_xy[1])

            draw_axes_on_floor(bgr, ksys, axis_len_m=0.60)
            draw_grid_points(bgr, ksys, GRID_SPACING_M)

            tracker.draw_debug(bgr)

            now = time.monotonic()
            marker_seen = bool(tracker.marker_seen_now or (robot is not None))

            if marker_seen:
                lost_since = None
                if mission.state == MissionState.PAUSED and paused_reason_lost:
                    paused_reason_lost = False
                    set_state(mission.paused_resume_state)
                    status_msg = "Robot visible again -> RESUMED."
                    ctrl.reset()
                    ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
            else:
                if mission.state not in (MissionState.NEED_HOME, MissionState.DONE, MissionState.PAUSED):
                    if lost_since is None:
                        lost_since = now
                    elif (now - lost_since) >= LOST_TIMEOUT_S:
                        if mission.state != MissionState.PAUSED:
                            mission.paused_resume_state = mission.state
                        paused_reason_lost = True
                        paused_manual = False
                        set_state(MissionState.PAUSED)
                        status_msg = "Robot lost -> PAUSED."

            if robot is not None:
                draw_robot(bgr, robot, ksys)

            # Generate targets once HOME + grid origin exist
            if ksys.grid_frame is not None and mission.home_xy is not None and not mission.targets_xy:
                mission.targets_xy = make_targets_columnwise_right_to_left_top_to_bottom()
                mission.target_index = 0
                targets_order_lines = build_target_order_lines(mission.targets_xy)

                print("=== TARGET ORDER (grid coords) ===")
                for line in targets_order_lines:
                    print(line)
                print("===================================")

                mission.next_after_home_approach = MissionState.GO_TARGET_APPROACH
                set_state(MissionState.GO_HOME_APPROACH)
                status_msg = "Targets generated. Going to home APPROACH."
                ctrl.reset()
                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)

            cur_t = mission.current_target()
            cur_a = mission.current_approach()
            cur_back = mission.current_backout_goal()
            home_a = mission.home_approach()

            if mission.home_xy is not None:
                draw_goal_point(bgr, ksys, mission.home_xy, "HOME", HOME_COLOR, radius=10)
                if home_a is not None:
                    draw_goal_point(bgr, ksys, home_a, "H-APP", APPROACH_COLOR, radius=8)

            if cur_t is not None:
                draw_goal_point(bgr, ksys, cur_a, "APP", APPROACH_COLOR, radius=8)
                draw_goal_point(bgr, ksys, cur_back, "BACK", (200, 50, 200), radius=7)
                draw_goal_point(bgr, ksys, cur_t, f"T{mission.target_index+1}", TARGET_COLOR, radius=10)

            # Draw order numbers on top. Preview if not generated yet.
            if ksys.grid_frame is not None:
                targets_for_numbers = mission.targets_xy if mission.targets_xy else make_targets_columnwise_right_to_left_top_to_bottom()
                draw_target_order_numbers(bgr, ksys, targets_for_numbers)

            # ---------- Control loop ----------
            if now >= next_control:
                next_control = now + dt
                last_pose_cmd = 0
                last_out = None
                last_sent = (0, 0, 0)

                if mission.state == MissionState.PAUSED or paused_manual:
                    commander.stop_robot()
                    ctrl.reset()
                    ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                    log_tick(now, mission.state, marker_seen, robot, None, None, last_sent, last_pose_cmd, "paused_or_manual_stop")

                else:
                    safe_stop = (not commander.is_connected()) or (ksys.grid_frame is None) or (robot is None)
                    if safe_stop:
                        commander.stop_robot()
                        ctrl.reset()
                        ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                        log_tick(now, mission.state, marker_seen, robot, None, None, last_sent, last_pose_cmd, "safe_stop_no_pose_or_no_grid")

                    else:
                        goal_for_log: Optional[Tuple[float, float]] = None

                        if mission.state == MissionState.NEED_HOME:
                            commander.stop_robot()
                            ctrl.reset()
                            ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                            log_tick(now, mission.state, marker_seen, robot, None, None, last_sent, last_pose_cmd, "need_home")

                        elif mission.state == MissionState.GO_TARGET_APPROACH:
                            goal_for_log = cur_a
                            out = ctrl.move_to_point_y_then_x(robot, cur_a)
                            last_out = out
                            if out.reached:
                                commander.stop_robot()
                                ctrl.reset()
                                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                set_state(MissionState.FACE_TARGET)
                                status_msg = "At target APPROACH. Micro-yaw (if needed) to face target..."
                            else:
                                commander.send_axes(out.ax, out.ay, out.az)
                                last_sent = (out.ax, out.ay, out.az)
                            log_tick(now, mission.state, marker_seen, robot, goal_for_log, last_out, last_sent, last_pose_cmd, "")

                        elif mission.state == MissionState.FACE_TARGET:
                            goal_for_log = cur_t
                            out = ctrl.turn_to_face_small(robot, cur_t)
                            last_out = out
                            if out.reached:
                                commander.stop_robot()
                                ctrl.reset()
                                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                set_state(MissionState.GO_TARGET_FINAL)
                                status_msg = "Facing target (or yaw skipped). Forward slowly..."
                            else:
                                commander.send_axes(out.ax, out.ay, out.az)
                                last_sent = (out.ax, out.ay, out.az)
                            log_tick(now, mission.state, marker_seen, robot, goal_for_log, last_out, last_sent, last_pose_cmd, "")

                        elif mission.state == MissionState.GO_TARGET_FINAL:
                            goal_for_log = cur_t
                            out = ctrl.forward_to_point(robot, cur_t)
                            last_out = out
                            if out.reached:
                                commander.stop_robot()
                                ctrl.reset()
                                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                set_state(MissionState.TARGET_LOWER)
                                status_msg = f"At target {mission.target_index+1}. Lowering body (POSE)..."
                                commander.set_motion_mode("pose")
                            else:
                                commander.send_axes(out.ax, out.ay, out.az)
                                last_sent = (out.ax, out.ay, out.az)
                            log_tick(now, mission.state, marker_seen, robot, goal_for_log, last_out, last_sent, last_pose_cmd, "")

                        elif mission.state == MissionState.TARGET_LOWER:
                            commander.send_axes(0, 0, 0)
                            t0 = mission.state_enter_time
                            cmd = _ramp_value(now, t0, LOWER_RAMP_S, start=0, end=pose_lower)
                            if (now - pose_last_send_t) >= pose_send_dt:
                                commander.send_body_height_axis_once(cmd)
                                pose_last_send_t = now
                            last_pose_cmd = cmd
                            if (now - t0) >= LOWER_RAMP_S:
                                set_state(MissionState.TARGET_HOLD_LOW)
                                status_msg = "Holding low for 5 seconds..."
                            log_tick(now, mission.state, marker_seen, robot, cur_t, None, (0, 0, 0), last_pose_cmd, "pose_lower")

                        elif mission.state == MissionState.TARGET_HOLD_LOW:
                            commander.send_axes(0, 0, 0)
                            t0 = mission.state_enter_time
                            cmd = pose_lower
                            if (now - pose_last_send_t) >= pose_send_dt:
                                commander.send_body_height_axis_once(cmd)
                                pose_last_send_t = now
                            last_pose_cmd = cmd
                            if (now - t0) >= LOW_HOLD_S:
                                set_state(MissionState.TARGET_RAISE)
                                status_msg = "Raising back to normal height..."
                            log_tick(now, mission.state, marker_seen, robot, cur_t, None, (0, 0, 0), last_pose_cmd, "pose_hold_low")

                        elif mission.state == MissionState.TARGET_RAISE:
                            commander.send_axes(0, 0, 0)
                            t0 = mission.state_enter_time
                            cmd = _ramp_value(now, t0, LOWER_RAMP_S, start=pose_lower, end=0)
                            if (now - pose_last_send_t) >= pose_send_dt:
                                commander.send_body_height_axis_once(cmd)
                                pose_last_send_t = now
                            last_pose_cmd = cmd
                            if (now - t0) >= LOWER_RAMP_S:
                                commander.set_motion_mode("move")
                                set_state(MissionState.BACK_TO_TARGET_APPROACH)
                                ctrl.reset()
                                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                status_msg = "Back to MOVE. Backing out beyond approach..."
                            log_tick(now, mission.state, marker_seen, robot, cur_t, None, (0, 0, 0), last_pose_cmd, "pose_raise")

                        elif mission.state == MissionState.BACK_TO_TARGET_APPROACH:
                            goal_for_log = cur_back
                            out = ctrl.backward_to_point(robot, cur_back)
                            last_out = out
                            if out.reached:
                                commander.stop_robot()
                                ctrl.reset()
                                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                mission.have_completed_first_target = True
                                mission.next_after_home_approach = MissionState.GO_HOME_FINAL
                                set_state(MissionState.GO_HOME_APPROACH)
                                status_msg = "Backed out. Going to HOME-APPROACH..."
                            else:
                                commander.send_axes(out.ax, out.ay, out.az)
                                last_sent = (out.ax, out.ay, out.az)
                            log_tick(now, mission.state, marker_seen, robot, goal_for_log, last_out, last_sent, last_pose_cmd, "")

                        elif mission.state == MissionState.GO_HOME_APPROACH:
                            goal = home_a if (home_a is not None) else mission.home_xy
                            goal_for_log = goal
                            out = ctrl.move_to_point_y_then_x(robot, goal)
                            last_out = out
                            if out.reached:
                                commander.stop_robot()
                                ctrl.reset()
                                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                nxt = mission.next_after_home_approach
                                set_state(nxt)
                                if nxt == MissionState.GO_TARGET_APPROACH:
                                    status_msg = "At HOME-APPROACH. Now going to target APPROACH..."
                                elif nxt == MissionState.GO_HOME_FINAL:
                                    status_msg = "At HOME-APPROACH. Translating to HOME (no yaw)..."
                                else:
                                    status_msg = f"At HOME-APPROACH. Next: {nxt}"
                            else:
                                commander.send_axes(out.ax, out.ay, out.az)
                                last_sent = (out.ax, out.ay, out.az)
                            log_tick(now, mission.state, marker_seen, robot, goal_for_log, last_out, last_sent, last_pose_cmd, "")

                        elif mission.state == MissionState.GO_HOME_FINAL:
                            goal_for_log = mission.home_xy
                            out = ctrl.move_to_point_y_then_x(robot, mission.home_xy)
                            last_out = out
                            if out.reached:
                                commander.stop_robot()
                                ctrl.reset()
                                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                set_state(MissionState.WAIT_HOME)
                                status_msg = "At HOME. Simulating pick-up..."
                            else:
                                commander.send_axes(out.ax, out.ay, out.az)
                                last_sent = (out.ax, out.ay, out.az)
                            log_tick(now, mission.state, marker_seen, robot, goal_for_log, last_out, last_sent, last_pose_cmd, "")

                        elif mission.state == MissionState.WAIT_HOME:
                            commander.stop_robot()
                            if (now - mission.state_enter_time) >= PAUSE_AT_HOME_S:
                                mission.target_index += 1
                                if mission.current_target() is None:
                                    set_state(MissionState.DONE)
                                    status_msg = "All targets done. DONE."
                                else:
                                    mission.next_after_home_approach = MissionState.GO_TARGET_APPROACH
                                    set_state(MissionState.GO_HOME_APPROACH)
                                    ctrl.reset()
                                    ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                    status_msg = f"Next target {mission.target_index+1} -> going to APPROACH"
                            log_tick(now, mission.state, marker_seen, robot, mission.home_xy, None, (0, 0, 0), 0, "wait_home")

                        elif mission.state == MissionState.DONE:
                            commander.stop_robot()
                            ctrl.reset()
                            ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                            log_tick(now, mission.state, marker_seen, robot, None, None, (0, 0, 0), 0, "done")

            if last_out is None:
                out_line = "CtrlOut: (none)"
            else:
                out_line = f"CtrlOut: phase={last_out.phase} ax={last_out.ax} ay={last_out.ay} az={last_out.az} reached={last_out.reached}"

            hud = [
                status_msg,
                f"UDP: RUNNING robot={robot_ip}:{robot_port} (MOTION={'POSE' if mission.state in (MissionState.TARGET_LOWER, MissionState.TARGET_HOLD_LOW, MissionState.TARGET_RAISE) else 'MOVE'})  LOG={log_name}",
                f"Mission: {mission.state}  target={mission.target_index+1 if mission.current_target() else '-'} / {len(mission.targets_xy) if mission.targets_xy else '-'}",
                f"Home: {'SET' if mission.home_xy else 'NOT SET'}  GridOrigin: {'SET' if ksys.grid_frame else 'NOT SET'}",
                f"RobotIDs: primary={ROBOT_ARUCO_ID} alt={ROBOT_ALT_IDS}  seen={marker_seen} pose={'OK' if robot else '---'} det_id={tracker.last_detected_id} strict={ARUCO_STRICTNESS:.2f}",
                f"Axis knobs: fwd_fast={gains.axis_fwd_fast} fwd_near={gains.axis_fwd_near} lat={gains.axis_lat_mag} yawT={gains.axis_yaw_mag_target} lat_sign={ctrl.get_lateral_sign():+d}",
                f"Pose knobs: lower_axis={pose_lower} ramp_s={LOWER_RAMP_S:.1f} hold_s={LOW_HOLD_S:.1f} back_extra={BACK_EXTRA_M:.2f}m",
                out_line,
                f"Plane: {'LOCKED' if ksys.plane_locked else 'CALIBRATING'} Fits: {ksys.fit_count}/{FITS_TO_LOCK}",
                "Keys: q/ESC quit, p pause/resume, SPACE stop, r recalibrate, h set HOME at robot position",
            ]
            if targets_order_lines:
                hud.append("Target order (grid):")
                hud.extend(targets_order_lines)

            draw_hud(bgr, hud)

            disp = cv2.resize(bgr, (display_w, display_h), interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if key == ord("p"):
                if mission.state != MissionState.PAUSED:
                    mission.paused_resume_state = mission.state
                if mission.state == MissionState.PAUSED and paused_manual:
                    paused_manual = False
                    set_state(mission.paused_resume_state)
                    status_msg = "Manual resume."
                else:
                    paused_manual = True
                    paused_reason_lost = False
                    set_state(MissionState.PAUSED)
                    status_msg = "Manual pause."
                commander.stop_robot()
                ctrl.reset()
                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)

            if key == 32:  # SPACE
                paused_manual = True
                paused_reason_lost = False
                set_state(MissionState.PAUSED)
                status_msg = "STOP. Press 'p' to resume."
                commander.stop_robot()
                ctrl.reset()
                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)

            if key == ord("r"):
                paused_manual = True
                paused_reason_lost = False
                commander.stop_robot()
                ctrl.reset()

                ksys.recalibrate_plane()
                ksys.grid_frame = None
                clicked_color_xy = None

                mission.home_xy = None
                mission.targets_xy = []
                targets_order_lines = []
                mission.target_index = 0
                mission.have_completed_first_target = False
                mission.next_after_home_approach = MissionState.GO_TARGET_APPROACH

                set_state(MissionState.NEED_HOME)
                status_msg = "Recalibrating. Left-click GRID CENTER again."
                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)

            if key == ord("h"):
                if robot is not None:
                    mission.home_xy = (float(robot.x), float(robot.y))
                    status_msg = f"HOME set: x={mission.home_xy[0]:+.2f}, y={mission.home_xy[1]:+.2f}"
                    paused_manual = False
                    paused_reason_lost = False
                    lost_since = None
                    ctrl.reset()
                    ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                else:
                    status_msg = "Cannot set HOME: robot pose not available."

            frame_idx += 1
            time.sleep(0.001)

    finally:
        try:
            commander.stop_robot()
            commander.set_motion_mode("pose")
            time.sleep(0.2)
        except Exception:
            pass
        commander.shutdown()

        try:
            ksys.close()
        except Exception:
            pass

        try:
            log_f.close()
        except Exception:
            pass

        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())