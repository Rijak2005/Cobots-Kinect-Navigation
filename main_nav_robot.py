from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import cv2
import numpy as np

from grid_core import KinectGridSystem, map_camera_point_to_color_xy, grid_xy_to_camera
from robot_tracker import ArucoRobotTrackerAuto, RobotPose2D
from cmd_vel_controller import AxisStepController, AxisGains
from lite3_udp_commander import Lite3UdpCommander, Lite3UdpConfig


WINDOW_NAME = "Kinect v2 - Robot Mission (HOME <-> GRID)"
DISPLAY_SCALE = 0.65

ARUCO_STRICTNESS = 0.30
ROBOT_ARUCO_ID = 871
PREFERRED_ARUCO_DICT = "DICT_4X4_1000"
HEADING_OFFSET_DEG = 0.0
MIN_MARKER_SIZE_PX = 40.0

GRID_SPACING_M = 0.60
APPROACH_BEFORE_X_M = 0.30

FIT_EVERY_N_FRAMES = 10
FITS_TO_LOCK = 25
MAX_SAMPLES = 8000
RANSAC_ITERS = 140
INLIER_THRESH_M = 0.015
ROI_X_FRAC = (0.10, 0.90)
ROI_Y_FRAC = (0.20, 0.95)

CONTROL_HZ = 12.0
PAUSE_AT_HOME_S = 1.0
PAUSE_AT_TARGET_S = 1.0

LOST_TIMEOUT_S = 0.80

# IMPORTANT: You confirmed -1 works in your setup
DEFAULT_LAT_SIGN = -1

POINT_COLOR = (0, 255, 0)
TARGET_COLOR = (0, 165, 255)
APPROACH_COLOR = (255, 0, 255)
HOME_COLOR = (255, 255, 0)
LINE_COLOR = (0, 165, 255)
ROBOT_COLOR = (0, 255, 255)
HUD_COLOR = (255, 0, 0)
CLICK_COLOR = (255, 0, 255)


class MissionState(str, Enum):
    NEED_HOME = "NEED_HOME"
    GO_TARGET_APPROACH = "GO_TARGET_APPROACH"
    FACE_TARGET = "FACE_TARGET"
    GO_TARGET_FINAL = "GO_TARGET_FINAL"
    WAIT_TARGET = "WAIT_TARGET"
    GO_HOME = "GO_HOME"
    WAIT_HOME = "WAIT_HOME"
    DONE = "DONE"
    PAUSED = "PAUSED"


@dataclass
class Mission:
    home_xy: Optional[Tuple[float, float]] = None
    targets_xy: List[Tuple[float, float]] = None
    target_index: int = 0
    state: MissionState = MissionState.NEED_HOME
    state_enter_time: float = 0.0
    paused_resume_state: MissionState = MissionState.NEED_HOME

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


def make_targets_row_by_row_like_image() -> List[Tuple[float, float]]:
    s = GRID_SPACING_M
    ys = [-s, 0.0, +s]
    xs = [+s, 0.0, -s]
    return [(x, y) for y in ys for x in xs]


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
        preferred_dict=PREFERRED_ARUCO_DICT,
        heading_offset_deg=HEADING_OFFSET_DEG,
        min_marker_size_px=MIN_MARKER_SIZE_PX,
    )

    # Forward/back is now genuinely slower (near speed is ~7000 = just above deadzone)
    gains = AxisGains(
        axis_fwd_fast=9000,
        axis_fwd_near=7000,
        axis_lat_mag=16000,
        axis_yaw_mag=11000,
        axis_tol_m=0.03,
        arrive_radius_m=0.07,
        stable_reach_ticks=4,
        slow_y_m=0.25,
        slow_x_m=0.25,
        turn_tol_deg=8.0,
        latch_turn=True,
        bump_over_deadzone=True,
    )
    ctrl = AxisStepController(gains)
    ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)

    clicked_color_xy: Optional[Tuple[float, float]] = None
    status_msg = "1) Wait plane lock, 2) Left-click GRID CENTER tape."

    mission = Mission(home_xy=None, targets_xy=[], target_index=0, state=MissionState.NEED_HOME, state_enter_time=time.monotonic())
    paused_manual = False
    paused_reason_lost = False
    lost_since: Optional[float] = None
    frame_idx = 0

    def set_state(s: MissionState) -> None:
        mission.state = s
        mission.state_enter_time = time.monotonic()

    def on_mouse(event, x, y, flags, userdata):
        nonlocal clicked_color_xy, status_msg
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
            mission.target_index = 0
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

            if clicked_color_xy is not None and ksys.grid_frame is None and ksys.plane is not None:
                ok = ksys.set_grid_center_from_color_click(clicked_color_xy, search_radius=160)
                status_msg = "Grid origin set ✅. Put robot at HOME and press 'h'." if ok else "Could not set origin yet. Click again or wait."

            if clicked_color_xy is not None:
                draw_marker_cross(bgr, clicked_color_xy[0], clicked_color_xy[1])

            draw_grid_points(bgr, ksys, GRID_SPACING_M)

            robot: Optional[RobotPose2D] = tracker.detect_and_estimate(bgr)
            tracker.draw_debug(bgr)

            now = time.monotonic()
            marker_seen = bool(tracker.marker_seen_now or (robot is not None))

            # LOST debounce + auto-resume
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
                mission.targets_xy = make_targets_row_by_row_like_image()
                mission.target_index = 0
                set_state(MissionState.GO_TARGET_APPROACH)
                status_msg = "Targets generated. Going to target APPROACH."
                ctrl.reset()
                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)

            cur_t = mission.current_target()
            cur_a = mission.current_approach()

            if mission.home_xy is not None:
                draw_goal_point(bgr, ksys, mission.home_xy, "HOME", HOME_COLOR, radius=10)
            if cur_t is not None:
                draw_goal_point(bgr, ksys, cur_a, "APP", APPROACH_COLOR, radius=8)
                draw_goal_point(bgr, ksys, cur_t, f"T{mission.target_index+1}", TARGET_COLOR, radius=10)

            if now >= next_control:
                next_control = now + dt

                if mission.state == MissionState.PAUSED or paused_manual:
                    commander.stop_robot()
                    ctrl.reset()
                    ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                else:
                    safe_stop = (not commander.is_connected()) or (ksys.grid_frame is None) or (robot is None)
                    if safe_stop:
                        commander.stop_robot()
                        ctrl.reset()
                        ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                    else:
                        if mission.state == MissionState.NEED_HOME:
                            commander.stop_robot()
                            ctrl.reset()
                            ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)

                        elif mission.state == MissionState.GO_TARGET_APPROACH:
                            out = ctrl.move_to_point_y_then_x(robot, cur_a)
                            if out.reached:
                                commander.stop_robot()
                                ctrl.reset()
                                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                set_state(MissionState.FACE_TARGET)
                                status_msg = "At APPROACH. Turning slowly to face target..."
                            else:
                                commander.send_axes(out.ax, out.ay, out.az)

                        elif mission.state == MissionState.FACE_TARGET:
                            out = ctrl.turn_to_face(robot, cur_t)
                            if out.reached:
                                commander.stop_robot()
                                ctrl.reset()
                                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                set_state(MissionState.GO_TARGET_FINAL)
                                status_msg = "Facing target. Forward slowly..."
                            else:
                                commander.send_axes(out.ax, out.ay, out.az)

                        elif mission.state == MissionState.GO_TARGET_FINAL:
                            out = ctrl.forward_to_point(robot, cur_t)
                            if out.reached:
                                commander.stop_robot()
                                ctrl.reset()
                                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                set_state(MissionState.WAIT_TARGET)
                                status_msg = f"At target {mission.target_index+1}. Simulating place..."
                            else:
                                commander.send_axes(out.ax, out.ay, out.az)

                        elif mission.state == MissionState.WAIT_TARGET:
                            commander.stop_robot()
                            if (now - mission.state_enter_time) >= PAUSE_AT_TARGET_S:
                                set_state(MissionState.GO_HOME)
                                ctrl.reset()
                                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                status_msg = "Returning HOME..."

                        elif mission.state == MissionState.GO_HOME:
                            out = ctrl.move_to_point_y_then_x(robot, mission.home_xy)
                            if out.reached:
                                commander.stop_robot()
                                ctrl.reset()
                                ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                set_state(MissionState.WAIT_HOME)
                                status_msg = "At HOME. Simulating pick-up..."
                            else:
                                commander.send_axes(out.ax, out.ay, out.az)

                        elif mission.state == MissionState.WAIT_HOME:
                            commander.stop_robot()
                            if (now - mission.state_enter_time) >= PAUSE_AT_HOME_S:
                                mission.target_index += 1
                                if mission.current_target() is None:
                                    set_state(MissionState.DONE)
                                    status_msg = "All targets done. DONE."
                                else:
                                    set_state(MissionState.GO_TARGET_APPROACH)
                                    ctrl.reset()
                                    ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)
                                    status_msg = f"Next target {mission.target_index+1} -> going to APPROACH"

                        elif mission.state == MissionState.DONE:
                            commander.stop_robot()
                            ctrl.reset()
                            ctrl.set_lateral_sign(DEFAULT_LAT_SIGN)

            hud = [
                status_msg,
                f"UDP: RUNNING robot={robot_ip}:{robot_port} (MOTION=MOVE)",
                f"Mission: {mission.state}  target={mission.target_index+1 if mission.current_target() else '-'} / {len(mission.targets_xy) if mission.targets_xy else '-'}",
                f"Home: {'SET' if mission.home_xy else 'NOT SET'}  GridOrigin: {'SET' if ksys.grid_frame else 'NOT SET'}",
                f"Robot{ROBOT_ARUCO_ID}: seen={marker_seen} pose={'OK' if robot else '---'} strict={ARUCO_STRICTNESS:.2f}",
                f"Axis knobs: fwd_fast={gains.axis_fwd_fast} fwd_near={gains.axis_fwd_near} lat={gains.axis_lat_mag} yaw={gains.axis_yaw_mag} lat_sign={ctrl.get_lateral_sign():+d}",
                f"Plane: {'LOCKED' if ksys.plane_locked else 'CALIBRATING'} Fits: {ksys.fit_count}/{FITS_TO_LOCK}",
                "Keys: q/ESC quit, p pause/resume, SPACE stop, r recalibrate, h set HOME at robot position",
            ]
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
                mission.target_index = 0
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
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())