from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import cv2
import numpy as np

from grid_core import (
    KinectGridSystem,
    map_camera_point_to_color_xy,
    grid_xy_to_camera,
)
from robot_tracker import ArucoRobotTrackerAuto, RobotPose2D
from cmd_vel_controller import TwoStageCmdVelController, TwoStageGains
from olo_rosbridge import RosbridgeCommander, RosbridgeConfig


# ---------------- Window / Display ----------------
WINDOW_NAME = "Kinect v2 - Robot Mission (HOME <-> GRID)"
DISPLAY_SCALE = 0.6

# ---------------- Robot / ArUco ----------------
ARUCO_STRICTNESS = 0.65
ROBOT_ARUCO_ID = 871
PREFERRED_ARUCO_DICT = "DICT_4X4_1000"

# ---------------- Grid ----------------
GRID_SPACING_M = 0.60  # 60 cm

# ---------------- Floor plane fitting ----------------
FIT_EVERY_N_FRAMES = 10
FITS_TO_LOCK = 25
MAX_SAMPLES = 8000
RANSAC_ITERS = 140
INLIER_THRESH_M = 0.015
ROI_X_FRAC = (0.10, 0.90)
ROI_Y_FRAC = (0.20, 0.95)

# ---------------- Control ----------------
CONTROL_HZ = 10.0
PAUSE_AT_HOME_S = 1.5     # simulate picking stilt
PAUSE_AT_TARGET_S = 1.5   # simulate placing stilt

# ---------------- Colors (BGR) ----------------
POINT_COLOR = (0, 255, 0)        # grid points
GOAL_COLOR = (0, 165, 255)       # goal marker
HOME_COLOR = (255, 255, 0)       # home marker (cyan-ish)
LINE_COLOR = (0, 165, 255)
ROBOT_COLOR = (0, 255, 255)
HUD_COLOR = (255, 255, 255)
CLICK_COLOR = (255, 0, 255)


class MissionState(str, Enum):
    NEED_HOME = "NEED_HOME"
    GO_HOME = "GO_HOME"
    WAIT_HOME = "WAIT_HOME"
    GO_TARGET = "GO_TARGET"
    WAIT_TARGET = "WAIT_TARGET"
    DONE = "DONE"
    PAUSED = "PAUSED"


@dataclass
class Mission:
    home_xy: Optional[Tuple[float, float]] = None
    targets_xy: List[Tuple[float, float]] = None
    target_index: int = 0
    state: MissionState = MissionState.NEED_HOME
    state_enter_time: float = 0.0

    def current_target(self) -> Optional[Tuple[float, float]]:
        if not self.targets_xy:
            return None
        if self.target_index < 0 or self.target_index >= len(self.targets_xy):
            return None
        return self.targets_xy[self.target_index]


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


def draw_goal_point(
    img: np.ndarray,
    ksys: KinectGridSystem,
    goal_xy: Optional[Tuple[float, float]],
    label: str,
    color: Tuple[int, int, int],
) -> Optional[Tuple[int, int]]:
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
        cv2.circle(img, (u, v), 10, color, -1, cv2.LINE_AA)
        cv2.circle(img, (u, v), 16, color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (u + 12, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return (u, v)
    return None


def draw_robot(img: np.ndarray, robot: RobotPose2D, ksys: KinectGridSystem) -> Optional[Tuple[int, int]]:
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
    """
    You asked: start far right of picture and move down row by row.
    Our grid coordinate system:
      +x is "right" on the image, +y is "down" if your basis was created that way.
    Your earlier grid overlay follows image directions; so:
      row1: (+x, -y), (0, -y), (-x, -y)
      row2: (+x, 0),  (0, 0),  (-x, 0)
      row3: (+x, +y), (0, +y), (-x, +y)
    """
    s = GRID_SPACING_M
    ys = [-s, 0.0, +s]
    xs = [+s, 0.0, -s]  # far right first
    out: List[Tuple[float, float]] = []
    for y in ys:
        for x in xs:
            out.append((x, y))
    return out


def main() -> int:
    ws_url = "wss://app.olo-robotics.com/rosbridge?robotId=6800d52b-5777-4411-8ead-9aa10662def6&token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjgxYWFmZTM2LWEyNmEtNDg4Ni04NDYwLTFjMzk5YTQ5M2FkMSIsInVzZXJuYW1lIjoibGVvbi5kcmVzZWxAc3Qub3RoLXJlZ2Vuc2J1cmcuZGUiLCJpYXQiOjE3Njc4ODA0NjUsImV4cCI6MTc2Nzk2Njg2NX0.X0bNHjun6DmeQp1M8tM_2DSBsG8bZ2OCL0ATTlSl4XI"
    if not ws_url:
        print("ERROR: Please set OLO_ROSBRIDGE_URL to your wss://... rosbridge URL.")
        return 1

    commander = RosbridgeCommander(
        RosbridgeConfig(
            url=ws_url,
            send_hz=10.0,
            verbose=True,
            # If your gateway is picky, you can set ping_interval_s=None/ping_timeout_s=None
            ping_interval_s=20.0,
            ping_timeout_s=20.0,
            debug_print_period_s=2.0,
        )
    )
    commander.start()
    commander.set_mode("move")

    # Kinect system
    ksys = KinectGridSystem(plane_smooth_alpha=0.15)
    display_w = int(ksys.color_w * DISPLAY_SCALE)
    display_h = int(ksys.color_h * DISPLAY_SCALE)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # Tracker
    tracker = ArucoRobotTrackerAuto(
        kinect_sys=ksys,
        strictness=ARUCO_STRICTNESS,
        robot_id=ROBOT_ARUCO_ID,
        preferred_dict=PREFERRED_ARUCO_DICT,
    )

    # Two-stage controller
    ctrl = TwoStageCmdVelController(
        TwoStageGains(
            coarse_radius_m=0.10,
            success_radius_m=0.04,
            stable_reach_ticks=5,
            # keep conservative max speed; adjust later
            coarse_max_lin=0.28,
            coarse_max_ang=0.85,
            fine_max_lin=0.10,
            fine_max_ang=0.45,
            allow_reverse_fine=False,
        )
    )

    clicked_color_xy: Optional[Tuple[float, float]] = None
    status_msg = "1) Wait for plane lock, 2) Left-click GRID CENTER tape."

    mission = Mission(home_xy=None, targets_xy=[], target_index=0, state=MissionState.NEED_HOME, state_enter_time=time.monotonic())

    def set_state(new_state: MissionState) -> None:
        mission.state = new_state
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
            status_msg = "Cleared origin + HOME. Left-click GRID CENTER again."

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # control timing
    dt = 1.0 / max(1e-6, CONTROL_HZ)
    next_control = time.monotonic()

    paused = False
    frame_idx = 0

    try:
        while True:
            # --- update frames ---
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

            # --- set grid origin from click ---
            if clicked_color_xy is not None and ksys.grid_frame is None and ksys.plane is not None:
                ok = ksys.set_grid_center_from_color_click(clicked_color_xy, search_radius=160)
                if ok:
                    status_msg = "Grid origin set ✅. Now RIGHT-CLICK HOME position on floor."
                    set_state(MissionState.NEED_HOME)
                else:
                    status_msg = "Could not set origin yet. Click again or wait."

            # Draw click cross
            if clicked_color_xy is not None:
                draw_marker_cross(bgr, clicked_color_xy[0], clicked_color_xy[1])

            # --- draw grid overlay ---
            draw_grid_points(bgr, ksys, GRID_SPACING_M)

            # --- tracking ---
            robot: Optional[RobotPose2D] = tracker.detect_and_estimate(bgr)
            tracker.draw_debug(bgr)
            robot_uv = draw_robot(bgr, robot, ksys) if robot is not None else None

            # --- HOME selection (right click) ---
            # We already used right click to clear everything; so use key 'h' to set HOME at robot position.
            # This avoids confusing UI and makes HOME easy: just put robot at home station and press 'h'.
            # (Less clicking on floor.)
            #
            # If you prefer mouse-based home marking, tell me and I’ll add it back.
            if mission.home_xy is not None:
                draw_goal_point(bgr, ksys, mission.home_xy, "HOME", HOME_COLOR)

            # --- create targets once we have origin + home ---
            if ksys.grid_frame is not None and mission.home_xy is not None and not mission.targets_xy:
                mission.targets_xy = make_targets_row_by_row_like_image()
                mission.target_index = 0
                set_state(MissionState.GO_HOME)
                status_msg = "Targets generated. Going HOME first."

            # --- mission goal selection ---
            current_target = mission.current_target()
            active_goal_xy: Optional[Tuple[float, float]] = None
            goal_label = ""

            if paused:
                set_state(MissionState.PAUSED)

            if mission.state == MissionState.NEED_HOME:
                # no driving
                active_goal_xy = None
                goal_label = ""
            elif mission.state in (MissionState.GO_HOME, MissionState.WAIT_HOME):
                active_goal_xy = mission.home_xy
                goal_label = "HOME"
            elif mission.state in (MissionState.GO_TARGET, MissionState.WAIT_TARGET):
                active_goal_xy = current_target
                goal_label = f"T{mission.target_index+1}"
            elif mission.state == MissionState.DONE:
                active_goal_xy = mission.home_xy
                goal_label = "HOME"
            else:
                active_goal_xy = None

            goal_uv = None
            if active_goal_xy is not None:
                color = HOME_COLOR if goal_label == "HOME" else GOAL_COLOR
                goal_uv = draw_goal_point(bgr, ksys, active_goal_xy, goal_label, color)
                if goal_uv is not None and robot_uv is not None:
                    cv2.line(bgr, robot_uv, goal_uv, LINE_COLOR, 2, cv2.LINE_AA)

            # --- control tick ---
            now = time.monotonic()
            if now >= next_control:
                next_control = now + dt

                if paused or (not commander.is_connected()) or ksys.grid_frame is None:
                    commander.stop_robot()
                    ctrl.reset()
                else:
                    # Decide what the robot should do based on mission state.
                    if mission.state == MissionState.NEED_HOME:
                        commander.stop_robot()
                        ctrl.reset()

                    elif mission.state == MissionState.GO_HOME:
                        out = ctrl.compute(robot, mission.home_xy)
                        if out.reached:
                            commander.stop_robot()
                            ctrl.reset()
                            set_state(MissionState.WAIT_HOME)
                        else:
                            commander.send_cmd_vel(out.linear_x, out.angular_z)

                    elif mission.state == MissionState.WAIT_HOME:
                        commander.stop_robot()
                        # simulate "pick stilt"
                        if (now - mission.state_enter_time) >= PAUSE_AT_HOME_S:
                            # after home wait -> go to target if any left
                            if current_target is None:
                                set_state(MissionState.DONE)
                            else:
                                set_state(MissionState.GO_TARGET)

                    elif mission.state == MissionState.GO_TARGET:
                        out = ctrl.compute(robot, current_target)
                        if out.reached:
                            commander.stop_robot()
                            ctrl.reset()
                            set_state(MissionState.WAIT_TARGET)
                        else:
                            commander.send_cmd_vel(out.linear_x, out.angular_z)

                    elif mission.state == MissionState.WAIT_TARGET:
                        commander.stop_robot()
                        # simulate "place stilt"
                        if (now - mission.state_enter_time) >= PAUSE_AT_TARGET_S:
                            # Finished placing -> return HOME
                            mission.target_index += 1
                            if mission.current_target() is None:
                                # All targets done -> return home and stop
                                set_state(MissionState.GO_HOME)
                                status_msg = "All targets done. Returning HOME."
                            else:
                                set_state(MissionState.GO_HOME)

                    elif mission.state == MissionState.DONE:
                        commander.stop_robot()
                        ctrl.reset()

                    elif mission.state == MissionState.PAUSED:
                        commander.stop_robot()
                        ctrl.reset()

            # --- HUD ---
            mode_seen = commander.last_mode_seen()
            hud = [
                f"ROS: {'CONNECTED' if commander.is_connected() else 'DISCONNECTED'}  mode={mode_seen}",
                f"Mission: {mission.state}  target={mission.target_index+1 if mission.current_target() else '-'} / {len(mission.targets_xy) if mission.targets_xy else '-'}",
                f"Home: {'SET' if mission.home_xy is not None else 'NOT SET'}  GridOrigin: {'SET' if ksys.grid_frame is not None else 'NOT SET'}",
                f"Robot: {'OK' if robot is not None else '---'}  ArucoID={ROBOT_ARUCO_ID} strict={ARUCO_STRICTNESS:.2f}",
                f"Plane: {'LOCKED' if ksys.plane_locked else 'CALIBRATING'}  Fits: {ksys.fit_count}/{FITS_TO_LOCK}",
                "Keys: q/ESC quit, p pause, SPACE emergency stop, r recalibrate, h set HOME at robot position",
            ]
            if status_msg:
                hud.insert(0, status_msg)
            draw_hud(bgr, hud)

            disp = cv2.resize(bgr, (display_w, display_h), interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if key == ord("p"):
                paused = not paused
                if paused:
                    status_msg = "Paused."
                    set_state(MissionState.PAUSED)
                    commander.stop_robot()
                    ctrl.reset()
                else:
                    status_msg = "Resumed."
                    # resume into a safe state
                    if mission.home_xy is None:
                        set_state(MissionState.NEED_HOME)
                    elif mission.targets_xy and mission.current_target() is not None:
                        set_state(MissionState.GO_HOME)
                    else:
                        set_state(MissionState.GO_HOME)

            if key == 32:  # SPACE
                paused = True
                status_msg = "EMERGENCY STOP. Press 'p' to resume."
                set_state(MissionState.PAUSED)
                commander.stop_robot()
                ctrl.reset()

            if key == ord("r"):
                paused = True
                commander.stop_robot()
                ctrl.reset()
                ksys.recalibrate_plane()
                ksys.grid_frame = None
                clicked_color_xy = None
                mission.home_xy = None
                mission.targets_xy = []
                mission.target_index = 0
                set_state(MissionState.NEED_HOME)
                status_msg = "Recalibrating plane. Left-click GRID CENTER again."

            if key == ord("h"):
                # Set HOME at the robot's current (x,y) in grid coordinates.
                if robot is not None:
                    mission.home_xy = (float(robot.x), float(robot.y))
                    status_msg = f"HOME set at robot position: x={mission.home_xy[0]:+.2f}, y={mission.home_xy[1]:+.2f}"
                    if ksys.grid_frame is not None and not mission.targets_xy:
                        # targets will be generated automatically in loop
                        pass
                else:
                    status_msg = "Cannot set HOME: robot not visible."

            frame_idx += 1
            time.sleep(0.001)

    finally:
        # Always stop robot and request stand
        try:
            commander.stop_robot()
            commander.set_mode("stand")
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
