from __future__ import annotations

import asyncio
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from grid_core import KinectGridSystem, map_camera_point_to_color_xy, grid_xy_to_camera
from robot_tracker import ArucoRobotTrackerAuto, RobotPose2D
from navigator import GridNavigator, NavConfig
from cmd_vel_controller import CmdVelController, CmdVelGains
from olo_rosbridge import OloRosbridgeClient, RosbridgeConfig

WINDOW_NAME = "Kinect v2 - Robot Navigation"
DISPLAY_SCALE = 0.6

# ---------------- User tuning ----------------
ARUCO_STRICTNESS = 0.65
ROBOT_ARUCO_ID = 871
PREFERRED_ARUCO_DICT = "DICT_4X4_1000"  # optional but matches ID range

# Navigation tolerances
POS_TOL_M = 0.05           # 5 cm
CONTROL_HZ = 10.0          # cmd_vel update rate
# --------------------------------------------

# Floor calibration
FIT_EVERY_N_FRAMES = 10
FITS_TO_LOCK = 25
MAX_SAMPLES = 8000
RANSAC_ITERS = 140
INLIER_THRESH_M = 0.015
ROI_X_FRAC = (0.10, 0.90)
ROI_Y_FRAC = (0.20, 0.95)

GRID_SPACING_M = 0.60

# Colors (BGR)
POINT_COLOR = (0, 255, 0)
GOAL_COLOR = (0, 165, 255)     # orange
LINE_COLOR = (0, 165, 255)
ROBOT_COLOR = (0, 255, 255)    # yellow
HUD_COLOR = (255, 255, 255)
CLICK_COLOR = (255, 0, 255)


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
    for y in (-spacing_m, 0.0, spacing_m):
        for x in (-spacing_m, 0.0, spacing_m):
            p_cam = grid_xy_to_camera(x, y, frame)
            uv = map_camera_point_to_color_xy(ksys.kinect, p_cam)
            if uv is None:
                continue
            u, v = int(uv[0]), int(uv[1])
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 6, POINT_COLOR, -1, cv2.LINE_AA)


def draw_goal(img: np.ndarray, ksys: KinectGridSystem, goal_xy: Tuple[float, float]) -> Optional[Tuple[int, int]]:
    frame = ksys.grid_frame
    if frame is None:
        return None
    p_cam = grid_xy_to_camera(goal_xy[0], goal_xy[1], frame)
    uv = map_camera_point_to_color_xy(ksys.kinect, p_cam)
    if uv is None:
        return None
    u, v = int(uv[0]), int(uv[1])
    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
        cv2.circle(img, (u, v), 10, GOAL_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, (u, v), 16, GOAL_COLOR, 2, cv2.LINE_AA)
        cv2.putText(img, "GOAL", (u + 12, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GOAL_COLOR, 2, cv2.LINE_AA)
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


async def run() -> int:
    # 1) Read websocket URL from environment (recommended).
    #    Set it in PowerShell:
    #      $env:OLO_ROSBRIDGE_URL="wss://..."
    # ws_url = os.environ.get("OLO_ROSBRIDGE_URL", "").strip()
    ws_url = "wss://app.olo-robotics.com/rosbridge?robotId=6800d52b-5777-4411-8ead-9aa10662def6&token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjgxYWFmZTM2LWEyNmEtNDg4Ni04NDYwLTFjMzk5YTQ5M2FkMSIsInVzZXJuYW1lIjoibGVvbi5kcmVzZWxAc3Qub3RoLXJlZ2Vuc2J1cmcuZGUiLCJpYXQiOjE3Njc4ODA0NjUsImV4cCI6MTc2Nzk2Njg2NX0.X0bNHjun6DmeQp1M8tM_2DSBsG8bZ2OCL0ATTlSl4XI"
    if not ws_url:
        print("ERROR: Please set OLO_ROSBRIDGE_URL environment variable to your wss://... rosbridge URL.")
        return 1

    # 2) Connect rosbridge
    client = OloRosbridgeClient(RosbridgeConfig(url=ws_url))
    print("Connecting to robot rosbridge...")
    await client.connect()
    print("Connected.")

    # Advertise topics we will publish
    await client.advertise("/cmd_vel", "geometry_msgs/msg/Twist")
    await client.advertise("/set_mode_cmd", "std_msgs/msg/String")

    # Set mode to move (matches your example script)
    current_mode = await client.get_current_mode()
    print("Robot current mode:", current_mode)
    print("Setting robot mode to 'move' ...")
    await client.set_mode("move")
    await asyncio.sleep(1.0)

    # 3) Init Kinect system
    ksys = KinectGridSystem(plane_smooth_alpha=0.15)
    display_w = int(ksys.color_w * DISPLAY_SCALE)
    display_h = int(ksys.color_h * DISPLAY_SCALE)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # 4) Tracker + Navigator + Controller
    tracker = ArucoRobotTrackerAuto(
        kinect_sys=ksys,
        strictness=ARUCO_STRICTNESS,
        robot_id=ROBOT_ARUCO_ID,
        preferred_dict=PREFERRED_ARUCO_DICT,
    )

    nav = GridNavigator(
        NavConfig(
            command_interval_s=0.25,   # still prints status at this interval (optional)
            pos_tolerance_m=POS_TOL_M,
            turn_start_deg=30.0,
            turn_stop_deg=12.0,
        )
    )
    ctrl = CmdVelController(CmdVelGains())

    # Clicked grid center
    clicked_color_xy: Optional[Tuple[float, float]] = None
    status_msg = "1) Wait for plane lock, 2) Left-click taped grid center."

    def on_mouse(event, x, y, flags, userdata):
        nonlocal clicked_color_xy, status_msg
        if event == cv2.EVENT_LBUTTONDOWN:
            cx = float(np.clip(x / DISPLAY_SCALE, 0, ksys.color_w - 1))
            cy = float(np.clip(y / DISPLAY_SCALE, 0, ksys.color_h - 1))
            clicked_color_xy = (cx, cy)
            status_msg = f"Clicked grid center at ({cx:.1f}, {cy:.1f}). Setting origin..."
        if event == cv2.EVENT_RBUTTONDOWN:
            clicked_color_xy = None
            ksys.grid_frame = None
            status_msg = "Cleared grid center. Left-click again."

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    targets_initialized = False
    paused = False
    frame_idx = 0

    # Control loop timing
    dt = 1.0 / CONTROL_HZ
    next_control_time = time.monotonic()

    try:
        while True:
            # --- Kinect update ---
            if not ksys.update_frames():
                await asyncio.sleep(0.002)
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
                await asyncio.sleep(0.002)
                continue
            bgr = bgra_to_bgr(bgra)

            # set origin from click
            if clicked_color_xy is not None and ksys.grid_frame is None and ksys.plane is not None:
                ok = ksys.set_grid_center_from_color_click(clicked_color_xy, search_radius=160)
                status_msg = "Grid origin set. Show robot marker." if ok else "Could not set origin yet. Click again or wait."

            if clicked_color_xy is not None:
                draw_marker_cross(bgr, clicked_color_xy[0], clicked_color_xy[1])

            draw_grid_points(bgr, ksys, GRID_SPACING_M)

            # --- Tracking ---
            robot: Optional[RobotPose2D] = tracker.detect_and_estimate(bgr)
            tracker.draw_debug(bgr)
            robot_uv = draw_robot(bgr, robot, ksys) if robot is not None else None

            # --- Targets init ---
            if (not targets_initialized) and (ksys.grid_frame is not None):
                targets = nav.make_targets_3x3_ordered_like_image(
                    frame=ksys.grid_frame,
                    kinect=ksys.kinect,
                    spacing_m=GRID_SPACING_M,
                )
                nav.set_targets(targets)
                targets_initialized = True
                status_msg = "Targets set. Robot driving. Press 'n' after placing stilt (or when safe)."

            goal_xy = nav.current_target() if targets_initialized else None
            goal_uv = None
            if goal_xy is not None:
                goal_uv = draw_goal(bgr, ksys, goal_xy)
                if goal_uv is not None and robot_uv is not None:
                    cv2.line(bgr, robot_uv, goal_uv, LINE_COLOR, 2, cv2.LINE_AA)

            # --- Control tick (send cmd_vel) ---
            now = time.monotonic()
            if now >= next_control_time:
                next_control_time = now + dt

                if paused or (not targets_initialized) or (ksys.grid_frame is None):
                    await client.send_cmd_vel(0.0, 0.0)
                    ctrl.reset()
                else:
                    out = ctrl.compute(robot, goal_xy, pos_tol_m=POS_TOL_M)

                    # If reached: stop and wait for 'n' like before (safe)
                    if out.reached:
                        await client.send_cmd_vel(0.0, 0.0)
                        nav._waiting_for_stilt = True  # keep existing workflow
                    else:
                        await client.send_cmd_vel(out.linear_x, out.angular_z)

                    # Optional: print status
                    # print(out.status)

            # --- HUD ---
            hud = [
                f"RobotID: {ROBOT_ARUCO_ID}  Strictness: {ARUCO_STRICTNESS:.2f}  CtrlHz: {CONTROL_HZ:.1f}",
                f"Depth: {'OK' if ksys.last_depth_1d is not None else '---'}  Plane: {'LOCKED' if ksys.plane_locked else 'CALIBRATING'}  Fits: {ksys.fit_count}/{FITS_TO_LOCK}",
                f"Grid origin: {'SET' if ksys.grid_frame is not None else 'NOT SET'}  Robot: {'OK' if robot is not None else '---'}",
                f"Aruco: last_dict={tracker.last_dict_used} ids={tracker.last_detected_ids} perim={tracker.last_perimeter_px:.0f}px",
                f"Nav: {'PAUSED' if paused else 'RUNNING'}  Target: {nav.current_index + 1 if targets_initialized and not nav.is_done() else '-'} / {len(nav.targets_xy) if targets_initialized else '-'}",
                "Keys: q/ESC quit, r recalibrate, n next target, p pause, SPACE emergency stop",
            ]
            if status_msg:
                hud.insert(0, status_msg)
            draw_hud(bgr, hud)

            disp = cv2.resize(bgr, (display_w, display_h), interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                ksys.recalibrate_plane()
                ksys.grid_frame = None
                clicked_color_xy = None
                targets_initialized = False
                status_msg = "Recalibrating plane. After lock, click taped grid center again."
                await client.send_cmd_vel(0.0, 0.0)
                ctrl.reset()
            if key == ord("p"):
                paused = not paused
                status_msg = "Paused." if paused else "Resumed."
                if paused:
                    await client.send_cmd_vel(0.0, 0.0)
                    ctrl.reset()
            if key == ord("n"):
                nav.confirm_stilt_and_advance()
                ctrl.reset()
                status_msg = "All targets completed." if nav.is_done() else f"Continuing to target {nav.current_index + 1}."
            if key == 32:  # SPACE
                paused = True
                status_msg = "EMERGENCY STOP (paused). Press 'p' to resume."
                await client.send_cmd_vel(0.0, 0.0)
                ctrl.reset()

            frame_idx += 1
            await asyncio.sleep(0.001)

    finally:
        # Always stop robot and set safe mode
        try:
            await client.stop()
            await client.set_mode("stand")
        except Exception:
            pass
        try:
            ksys.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        await client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))