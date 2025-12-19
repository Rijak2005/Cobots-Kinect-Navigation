from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from grid_core import KinectGridSystem, map_camera_point_to_color_xy, grid_xy_to_camera
from robot_tracker import ArucoRobotTrackerAuto, RobotPose2D
from navigator import GridNavigator, NavConfig

WINDOW_NAME = "Kinect v2 - Navigation"
DISPLAY_SCALE = 0.6

# ---- ONE tuning knob ----
ARUCO_STRICTNESS = 0.80
# -------------------------

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
AXIS_X_COLOR = (0, 0, 255)
AXIS_Y_COLOR = (255, 0, 0)
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


def draw_grid_overlay(img: np.ndarray, ksys: KinectGridSystem, spacing_m: float) -> None:
    frame = ksys.grid_frame
    if frame is None:
        return

    origin_uv = map_camera_point_to_color_xy(ksys.kinect, frame.origin_cam)
    if origin_uv is not None:
        ox, oy = int(origin_uv[0]), int(origin_uv[1])
        cv2.circle(img, (ox, oy), 7, (255, 255, 255), -1, cv2.LINE_AA)

        x_end = frame.origin_cam + 1.0 * frame.e1
        y_end = frame.origin_cam + 1.0 * frame.e2
        x_uv = map_camera_point_to_color_xy(ksys.kinect, x_end)
        y_uv = map_camera_point_to_color_xy(ksys.kinect, y_end)

        if x_uv is not None:
            cv2.arrowedLine(img, (ox, oy), (int(x_uv[0]), int(x_uv[1])), AXIS_X_COLOR, 2, cv2.LINE_AA, tipLength=0.03)
        if y_uv is not None:
            cv2.arrowedLine(img, (ox, oy), (int(y_uv[0]), int(y_uv[1])), AXIS_Y_COLOR, 2, cv2.LINE_AA, tipLength=0.03)

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


def draw_robot_overlay(img: np.ndarray, robot: RobotPose2D, ksys: KinectGridSystem) -> Optional[Tuple[int, int]]:
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

        # heading arrow
        arrow_len = 0.20
        hx = robot.x + arrow_len * float(np.cos(robot.heading))
        hy = robot.y + arrow_len * float(np.sin(robot.heading))
        p_head = grid_xy_to_camera(hx, hy, frame)
        uvh = map_camera_point_to_color_xy(ksys.kinect, p_head)
        if uvh is not None:
            cv2.arrowedLine(img, (u, v), (int(uvh[0]), int(uvh[1])), ROBOT_COLOR, 2, cv2.LINE_AA, tipLength=0.2)

        cv2.putText(
            img,
            f"Robot: x={robot.x:+.2f} y={robot.y:+.2f} conf={robot.confidence:.2f} id={robot.marker_id} {robot.dict_name}",
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            ROBOT_COLOR,
            2,
            cv2.LINE_AA,
        )
        return (u, v)
    return None


def main() -> int:
    if not hasattr(cv2, "aruco"):
        print("ERROR: cv2.aruco missing. Install: pip install opencv-contrib-python")
        return 1

    ksys = KinectGridSystem(plane_smooth_alpha=0.15)
    display_w = int(ksys.color_w * DISPLAY_SCALE)
    display_h = int(ksys.color_h * DISPLAY_SCALE)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

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

    tracker = ArucoRobotTrackerAuto(ksys, strictness=ARUCO_STRICTNESS)

    nav = GridNavigator(
        NavConfig(
            command_interval_s=0.25,
            pos_tolerance_m=0.05,
            turn_start_deg=30.0,
            turn_stop_deg=12.0,
        )
    )
    targets_initialized = False
    paused = False
    frame_idx = 0

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
                continue
            bgr = bgra_to_bgr(bgra)

            if clicked_color_xy is not None and ksys.grid_frame is None and ksys.plane is not None:
                ok = ksys.set_grid_center_from_color_click(clicked_color_xy, search_radius=160)
                status_msg = "Grid origin set. Show marker." if ok else "Could not set origin yet. Click again or wait."

            if clicked_color_xy is not None:
                draw_marker_cross(bgr, clicked_color_xy[0], clicked_color_xy[1])

            draw_grid_overlay(bgr, ksys, GRID_SPACING_M)

            robot = tracker.detect_and_estimate(bgr)
            tracker.draw_debug(bgr)

            robot_uv = None
            if robot is not None:
                robot_uv = draw_robot_overlay(bgr, robot, ksys)

            if (not targets_initialized) and (ksys.grid_frame is not None):
                targets = nav.make_targets_3x3_ordered_like_image(
                    frame=ksys.grid_frame,
                    kinect=ksys.kinect,
                    spacing_m=GRID_SPACING_M,
                )
                nav.set_targets(targets)
                targets_initialized = True
                status_msg = "Targets set. Navigation running. Press 'n' after placing stilt."

            # Goal visualization
            goal_xy = nav.current_target() if targets_initialized else None
            goal_uv = None
            if goal_xy is not None:
                goal_uv = draw_goal(bgr, ksys, goal_xy)
                if goal_uv is not None and robot_uv is not None:
                    cv2.line(bgr, robot_uv, goal_uv, LINE_COLOR, 2, cv2.LINE_AA)

            # Command output
            now = time.monotonic()
            if targets_initialized and (not paused):
                cmd = nav.update(robot, now_s=now)
                if cmd:
                    print(cmd)

            hud = [
                f"Strictness: {ARUCO_STRICTNESS:.2f}",
                f"Depth: {'OK' if ksys.last_depth_1d is not None else '---'}  Plane: {'LOCKED' if ksys.plane_locked else 'CALIBRATING'}  Fits: {ksys.fit_count}/{FITS_TO_LOCK}",
                f"Grid origin: {'SET' if ksys.grid_frame is not None else 'NOT SET'}  Robot: {'OK' if robot is not None else '---'}",
                f"Aruco: active=({tracker.active_dict_name},{tracker.active_id})  last=({tracker.last_dict_used}) ids={tracker.last_detected_ids} perim>={tracker.min_perimeter_px:.0f}px last={tracker.last_perimeter_px:.0f}px",
                f"Nav: {'PAUSED' if paused else 'RUNNING'}  Target: {nav.current_index + 1 if targets_initialized and not nav.is_done() else '-'} / {len(nav.targets_xy) if targets_initialized else '-'}",
                "Keys: q/ESC quit, r recalibrate, n next target, p pause, RightClick clear origin",
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
            if key == ord("p"):
                paused = not paused
                status_msg = "Paused." if paused else "Resumed."
            if key == ord("n"):
                nav.confirm_stilt_and_advance()
                status_msg = "All targets completed." if nav.is_done() else f"Continuing to target {nav.current_index + 1}."

            frame_idx += 1
            time.sleep(0.001)

    finally:
        ksys.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
