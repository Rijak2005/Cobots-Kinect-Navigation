from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Set

import numpy as np
import cv2

from grid_core import (
    KinectGridSystem,
    GridFrame,
    find_depth_pixel_for_color_xy,
    depth_pixel_to_camera_point,
    project_point_to_plane,
    camera_to_grid_xy,
)


@dataclass
class RobotPose2D:
    x: float
    y: float
    heading: float          # radians in grid frame
    marker_id: int
    dict_name: str
    confidence: float       # 0..1


def _wrap_angle(rad: float) -> float:
    return (rad + np.pi) % (2.0 * np.pi) - np.pi


class ArucoRobotTrackerAuto:
    """
    Detects ONE robot marker (with optional alternate IDs) and draws ONLY that marker.

    Key behaviors:
      - Two-pass ArUco detection:
          (1) raw gray
          (2) CLAHE + slight blur (helps Kinect auto-exposure & low-contrast regions)
      - Looser DetectorParameters for Kinect RGB robustness
      - Pose-hold for brief dropouts:
          * if depth projection fails
          * OR if marker detection fails briefly
    """

    POSE_HOLD_SECONDS = 0.60

    def __init__(
        self,
        kinect_sys: KinectGridSystem,
        strictness: float = 0.60,
        robot_id: int = 871,
        alt_ids: Optional[list[int]] = None,
        preferred_dict: str = "DICT_4X4_1000",
        heading_offset_deg: float = 0.0,
        min_marker_size_px: float = 40.0,
    ) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco not found. Install: pip install opencv-contrib-python")

        self.kinect_sys = kinect_sys
        self.strictness = float(np.clip(strictness, 0.0, 1.0))

        self.robot_id = int(robot_id)
        self.alt_ids = [int(x) for x in (alt_ids or [])]
        self.allowed_ids: Set[int] = {self.robot_id, *self.alt_ids}

        self.preferred_dict_name = str(preferred_dict)
        self.heading_offset_rad = float(np.deg2rad(heading_offset_deg))
        self.min_marker_size_px = float(min_marker_size_px)

        aruco_dict_id = getattr(cv2.aruco, self.preferred_dict_name, None)
        if aruco_dict_id is None:
            raise ValueError(f"Unknown aruco dict: {self.preferred_dict_name}")
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)

        # ----------------- DETECTOR PARAMETERS -----------------
        self.params = cv2.aruco.DetectorParameters()

        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.params.adaptiveThreshWinSizeMin = 3
        self.params.adaptiveThreshWinSizeMax = 95
        self.params.adaptiveThreshWinSizeStep = 10
        self.params.adaptiveThreshConstant = 7

        self.params.minMarkerPerimeterRate = 0.01
        self.params.maxMarkerPerimeterRate = 10.0

        self.params.minCornerDistanceRate = 0.02
        self.params.minDistanceToBorder = 3

        if hasattr(self.params, "detectInvertedMarker"):
            self.params.detectInvertedMarker = True

        if hasattr(self.params, "useAruco3Detection"):
            self.params.useAruco3Detection = True
        # -------------------------------------------------------

        # Prefer new API if available
        self._detector: Optional[cv2.aruco.ArucoDetector] = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)

        # strictness affects depth matching (not detection)
        self.depth_search_radius = int(round(np.interp(self.strictness, [0.0, 1.0], [220.0, 120.0])))

        # Seeds for depth matching
        self._seed_center_uv: Optional[Tuple[int, int]] = None
        self._seed_top_uv: Optional[Tuple[int, int]] = None

        # Debug state
        self.marker_seen_now: bool = False
        self.last_marker_corners: Optional[np.ndarray] = None   # (4,2)
        self.last_detected_id: Optional[int] = None
        self.last_color_center_xy: Optional[Tuple[float, float]] = None
        self.last_color_topmid_xy: Optional[Tuple[float, float]] = None

        # Pose hold state
        self._last_pose: Optional[RobotPose2D] = None
        self._last_pose_t: float = 0.0

        # smoothing (position + heading)
        self._alpha = float(np.interp(self.strictness, [0.0, 1.0], [0.30, 0.18]))

        # CLAHE for second-pass robustness
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @staticmethod
    def _mean_side_len(c4: np.ndarray) -> float:
        d = np.linalg.norm(np.roll(c4, -1, axis=0) - c4, axis=1)
        return float(np.mean(d))

    def _aruco_detect(self, gray: np.ndarray):
        if self._detector is not None:
            return self._detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

    def _detect_robot_marker(self, gray: np.ndarray) -> Optional[np.ndarray]:
        self.marker_seen_now = False
        self.last_marker_corners = None
        self.last_detected_id = None

        def pick_best_robot(corners_list, ids) -> Optional[tuple[np.ndarray, int]]:
            if ids is None or len(ids) == 0:
                return None
            ids_flat = ids.flatten().astype(int)
            keep = [i for i, mid in enumerate(ids_flat.tolist()) if mid in self.allowed_ids]
            if not keep:
                return None

            best = None
            best_size = -1.0
            best_id: Optional[int] = None

            for i in keep:
                mid = int(ids_flat[i])
                c4 = corners_list[i].reshape(4, 2).astype(np.float64)
                size = self._mean_side_len(c4)
                if size > best_size:
                    best_size = size
                    best = c4
                    best_id = mid

            if best is None or best_id is None:
                return None
            if best_size < self.min_marker_size_px:
                return None
            return best, best_id

        # Pass 1
        corners_list, ids, _ = self._aruco_detect(gray)
        hit = pick_best_robot(corners_list, ids)
        if hit is not None:
            best, mid = hit
            self.marker_seen_now = True
            self.last_marker_corners = best
            self.last_detected_id = mid
            return best

        # Pass 2: CLAHE + blur
        g2 = self._clahe.apply(gray)
        g2 = cv2.GaussianBlur(g2, (3, 3), 0)
        corners_list, ids, _ = self._aruco_detect(g2)
        hit = pick_best_robot(corners_list, ids)
        if hit is not None:
            best, mid = hit
            self.marker_seen_now = True
            self.last_marker_corners = best
            self.last_detected_id = mid
            return best

        return None

    def _color_xy_to_floor_point(
        self,
        frame: GridFrame,
        target_xy: Tuple[float, float],
        seed_uv: Optional[Tuple[int, int]],
        search_radius: int,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        depth_1d = self.kinect_sys.last_depth_1d
        if depth_1d is None:
            return None, seed_uv

        depth_w = self.kinect_sys.depth_w
        depth_h = self.kinect_sys.depth_h

        uv = find_depth_pixel_for_color_xy(
            self.kinect_sys.kinect,
            depth_1d,
            depth_w,
            depth_h,
            target_color_xy=target_xy,
            seed_depth_uv=seed_uv,
            search_radius=search_radius,
        )
        if uv is None:
            return None, seed_uv

        u, v = uv
        if not (0 <= u < depth_w and 0 <= v < depth_h):
            return None, uv

        depth_mm = int(depth_1d[v * depth_w + u])
        if depth_mm <= 0:
            return None, uv

        p_cam = depth_pixel_to_camera_point(self.kinect_sys.kinect, u, v, depth_mm)
        if p_cam is None:
            return None, uv

        p_floor = project_point_to_plane(p_cam, frame.plane)
        return p_floor, uv

    def _smooth_pose(self, raw: RobotPose2D) -> RobotPose2D:
        if self._last_pose is None:
            self._last_pose = raw
            self._last_pose_t = float(cv2.getTickCount() / cv2.getTickFrequency())
            return raw

        a = self._alpha
        x = (1 - a) * self._last_pose.x + a * raw.x
        y = (1 - a) * self._last_pose.y + a * raw.y

        v_prev = np.array([np.cos(self._last_pose.heading), np.sin(self._last_pose.heading)], dtype=np.float64)
        v_new = np.array([np.cos(raw.heading), np.sin(raw.heading)], dtype=np.float64)
        v = (1 - a) * v_prev + a * v_new
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            v /= n
        heading = float(np.arctan2(v[1], v[0]))

        sm = RobotPose2D(
            x=float(x),
            y=float(y),
            heading=float(heading),
            marker_id=raw.marker_id,
            dict_name=raw.dict_name,
            confidence=raw.confidence,
        )
        self._last_pose = sm
        self._last_pose_t = float(cv2.getTickCount() / cv2.getTickFrequency())
        return sm

    def _hold_last_pose_if_recent(self) -> Optional[RobotPose2D]:
        if self._last_pose is None:
            return None
        now = float(cv2.getTickCount() / cv2.getTickFrequency())
        if (now - self._last_pose_t) <= self.POSE_HOLD_SECONDS:
            return RobotPose2D(
                x=self._last_pose.x,
                y=self._last_pose.y,
                heading=self._last_pose.heading,
                marker_id=self._last_pose.marker_id,
                dict_name=self._last_pose.dict_name,
                confidence=max(0.05, self._last_pose.confidence * 0.75),
            )
        return None

    def detect_and_estimate(self, bgr: np.ndarray) -> Optional[RobotPose2D]:
        frame = self.kinect_sys.grid_frame
        if frame is None:
            self.marker_seen_now = False
            return None

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners = self._detect_robot_marker(gray)
        if corners is None:
            # Hold pose briefly even if detection drops
            return self._hold_last_pose_if_recent()

        center = corners.mean(axis=0)
        top_mid = 0.5 * (corners[0] + corners[1])

        self.last_color_center_xy = (float(center[0]), float(center[1]))
        self.last_color_topmid_xy = (float(top_mid[0]), float(top_mid[1]))

        p_center, self._seed_center_uv = self._color_xy_to_floor_point(
            frame, self.last_color_center_xy, self._seed_center_uv, search_radius=self.depth_search_radius
        )
        p_top, self._seed_top_uv = self._color_xy_to_floor_point(
            frame, self.last_color_topmid_xy, self._seed_top_uv, search_radius=int(self.depth_search_radius * 1.8)
        )

        if p_center is None:
            return self._hold_last_pose_if_recent()

        cx, cy = camera_to_grid_xy(p_center, frame)

        if p_top is None and self._last_pose is not None:
            heading = self._last_pose.heading
        elif p_top is None:
            return self._hold_last_pose_if_recent()
        else:
            tx, ty = camera_to_grid_xy(p_top, frame)
            forward = np.array([tx - cx, ty - cy], dtype=np.float64)
            n = float(np.linalg.norm(forward))
            if n < 1e-9:
                return self._hold_last_pose_if_recent()
            forward /= n
            heading = float(np.arctan2(forward[1], forward[0]))

        heading = _wrap_angle(heading + self.heading_offset_rad)

        size_px = self._mean_side_len(corners)
        conf = float(np.clip(size_px / 140.0, 0.0, 1.0))

        raw = RobotPose2D(
            x=float(cx),
            y=float(cy),
            heading=float(heading),
            marker_id=self.robot_id,
            dict_name=self.preferred_dict_name,
            confidence=conf,
        )
        return self._smooth_pose(raw)

    def draw_debug(self, img: np.ndarray) -> None:
        if not self.marker_seen_now or self.last_marker_corners is None:
            return

        pts = self.last_marker_corners.astype(np.int32).reshape(4, 2)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)

        center = pts.mean(axis=0).astype(np.float32)
        top_mid = 0.5 * (pts[0].astype(np.float32) + pts[1].astype(np.float32))
        dir_vec = top_mid - center
        arrow_end = center + 1.5 * dir_vec

        c = tuple(center.astype(int))
        a = tuple(arrow_end.astype(int))

        cv2.circle(img, c, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(img, c, a, (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.25)

        det_id_txt = self.last_detected_id if self.last_detected_id is not None else self.robot_id
        cv2.putText(
            img,
            f"ID:{det_id_txt}",
            tuple(pts[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
