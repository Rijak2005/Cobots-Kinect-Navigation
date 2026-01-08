from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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
    Robot tracker for Kinect ceiling setup.

    Key changes for your project:
      - Detect ONLY robot_id (e.g. 871) so we do not draw / consider other markers.
      - Prefer ONE dictionary (e.g. DICT_4X4_1000) to reduce false positives.
      - Compute heading as vector from marker center -> top edge midpoint (like your example).
      - Draw ONLY the robot marker + heading arrow. If not visible, draw nothing.
    """

    def __init__(
        self,
        kinect_sys: KinectGridSystem,
        strictness: float = 0.80,
        robot_id: int = 871,
        preferred_dict: str = "DICT_4X4_1000",
        heading_offset_deg: float = 0.0,     # set to 180 if arrow points backward
        min_marker_size_px: float = 40.0,    # reject too-small detections (helps false positives)
    ) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco not found. Install: pip install opencv-contrib-python")

        self.kinect_sys = kinect_sys
        self.strictness = float(np.clip(strictness, 0.0, 1.0))

        self.robot_id = int(robot_id)
        self.preferred_dict_name = str(preferred_dict)
        self.heading_offset_rad = float(np.deg2rad(heading_offset_deg))

        self.min_marker_size_px = float(min_marker_size_px)

        # Dictionary
        aruco_dict_id = getattr(cv2.aruco, self.preferred_dict_name, None)
        if aruco_dict_id is None:
            raise ValueError(f"Unknown aruco dict: {self.preferred_dict_name}")
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)

        # Detector params (kept fairly stable)
        self.params = cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementWinSize = 5
        self.params.cornerRefinementMaxIterations = 30
        self.params.cornerRefinementMinAccuracy = 0.1

        # reduce false positives (similar spirit to your webcam script)
        self.params.adaptiveThreshWinSizeMin = 5
        self.params.adaptiveThreshWinSizeMax = 35
        self.params.adaptiveThreshWinSizeStep = 5
        self.params.adaptiveThreshConstant = 7
        self.params.minDistanceToBorder = 3

        # These exist in some OpenCV versions; set if present
        if hasattr(self.params, "detectInvertedMarker"):
            self.params.detectInvertedMarker = True

        # Derived thresholds from strictness
        self.depth_search_radius = int(round(np.interp(self.strictness, [0.0, 1.0], [180.0, 90.0])))

        # Seeds for faster color->depth matching
        self._seed_center_uv: Optional[Tuple[int, int]] = None
        self._seed_top_uv: Optional[Tuple[int, int]] = None
        self._seed_bottom_uv: Optional[Tuple[int, int]] = None

        # Debug info
        self.last_seen: bool = False
        self.last_color_center_xy: Optional[Tuple[float, float]] = None
        self.last_color_topmid_xy: Optional[Tuple[float, float]] = None
        self.last_marker_corners: Optional[np.ndarray] = None  # (4,2) float
        self.last_perimeter_px: float = 0.0

        # Pose smoothing
        self._last_pose: Optional[RobotPose2D] = None
        self._alpha = float(np.interp(self.strictness, [0.0, 1.0], [0.28, 0.18]))

    @staticmethod
    def _perimeter(c4: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.roll(c4, -1, axis=0) - c4, axis=1)))

    @staticmethod
    def _mean_side_len(c4: np.ndarray) -> float:
        d = np.linalg.norm(np.roll(c4, -1, axis=0) - c4, axis=1)
        return float(np.mean(d))

    def _detect_robot_marker(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns corners (4,2) float64 for robot marker_id if detected, else None.
        """
        corners_list, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        self.last_seen = False
        self.last_marker_corners = None
        self.last_perimeter_px = 0.0

        if ids is None or len(ids) == 0:
            return None

        ids_flat = ids.flatten().astype(int)
        # filter only robot_id
        keep = [i for i, mid in enumerate(ids_flat.tolist()) if mid == self.robot_id]
        if not keep:
            return None

        # If multiple, pick the largest by perimeter
        best_i = None
        best_per = -1.0
        best_c4 = None

        for i in keep:
            c4 = corners_list[i].reshape(4, 2).astype(np.float64)
            per = self._perimeter(c4)
            if per > best_per:
                best_per = per
                best_i = i
                best_c4 = c4

        if best_c4 is None:
            return None

        # size gate (cuts many false positives)
        if self._mean_side_len(best_c4) < self.min_marker_size_px:
            return None

        self.last_seen = True
        self.last_marker_corners = best_c4
        self.last_perimeter_px = float(best_per)
        return best_c4

    def _smooth_pose(self, raw: RobotPose2D) -> RobotPose2D:
        if self._last_pose is None:
            self._last_pose = raw
            return raw

        a = self._alpha
        x = (1 - a) * self._last_pose.x + a * raw.x
        y = (1 - a) * self._last_pose.y + a * raw.y

        # Smooth heading using unit vectors (avoids wrap issues)
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
        return sm

    def _color_xy_to_floor_point(
        self,
        frame: GridFrame,
        target_xy: Tuple[float, float],
        seed_uv: Optional[Tuple[int, int]],
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        depth_1d = self.kinect_sys.last_depth_1d
        if depth_1d is None:
            return None, seed_uv

        depth_w = self.kinect_sys.depth_w
        depth_h = self.kinect_sys.depth_h

        # Try with seed
        uv = find_depth_pixel_for_color_xy(
            self.kinect_sys.kinect,
            depth_1d,
            depth_w,
            depth_h,
            target_color_xy=target_xy,
            seed_depth_uv=seed_uv,
            search_radius=self.depth_search_radius,
        )

        # Retry with no seed and bigger radius if needed
        if uv is None:
            uv = find_depth_pixel_for_color_xy(
                self.kinect_sys.kinect,
                depth_1d,
                depth_w,
                depth_h,
                target_color_xy=target_xy,
                seed_depth_uv=None,
                search_radius=int(self.depth_search_radius * 2.0),
            )
            if uv is None:
                return None, seed_uv

        u, v = uv
        if not (0 <= u < depth_w and 0 <= v < depth_h):
            return None, seed_uv

        depth_mm = int(depth_1d[v * depth_w + u])
        if depth_mm <= 0:
            return None, uv

        p_cam = depth_pixel_to_camera_point(self.kinect_sys.kinect, u, v, depth_mm)
        if p_cam is None:
            return None, uv

        p_floor = project_point_to_plane(p_cam, frame.plane)
        return p_floor, uv

    def detect_and_estimate(self, bgr: np.ndarray) -> Optional[RobotPose2D]:
        frame = self.kinect_sys.grid_frame
        if frame is None:
            self.last_seen = False
            return None

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners = self._detect_robot_marker(gray)
        if corners is None:
            return None

        # Marker center + top edge midpoint in color pixel coordinates (like your example)
        center = corners.mean(axis=0)
        top_mid = 0.5 * (corners[0] + corners[1])
        bottom_mid = 0.5 * (corners[3] + corners[2])

        self.last_color_center_xy = (float(center[0]), float(center[1]))
        self.last_color_topmid_xy = (float(top_mid[0]), float(top_mid[1]))

        # Map those color pixels onto floor plane to build heading in grid coordinates
        p_center, self._seed_center_uv = self._color_xy_to_floor_point(frame, self.last_color_center_xy, self._seed_center_uv)
        p_top, self._seed_top_uv = self._color_xy_to_floor_point(frame, self.last_color_topmid_xy, self._seed_top_uv)
        p_bottom, self._seed_bottom_uv = self._color_xy_to_floor_point(frame, (float(bottom_mid[0]), float(bottom_mid[1])), self._seed_bottom_uv)

        if p_center is None or p_top is None or p_bottom is None:
            return None

        # Convert to grid coordinates
        cx, cy = camera_to_grid_xy(p_center, frame)
        tx, ty = camera_to_grid_xy(p_top, frame)
        bx, by = camera_to_grid_xy(p_bottom, frame)

        # Forward direction: center -> top edge (like your example)
        forward = np.array([tx - cx, ty - cy], dtype=np.float64)
        n = float(np.linalg.norm(forward))
        if n < 1e-9:
            return None
        forward /= n

        heading = float(np.arctan2(forward[1], forward[0]))
        heading = _wrap_angle(heading + self.heading_offset_rad)

        # Confidence from size (simple)
        conf = float(np.clip(self.last_perimeter_px / 900.0, 0.0, 1.0))

        raw = RobotPose2D(
            x=float(cx),
            y=float(cy),
            heading=heading,
            marker_id=self.robot_id,
            dict_name=self.preferred_dict_name,
            confidence=conf,
        )
        return self._smooth_pose(raw)

    def draw_debug(self, img: np.ndarray) -> None:
        """
        Draw ONLY:
          - The robot marker 871 outline
          - A heading arrow from center to the marker's top edge midpoint (in image pixels)
        If robot not visible -> draw nothing.
        """
        if not self.last_seen or self.last_marker_corners is None:
            return

        pts = self.last_marker_corners.astype(np.int32).reshape(4, 2)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)

        center = pts.mean(axis=0).astype(np.float32)
        top_mid = 0.5 * (pts[0].astype(np.float32) + pts[1].astype(np.float32))

        # Heading arrow like your example: arrow_end = center + scale*(top_mid-center)
        dir_vec = top_mid - center
        arrow_end = center + 1.5 * dir_vec

        c = tuple(center.astype(int))
        a = tuple(arrow_end.astype(int))

        cv2.circle(img, c, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(img, c, a, (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.25)

        # Label
        cv2.putText(
            img,
            f"ID:{self.robot_id}",
            tuple(pts[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
