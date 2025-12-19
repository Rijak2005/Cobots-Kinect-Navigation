from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

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
    heading: float          # radians in grid frame (+X is 0)
    marker_id: int
    confidence: float       # 0..1


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _lerp(a: float, b: float, t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return a + (b - a) * t


class ArucoRobotTrackerAuto:
    """
    Fully automatic ArUco tracking:
      - Uses ONE dictionary (DICT_4X4_50) to reduce false positives.
      - Auto-selects the "real" marker ID by consistency over recent frames.
      - Uses ONE knob ARUCO_STRICTNESS (0..1) to control:
          * min marker size (pixel perimeter)
          * decoder strictness
          * jump rejection
          * smoothing amount
          * color->depth search radius (speed vs robustness)
    """

    def __init__(
        self,
        kinect_sys: KinectGridSystem,
        strictness: float = 0.75,          # <<<<<< single tuning knob
    ) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco not found. Install: pip install opencv-contrib-python")

        self.kinect_sys = kinect_sys
        self.strictness = float(np.clip(strictness, 0.0, 1.0))

        # Use one dictionary to narrow the search space (important for false positives)
        self.dict_name = "DICT_4X4_50"
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        # Detector parameters (we'll make them stricter as strictness increases)
        self.params = cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementWinSize = 5
        self.params.cornerRefinementMaxIterations = 30
        self.params.cornerRefinementMinAccuracy = 0.1

        self.params.adaptiveThreshWinSizeMin = 7
        self.params.adaptiveThreshWinSizeMax = 45
        self.params.adaptiveThreshWinSizeStep = 10
        self.params.adaptiveThreshConstant = 7

        if hasattr(self.params, "detectInvertedMarker"):
            self.params.detectInvertedMarker = True

        # --- Strictness mapping (ONE knob drives these) ---
        # 1) How big marker must be in the image (pixel perimeter)
        #    Bigger threshold => rejects tiny "fake" markers on the floor.
        self.min_perimeter_px = _lerp(110.0, 260.0, self.strictness)

        # 2) Jump rejection in meters
        #    Smaller max jump => rejects teleporting to random floor spots.
        self.max_jump_m = _lerp(0.70, 0.22, self.strictness)

        # 3) Smoothing for x/y/heading
        self.smooth_alpha = _lerp(0.45, 0.18, self.strictness)

        # 4) Color->depth search radius (pixels in depth image)
        #    Smaller radius => faster and less jitter, but needs reasonable continuity.
        self.search_radius = int(round(_lerp(160.0, 70.0, self.strictness)))

        # 5) Extra strict decoder settings if OpenCV exposes them
        #    Higher strictness => lower tolerance for decoding errors.
        if hasattr(self.params, "maxErroneousBitsInBorderRate"):
            self.params.maxErroneousBitsInBorderRate = _lerp(0.35, 0.08, self.strictness)
        if hasattr(self.params, "errorCorrectionRate"):
            self.params.errorCorrectionRate = _lerp(0.80, 0.45, self.strictness)
        if hasattr(self.params, "minCornerDistanceRate"):
            self.params.minCornerDistanceRate = _lerp(0.02, 0.06, self.strictness)
        if hasattr(self.params, "polygonalApproxAccuracyRate"):
            self.params.polygonalApproxAccuracyRate = _lerp(0.05, 0.03, self.strictness)

        # --- Auto-ID memory ---
        # We keep a recent history of detected IDs and choose the one that appears most often.
        self._id_history: deque[int] = deque(maxlen=30)
        self.robot_id: Optional[int] = None

        # If robot_id disappears for too long, we forget it and relearn.
        self._miss_count: int = 0
        self._miss_to_forget: int = int(round(_lerp(18, 10, self.strictness)))

        # Seeds to make color->depth mapping fast (re-using last good UV as "starting point")
        self._seed_center_uv: Optional[Tuple[int, int]] = None
        self._seed_top_uv: Optional[Tuple[int, int]] = None
        self._seed_bottom_uv: Optional[Tuple[int, int]] = None

        # State for stability
        self.last_detected_ids: List[int] = []
        self._last_pose_raw: Optional[RobotPose2D] = None
        self._last_pose_smooth: Optional[RobotPose2D] = None

        # Debug: last accepted perimeter
        self.last_perimeter_px: float = 0.0

    @staticmethod
    def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 0.5 * (a + b)

    @staticmethod
    def _perimeter(c4: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.roll(c4, -1, axis=0) - c4, axis=1)))

    def _auto_choose_id(self) -> Optional[int]:
        if len(self._id_history) < 6:
            return None
        vals, counts = np.unique(np.array(self._id_history), return_counts=True)
        best_i = int(np.argmax(counts))
        best_id = int(vals[best_i])
        best_count = int(counts[best_i])

        # Require some consistency
        if best_count >= 5:
            return best_id
        return None

    def _smooth_pose(self, raw: RobotPose2D) -> RobotPose2D:
        if self._last_pose_smooth is None:
            self._last_pose_smooth = raw
            return raw

        a = self.smooth_alpha
        sx = (1 - a) * self._last_pose_smooth.x + a * raw.x
        sy = (1 - a) * self._last_pose_smooth.y + a * raw.y

        # Smooth heading using unit vectors (avoids wrap-around issues)
        v_prev = np.array([np.cos(self._last_pose_smooth.heading), np.sin(self._last_pose_smooth.heading)], dtype=np.float64)
        v_new = np.array([np.cos(raw.heading), np.sin(raw.heading)], dtype=np.float64)
        v = (1 - a) * v_prev + a * v_new
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            v /= n
        shead = float(np.arctan2(v[1], v[0]))

        sm = RobotPose2D(
            x=float(sx),
            y=float(sy),
            heading=shead,
            marker_id=raw.marker_id,
            confidence=raw.confidence,
        )
        self._last_pose_smooth = sm
        return sm

    def detect_and_estimate(self, bgr: np.ndarray) -> Optional[RobotPose2D]:
        frame: Optional[GridFrame] = self.kinect_sys.grid_frame
        if frame is None:
            return None
        if self.kinect_sys.last_depth_1d is None:
            return None

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

        if ids is None or len(ids) == 0:
            self.last_detected_ids = []
            self._miss_count += 1
            if self._miss_count >= self._miss_to_forget:
                self.robot_id = None
                self._id_history.clear()
                self._last_pose_raw = None
                self._last_pose_smooth = None
            return None

        ids_flat = ids.flatten().astype(int)
        self.last_detected_ids = ids_flat.tolist()
        self._miss_count = 0

        # Reject small markers first; build candidates
        candidates: list[tuple[int, float]] = []  # (index, perimeter)
        for i, c in enumerate(corners_list):
            c4 = c.reshape(4, 2).astype(np.float64)
            per = self._perimeter(c4)
            if per >= self.min_perimeter_px:
                candidates.append((i, per))

        if not candidates:
            return None

        # Update id history from the best-looking candidate (largest perimeter)
        best_i, best_per = max(candidates, key=lambda t: t[1])
        seen_id = int(ids_flat[best_i])
        self._id_history.append(seen_id)
        if self.robot_id is None:
            self.robot_id = self._auto_choose_id()

        # Choose which marker to use:
        # - If we have an auto robot_id, use that if present among candidates.
        # - Otherwise use the largest candidate.
        chosen_idx = best_i
        if self.robot_id is not None:
            for i, per in candidates:
                if int(ids_flat[i]) == int(self.robot_id):
                    chosen_idx = i
                    best_per = per
                    break

        marker_id = int(ids_flat[chosen_idx])
        corners = corners_list[chosen_idx].reshape((4, 2)).astype(np.float64)
        self.last_perimeter_px = float(best_per)

        # Compute marker center + forward direction in COLOR pixels
        center = corners.mean(axis=0)
        top_mid = self._mid(corners[0], corners[1])
        bottom_mid = self._mid(corners[3], corners[2])

        depth_w = self.kinect_sys.depth_w
        depth_h = self.kinect_sys.depth_h
        depth_1d = self.kinect_sys.last_depth_1d

        # Map (color pixel) -> (depth pixel) by searching near previous result.
        def color_xy_to_floor_point(
            xy: Tuple[float, float],
            seed: Optional[Tuple[int, int]],
        ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
            uv = find_depth_pixel_for_color_xy(
                self.kinect_sys.kinect,
                depth_1d,
                depth_w,
                depth_h,
                target_color_xy=xy,
                seed_depth_uv=seed,
                search_radius=self.search_radius,
            )
            if uv is None:
                return None, seed

            u, v = uv
            if not (0 <= u < depth_w and 0 <= v < depth_h):
                return None, seed

            depth_mm = int(depth_1d[v * depth_w + u])
            p_cam = depth_pixel_to_camera_point(self.kinect_sys.kinect, u, v, depth_mm)
            if p_cam is None:
                return None, uv

            p_floor = project_point_to_plane(p_cam, frame.plane)
            return p_floor, uv

        p_center, self._seed_center_uv = color_xy_to_floor_point((float(center[0]), float(center[1])), self._seed_center_uv)
        p_top, self._seed_top_uv = color_xy_to_floor_point((float(top_mid[0]), float(top_mid[1])), self._seed_top_uv)
        p_bottom, self._seed_bottom_uv = color_xy_to_floor_point((float(bottom_mid[0]), float(bottom_mid[1])), self._seed_bottom_uv)

        if p_center is None or p_top is None or p_bottom is None:
            return None

        cx, cy = camera_to_grid_xy(p_center, frame)
        tx, ty = camera_to_grid_xy(p_top, frame)
        bx, by = camera_to_grid_xy(p_bottom, frame)

        forward = np.array([tx - bx, ty - by], dtype=np.float64)
        n = float(np.linalg.norm(forward))
        if n < 1e-9:
            return None
        forward /= n
        heading = float(np.arctan2(forward[1], forward[0]))

        # Confidence: bigger marker in image => higher confidence
        confidence = float(np.clip(best_per / 900.0, 0.0, 1.0))

        raw = RobotPose2D(
            x=float(cx),
            y=float(cy),
            heading=heading,
            marker_id=marker_id,
            confidence=confidence,
        )

        # Jump reject to kill false-positive teleports
        if self._last_pose_raw is not None:
            jump = float(np.hypot(raw.x - self._last_pose_raw.x, raw.y - self._last_pose_raw.y))
            if jump > self.max_jump_m:
                return None

        self._last_pose_raw = raw
        return self._smooth_pose(raw)

    def draw_debug(self, img: np.ndarray) -> None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        if ids is None:
            return
        cv2.aruco.drawDetectedMarkers(img, corners_list, ids)
