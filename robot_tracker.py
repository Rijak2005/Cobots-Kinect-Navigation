from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import time
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
    dict_name: str
    confidence: float       # 0..1


def _lerp(a: float, b: float, t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return a + (b - a) * t


class ArucoRobotTrackerAuto:
    """
    Robust, automatic ArUco robot tracking for Kinect ceiling setup.

    Key behaviors:
      - Detects markers in color image (aruco).
      - Computes robot floor position by mapping color->depth->3D->floor plane.
      - NEVER gets stuck after temporary misses:
          * If robot is missed or pose fails for a bit, it resets internal state.
      - Optional hardcoded robot_id (recommended for your case: 871).
    """

    def __init__(
        self,
        kinect_sys: KinectGridSystem,
        strictness: float = 0.80,
        robot_id: Optional[int] = None,              # <-- set to 871 to hardcode
        preferred_dict: Optional[str] = None,        # e.g. "DICT_4X4_1000" (optional)
    ) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco not found. Install: pip install opencv-contrib-python")

        self.kinect_sys = kinect_sys
        self.strictness = float(np.clip(strictness, 0.0, 1.0))

        # Hardcode ID if provided
        self.robot_id: Optional[int] = int(robot_id) if robot_id is not None else None
        self.preferred_dict: Optional[str] = preferred_dict

        # Try multiple 4x4 dictionaries (important because your ID=871 indicates 4x4_1000 most likely)
        self.dicts: list[tuple[str, cv2.aruco.Dictionary]] = [
            ("DICT_4X4_1000", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)),
            ("DICT_4X4_250", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)),
            ("DICT_4X4_100", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)),
            ("DICT_4X4_50", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)),
        ]

        # Detector parameters (safe defaults + some stability)
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

        # ---- Single knob -> internal thresholds ----
        # Keep min perimeter low enough that detection doesn't die.
        self.min_perimeter_px = _lerp(45.0, 140.0, self.strictness)

        # Jump reject: prevents teleporting to false spots,
        # but we will DISABLE it during reacquisition (see below).
        self.max_jump_m = _lerp(0.90, 0.30, self.strictness)

        # Smoothing for stability
        self.smooth_alpha = _lerp(0.25, 0.18, self.strictness)

        # Depth search radius (will auto-expand on failure)
        self.base_search_radius = int(round(_lerp(170.0, 85.0, self.strictness)))
        # -------------------------------------------

        # Seeds for fast mapping color->depth
        self._seed_center_uv: Optional[Tuple[int, int]] = None
        self._seed_top_uv: Optional[Tuple[int, int]] = None
        self._seed_bottom_uv: Optional[Tuple[int, int]] = None

        # Debug/HUD info
        self.last_detected_ids: list[int] = []
        self.last_dict_used: str = ""
        self.last_perimeter_px: float = 0.0

        # Tracking state
        self._last_pose_raw: Optional[RobotPose2D] = None
        self._last_pose_smooth: Optional[RobotPose2D] = None

        # Reacquisition logic:
        self._last_good_time = 0.0
        self._consecutive_failures = 0
        self._failures_to_reset = 12  # ~12 frames before we reset (no stuck state)

    @staticmethod
    def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 0.5 * (a + b)

    @staticmethod
    def _perimeter(c4: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.roll(c4, -1, axis=0) - c4, axis=1)))

    def _reset_tracking_state(self) -> None:
        # Reset anything that can make us "stuck" after a miss
        self._seed_center_uv = None
        self._seed_top_uv = None
        self._seed_bottom_uv = None
        self._last_pose_raw = None
        self._last_pose_smooth = None
        self._consecutive_failures = 0

    def _smooth_pose(self, raw: RobotPose2D) -> RobotPose2D:
        if self._last_pose_smooth is None:
            self._last_pose_smooth = raw
            return raw

        a = self.smooth_alpha
        sx = (1 - a) * self._last_pose_smooth.x + a * raw.x
        sy = (1 - a) * self._last_pose_smooth.y + a * raw.y

        # Smooth heading using unit vectors
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
            dict_name=raw.dict_name,
            confidence=raw.confidence,
        )
        self._last_pose_smooth = sm
        return sm

    def _detect_candidates(self, gray: np.ndarray):
        """
        Return list of candidates across dictionaries:
          (dict_name, marker_id, corners4x2, perimeter_px)
        """
        candidates = []
        seen_ids = set()

        dict_order = self.dicts
        if self.preferred_dict is not None:
            # try preferred dict first
            pref = [d for d in self.dicts if d[0] == self.preferred_dict]
            rest = [d for d in self.dicts if d[0] != self.preferred_dict]
            dict_order = pref + rest

        for dict_name, d in dict_order:
            corners_list, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=self.params)
            if ids is None or len(ids) == 0:
                continue

            ids_flat = ids.flatten().astype(int)
            for mid in ids_flat.tolist():
                seen_ids.add(int(mid))

            for i, c in enumerate(corners_list):
                c4 = c.reshape(4, 2).astype(np.float64)
                per = self._perimeter(c4)
                if per >= self.min_perimeter_px:
                    mid = int(ids_flat[i])
                    candidates.append((dict_name, mid, c4, float(per)))

            # Optimization: if robot_id is hardcoded and we already found it in this dict, no need to try others
            if self.robot_id is not None and any((mid == self.robot_id and dn == dict_name) for (dn, mid, _, _) in candidates):
                self.last_dict_used = dict_name
                self.last_detected_ids = sorted(list(seen_ids))
                return candidates

        self.last_detected_ids = sorted(list(seen_ids))
        return candidates

    def detect_and_estimate(self, bgr: np.ndarray) -> Optional[RobotPose2D]:
        frame: Optional[GridFrame] = self.kinect_sys.grid_frame
        if frame is None or self.kinect_sys.last_depth_1d is None:
            return None

        now = time.monotonic()

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        candidates = self._detect_candidates(gray)

        if not candidates:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failures_to_reset:
                self._reset_tracking_state()
            return None

        # Choose which marker is the robot:
        # 1) If hardcoded robot_id exists -> choose that (largest perimeter if multiple)
        chosen = None
        if self.robot_id is not None:
            same_id = [c for c in candidates if c[1] == self.robot_id]
            if same_id:
                chosen = max(same_id, key=lambda t: t[3])

        # 2) Otherwise choose the biggest marker (most reliable)
        if chosen is None:
            chosen = max(candidates, key=lambda t: t[3])

        dict_name, marker_id, corners, perim_px = chosen
        self.last_dict_used = dict_name
        self.last_perimeter_px = float(perim_px)

        # Confidence from marker size
        confidence = float(np.clip(perim_px / 900.0, 0.0, 1.0))

        center = corners.mean(axis=0)
        top_mid = self._mid(corners[0], corners[1])
        bottom_mid = self._mid(corners[3], corners[2])

        depth_w = self.kinect_sys.depth_w
        depth_h = self.kinect_sys.depth_h
        depth_1d = self.kinect_sys.last_depth_1d

        def color_xy_to_floor_point(
            xy: Tuple[float, float],
            seed: Optional[Tuple[int, int]],
        ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
            # Try normal search (fast)
            uv = find_depth_pixel_for_color_xy(
                self.kinect_sys.kinect,
                depth_1d,
                depth_w,
                depth_h,
                target_color_xy=xy,
                seed_depth_uv=seed,
                search_radius=self.base_search_radius,
            )
            if uv is None:
                # Retry with no seed + bigger radius (reacquisition after miss)
                uv = find_depth_pixel_for_color_xy(
                    self.kinect_sys.kinect,
                    depth_1d,
                    depth_w,
                    depth_h,
                    target_color_xy=xy,
                    seed_depth_uv=None,
                    search_radius=int(self.base_search_radius * 2.2),
                )
                if uv is None:
                    return None, seed

            u, v = uv
            if not (0 <= u < depth_w and 0 <= v < depth_h):
                return None, seed

            depth_mm = int(depth_1d[v * depth_w + u])
            if depth_mm <= 0:
                return None, uv

            p_cam = depth_pixel_to_camera_point(self.kinect_sys.kinect, u, v, depth_mm)
            if p_cam is None:
                return None, uv

            p_floor = project_point_to_plane(p_cam, frame.plane)
            return p_floor, uv

        p_center, self._seed_center_uv = color_xy_to_floor_point((float(center[0]), float(center[1])), self._seed_center_uv)
        p_top, self._seed_top_uv = color_xy_to_floor_point((float(top_mid[0]), float(top_mid[1])), self._seed_top_uv)
        p_bottom, self._seed_bottom_uv = color_xy_to_floor_point((float(bottom_mid[0]), float(bottom_mid[1])), self._seed_bottom_uv)

        if p_center is None or p_top is None or p_bottom is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failures_to_reset:
                self._reset_tracking_state()
            return None

        cx, cy = camera_to_grid_xy(p_center, frame)
        tx, ty = camera_to_grid_xy(p_top, frame)
        bx, by = camera_to_grid_xy(p_bottom, frame)

        forward = np.array([tx - bx, ty - by], dtype=np.float64)
        n = float(np.linalg.norm(forward))
        if n < 1e-9:
            self._consecutive_failures += 1
            return None
        forward /= n
        heading = float(np.arctan2(forward[1], forward[0]))

        raw = RobotPose2D(
            x=float(cx),
            y=float(cy),
            heading=heading,
            marker_id=int(marker_id),
            dict_name=dict_name,
            confidence=confidence,
        )

        # If we haven't had a valid pose recently, allow "teleport" (reacquire)
        time_since_good = now - self._last_good_time
        allow_reacquire_jump = time_since_good > 0.6  # marker was missing/unstable

        if self._last_pose_raw is not None and not allow_reacquire_jump:
            jump = float(np.hypot(raw.x - self._last_pose_raw.x, raw.y - self._last_pose_raw.y))
            if jump > self.max_jump_m:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._failures_to_reset:
                    self._reset_tracking_state()
                return None

        # Success: update good-state timers/counters
        self._last_pose_raw = raw
        self._last_good_time = now
        self._consecutive_failures = 0
        return self._smooth_pose(raw)

    def draw_debug(self, img: np.ndarray) -> None:
        """
        Draw outlines for the best dictionary result. This does NOT guarantee we computed robot pose.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Prefer drawing from the dictionary that most recently worked
        dict_order = self.dicts
        if self.last_dict_used:
            pref = [d for d in self.dicts if d[0] == self.last_dict_used]
            rest = [d for d in self.dicts if d[0] != self.last_dict_used]
            dict_order = pref + rest

        for dict_name, d in dict_order:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=self.params)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
                return
