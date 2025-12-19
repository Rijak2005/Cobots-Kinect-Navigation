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
    dict_name: str
    confidence: float       # 0..1


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _lerp(a: float, b: float, t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return a + (b - a) * t


class ArucoRobotTrackerAuto:
    """
    Fully automatic, robust ArUco robot tracker.

    Core idea:
      - We DO NOT ask the user to "lock" anything.
      - We learn the most stable (dictionary, ID) pair automatically
        from recent detections and then follow only that pair.
      - If it disappears for long enough, we re-learn.

    One tuning knob:
      strictness in [0..1]
        higher => fewer false positives, more stable (but can miss if marker is tiny/blurred)
        lower  => detects easier (but more false positives)
    """

    def __init__(
        self,
        kinect_sys: KinectGridSystem,
        strictness: float = 0.80,
    ) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco not found. Install: pip install opencv-contrib-python")

        self.kinect_sys = kinect_sys
        self.strictness = float(np.clip(strictness, 0.0, 1.0))

        # Try multiple 4x4 dictionaries (this is what makes it "work again" reliably)
        self.dicts: list[tuple[str, cv2.aruco.Dictionary]] = [
            ("DICT_4X4_50", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)),
            ("DICT_4X4_100", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)),
            ("DICT_4X4_250", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)),
            ("DICT_4X4_1000", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)),
        ]

        # Detector parameters (safe defaults)
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

        # ---- Strictness mapping (ONE knob controls these) ----
        # minimum marker size in the COLOR image (pixel perimeter)
        # very important: too high => "robot not found"
        self.min_perimeter_px = _lerp(35.0, 160.0, self.strictness)

        # reject teleport jumps (meters)
        self.max_jump_m = _lerp(0.90, 0.25, self.strictness)

        # smoothing amount for pose (higher strictness => smoother)
        self.smooth_alpha = _lerp(0.30, 0.18, self.strictness)

        # depth search radius (speed + stability)
        self.search_radius = int(round(_lerp(180.0, 75.0, self.strictness)))

        # optional stricter decoding (if available)
        if hasattr(self.params, "maxErroneousBitsInBorderRate"):
            self.params.maxErroneousBitsInBorderRate = _lerp(0.35, 0.10, self.strictness)
        if hasattr(self.params, "errorCorrectionRate"):
            self.params.errorCorrectionRate = _lerp(0.85, 0.50, self.strictness)
        # -----------------------------------------------------

        # Learned "best" signature (dict + id) automatically
        self.active_dict_name: Optional[str] = None
        self.active_id: Optional[int] = None
        self._signature_history: deque[tuple[str, int]] = deque(maxlen=40)

        # Forget active target if it disappears for a while
        self._miss_count = 0
        self._miss_to_forget = int(round(_lerp(25, 12, self.strictness)))

        # Seeds for faster color->depth matching
        self._seed_center_uv: Optional[Tuple[int, int]] = None
        self._seed_top_uv: Optional[Tuple[int, int]] = None
        self._seed_bottom_uv: Optional[Tuple[int, int]] = None

        # Debug / HUD info
        self.last_detected_ids: list[int] = []
        self.last_dict_used: str = ""
        self.last_perimeter_px: float = 0.0

        # Temporal stability
        self._last_pose_raw: Optional[RobotPose2D] = None
        self._last_pose_smooth: Optional[RobotPose2D] = None

    @staticmethod
    def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 0.5 * (a + b)

    @staticmethod
    def _perimeter(c4: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.roll(c4, -1, axis=0) - c4, axis=1)))

    def _smooth_pose(self, raw: RobotPose2D) -> RobotPose2D:
        if self._last_pose_smooth is None:
            self._last_pose_smooth = raw
            return raw

        a = self.smooth_alpha
        sx = (1 - a) * self._last_pose_smooth.x + a * raw.x
        sy = (1 - a) * self._last_pose_smooth.y + a * raw.y

        # Smooth heading via unit vectors
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

    def _choose_active_signature(self) -> None:
        """
        From recent history, pick the (dict,id) that appears most often.
        This is the fully automatic replacement for manual locking.
        """
        if len(self._signature_history) < 8:
            return
        arr = list(self._signature_history)
        # count frequencies
        counts: dict[tuple[str, int], int] = {}
        for sig in arr:
            counts[sig] = counts.get(sig, 0) + 1
        best_sig = max(counts.items(), key=lambda kv: kv[1])[0]
        best_count = counts[best_sig]

        # require consistency
        if best_count >= 6:
            self.active_dict_name, self.active_id = best_sig

    def _forget_active(self) -> None:
        self.active_dict_name = None
        self.active_id = None
        self._signature_history.clear()
        self._last_pose_raw = None
        self._last_pose_smooth = None
        self._seed_center_uv = None
        self._seed_top_uv = None
        self._seed_bottom_uv = None

    def _detect_markers(self, gray: np.ndarray):
        """
        Detection strategy:
          - If we have an active dictionary, try it first.
          - Otherwise, try all dictionaries.
        Returns a list of detections: (dict_name, corners_list, ids)
        """
        dets = []

        # try active dict first (fast path)
        if self.active_dict_name is not None:
            for name, d in self.dicts:
                if name == self.active_dict_name:
                    corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=self.params)
                    if ids is not None and len(ids) > 0:
                        dets.append((name, corners, ids))
                    break
            if dets:
                return dets

        # fall back: try all
        for name, d in self.dicts:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=self.params)
            if ids is not None and len(ids) > 0:
                dets.append((name, corners, ids))
        return dets

    def detect_and_estimate(self, bgr: np.ndarray) -> Optional[RobotPose2D]:
        frame: Optional[GridFrame] = self.kinect_sys.grid_frame
        if frame is None:
            return None
        if self.kinect_sys.last_depth_1d is None:
            return None

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        detections = self._detect_markers(gray)

        if not detections:
            self.last_detected_ids = []
            self.last_dict_used = ""
            self._miss_count += 1
            if self._miss_count >= self._miss_to_forget:
                self._forget_active()
                self._miss_count = 0
            return None

        self._miss_count = 0

        # Build candidates across all dicts
        candidates = []  # (dict_name, idx, perim_px, marker_id, corners4x2)
        all_ids_for_hud: list[int] = []
        for dict_name, corners_list, ids in detections:
            ids_flat = ids.flatten().astype(int)
            all_ids_for_hud.extend(ids_flat.tolist())
            for i, c in enumerate(corners_list):
                c4 = c.reshape(4, 2).astype(np.float64)
                per = self._perimeter(c4)
                if per >= self.min_perimeter_px:
                    mid = int(ids_flat[i])
                    candidates.append((dict_name, i, per, mid, c4, corners_list, ids))

        self.last_detected_ids = sorted(list(set(all_ids_for_hud)))

        if not candidates:
            return None

        # Prefer active signature if set
        chosen = None
        if self.active_dict_name is not None and self.active_id is not None:
            for cand in candidates:
                if cand[0] == self.active_dict_name and cand[3] == self.active_id:
                    chosen = cand
                    break

        # Otherwise choose biggest perimeter overall (best signal)
        if chosen is None:
            chosen = max(candidates, key=lambda t: t[2])

        dict_name, idx, perim_px, marker_id, corners4, corners_list, ids = chosen
        self.last_dict_used = dict_name
        self.last_perimeter_px = float(perim_px)

        # Update signature history and possibly choose a stable signature automatically
        self._signature_history.append((dict_name, marker_id))
        if self.active_dict_name is None or self.active_id is None:
            self._choose_active_signature()

        # corners4 is shape (4,2), used for pose estimation
        corners = corners4
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

        # Confidence: bigger marker => higher confidence
        confidence = float(np.clip(perim_px / 900.0, 0.0, 1.0))

        raw = RobotPose2D(
            x=float(cx),
            y=float(cy),
            heading=heading,
            marker_id=marker_id,
            dict_name=dict_name,
            confidence=confidence,
        )

        # Teleport rejection
        if self._last_pose_raw is not None:
            jump = float(np.hypot(raw.x - self._last_pose_raw.x, raw.y - self._last_pose_raw.y))
            if jump > self.max_jump_m:
                return None

        self._last_pose_raw = raw
        return self._smooth_pose(raw)

    def draw_debug(self, img: np.ndarray) -> None:
        """
        Draw only the currently 'best' dictionary's detections to avoid extra CPU.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try active dict first
        if self.active_dict_name is not None:
            for name, d in self.dicts:
                if name == self.active_dict_name:
                    corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=self.params)
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(img, corners, ids)
                    return

        # Otherwise draw detections from the first dict that finds anything
        for name, d in self.dicts:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=self.params)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
                return
