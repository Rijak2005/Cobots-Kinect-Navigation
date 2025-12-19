from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from grid_core import GridFrame, grid_xy_to_camera, map_camera_point_to_color_xy
from robot_tracker import RobotPose2D


def wrap_angle(rad: float) -> float:
    return float((rad + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class NavConfig:
    command_interval_s: float = 0.5

    # Position success threshold
    pos_tolerance_m: float = 0.04  # 4cm (start here; tighten later)

    # Turn hysteresis:
    # - start turning if error > turn_start_deg
    # - stop turning only once error < turn_stop_deg
    turn_start_deg: float = 25.0
    turn_stop_deg: float = 10.0

    # When close to the goal, we prefer smaller corrections
    close_distance_m: float = 0.10  # 10 cm

    min_confidence: float = 0.15


class GridNavigator:
    """
    Prints simple commands at fixed intervals to reach each target:
      - turn left / turn right
      - move forward
      - move back (rare fallback)
    """

    def __init__(self, cfg: NavConfig) -> None:
        self.cfg = cfg
        self.targets_xy: List[Tuple[float, float]] = []
        self._idx: int = 0
        self._waiting_for_stilt: bool = False
        self._last_cmd_time: float = 0.0

        # Turning state to avoid oscillation
        self._turning_dir: Optional[str] = None  # "left" / "right" / None

    @property
    def waiting_for_stilt(self) -> bool:
        return self._waiting_for_stilt

    @property
    def current_index(self) -> int:
        return self._idx

    def set_targets(self, targets_xy: List[Tuple[float, float]]) -> None:
        self.targets_xy = list(targets_xy)
        self._idx = 0
        self._waiting_for_stilt = False
        self._turning_dir = None
        self._last_cmd_time = 0.0

    def confirm_stilt_and_advance(self) -> None:
        if self._waiting_for_stilt:
            self._waiting_for_stilt = False
            self._turning_dir = None
            self._idx += 1

    def is_done(self) -> bool:
        return self._idx >= len(self.targets_xy)

    def update(self, robot: Optional[RobotPose2D], now_s: float) -> Optional[str]:
        if now_s - self._last_cmd_time < self.cfg.command_interval_s:
            return None
        self._last_cmd_time = now_s

        if self.is_done():
            return "All targets completed."

        if self._waiting_for_stilt:
            return "SUCCESS: Place Stilt on current position. (press 'n' to continue)"

        if robot is None:
            self._turning_dir = None
            return "Robot not visible. Hold position."

        if robot.confidence < self.cfg.min_confidence:
            self._turning_dir = None
            return "Robot detection weak. Hold position (improve marker visibility)."

        tx, ty = self.targets_xy[self._idx]
        dx = tx - robot.x
        dy = ty - robot.y
        dist = float(np.hypot(dx, dy))

        if dist <= self.cfg.pos_tolerance_m:
            self._waiting_for_stilt = True
            self._turning_dir = None
            return f"SUCCESS: Reached target {self._idx + 1}/{len(self.targets_xy)}. Place Stilt on current position. (press 'n')"

        angle_to_target = float(np.arctan2(dy, dx))
        err = wrap_angle(angle_to_target - robot.heading)

        turn_start = float(np.deg2rad(self.cfg.turn_start_deg))
        turn_stop = float(np.deg2rad(self.cfg.turn_stop_deg))

        # Decide if we should be turning (hysteresis)
        if self._turning_dir is None:
            if abs(err) > turn_start:
                self._turning_dir = "left" if err > 0 else "right"
        else:
            # Keep turning until we're well aligned
            if abs(err) < turn_stop:
                self._turning_dir = None
            else:
                # If error changes sign strongly, switch direction
                if err > 0 and self._turning_dir == "right":
                    self._turning_dir = "left"
                elif err < 0 and self._turning_dir == "left":
                    self._turning_dir = "right"

        # If we are turning, print turning command
        if self._turning_dir is not None:
            return f"turn {self._turning_dir} (angle error {np.rad2deg(err):+.1f}°, dist {dist*100:.1f} cm)"

        # Otherwise, we're aligned enough -> move
        # If very close, just move forward slowly (still a simple command text)
        if dist < self.cfg.close_distance_m:
            return f"move forward (close: {dist*100:.1f} cm)"

        # Forward/back decision (rarely needed, but keeps it sane)
        if np.cos(err) < 0.0:
            return f"move back (dist {dist*100:.1f} cm)"
        return f"move forward (dist {dist*100:.1f} cm)"

    @staticmethod
    def make_targets_3x3_ordered_like_image(
        frame: GridFrame,
        kinect,
        spacing_m: float = 0.60,
    ) -> List[Tuple[float, float]]:
        coords = [-spacing_m, 0.0, spacing_m]
        items = []
        for y in coords:
            for x in coords:
                p_cam = grid_xy_to_camera(x, y, frame)
                uv = map_camera_point_to_color_xy(kinect, p_cam)
                if uv is None:
                    items.append((x, y, -1e9, 1e9))
                else:
                    u, v = uv
                    items.append((x, y, float(u), float(v)))

        # Rightmost first, then top->bottom
        items.sort(key=lambda t: (-t[2], t[3]))
        return [(x, y) for (x, y, _, _) in items]
