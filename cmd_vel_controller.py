from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from robot_tracker import RobotPose2D


def wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class CmdVelGains:
    # Max speeds (start conservative)
    max_lin: float = 0.25
    max_ang: float = 0.70

    # Proportional gains
    k_lin: float = 0.8         # linear = k_lin * distance
    k_ang: float = 1.8         # angular = k_ang * angle_error

    # Alignment logic (degrees)
    turn_start_deg: float = 25.0
    turn_stop_deg: float = 10.0

    # When close to the goal, slow down
    slow_radius_m: float = 0.25
    min_lin: float = 0.05      # helps overcome stiction (optional)

    # If detection confidence is low, stop
    min_confidence: float = 0.18


@dataclass
class CmdVelOutput:
    linear_x: float
    angular_z: float
    reached: bool
    status: str


class CmdVelController:
    """
    Stateful controller with hysteresis:
      - If angle error large: turn in place
      - Once aligned: move forward while applying small angular correction
    """

    def __init__(self, gains: CmdVelGains) -> None:
        self.g = gains
        self._turning: Optional[str] = None  # "left"/"right"/None

    def reset(self) -> None:
        self._turning = None

    def compute(self, robot: Optional[RobotPose2D], target_xy: Optional[Tuple[float, float]], pos_tol_m: float) -> CmdVelOutput:
        if target_xy is None:
            self.reset()
            return CmdVelOutput(0.0, 0.0, True, "No target")

        if robot is None:
            self.reset()
            return CmdVelOutput(0.0, 0.0, False, "Robot not visible -> stop")

        if robot.confidence < self.g.min_confidence:
            self.reset()
            return CmdVelOutput(0.0, 0.0, False, "Robot confidence low -> stop")

        tx, ty = target_xy
        dx = tx - robot.x
        dy = ty - robot.y
        dist = math.hypot(dx, dy)

        if dist <= pos_tol_m:
            self.reset()
            return CmdVelOutput(0.0, 0.0, True, "Reached target")

        angle_to_target = math.atan2(dy, dx)
        err = wrap_angle(angle_to_target - robot.heading)

        turn_start = math.radians(self.g.turn_start_deg)
        turn_stop = math.radians(self.g.turn_stop_deg)

        # Turning hysteresis
        if self._turning is None:
            if abs(err) > turn_start:
                self._turning = "left" if err > 0 else "right"
        else:
            if abs(err) < turn_stop:
                self._turning = None
            else:
                # allow switch if sign changes strongly
                if err > 0 and self._turning == "right":
                    self._turning = "left"
                elif err < 0 and self._turning == "left":
                    self._turning = "right"

        # Turn in place if needed
        if self._turning is not None:
            ang = clamp(self.g.k_ang * err, -self.g.max_ang, self.g.max_ang)
            return CmdVelOutput(0.0, ang, False, f"Turning {self._turning} (err={math.degrees(err):+.1f}°)")

        # Move forward with a little steering
        lin = clamp(self.g.k_lin * dist, 0.0, self.g.max_lin)

        # slow down near goal
        if dist < self.g.slow_radius_m:
            lin *= (dist / self.g.slow_radius_m)

        # ensure minimum if we decided to move
        if lin > 0.0:
            lin = max(lin, self.g.min_lin)

        ang = clamp(0.8 * self.g.k_ang * err, -self.g.max_ang, self.g.max_ang)

        return CmdVelOutput(lin, ang, False, f"Moving (dist={dist*100:.1f}cm err={math.degrees(err):+.1f}°)")
