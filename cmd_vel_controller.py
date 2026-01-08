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
class TwoStageGains:
    # thresholds
    coarse_radius_m: float = 0.10     # switch to fine control inside this radius (10 cm)
    success_radius_m: float = 0.04    # consider goal reached inside this radius (4 cm)

    # how many consecutive "inside success radius" ticks before we accept reach
    stable_reach_ticks: int = 5

    # confidence gate
    min_confidence: float = 0.18

    # --- coarse stage speeds ---
    coarse_max_lin: float = 0.30
    coarse_max_ang: float = 0.90
    coarse_k_lin: float = 0.9
    coarse_k_ang: float = 2.0

    # --- fine stage speeds ---
    fine_max_lin: float = 0.10
    fine_max_ang: float = 0.45
    fine_k_lin: float = 0.7
    fine_k_ang: float = 1.6

    # heading behavior (degrees)
    turn_in_place_start_deg: float = 30.0
    turn_in_place_stop_deg: float = 12.0

    # allow a small reverse in fine stage (helps if it overshoots)
    allow_reverse_fine: bool = False
    max_reverse_lin: float = -0.05


@dataclass
class CmdVelOutput:
    linear_x: float
    angular_z: float
    reached: bool
    stage: str
    status: str


class TwoStageCmdVelController:
    """
    Two-stage polar controller (coarse then fine).
    Also requires the robot to be within success radius for N consecutive ticks
    before declaring "reached" (prevents flicker).
    """

    def __init__(self, g: TwoStageGains) -> None:
        self.g = g
        self._turning: Optional[str] = None
        self._reach_streak = 0

    def reset(self) -> None:
        self._turning = None
        self._reach_streak = 0

    def compute(self, robot: Optional[RobotPose2D], target_xy: Optional[Tuple[float, float]]) -> CmdVelOutput:
        if target_xy is None:
            self.reset()
            return CmdVelOutput(0.0, 0.0, True, "none", "No target")

        if robot is None:
            self.reset()
            return CmdVelOutput(0.0, 0.0, False, "none", "Robot not visible -> stop")

        if robot.confidence < self.g.min_confidence:
            self.reset()
            return CmdVelOutput(0.0, 0.0, False, "none", "Robot confidence low -> stop")

        tx, ty = target_xy
        dx = tx - robot.x
        dy = ty - robot.y
        dist = math.hypot(dx, dy)

        # stable reach gating
        if dist <= self.g.success_radius_m:
            self._reach_streak += 1
        else:
            self._reach_streak = 0

        if self._reach_streak >= self.g.stable_reach_ticks:
            self.reset()
            return CmdVelOutput(0.0, 0.0, True, "reached", f"Reached (stable {self.g.stable_reach_ticks} ticks)")

        # polar angle error
        angle_to_target = math.atan2(dy, dx)
        err = wrap_angle(angle_to_target - robot.heading)

        turn_start = math.radians(self.g.turn_in_place_start_deg)
        turn_stop = math.radians(self.g.turn_in_place_stop_deg)

        # Hysteresis for turning-in-place
        if self._turning is None:
            if abs(err) > turn_start:
                self._turning = "left" if err > 0 else "right"
        else:
            if abs(err) < turn_stop:
                self._turning = None
            else:
                # allow switching if sign flips
                if err > 0 and self._turning == "right":
                    self._turning = "left"
                elif err < 0 and self._turning == "left":
                    self._turning = "right"

        stage = "coarse" if dist > self.g.coarse_radius_m else "fine"

        if stage == "coarse":
            max_lin = self.g.coarse_max_lin
            max_ang = self.g.coarse_max_ang
            k_lin = self.g.coarse_k_lin
            k_ang = self.g.coarse_k_ang
        else:
            max_lin = self.g.fine_max_lin
            max_ang = self.g.fine_max_ang
            k_lin = self.g.fine_k_lin
            k_ang = self.g.fine_k_ang

        # If we need to turn in place, do it (both stages)
        if self._turning is not None:
            ang = clamp(k_ang * err, -max_ang, max_ang)
            return CmdVelOutput(0.0, ang, False, stage, f"Turn-in-place (err={math.degrees(err):+.1f}°)")

        # Otherwise move + steer
        lin = clamp(k_lin * dist, 0.0, max_lin)

        # Fine stage: optionally allow a tiny reverse if angle error is big and we are very close
        # (Usually not needed; can be enabled later.)
        if stage == "fine" and self.g.allow_reverse_fine:
            if dist < 0.06 and abs(err) > math.radians(45):
                lin = max(lin, self.g.max_reverse_lin)

        # steering
        ang = clamp(0.8 * k_ang * err, -max_ang, max_ang)

        return CmdVelOutput(lin, ang, False, stage, f"{stage} move (dist={dist*100:.1f}cm err={math.degrees(err):+.1f}°)")
