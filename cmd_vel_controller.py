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
    # Reach hysteresis (fix hover)
    arrive_radius_m: float = 0.08    # declare reached inside 8 cm
    depart_radius_m: float = 0.12    # only "unreach" if outside 12 cm
    stable_reach_ticks: int = 3      # require N consecutive ticks inside ARRIVE

    # stage boundary
    coarse_radius_m: float = 0.20    # switch to fine inside 20 cm

    # confidence gate
    min_confidence: float = 0.18

    # coarse speeds/gains
    coarse_max_lin: float = 0.28
    coarse_max_ang: float = 0.85
    coarse_k_lin: float = 0.9
    coarse_k_ang: float = 2.0

    # fine speeds/gains
    fine_max_lin: float = 0.10
    fine_max_ang: float = 0.45
    fine_k_lin: float = 0.7
    fine_k_ang: float = 1.6

    # turn-in-place (degrees)
    turn_start_deg: float = 30.0
    turn_stop_deg: float = 12.0

    # prevent "spin forever" very close to target
    disable_turn_in_place_within_m: float = 0.15

    # minimum forward speed when we choose to move (helps deadband)
    min_lin_mps: float = 0.06


@dataclass
class CmdVelOutput:
    linear_x: float
    angular_z: float
    reached: bool
    stage: str
    status: str


class TwoStageCmdVelController:
    """
    Two-stage polar controller + arrival hysteresis.
    """

    def __init__(self, g: TwoStageGains) -> None:
        self.g = g
        self._turning: Optional[str] = None
        self._reach_streak = 0
        self._reached_latched = False

    def reset(self) -> None:
        self._turning = None
        self._reach_streak = 0
        self._reached_latched = False

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

        # Latch reached with hysteresis
        if self._reached_latched:
            if dist > self.g.depart_radius_m:
                self._reached_latched = False
                self._reach_streak = 0
            else:
                return CmdVelOutput(0.0, 0.0, True, "reached", "Reached (latched)")

        # stable reach streak (prevents jitter)
        if dist <= self.g.arrive_radius_m:
            self._reach_streak += 1
        else:
            self._reach_streak = 0

        if self._reach_streak >= self.g.stable_reach_ticks:
            self._reached_latched = True
            return CmdVelOutput(0.0, 0.0, True, "reached", f"Reached (stable {self.g.stable_reach_ticks} ticks)")

        # polar angle error
        angle_to_target = math.atan2(dy, dx)
        err = wrap_angle(angle_to_target - robot.heading)

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

        # turn-in-place hysteresis (unless very close)
        turn_start = math.radians(self.g.turn_start_deg)
        turn_stop = math.radians(self.g.turn_stop_deg)

        allow_turn_in_place = dist > self.g.disable_turn_in_place_within_m

        if allow_turn_in_place:
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
        else:
            self._turning = None

        if self._turning is not None:
            ang = clamp(k_ang * err, -max_ang, max_ang)
            return CmdVelOutput(0.0, ang, False, stage, f"Turn-in-place (err={math.degrees(err):+.1f}°)")

        # move + steer
        lin = clamp(k_lin * dist, 0.0, max_lin)
        if lin > 0.0:
            lin = max(lin, self.g.min_lin_mps)

        # steering
        ang = clamp(0.8 * k_ang * err, -max_ang, max_ang)

        return CmdVelOutput(lin, ang, False, stage, f"{stage} move dist={dist*100:.1f}cm err={math.degrees(err):+.1f}°")
