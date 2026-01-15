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
class TurnStraightGains:
    # --- reach logic ---
    arrive_radius_m: float = 0.08        # declare reached inside this
    depart_radius_m: float = 0.12        # “unreach” only outside this
    stable_reach_ticks: int = 3

    # --- phase switching ---
    turn_tol_deg: float = 10.0           # when yaw error below this, start driving straight
    re_turn_tol_deg: float = 18.0        # if yaw error grows above this while driving, go back to turn
    re_turn_min_dist_m: float = 0.20     # only re-turn if still far enough

    # --- speeds (m/s, rad/s) ---
    max_lin: float = 0.28
    max_ang: float = 0.85
    min_lin: float = 0.06               # helps motor deadband

    # slow down near goal
    slow_dist_m: float = 0.30           # start slowing inside this
    min_lin_near: float = 0.03

    # --- gains ---
    k_turn: float = 2.0                 # yaw error -> angular z in TURN phase
    k_drive_yaw: float = 1.2            # yaw error -> angular z in DRIVE phase
    k_lin: float = 1.0                  # distance -> linear x

    # smoothing for yaw in DRIVE (reduces chaotic curves)
    yaw_err_lpf_alpha: float = 0.25     # 0..1 (higher = less smoothing)

    # confidence gating
    min_confidence: float = 0.12


@dataclass
class CmdVelOutput:
    linear_x: float
    angular_z: float
    reached: bool
    phase: str
    status: str


class TurnThenStraightController:
    """
    Deterministic, non-chaotic motion:
      1) TURN in place until facing goal (within turn_tol)
      2) DRIVE straight, using a smoothed yaw correction
      3) Uses hysteresis (arrive/depart) so it doesn't hover forever due to jitter.
    """

    def __init__(self, g: TurnStraightGains) -> None:
        self.g = g
        self._reached_latched = False
        self._reach_streak = 0

        self._phase: str = "TURN"
        self._locked_bearing: Optional[float] = None
        self._yaw_err_filt: float = 0.0

        self._last_goal: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._reached_latched = False
        self._reach_streak = 0
        self._phase = "TURN"
        self._locked_bearing = None
        self._yaw_err_filt = 0.0
        self._last_goal = None

    def _new_goal_if_needed(self, goal: Tuple[float, float]) -> None:
        if self._last_goal != goal:
            self._last_goal = goal
            self._phase = "TURN"
            self._locked_bearing = None
            self._yaw_err_filt = 0.0
            self._reached_latched = False
            self._reach_streak = 0

    def compute(self, robot: Optional[RobotPose2D], goal_xy: Optional[Tuple[float, float]]) -> CmdVelOutput:
        if goal_xy is None:
            self.reset()
            return CmdVelOutput(0.0, 0.0, True, "NONE", "No goal")

        self._new_goal_if_needed(goal_xy)

        if robot is None:
            return CmdVelOutput(0.0, 0.0, False, "NONE", "Robot pose unavailable -> stop")

        if robot.confidence < self.g.min_confidence:
            return CmdVelOutput(0.0, 0.0, False, "NONE", "Robot confidence low -> stop")

        gx, gy = goal_xy
        dx = gx - robot.x
        dy = gy - robot.y
        dist = math.hypot(dx, dy)

        # reached latch with hysteresis
        if self._reached_latched:
            if dist > self.g.depart_radius_m:
                self._reached_latched = False
                self._reach_streak = 0
            else:
                return CmdVelOutput(0.0, 0.0, True, "REACHED", "Reached (latched)")

        if dist <= self.g.arrive_radius_m:
            self._reach_streak += 1
        else:
            self._reach_streak = 0

        if self._reach_streak >= self.g.stable_reach_ticks:
            self._reached_latched = True
            return CmdVelOutput(0.0, 0.0, True, "REACHED", "Reached (stable)")

        bearing = math.atan2(dy, dx)

        # lock bearing when we leave TURN -> DRIVE (to keep a straight line)
        if self._phase == "TURN":
            yaw_err = wrap_angle(bearing - robot.heading)
            turn_tol = math.radians(self.g.turn_tol_deg)

            if abs(yaw_err) <= turn_tol:
                self._phase = "DRIVE"
                self._locked_bearing = bearing
                self._yaw_err_filt = 0.0
                return CmdVelOutput(0.0, 0.0, False, "DRIVE", "Aligned -> start drive")

            ang = clamp(self.g.k_turn * yaw_err, -self.g.max_ang, self.g.max_ang)
            return CmdVelOutput(0.0, ang, False, "TURN", f"Turning (err={math.degrees(yaw_err):+.1f}°)")

        # DRIVE phase
        assert self._locked_bearing is not None
        yaw_err_drive = wrap_angle(self._locked_bearing - robot.heading)

        # low-pass filter yaw error (prevents left/right oscillation)
        a = self.g.yaw_err_lpf_alpha
        self._yaw_err_filt = (1 - a) * self._yaw_err_filt + a * yaw_err_drive

        # if we drift off heading significantly (and still far), re-enter TURN
        re_turn_tol = math.radians(self.g.re_turn_tol_deg)
        if dist > self.g.re_turn_min_dist_m and abs(yaw_err_drive) > re_turn_tol:
            self._phase = "TURN"
            self._locked_bearing = None
            return CmdVelOutput(0.0, 0.0, False, "TURN", "Re-acquire heading")

        # linear speed proportional to distance
        lin = clamp(self.g.k_lin * dist, 0.0, self.g.max_lin)

        # slow down near goal
        if dist < self.g.slow_dist_m:
            # scale linearly to a minimum
            t = dist / max(1e-6, self.g.slow_dist_m)
            lin = max(self.g.min_lin_near, lin * t)

        if lin > 0:
            lin = max(lin, self.g.min_lin)

        # yaw correction during drive (smoothed)
        ang = clamp(self.g.k_drive_yaw * self._yaw_err_filt, -self.g.max_ang, self.g.max_ang)

        return CmdVelOutput(lin, ang, False, "DRIVE", f"Driving dist={dist*100:.1f}cm err={math.degrees(yaw_err_drive):+.1f}°")