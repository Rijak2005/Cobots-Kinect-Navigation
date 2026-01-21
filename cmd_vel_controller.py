from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from robot_tracker import RobotPose2D

# Deadzones (confirmed)
DZ_X = 6553
DZ_Y = 12553
DZ_YAW = 9553

AXIS_MIN = -32767
AXIS_MAX = 32767


def clamp_axis(v: int) -> int:
    return max(AXIS_MIN, min(AXIS_MAX, int(v)))


def wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def world_to_body(vx_w: float, vy_w: float, heading_rad: float) -> tuple[float, float]:
    """
    Convert WORLD vector -> BODY components.

    heading_rad: robot heading in WORLD; 0 means robot faces +WORLD_X
    BODY:
      +x forward
      +y right (positive axis_y moves robot to the right)
    """
    c = math.cos(heading_rad)
    s = math.sin(heading_rad)
    vx_b = c * vx_w + s * vy_w
    vy_b = -s * vx_w + c * vy_w
    return vx_b, vy_b


@dataclass
class AxisGains:
    # ======== USER KNOBS ========
    axis_fwd_fast: int = 9000
    axis_fwd_near: int = 7000  # just above deadzone
    axis_lat_mag: int = 16000

    # yaw magnitudes (micro yaw used at target approach)
    axis_yaw_mag: int = 9600
    axis_yaw_mag_target: int = 9600
    # ============================

    # tolerances
    axis_tol_m: float = 0.03
    arrive_radius_m: float = 0.07
    stable_reach_ticks: int = 4

    # slow down zones (meters)
    slow_y_m: float = 0.25
    slow_x_m: float = 0.25

    # heading alignment
    turn_tol_deg: float = 8.0

    # latch turn direction to avoid oscillation
    latch_turn: bool = True

    # bump above deadzone if needed
    bump_over_deadzone: bool = True

    # target yaw behavior
    target_yaw_start_deg: float = 15.0
    yaw_pulse_period: int = 3
    yaw_pulse_on: int = 1

    # NEW: if Y error grows again while doing X leg, switch back to Y (prevents "stuck off to the side")
    reenter_y_hysteresis: float = 2.0  # multiplier on axis_tol_m


@dataclass
class AxisOut:
    ax: int
    ay: int
    az: int
    reached: bool
    phase: str
    status: str


class AxisStepController:
    """
    Translation-first controller:
      - move_to_point_y_then_x: translation only (NO yaw), but can re-enter Y if drift occurs (NEW)
      - turn_to_face_small: micro-yaw (pulsed) with skip threshold (used at TARGET approach)
      - forward_to_point: forward/back, but with Y-guard correction (NEW) for reliability
      - backward_to_point: reverse slowly, with optional Y-guard correction (NEW)
    """

    def __init__(self, g: AxisGains) -> None:
        self.g = g
        self.reset()

    def reset(self) -> None:
        self._reach_streak = 0
        self._last_goal: Optional[Tuple[float, float]] = None
        self._phase_axis = "Y"
        self._turn_dir: Optional[int] = None
        self._lat_sign: int = -1

        self._yaw_tick: int = 0
        self._yaw_target_lock: Optional[Tuple[float, float]] = None

    def set_lateral_sign(self, lat_sign: int) -> None:
        self._lat_sign = +1 if lat_sign >= 0 else -1

    def get_lateral_sign(self) -> int:
        return self._lat_sign

    def _new_goal_if_needed(self, goal: Tuple[float, float]) -> None:
        if self._last_goal != goal:
            self._last_goal = goal
            self._phase_axis = "Y"
            self._reach_streak = 0
            self._turn_dir = None

    def _reached(self, robot: RobotPose2D, goal: Tuple[float, float]) -> bool:
        dx = goal[0] - robot.x
        dy = goal[1] - robot.y
        dist = math.hypot(dx, dy)
        if dist <= self.g.arrive_radius_m:
            self._reach_streak += 1
        else:
            self._reach_streak = 0
        return self._reach_streak >= self.g.stable_reach_ticks

    def _bump(self, v: int, dead: int) -> int:
        if v == 0:
            return 0
        if not self.g.bump_over_deadzone:
            return v
        if abs(v) <= dead:
            return (dead + 1) if v > 0 else -(dead + 1)
        return v

    def _pick_fwd_mag(self, err_m: float, slow_m: float) -> int:
        return self.g.axis_fwd_near if abs(err_m) < slow_m else self.g.axis_fwd_fast

    def _dir_world_to_axes(
        self, vx_w: float, vy_w: float, heading: float, fwd_mag: int, lat_mag: int
    ) -> tuple[int, int]:
        vx_b, vy_b = world_to_body(vx_w, vy_w, heading)

        m = max(abs(vx_b), abs(vy_b))
        if m < 1e-9:
            return 0, 0
        nx = vx_b / m
        ny = vy_b / m

        ax = int(round(nx * fwd_mag))
        ay = int(round(ny * lat_mag))

        ay = int(self._lat_sign * ay)

        ax = clamp_axis(ax)
        ay = clamp_axis(ay)

        ax = self._bump(ax, DZ_X)
        ay = self._bump(ay, DZ_Y)
        return ax, ay

    # -------- translation-only travel (NO yaw) --------
    def move_to_point_y_then_x(self, robot: RobotPose2D, goal: Tuple[float, float]) -> AxisOut:
        self._new_goal_if_needed(goal)

        ex = goal[0] - robot.x
        ey = goal[1] - robot.y

        if self._reached(robot, goal):
            return AxisOut(0, 0, 0, True, "REACHED", "Reached (stable)")

        # NEW: if we were in X phase but lateral drift grows again, re-enter Y phase.
        # This fixes the "pause/resume makes it correct Y" behavior.
        y_reenter = self.g.axis_tol_m * max(1.0, float(self.g.reenter_y_hysteresis))
        if self._phase_axis == "X" and abs(ey) > y_reenter:
            self._phase_axis = "Y"

        # WORLD Y first
        if self._phase_axis == "Y":
            if abs(ey) <= self.g.axis_tol_m:
                self._phase_axis = "X"
            else:
                vy_w = 1.0 if ey > 0 else -1.0
                fwd_mag = self._pick_fwd_mag(ey, self.g.slow_y_m)
                ax, ay = self._dir_world_to_axes(
                    0.0, vy_w, robot.heading, fwd_mag=fwd_mag, lat_mag=self.g.axis_lat_mag
                )
                return AxisOut(ax, ay, 0, False, "MOVE_WY", f"Reduce world-Y ey={ey:+.3f}m fwd_mag={fwd_mag}")

        # then WORLD X
        if abs(ex) > self.g.axis_tol_m:
            vx_w = 1.0 if ex > 0 else -1.0
            fwd_mag = self._pick_fwd_mag(ex, self.g.slow_x_m)
            ax, ay = self._dir_world_to_axes(
                vx_w, 0.0, robot.heading, fwd_mag=fwd_mag, lat_mag=self.g.axis_lat_mag
            )
            return AxisOut(ax, ay, 0, False, "MOVE_WX", f"Reduce world-X ex={ex:+.3f}m fwd_mag={fwd_mag}")

        return AxisOut(0, 0, 0, False, "HOLD", "Inside axis tolerance, holding")

    # -------- yaw helpers (micro yaw) --------
    def _yaw_cmd(self, err_rad: float, yaw_mag: int, pulse: bool) -> AxisOut:
        tol = math.radians(self.g.turn_tol_deg)
        if abs(err_rad) <= tol:
            self._turn_dir = None
            return AxisOut(0, 0, 0, True, "TURN_DONE", f"Facing OK err={math.degrees(err_rad):+.1f}°")

        if self.g.latch_turn:
            if self._turn_dir is None:
                self._turn_dir = +1 if err_rad > 0 else -1
            az = yaw_mag * self._turn_dir
        else:
            az = yaw_mag if err_rad > 0 else -yaw_mag

        az = clamp_axis(az)
        if abs(az) <= DZ_YAW:
            az = (DZ_YAW + 1) if az > 0 else -(DZ_YAW + 1)

        if not pulse:
            return AxisOut(0, 0, az, False, "TURN", f"Turn err={math.degrees(err_rad):+.1f}°")

        # pulsed yaw to reduce overshoot
        self._yaw_tick += 1
        period = max(1, int(self.g.yaw_pulse_period))
        on = max(0, min(period, int(self.g.yaw_pulse_on)))
        in_on_window = ((self._yaw_tick - 1) % period) < on
        az_out = az if in_on_window else 0

        return AxisOut(0, 0, az_out, False, "TURN_PULSE", f"Pulsed yaw err={math.degrees(err_rad):+.1f}° az={'ON' if in_on_window else 'OFF'}")

    def turn_to_face_small(self, robot: RobotPose2D, face_goal: Tuple[float, float]) -> AxisOut:
        """
        For TARGET approach only:
          - skip yaw if error is small
          - otherwise pulsed micro-yaw
        """
        if self._yaw_target_lock != face_goal:
            self._yaw_target_lock = face_goal
            self._yaw_tick = 0
            self._turn_dir = None

        dx = face_goal[0] - robot.x
        dy = face_goal[1] - robot.y
        desired = math.atan2(dy, dx)
        err = wrap_angle(desired - robot.heading)

        if abs(err) < math.radians(self.g.target_yaw_start_deg):
            self._turn_dir = None
            return AxisOut(0, 0, 0, True, "TURN_SKIPPED", f"Skip small yaw err={math.degrees(err):+.1f}°")

        return self._yaw_cmd(err, yaw_mag=int(self.g.axis_yaw_mag_target), pulse=True)

    # -------- final approach (TARGET): forward/back, but with a Y-guard correction (NEW) --------
    def forward_to_point(self, robot: RobotPose2D, goal: Tuple[float, float]) -> AxisOut:
        if self._reached(robot, goal):
            return AxisOut(0, 0, 0, True, "REACHED", "Reached (stable)")

        dx = goal[0] - robot.x
        dy = goal[1] - robot.y

        # NEW: if lateral error is significant, correct Y first (translation-only).
        if abs(dy) > self.g.axis_tol_m:
            vy_w = 1.0 if dy > 0 else -1.0
            fwd_mag = self._pick_fwd_mag(dy, self.g.slow_y_m)
            ax, ay = self._dir_world_to_axes(
                0.0, vy_w, robot.heading, fwd_mag=fwd_mag, lat_mag=self.g.axis_lat_mag
            )
            return AxisOut(ax, ay, 0, False, "FINAL_LAT", f"Final: reduce world-Y dy={dy:+.3f}m")

        mag = self._pick_fwd_mag(dx, slow_m=0.35)
        ax = mag if dx >= 0 else -mag
        ax = self._bump(clamp_axis(ax), DZ_X)
        return AxisOut(ax, 0, 0, False, "FINAL_FWD", f"Forward only mag={mag}")

    # -------- reverse: back out slowly, with optional Y-guard (NEW) --------
    def backward_to_point(self, robot: RobotPose2D, goal: Tuple[float, float]) -> AxisOut:
        if self._reached(robot, goal):
            return AxisOut(0, 0, 0, True, "REACHED", "Reached (stable)")

        dx = goal[0] - robot.x
        dy = goal[1] - robot.y

        # NEW: if lateral error is significant, correct Y first (translation-only) before backing.
        if abs(dy) > self.g.axis_tol_m:
            vy_w = 1.0 if dy > 0 else -1.0
            fwd_mag = self._pick_fwd_mag(dy, self.g.slow_y_m)
            ax, ay = self._dir_world_to_axes(
                0.0, vy_w, robot.heading, fwd_mag=fwd_mag, lat_mag=self.g.axis_lat_mag
            )
            dist = math.hypot(dx, dy)
            return AxisOut(ax, ay, 0, False, "BACK_LAT", f"Back: reduce world-Y dy={dy:+.3f}m dist={dist:.2f}m")

        mag = int(self.g.axis_fwd_near)
        ax = -mag
        ax = self._bump(clamp_axis(ax), DZ_X)

        dist = math.hypot(dx, dy)
        return AxisOut(ax, 0, 0, False, "BACK", f"Backward only mag={mag} dist={dist:.2f}m")
