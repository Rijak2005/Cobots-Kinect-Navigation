from __future__ import annotations

"""
Lite3 UDP commander (SAFE SUBSET).

We use:
- heartbeat
- control mode: MANUAL
- motion mode: MOVE / POSE
- MOVE axes: X / Y / YAW
- POSE axis: BODY HEIGHT (for gentle lowering at target)

No action skills, no gait changes, no flips.
"""

import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Tuple

# ---------------- Safe commands ----------------
CMD_HEARTBEAT = 0x21040001

# Control mode (keep MANUAL; do NOT use navigation mode)
CMD_CONTROL_MODE_MANUAL = 0x21010C02

# Motion mode
CMD_MODE_POSE = 0x21010D05
CMD_MODE_MOVE = 0x21010D06

# Optional (manual key only)
CMD_STAND_SIT_TOGGLE = 0x21010202

# MOVE axes
CMD_AXIS_X = 0x21010130   # forward/back
CMD_AXIS_Y = 0x21010131   # left/right (positive -> RIGHT)
CMD_AXIS_YAW = 0x21010135 # turning (firmware often: positive -> RIGHT)

# POSE axis (body height)
CMD_AXIS_BODY_HEIGHT = 0x21010102  # pose mode: adjust body height

# ----------------------------------------------

AXIS_MIN = -32767
AXIS_MAX = 32767

# Deadzones you confirmed (MOVE mode)
DEADZONE_X = 6553
DEADZONE_Y = 12553
DEADZONE_YAW = 9553

# Pose-mode body height deadzone from your manual/script: [-20000, 20000] treated as 0
DEADZONE_BODY_HEIGHT = 20000


@dataclass(frozen=True)
class Lite3UdpConfig:
    robot_ip: str
    robot_port: int = 43893
    local_port: int = 12345

    heartbeat_period_s: float = 0.5
    send_hz: float = 50.0
    command_stale_s: float = 0.30

    # If True, flip yaw sign before sending (controller + = CCW/left, robot + often = right)
    yaw_positive_is_right: bool = True

    verbose: bool = True


def _clamp_axis(v: int) -> int:
    return max(AXIS_MIN, min(AXIS_MAX, int(v)))


def _to_u32(v: int) -> int:
    return v + (1 << 32) if v < 0 else v


def build_simple_command(code: int, value: int = 0, cmd_type: int = 0) -> bytes:
    return struct.pack("<III", int(code), _to_u32(int(value)), int(cmd_type))


def _apply_deadzone_or_zero(v: int, dead: int) -> int:
    return 0 if abs(v) <= dead else v


def _ensure_outside_deadzone(v: int, dead: int, bump: int = 500) -> int:
    """
    If abs(v) <= dead, robot treats it as 0. Push just outside.
    """
    if v == 0:
        return 0
    if abs(v) <= dead:
        v = (dead + bump) if v > 0 else -(dead + bump)
    return _clamp_axis(v)


class Lite3UdpCommander:
    def __init__(self, cfg: Lite3UdpConfig) -> None:
        self.cfg = cfg
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", int(cfg.local_port)))
        self._addr: Tuple[str, int] = (cfg.robot_ip, int(cfg.robot_port))

        self._stop_evt = threading.Event()
        self._hb_thread: threading.Thread | None = None
        self._tx_thread: threading.Thread | None = None

        self._lock = threading.Lock()
        self._latest_axes = (0, 0, 0)  # x_axis, y_axis, yaw_axis (robot sign)
        self._latest_t = 0.0
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._hb_thread.start()
        self._tx_thread.start()
        if self.cfg.verbose:
            print(f"[lite3-udp] started -> {self._addr[0]}:{self._addr[1]}")

    def shutdown(self) -> None:
        self._stop_evt.set()
        try:
            self.stop_robot(buffer_s=0.3)
        except Exception:
            pass
        if self._hb_thread:
            self._hb_thread.join(timeout=2.0)
        if self._tx_thread:
            self._tx_thread.join(timeout=2.0)
        try:
            self._sock.close()
        except Exception:
            pass
        if self.cfg.verbose:
            print("[lite3-udp] shutdown")

    def is_connected(self) -> bool:
        return self._started and not self._stop_evt.is_set()

    def _send_simple(self, code: int, value: int = 0) -> None:
        self._sock.sendto(build_simple_command(code, value, 0), self._addr)

    def set_control_mode_manual(self) -> None:
        for _ in range(5):
            self._send_simple(CMD_CONTROL_MODE_MANUAL, 0)
            time.sleep(0.05)
        if self.cfg.verbose:
            print("[lite3-udp] control mode -> MANUAL")

    def set_motion_mode(self, mode: str) -> None:
        m = mode.strip().lower()
        if m == "move":
            for _ in range(5):
                self._send_simple(CMD_MODE_MOVE, 0)
                time.sleep(0.05)
            if self.cfg.verbose:
                print("[lite3-udp] motion mode -> MOVE")
        elif m == "pose":
            for _ in range(5):
                self._send_simple(CMD_MODE_POSE, 0)
                time.sleep(0.05)
            if self.cfg.verbose:
                print("[lite3-udp] motion mode -> POSE")
        else:
            raise ValueError("mode must be 'move' or 'pose'")

    def stand_sit_toggle(self) -> None:
        for _ in range(3):
            self._send_simple(CMD_STAND_SIT_TOGGLE, 0)
            time.sleep(0.1)

    # ------------------ RAW AXIS CONTROL ------------------
    def send_axes(self, axis_x: int, axis_y: int, axis_yaw: int) -> None:
        """
        Send raw axis commands:
          axis_x: forward/back    (+ = forward)
          axis_y: left/right      (+ = right)  <-- you confirmed
          axis_yaw: turn          (robot's convention; we flip if configured)

        IMPORTANT:
        We flip yaw if cfg.yaw_positive_is_right, because your controller uses + = CCW/left.
        Here we assume axis_yaw passed in is CONTROLLER SIGN (+ = left/CCW).
        """
        ax = _clamp_axis(axis_x)
        ay = _clamp_axis(axis_y)
        az = _clamp_axis(axis_yaw)

        if self.cfg.yaw_positive_is_right:
            az = -az

        with self._lock:
            self._latest_axes = (ax, ay, az)
            self._latest_t = time.monotonic()

    def stop_robot(self, buffer_s: float = 0.0) -> None:
        self.send_axes(0, 0, 0)
        if buffer_s > 0:
            end = time.monotonic() + float(buffer_s)
            while time.monotonic() < end:
                self._send_axis_packet(0, 0, 0)
                time.sleep(0.02)

    def _send_axis_packet(self, x_axis: int, y_axis: int, yaw_axis: int) -> None:
        self._sock.sendto(build_simple_command(CMD_AXIS_X, x_axis, 0), self._addr)
        self._sock.sendto(build_simple_command(CMD_AXIS_Y, y_axis, 0), self._addr)
        self._sock.sendto(build_simple_command(CMD_AXIS_YAW, yaw_axis, 0), self._addr)

    # ------------------ POSE: BODY HEIGHT ------------------
    def send_body_height_axis_once(self, axis_value: int) -> None:
        """
        Pose mode body height axis.
        Must be outside deadzone (|v| > 20000) or it does nothing.
        """
        v = int(axis_value)
        v = _ensure_outside_deadzone(v, DEADZONE_BODY_HEIGHT, bump=500)
        self._sock.sendto(build_simple_command(CMD_AXIS_BODY_HEIGHT, v, 0), self._addr)

    # ------------------ Threads ------------------
    def _heartbeat_loop(self) -> None:
        pkt = build_simple_command(CMD_HEARTBEAT, 0, 0)
        while not self._stop_evt.is_set():
            try:
                self._sock.sendto(pkt, self._addr)
            except Exception:
                pass
            time.sleep(self.cfg.heartbeat_period_s)

    def _tx_loop(self) -> None:
        period = 1.0 / max(1e-6, float(self.cfg.send_hz))
        next_t = time.monotonic()

        while not self._stop_evt.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue
            next_t += period

            with self._lock:
                ax, ay, az = self._latest_axes
                age = now - self._latest_t

            if age > self.cfg.command_stale_s:
                ax = ay = az = 0

            # Apply deadzones (keeps tiny jitters from moving robot)
            ax = _apply_deadzone_or_zero(ax, DEADZONE_X)
            ay = _apply_deadzone_or_zero(ay, DEADZONE_Y)
            az = _apply_deadzone_or_zero(az, DEADZONE_YAW)

            try:
                self._send_axis_packet(ax, ay, az)
            except Exception:
                pass
