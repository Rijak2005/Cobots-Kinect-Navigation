from __future__ import annotations

"""Lite3 UDP commander (SAFE SUBSET).

This module replaces the previous rosbridge/cmd_vel publisher with direct UDP
commands to the Lite3 motion host.

Safety rules in this file:
  - We ONLY use the same basic command codes as the provided working example:
      * Heartbeat
      * Switch motion mode to MOVE / POSE
      * Axis commands for forward/back, left/right, and yaw
  - NO gait switching, NO flips, NO special actions.
  - If the main loop stops updating commands, we automatically fall back to STOP.

The motion host expects axis commands at >= 20 Hz while in MOVE mode.
We therefore run a dedicated sender thread at 50 Hz.
"""

import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Tuple


# ---------------- Lite3 command codes (simple commands) ----------------
# These match the user's working UDP sample script.
CMD_HEARTBEAT = 0x21040001
CMD_STAND_SIT_TOGGLE = 0x21010202  # NOTE: this is a toggle in many firmwares

CMD_MODE_POSE = 0x21010D05
CMD_MODE_MOVE = 0x21010D06

# Axis in MOVE mode
CMD_AXIS_Y = 0x21010131  # Translation (Left/Right)
CMD_AXIS_X = 0x21010130  # Translation (Forward/Backward)
CMD_AXIS_YAW = 0x21010135  # Turning Left/Right


# ---------------- Axis value constraints ----------------
AXIS_MAX_ABS = 32700  # keep slightly below full scale

# Dead-zones from the vendor manual (values inside these are treated as 0).
DEADZONE_X = 6553
DEADZONE_Y = 12553
DEADZONE_YAW = 9553


@dataclass(frozen=True)
class Lite3UdpConfig:
    robot_ip: str
    robot_port: int = 43893
    local_port: int = 12345

    # Heartbeat period (sample uses 0.5s)
    heartbeat_period_s: float = 0.5

    # Axis stream rate. Must be >= 20 Hz in MOVE mode.
    send_hz: float = 50.0

    # If no fresh command for this long, stream STOP instead.
    command_stale_s: float = 0.20

    # Mapping from (m/s, rad/s) -> axis integers.
    # We map "controller max" -> full axis range.
    # Keeping these equal to the controller's max values makes tuning intuitive.
    lin_full_scale_mps: float = 0.28
    yaw_full_scale_rps: float = 0.85
    lat_full_scale_mps: float = 0.28

    verbose: bool = True


def _clamp_i32(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _apply_deadzone(v: int, deadzone_abs: int) -> int:
    return 0 if abs(v) <= deadzone_abs else v


def _to_u32(v: int) -> int:
    """Convert signed int32 to uint32 for struct packing."""
    if v < 0:
        return v + (1 << 32)
    return v


def build_simple_command(code: int, value: int = 0, cmd_type: int = 0) -> bytes:
    """Pack a 'simple command' frame as used by the vendor example."""
    return struct.pack("<III", int(code), _to_u32(int(value)), int(cmd_type))


class Lite3UdpCommander:
    """Threaded UDP sender with heartbeat + stale-command watchdog."""

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
        self._latest_cmd = (0.0, 0.0, 0.0)  # (lin_x, lin_y, yaw_z)
        self._latest_cmd_t = 0.0

        self._started = False

    # ---------------- public API ----------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True

        self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="lite3-heartbeat", daemon=True)
        self._tx_thread = threading.Thread(target=self._tx_loop, name="lite3-axis-tx", daemon=True)
        self._hb_thread.start()
        self._tx_thread.start()

        if self.cfg.verbose:
            print(f"[lite3-udp] Heartbeat started -> {self._addr[0]}:{self._addr[1]}")

    def shutdown(self) -> None:
        self._stop_evt.set()
        # Stream stop for a short moment before closing.
        try:
            self.stop_robot(buffer_s=0.3)
        except Exception:
            pass

        if self._hb_thread is not None:
            self._hb_thread.join(timeout=2.0)
            self._hb_thread = None
        if self._tx_thread is not None:
            self._tx_thread.join(timeout=2.0)
            self._tx_thread = None

        try:
            self._sock.close()
        except Exception:
            pass

        if self.cfg.verbose:
            print("[lite3-udp] shutdown")

    def is_connected(self) -> bool:
        """UDP has no connection state. If we started, treat as 'connected'."""
        return self._started and not self._stop_evt.is_set()

    def set_motion_mode(self, mode: str) -> None:
        """Switch between 'move' and 'pose' motion modes."""
        m = mode.strip().lower()
        if m == "move":
            self._send_simple(CMD_MODE_MOVE, 0)
            # Like the sample: send multiple times to make it stick.
            for _ in range(4):
                time.sleep(0.05)
                self._send_simple(CMD_MODE_MOVE, 0)
            if self.cfg.verbose:
                print("[lite3-udp] set_motion_mode -> MOVE")
        elif m == "pose":
            self._send_simple(CMD_MODE_POSE, 0)
            for _ in range(4):
                time.sleep(0.05)
                self._send_simple(CMD_MODE_POSE, 0)
            if self.cfg.verbose:
                print("[lite3-udp] set_motion_mode -> POSE")
        else:
            raise ValueError("mode must be 'move' or 'pose'")

    def stand_sit_toggle(self) -> None:
        """Toggle stand/sit. WARNING: This is a toggle in many firmwares."""
        # Keep as an explicit call only. We do NOT auto-call this.
        self._send_simple(CMD_STAND_SIT_TOGGLE, 0)
        for _ in range(2):
            time.sleep(0.1)
            self._send_simple(CMD_STAND_SIT_TOGGLE, 0)
        if self.cfg.verbose:
            print("[lite3-udp] stand/sit TOGGLE sent")

    def send_cmd_vel(self, linear_x: float, angular_z: float, linear_y: float = 0.0) -> None:
        """Set desired velocities in (m/s, rad/s). They will be streamed as axis commands."""
        with self._lock:
            self._latest_cmd = (float(linear_x), float(linear_y), float(angular_z))
            self._latest_cmd_t = time.monotonic()

    def stop_robot(self, buffer_s: float = 0.0) -> None:
        """Request stop and (optionally) stream stop for a short buffer time."""
        self.send_cmd_vel(0.0, 0.0, 0.0)
        if buffer_s > 0:
            end = time.monotonic() + float(buffer_s)
            while time.monotonic() < end:
                self._send_axis(0, 0, 0)
                time.sleep(0.02)

    # ---------------- internals ----------------

    def _send_simple(self, code: int, value: int = 0) -> None:
        pkt = build_simple_command(code, value, 0)
        self._sock.sendto(pkt, self._addr)

    def _heartbeat_loop(self) -> None:
        pkt = build_simple_command(CMD_HEARTBEAT, 0, 0)
        while not self._stop_evt.is_set():
            try:
                self._sock.sendto(pkt, self._addr)
            except Exception:
                # UDP send errors are rare; if they happen, keep trying.
                pass
            time.sleep(self.cfg.heartbeat_period_s)

    def _mps_to_axis(self, v_mps: float, full_scale_mps: float) -> int:
        fs = float(full_scale_mps)
        if fs <= 1e-9:
            return 0
        axis = int(round((float(v_mps) / fs) * AXIS_MAX_ABS))
        return _clamp_i32(axis, -AXIS_MAX_ABS, AXIS_MAX_ABS)

    def _rps_to_axis(self, v_rps: float, full_scale_rps: float) -> int:
        fs = float(full_scale_rps)
        if fs <= 1e-9:
            return 0
        axis = int(round((float(v_rps) / fs) * AXIS_MAX_ABS))
        return _clamp_i32(axis, -AXIS_MAX_ABS, AXIS_MAX_ABS)

    def _send_axis(self, x_axis: int, y_axis: int, yaw_axis: int) -> None:
        """Send MOVE-mode axis commands (3 packets)."""
        try:
            self._sock.sendto(build_simple_command(CMD_AXIS_X, x_axis, 0), self._addr)
            self._sock.sendto(build_simple_command(CMD_AXIS_Y, y_axis, 0), self._addr)
            self._sock.sendto(build_simple_command(CMD_AXIS_YAW, yaw_axis, 0), self._addr)
        except Exception:
            # If UDP send fails, we cannot do much besides retry next tick.
            pass

    def _tx_loop(self) -> None:
        period = 1.0 / max(1e-6, float(self.cfg.send_hz))
        next_t = time.monotonic()
        last_debug = 0.0

        while not self._stop_evt.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue
            next_t += period

            with self._lock:
                lin_x, lin_y, yaw = self._latest_cmd
                age = now - self._latest_cmd_t

            if age > self.cfg.command_stale_s:
                lin_x = 0.0
                lin_y = 0.0
                yaw = 0.0

            # Convert to axis integers.
            ax_x = self._mps_to_axis(lin_x, self.cfg.lin_full_scale_mps)
            ax_y = self._mps_to_axis(lin_y, self.cfg.lat_full_scale_mps)
            ax_yaw = self._rps_to_axis(yaw, self.cfg.yaw_full_scale_rps)

            # Apply dead-zones (robot treats these as 0 anyway).
            ax_x = _apply_deadzone(ax_x, DEADZONE_X)
            ax_y = _apply_deadzone(ax_y, DEADZONE_Y)
            ax_yaw = _apply_deadzone(ax_yaw, DEADZONE_YAW)

            self._send_axis(ax_x, ax_y, ax_yaw)

            if self.cfg.verbose and (now - last_debug) >= 2.0:
                last_debug = now
                print(
                    f"[lite3-udp] TX axis x={ax_x:+6d} y={ax_y:+6d} yaw={ax_yaw:+6d} "
                    f"(lin={lin_x:+.2f} m/s, yaw={yaw:+.2f} rad/s, age={age*1000:.0f}ms)"
                )
