import socket
import struct
import time
import threading
from dataclasses import dataclass

# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class RobotConfig:
    robot_ip: str = "192.168.1.120"
    robot_port: int = 43893
    local_port: int = 12345

# ----------------------------
# Lite3 UDP helper
# ----------------------------

class Lite3UDP:
    """
    Implements the "simple command" UDP packet:
      struct CommandHead { uint32_t code; uint32_t paramters_size; uint32_t type; }
    Stored little-endian. :contentReference[oaicite:4]{index=4}
    """

    # --- codes (Motion Host Communication Interface manual) ---
    HEARTBEAT = 0x21040001            # >= 2Hz :contentReference[oaicite:5]{index=5}
    STAND_SIT = 0x21010202
    STOP = 0x21020C0E

    POSE_MODE = 0x21010D05            # :contentReference[oaicite:6]{index=6}
    MOVE_MODE = 0x21010D06            # :contentReference[oaicite:7]{index=7}

    MANUAL_MODE = 0x21010C02          # manual vs navigation :contentReference[oaicite:8]{index=8}

    # Pose-mode axis:
    AXIS_BODY_HEIGHT = 0x21010102     # :contentReference[oaicite:9]{index=9}

    # From manual (Pose Mode deadzone table):
    # Adjust Body Height deadzone range is [-20000, 20000]. :contentReference[oaicite:10]{index=10}
    BODY_HEIGHT_DEADZONE = 20000

    # Axis command overall range [-32767, 32767]. :contentReference[oaicite:11]{index=11}
    AXIS_MIN = -32767
    AXIS_MAX =  32767

    def __init__(self, cfg: RobotConfig) -> None:
        self.cfg = cfg
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", cfg.local_port))

        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None

    @staticmethod
    def _to_u32(val: int) -> int:
        # Represent signed int32 inside uint32 field (two's complement).
        return val & 0xFFFFFFFF

    def build_simple(self, code: int, value: int = 0, cmd_type: int = 0) -> bytes:
        return struct.pack("<III", self._to_u32(code), self._to_u32(value), self._to_u32(cmd_type))

    def send_simple(self, code: int, value: int = 0, cmd_type: int = 0, repeat: int = 1, dt: float = 0.05) -> None:
        pkt = self.build_simple(code, value, cmd_type)
        for _ in range(max(1, repeat)):
            self.sock.sendto(pkt, (self.cfg.robot_ip, self.cfg.robot_port))
            time.sleep(dt)

    # ----------------------------
    # Heartbeat
    # ----------------------------

    def start_heartbeat(self, hz: float = 2.0) -> None:
        # Heartbeat should be >= 2Hz. :contentReference[oaicite:12]{index=12}
        period = 1.0 / max(hz, 0.5)
        self._hb_stop.clear()
        pkt = self.build_simple(self.HEARTBEAT, 0, 0)

        def loop() -> None:
            while not self._hb_stop.is_set():
                try:
                    self.sock.sendto(pkt, (self.cfg.robot_ip, self.cfg.robot_port))
                except OSError:
                    break
                time.sleep(period)

        self._hb_thread = threading.Thread(target=loop, daemon=True)
        self._hb_thread.start()

    def stop_heartbeat(self) -> None:
        self._hb_stop.set()

    # ----------------------------
    # Safe control helpers
    # ----------------------------

    def emergency_stop(self) -> None:
        self.send_simple(self.STOP, 0, 0, repeat=8, dt=0.05)

    def stand_toggle(self) -> None:
        self.send_simple(self.STAND_SIT, 0, 0, repeat=3, dt=0.1)

    def set_manual_mode(self) -> None:
        # Control mode: manual. :contentReference[oaicite:13]{index=13}
        self.send_simple(self.MANUAL_MODE, 0, 0, repeat=5, dt=0.05)

    def enter_pose_mode(self) -> None:
        self.send_simple(self.POSE_MODE, 0, 0, repeat=5, dt=0.05)

    def enter_move_mode(self) -> None:
        self.send_simple(self.MOVE_MODE, 0, 0, repeat=5, dt=0.05)

    # ----------------------------
    # Pose-mode body height control (deadzone-safe)
    # ----------------------------

    def _clamp_axis(self, v: int) -> int:
        return max(self.AXIS_MIN, min(self.AXIS_MAX, v))

    def _ensure_outside_deadzone(self, v: int) -> int:
        """
        Manual: if within deadzone, treated as 0. :contentReference[oaicite:14]{index=14}
        So if abs(v) <= 20000, it does NOTHING. Push it just past the deadzone.
        """
        if v == 0:
            return 0
        if abs(v) <= self.BODY_HEIGHT_DEADZONE:
            v = (self.BODY_HEIGHT_DEADZONE + 500) * (1 if v > 0 else -1)
        return self._clamp_axis(v)

    def send_body_height_axis(self, axis_value: int, duration_s: float, hz: float = 50.0) -> None:
        """
        Pose Mode: if no axis command received for >1s, it returns to normal standing. :contentReference[oaicite:15]{index=15}
        So we keep sending for the entire duration.
        """
        axis_value = self._ensure_outside_deadzone(axis_value)

        dt = 1.0 / max(hz, 5.0)
        pkt = self.build_simple(self.AXIS_BODY_HEIGHT, axis_value, 0)

        t0 = time.time()
        while (time.time() - t0) < duration_s:
            self.sock.sendto(pkt, (self.cfg.robot_ip, self.cfg.robot_port))
            time.sleep(dt)

    def ramp_body_height(self, start: int, end: int, ramp_s: float, hz: float = 50.0) -> None:
        """
        Smoothly ramps the commanded axis value from start to end.
        """
        steps = max(1, int(ramp_s * hz))
        for i in range(steps):
            a = i / steps
            v = int(round(start + (end - start) * a))
            self.send_body_height_axis(v, duration_s=1.0 / hz, hz=hz)

        # Ensure we land on exact end value
        self.send_body_height_axis(end, duration_s=0.2, hz=hz)

    def close(self) -> None:
        try:
            self.stop_heartbeat()
        finally:
            self.sock.close()

# ----------------------------
# Main: lower then raise
# ----------------------------

def main() -> None:
    cfg = RobotConfig()
    bot = Lite3UDP(cfg)

    # This is the key fix:
    # Must be outside deadzone (|v| > 20000) or it becomes 0. :contentReference[oaicite:16]{index=16}
    #
    # Start conservative. You can try -22500, -24500, -26500, etc.
    TARGET_LOWER = -35000

    try:
        input(
            "Safety:\n"
            "- Clear area around the robot\n"
            "- Be ready to hit STOP\n"
            "Press Enter to start..."
        )

        bot.start_heartbeat(hz=2.0)  # :contentReference[oaicite:17]{index=17}
        bot.set_manual_mode()        # safer if anything was in navigation mode :contentReference[oaicite:18]{index=18}

        # Stand up
        print("Standing up...")
        bot.stand_toggle()
        time.sleep(5.0)

        # Pose mode
        print("Entering Pose Mode...")
        bot.enter_pose_mode()
        time.sleep(0.5)

        # Ramp down gently (no sudden drop)
        print(f"Lowering (deadzone-safe) to {TARGET_LOWER} ...")
        bot.ramp_body_height(start=0, end=TARGET_LOWER, ramp_s=2.0, hz=50.0)

        # Hold low for a moment (keep sending or it resets after ~1s) :contentReference[oaicite:19]{index=19}
        print("Holding low...")
        bot.send_body_height_axis(TARGET_LOWER, duration_s=2.0, hz=30.0)

        # Ramp back up to normal
        print("Raising back to normal...")
        bot.ramp_body_height(start=TARGET_LOWER, end=0, ramp_s=2.0, hz=50.0)

        # Back to Move Mode (optional, doesn’t change gait)
        bot.enter_move_mode()
        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted -> STOP")
        bot.emergency_stop()
    except Exception as e:
        print(f"\nError: {e}\nSTOP for safety.")
        bot.emergency_stop()
        raise
    finally:
        bot.close()

if __name__ == "__main__":
    main()