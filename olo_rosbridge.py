from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import websockets


@dataclass
class RosbridgeConfig:
    url: str
    cmd_vel_topic: str = "/cmd_vel"
    mode_topic: str = "/set_mode_cmd"

    # How often we send cmd_vel (Hz)
    send_hz: float = 10.0

    # Websocket keepalive
    ping_interval_s: float = 10.0
    ping_timeout_s: float = 10.0

    # Reconnect behavior
    reconnect_backoff_s: float = 2.0
    connect_timeout_s: float = 10.0

    # Debug prints
    verbose: bool = True


class RosbridgeCommander:
    """
    Runs rosbridge websocket in a dedicated thread (so OpenCV/Kinect can't block it).

    Public (thread-safe) methods:
      - start()
      - send_cmd_vel(linear_x, angular_z)
      - stop_robot()
      - set_mode(mode)
      - is_connected()
      - shutdown()
    """

    def __init__(self, cfg: RosbridgeConfig) -> None:
        self.cfg = cfg

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # Latest command buffer (protected by lock)
        self._lock = threading.Lock()
        self._latest_cmd: Tuple[float, float] = (0.0, 0.0)  # (linear_x, angular_z)
        self._latest_mode: Optional[str] = None

        # Connection state
        self._connected = False
        self._connected_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main, name="rosbridge-commander", daemon=True)
        self._thread.start()

    def is_connected(self) -> bool:
        with self._connected_lock:
            return self._connected

    def _set_connected(self, v: bool) -> None:
        with self._connected_lock:
            self._connected = v

    def send_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        with self._lock:
            self._latest_cmd = (float(linear_x), float(angular_z))

    def stop_robot(self) -> None:
        self.send_cmd_vel(0.0, 0.0)

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._latest_mode = str(mode)

    def shutdown(self) -> None:
        self._stop_evt.set()
        # ensure we request a stop immediately
        self.stop_robot()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # -------------------- Thread / asyncio side --------------------

    def _thread_main(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        """
        Reconnect loop. Never crashes the main program.
        """
        backoff = self.cfg.reconnect_backoff_s

        while not self._stop_evt.is_set():
            try:
                await self._connect_and_loop()
            except Exception as e:
                self._set_connected(False)
                if self.cfg.verbose:
                    print(f"[rosbridge] connection error: {e!r}")
                # small backoff before reconnect
                await asyncio.sleep(backoff)

    async def _connect_and_loop(self) -> None:
        if self.cfg.verbose:
            print("[rosbridge] connecting...")

        # Use async context manager like your example script :contentReference[oaicite:1]{index=1}
        async with websockets.connect(
            self.cfg.url,
            ping_interval=self.cfg.ping_interval_s,
            ping_timeout=self.cfg.ping_timeout_s,
            open_timeout=self.cfg.connect_timeout_s,
            close_timeout=2.0,
            max_queue=32,
        ) as ws:
            self._set_connected(True)
            if self.cfg.verbose:
                print("[rosbridge] connected ✅")

            # Advertise topics (same types as example script) :contentReference[oaicite:2]{index=2}
            await ws.send(json.dumps({
                "op": "advertise",
                "topic": self.cfg.cmd_vel_topic,
                "type": "geometry_msgs/msg/Twist",
            }))
            await ws.send(json.dumps({
                "op": "advertise",
                "topic": self.cfg.mode_topic,
                "type": "std_msgs/msg/String",
            }))

            # If caller already requested a mode, send it once after connect.
            mode = None
            with self._lock:
                mode = self._latest_mode
            if mode:
                await ws.send(json.dumps({
                    "op": "publish",
                    "topic": self.cfg.mode_topic,
                    "msg": {"data": mode},
                }))
                if self.cfg.verbose:
                    print(f"[rosbridge] set_mode -> {mode}")

            # Main send loop
            period = 1.0 / max(1e-6, float(self.cfg.send_hz))
            next_t = time.monotonic()

            # We re-send cmd_vel continuously (robot expects a stream)
            while not self._stop_evt.is_set():
                now = time.monotonic()
                if now < next_t:
                    await asyncio.sleep(min(0.02, next_t - now))
                    continue
                next_t += period

                # Send mode if updated
                mode_to_send = None
                with self._lock:
                    if self._latest_mode is not None:
                        mode_to_send = self._latest_mode
                        self._latest_mode = None

                if mode_to_send is not None:
                    await ws.send(json.dumps({
                        "op": "publish",
                        "topic": self.cfg.mode_topic,
                        "msg": {"data": mode_to_send},
                    }))
                    if self.cfg.verbose:
                        print(f"[rosbridge] set_mode -> {mode_to_send}")

                # Send cmd_vel
                with self._lock:
                    lin, ang = self._latest_cmd

                msg = {
                    "op": "publish",
                    "topic": self.cfg.cmd_vel_topic,
                    "msg": {
                        "linear": {"x": float(lin), "y": 0.0, "z": 0.0},
                        "angular": {"x": 0.0, "y": 0.0, "z": float(ang)},
                    },
                }
                await ws.send(json.dumps(msg))

            # Before exiting connection, stop robot once
            try:
                await ws.send(json.dumps({
                    "op": "publish",
                    "topic": self.cfg.cmd_vel_topic,
                    "msg": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}},
                }))
            except Exception:
                pass

        self._set_connected(False)
        if self.cfg.verbose:
            print("[rosbridge] disconnected")
