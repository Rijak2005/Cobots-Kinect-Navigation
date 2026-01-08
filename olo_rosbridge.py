from __future__ import annotations

import asyncio
import json
import threading
import time
import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import websockets
from websockets.exceptions import ConnectionClosed


@dataclass
class RosbridgeConfig:
    url: str
    cmd_vel_topic: str = "/cmd_vel"
    mode_cmd_topic: str = "/set_mode_cmd"
    current_mode_topic: str = "/current_mode"

    send_hz: float = 10.0

    # Keepalive. Start with defaults similar to websockets defaults.
    ping_interval_s: Optional[float] = 20.0
    ping_timeout_s: Optional[float] = 20.0

    connect_timeout_s: float = 10.0
    close_timeout_s: float = 2.0

    reconnect_backoff_s: float = 2.0
    verbose: bool = True

    # Debug print sent cmd_vel every N seconds (0 disables)
    debug_print_period_s: float = 2.0


class RosbridgeCommander:
    """
    Rosbridge client running in a dedicated thread.

    IMPORTANT: We run a background receiver that continuously reads from the socket.
    Without this, the incoming message queue can fill and the library can hit
    keepalive ping timeouts even if sending works.
    """

    def __init__(self, cfg: RosbridgeConfig) -> None:
        self.cfg = cfg
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest_cmd: Tuple[float, float] = (0.0, 0.0)
        self._pending_mode: Optional[str] = None

        self._connected = False
        self._conn_lock = threading.Lock()

        # simple telemetry
        self._last_rx_any: float = 0.0
        self._last_tx_any: float = 0.0
        self._last_mode_seen: str = "unknown"

    # ---------- public API (thread-safe) ----------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main, name="rosbridge-commander", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_evt.set()
        self.stop_robot()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def is_connected(self) -> bool:
        with self._conn_lock:
            return self._connected

    def last_mode_seen(self) -> str:
        with self._lock:
            return self._last_mode_seen

    def send_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        with self._lock:
            self._latest_cmd = (float(linear_x), float(angular_z))

    def stop_robot(self) -> None:
        self.send_cmd_vel(0.0, 0.0)

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._pending_mode = str(mode)

    # ---------- internal ----------

    def _set_connected(self, v: bool) -> None:
        with self._conn_lock:
            self._connected = v

    def _thread_main(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        while not self._stop_evt.is_set():
            try:
                await self._connect_and_run()
            except Exception as e:
                self._set_connected(False)
                if self.cfg.verbose:
                    print(f"[rosbridge] connection error: {e!r}")
                await asyncio.sleep(self.cfg.reconnect_backoff_s)

    async def _connect_and_run(self) -> None:
        if self.cfg.verbose:
            print("[rosbridge] connecting...")

        async with websockets.connect(
            self.cfg.url,
            ping_interval=self.cfg.ping_interval_s,
            ping_timeout=self.cfg.ping_timeout_s,
            open_timeout=self.cfg.connect_timeout_s,
            close_timeout=self.cfg.close_timeout_s,
            max_queue=256,  # allow more incoming before backpressure
        ) as ws:
            self._set_connected(True)
            if self.cfg.verbose:
                print("[rosbridge] connected ✅")

            # Start background receiver/drain FIRST.
            rx_task = asyncio.create_task(self._rx_loop(ws))

            # Advertise topics (same pattern as your example) :contentReference[oaicite:2]{index=2}
            await self._send(ws, {"op": "advertise", "topic": self.cfg.cmd_vel_topic, "type": "geometry_msgs/msg/Twist"})
            await self._send(ws, {"op": "advertise", "topic": self.cfg.mode_cmd_topic, "type": "std_msgs/msg/String"})

            # Subscribe to current mode and wait briefly for one message (like your example) :contentReference[oaicite:3]{index=3}
            await self._send(ws, {"op": "subscribe", "topic": self.cfg.current_mode_topic, "type": "std_msgs/msg/String"})
            await asyncio.sleep(0.2)  # give it a moment to deliver

            # If a mode is pending, send it once after connect
            mode = None
            with self._lock:
                mode = self._pending_mode
                self._pending_mode = None
            if mode:
                await self._send(ws, {"op": "publish", "topic": self.cfg.mode_cmd_topic, "msg": {"data": mode}})
                if self.cfg.verbose:
                    print(f"[rosbridge] set_mode -> {mode}")

            # Main TX loop
            period = 1.0 / max(1e-6, float(self.cfg.send_hz))
            next_t = time.monotonic()

            last_debug = 0.0

            while not self._stop_evt.is_set():
                now = time.monotonic()
                if now < next_t:
                    await asyncio.sleep(min(0.02, next_t - now))
                    continue
                next_t += period

                # Send pending mode (if any)
                mode_to_send = None
                with self._lock:
                    if self._pending_mode is not None:
                        mode_to_send = self._pending_mode
                        self._pending_mode = None
                if mode_to_send is not None:
                    await self._send(ws, {"op": "publish", "topic": self.cfg.mode_cmd_topic, "msg": {"data": mode_to_send}})
                    if self.cfg.verbose:
                        print(f"[rosbridge] set_mode -> {mode_to_send}")

                # Send cmd_vel continuously (robot expects a stream)
                with self._lock:
                    lin, ang = self._latest_cmd

                await self._send(
                    ws,
                    {
                        "op": "publish",
                        "topic": self.cfg.cmd_vel_topic,
                        "msg": {
                            "linear": {"x": float(lin), "y": 0.0, "z": 0.0},
                            "angular": {"x": 0.0, "y": 0.0, "z": float(ang)},
                        },
                    },
                )

                # Optional debug print so you KNOW we're sending
                if self.cfg.debug_print_period_s > 0 and (now - last_debug) >= self.cfg.debug_print_period_s:
                    last_debug = now
                    mode_seen = self.last_mode_seen()
                    if self.cfg.verbose:
                        print(f"[rosbridge] TX cmd_vel lin={lin:+.2f} ang={ang:+.2f} | last_mode={mode_seen}")

            # Stop once on exit
            try:
                await self._send(
                    ws,
                    {
                        "op": "publish",
                        "topic": self.cfg.cmd_vel_topic,
                        "msg": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}},
                    },
                )
            except Exception:
                pass

            # End receiver task
            rx_task.cancel()
            with contextlib.suppress(Exception):
                await rx_task

        self._set_connected(False)
        if self.cfg.verbose:
            print("[rosbridge] disconnected")

    async def _send(self, ws, payload: dict[str, Any]) -> None:
        await ws.send(json.dumps(payload))
        with self._lock:
            self._last_tx_any = time.monotonic()

    async def _rx_loop(self, ws) -> None:
        """
        Drain all incoming messages so the websocket stays healthy.
        Also capture current_mode if it arrives.
        """
        try:
            while True:
                raw = await ws.recv()
                with self._lock:
                    self._last_rx_any = time.monotonic()

                # Parse and optionally track current mode.
                try:
                    pkt = json.loads(raw)
                except Exception:
                    continue

                if pkt.get("topic") == self.cfg.current_mode_topic:
                    msg = pkt.get("msg", {})
                    mode = str(msg.get("data", "unknown"))
                    with self._lock:
                        self._last_mode_seen = mode

                    # This mirrors what your example prints :contentReference[oaicite:4]{index=4}
                    if self.cfg.verbose:
                        print(f"[rosbridge] current_mode <- {mode}")

                # Otherwise discard.
        except asyncio.CancelledError:
            return
        except ConnectionClosed:
            return
        except Exception:
            return