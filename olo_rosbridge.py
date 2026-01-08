from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

import websockets


@dataclass
class RosbridgeConfig:
    url: str
    connect_timeout_s: float = 10.0
    recv_timeout_s: float = 2.0
    ping_interval_s: float = 20.0
    ping_timeout_s: float = 20.0


class OloRosbridgeClient:
    """
    Minimal rosbridge websocket client:
      - advertise(topic, type)
      - publish(topic, msg)
      - subscribe_once(topic, type, timeout)  (useful for /current_mode)
      - set_mode(mode) helper
      - stop() helper (publish zero cmd_vel)
    """

    def __init__(self, cfg: RosbridgeConfig) -> None:
        self.cfg = cfg
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self) -> None:
        self._ws = await asyncio.wait_for(
            websockets.connect(
                self.cfg.url,
                ping_interval=self.cfg.ping_interval_s,
                ping_timeout=self.cfg.ping_timeout_s,
                max_queue=32,
            ),
            timeout=self.cfg.connect_timeout_s,
        )

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    def _require_ws(self) -> websockets.WebSocketClientProtocol:
        if self._ws is None:
            raise RuntimeError("Not connected to rosbridge websocket.")
        return self._ws

    async def send_raw(self, payload: dict[str, Any]) -> None:
        ws = self._require_ws()
        await ws.send(json.dumps(payload))

    async def recv_raw(self, timeout_s: Optional[float] = None) -> dict[str, Any]:
        ws = self._require_ws()
        t = self.cfg.recv_timeout_s if timeout_s is None else timeout_s
        data = await asyncio.wait_for(ws.recv(), timeout=t)
        return json.loads(data)

    async def advertise(self, topic: str, msg_type: str) -> None:
        await self.send_raw({"op": "advertise", "topic": topic, "type": msg_type})

    async def publish(self, topic: str, msg: dict[str, Any]) -> None:
        await self.send_raw({"op": "publish", "topic": topic, "msg": msg})

    async def subscribe_once(self, topic: str, msg_type: str, timeout_s: float = 2.0) -> Optional[dict[str, Any]]:
        """
        Subscribe and wait for ONE message on that topic.
        """
        await self.send_raw({"op": "subscribe", "topic": topic, "type": msg_type})
        try:
            while True:
                packet = await self.recv_raw(timeout_s=timeout_s)
                if packet.get("topic") == topic:
                    return packet.get("msg", None)
        except asyncio.TimeoutError:
            return None

    async def set_mode(self, mode: str) -> None:
        await self.publish("/set_mode_cmd", {"data": mode})

    async def get_current_mode(self) -> str:
        msg = await self.subscribe_once("/current_mode", "std_msgs/msg/String", timeout_s=2.0)
        if msg is None:
            return "unknown"
        return str(msg.get("data", "unknown"))

    async def send_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        await self.publish(
            "/cmd_vel",
            {
                "linear": {"x": float(linear_x), "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": float(angular_z)},
            },
        )

    async def stop(self) -> None:
        await self.send_cmd_vel(0.0, 0.0)
