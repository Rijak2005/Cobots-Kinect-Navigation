from __future__ import annotations

import socket
import threading
import time
import queue
import os
from dataclasses import dataclass
from typing import Optional, Tuple


def _enc_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return len(b).to_bytes(2, "big") + b


def _enc_remaining_length(n: int) -> bytes:
    # MQTT variable length encoding
    out = bytearray()
    while True:
        d = n % 128
        n //= 128
        if n > 0:
            d |= 0x80
        out.append(d)
        if n == 0:
            break
    return bytes(out)


def _build_connect(client_id: str, keepalive_s: int = 30) -> bytes:
    proto_name = _enc_str("MQTT")
    proto_level = b"\x04"  # MQTT 3.1.1
    connect_flags = b"\x02"  # clean session
    keepalive = int(keepalive_s).to_bytes(2, "big")
    vh = proto_name + proto_level + connect_flags + keepalive
    payload = _enc_str(client_id)
    rl = _enc_remaining_length(len(vh) + len(payload))
    return b"\x10" + rl + vh + payload


def _build_publish(topic: str, payload: str) -> bytes:
    # QoS 0, retain 0
    t = _enc_str(topic)
    p = payload.encode("utf-8")
    rl = _enc_remaining_length(len(t) + len(p))
    return b"\x30" + rl + t + p


@dataclass(frozen=True)
class MqttConfig:
    host: str = "broker.hivemq.com"
    port: int = 1883
    topic_prefix: str = "rijakisthebest/cobots"  # your prefix
    connect_timeout_s: float = 2.0
    keepalive_s: int = 30


class GripperMqttPublisher:
    """
    Minimal MQTT publisher (QoS0) using raw TCP sockets (no dependencies).
    Runs a background worker thread so movement control never blocks.
    """

    def __init__(self, cfg: MqttConfig, verbose: bool = True) -> None:
        self.cfg = cfg
        self.verbose = verbose
        self._q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

        # stable unique-ish id
        pid = os.getpid()
        self._client_id = f"pc-gripper-{pid}-{int(time.time())}"

    def start(self) -> None:
        if self._th is not None:
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        if self.verbose:
            print(f"[mqtt] publisher thread started -> {self.cfg.host}:{self.cfg.port}")

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.5)
        self._th = None

    def _topic(self, leaf: str) -> str:
        p = self.cfg.topic_prefix.strip()
        if not p:
            return leaf
        if p.endswith("/"):
            return p + leaf
        return p + "/" + leaf

    def publish_turn(self, payload: str) -> None:
        self._q.put((self._topic("turn"), str(payload)))

    def publish_grip(self, payload: str) -> None:
        self._q.put((self._topic("grip"), str(payload)))

    def _send_once(self, topic: str, payload: str) -> bool:
        s: Optional[socket.socket] = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(float(self.cfg.connect_timeout_s))
            s.connect((self.cfg.host, int(self.cfg.port)))

            # CONNECT
            s.sendall(_build_connect(self._client_id, keepalive_s=int(self.cfg.keepalive_s)))

            # Read CONNACK (expected 4 bytes: 0x20 0x02 0x00 0x00)
            connack = s.recv(4)
            if len(connack) < 4 or connack[0] != 0x20 or connack[1] != 0x02 or connack[3] != 0x00:
                if self.verbose:
                    print(f"[mqtt] bad CONNACK: {connack!r}")
                return False

            # PUBLISH
            s.sendall(_build_publish(topic, payload))
            return True

        except Exception as e:
            if self.verbose:
                print(f"[mqtt] send failed topic={topic} payload={payload} err={e}")
            return False
        finally:
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                topic, payload = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            ok = self._send_once(topic, payload)
            if self.verbose:
                print(f"[mqtt] PUB {'OK' if ok else 'FAIL'}  {topic} -> {payload}")