"""WebSocket connection hub / event bus.

Prototype scope: an in-process broadcast list. The seam is deliberate — swap
`EventBus` for a Redis Pub/Sub or Kafka-backed implementation (same public
interface: `subscribe`, `unsubscribe`, `publish`) to scale beyond a single
backend process, per docs/ARCHITECTURE.md roadmap phase 3.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket


class EventBus:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def unsubscribe(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def publish(self, message: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        async with self._lock:
            connections = list(self._connections)
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)


event_bus = EventBus()
