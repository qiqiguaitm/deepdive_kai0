"""StatusBar 的真值聚合 + WS 广播。
独立模块，单向汇集来自 ros_bridge / recorder / stats / disk 的最新状态。
"""
from __future__ import annotations

import asyncio
import shutil
import time
from typing import Set

from fastapi import WebSocket

from .config import DATA_ROOT, STATUS_BROADCAST_HZ
from .recorder import dated_task_name, recorder
from .ros_bridge import bridge
from .stats_service import service as stats


class StatusHub:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._last_size_bytes = 0
        self._last_sample_at = time.time()

    def snapshot(self) -> dict:
        try:
            du = shutil.disk_usage(DATA_ROOT if DATA_ROOT.exists() else DATA_ROOT.parent)
            free_gb = round(du.free / 1e9, 1)
        except OSError:
            free_gb = -1
        st = stats.stats()
        # 写入速率：以总 size 增量除以采样间隔
        now = time.time()
        delta = max(0, st.total_size_bytes - self._last_size_bytes)
        dt = max(1e-3, now - self._last_sample_at)
        write_mbs = round(delta / 1e6 / dt, 2)
        self._last_size_bytes = st.total_size_bytes
        self._last_sample_at = now

        rec = recorder.snapshot()
        # 同样要传带日期的目录名, 否则 next_ep 永远是 0 (历史 bug, 见
        # stats_service.next_episode_id 注释)
        next_ep = (
            stats.next_episode_id(dated_task_name(rec["task_id"]), rec["subset"])
            if rec["task_id"] and rec["subset"]
            else None
        )

        warnings: list[str] = []
        if free_gb >= 0 and free_gb < 10:
            warnings.append(f"low_disk:{free_gb}GB")
        cam_health = bridge.get_camera_health()

        return {
            "ts": now,
            "health": bridge.get_health(),
            "cameras": cam_health,
            "recorder": rec,
            "next_episode_id": next_ep,
            "stats_total": st.total,
            "stats_today": st.today,
            "disk_free_gb": free_gb,
            "write_mbps": write_mbs,
            "warnings": warnings,
        }

    async def register(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def unregister(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast_loop(self) -> None:
        period = 1.0 / STATUS_BROADCAST_HZ
        while True:
            try:
                payload = self.snapshot()
            except Exception as e:
                payload = {"error": str(e)}
            dead = []
            async with self._lock:
                clients = list(self._clients)
            for ws in clients:
                try:
                    await ws.send_json(payload)
                except Exception:
                    dead.append(ws)
            if dead:
                async with self._lock:
                    for ws in dead:
                        self._clients.discard(ws)
            await asyncio.sleep(period)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.broadcast_loop())


hub = StatusHub()
