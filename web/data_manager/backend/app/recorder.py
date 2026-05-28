"""录制状态机：开始 / 保存 / 丢弃 三按钮。

后台 30Hz 采集线程从 ros_bridge 拉帧 + 关节快照，PyAV(libsvtav1) 编码为
480x640 AV1 mp4，pyarrow 写 LeRobot v2.1 parquet。discard 时删除半成品文件。
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from .config import DATA_ROOT
from .dataset_writer import (
    CAMERAS,
    DEPTH_CAMERAS,
    EpisodeWriter as _EpisodeWriter,
    FPS,
    HEIGHT,
    WIDTH,
    features_block,
    pick_codec as _pick_codec,
    task_subset_root as _task_subset_root,
    update_info_json,
    write_episode_meta,
)
from .layout import (
    compound_to_subset_root,
    new_task_subset_root,
    today_compound,
)
from .models import RecState, SaveRecordingReq, StartRecordingReq
from .ros_bridge import bridge
from .stats_service import service as stats
from .sync import sync_episode_files
from .templates import store as templates


log = logging.getLogger(__name__)


def dated_task_name(task: str) -> str:
    """Compatibility shim: `'Task_A'` → `'Task_A_2026-04-16'` (compound task_id).
    保留此名字以兼容外部脚本; 新代码请直接用 layout.today_compound."""
    return today_compound(task)



class Recorder:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.state: RecState = "IDLE"
        self.task_id: Optional[str] = None
        self.subset: Optional[str] = None
        self.prompt: Optional[str] = None
        self.operator: Optional[str] = None
        self.template_id: Optional[str] = None
        self.started_at: Optional[float] = None
        self.episode_id: Optional[int] = None
        self.error: Optional[str] = None

        # 踏板 toggle 用: 记住上一次通过 /api/recorder/start 传入的 template_id +
        # operator, 这样 IDLE 下只有外设触发也能启动 (首次仍需 UI 点一次"开始",
        # 或在 IDLE 下用 UI 只选择/提交而已). 不写磁盘, 进程重启后清空.
        self.last_template_id: Optional[str] = os.environ.get("KAI0_DEFAULT_TEMPLATE") or None
        self.last_operator: Optional[str] = os.environ.get("KAI0_DEFAULT_OPERATOR") or None

        self._writer: Optional[_EpisodeWriter] = None
        self._worker: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = (time.time() - self.started_at) if (self.started_at and self.state == "RECORDING") else 0.0
            return {
                "state": self.state,
                "task_id": self.task_id,
                "subset": self.subset,
                "prompt": self.prompt,
                "operator": self.operator,
                "template_id": self.template_id,
                "episode_id": self.episode_id,
                "elapsed_s": round(elapsed, 2),
                "error": self.error,
            }

    # ---------- 公共 API ----------
    def start(self, req: StartRecordingReq) -> dict:
        with self._lock:
            if self.state != "IDLE":
                raise RuntimeError(f"recorder not idle (state={self.state})")
            tpl = templates.get(req.template_id)
            if tpl is None or not tpl.enabled:
                raise ValueError("invalid or disabled template_id")
            self.task_id = tpl.task_id
            self.subset = tpl.subset
            self.prompt = tpl.prompt
            self.operator = req.operator
            self.template_id = tpl.id
            # toggle 用的粘性记忆: 保存到任何 start 成功就会更新一次
            self.last_template_id = tpl.id
            self.last_operator = req.operator
            # 必须传 *已带日期* 的目录名 (与磁盘上 _task_subset_root 一致),
            # 否则 next_episode_id 在 DB / 目录里都查不到匹配, 永远返回 0,
            # 每次保存都用 episode_000000.* 覆盖前一条 (这就是历史 bug 的来源).
            self.episode_id = stats.next_episode_id(dated_task_name(tpl.task_id), tpl.subset)
            self.started_at = time.time()
            self.error = None
            try:
                self._writer = _EpisodeWriter(
                    task=tpl.task_id, subset=tpl.subset, ep=self.episode_id,
                    prompt=tpl.prompt, template_id=tpl.id, operator=req.operator,
                )
            except Exception as e:
                self._reset()
                raise RuntimeError(f"writer init failed: {e}") from e
            self._stop_evt.clear()
            self._worker = threading.Thread(
                target=self._capture_loop, daemon=True, name="rec-capture"
            )
            self._worker.start()
            self.state = "RECORDING"
            return self.snapshot()

    def discard(self) -> dict:
        with self._lock:
            self._stop_worker()
            if self._writer is not None:
                try:
                    self._writer.abort()
                except Exception as e:  # noqa: BLE001
                    log.warning("discard abort failed: %s", e)
                self._writer = None
            self._reset()
            return self.snapshot()

    def save(self, req: SaveRecordingReq) -> dict:
        with self._lock:
            if self.state != "RECORDING":
                raise RuntimeError(f"cannot save in state={self.state}")
            self.state = "SAVING"
            duration = time.time() - (self.started_at or time.time())
            self._stop_worker()
            writer = self._writer
            self._writer = None
            ep_id, task, subset = self.episode_id, self.task_id, self.subset
            try:
                assert writer is not None
                writer.finalize()
                self._write_meta(writer, duration, req)
                self._update_info_json(task, subset)
            except Exception as e:
                self.state = "ERROR"
                self.error = str(e)
                try:
                    if writer is not None:
                        writer.abort()
                except Exception:  # noqa: BLE001
                    pass
                raise
            self._reset()
        if task and subset and ep_id is not None:
            # task 只是裸名 ('Task_A'); 新 episode 刚写到 new_task_subset_root 下面,
            # upsert_one 需要真实 parquet 路径。
            pq_path = new_task_subset_root(task, subset) / "data" / "chunk-000" / f"episode_{ep_id:06d}.parquet"
            stats.upsert_one(pq_path)
            # post-save 异步单 episode 推送: 只 rsync 该 ep 的 parquet/mp4/zarr + meta,
            # 用 --files-from 跳过全树 stat, 即使 subset 已有 10k+ 文件也几秒完成.
            today = datetime.date.today().strftime("%Y-%m-%d")
            sync_episode_files(task, today, subset, ep_id)
        return {"saved_episode_id": ep_id, "task_id": task, "subset": subset}

    def toggle(self, snapshot_fn) -> dict:
        """鼠标按钮之外的第二路启停入口 (踏板用).

        snapshot_fn: 延迟调用 status_hub.snapshot, 避免 recorder → status_hub 的反向
                     依赖 (status_hub 已经 import recorder).

        返回 {"action": "started"|"saved"|"rejected", "reason": ..., ...}
        状态码层面由 main.py 翻译: rejected → 409, 其他 → 200.
        """
        from .preflight import collect_failures

        with self._lock:
            st = self.state
            if st == "SAVING":
                return {"action": "rejected", "reason": "saving in progress",
                        "state": st}
            if st == "ERROR":
                return {"action": "rejected", "reason": f"recorder error: {self.error}",
                        "state": st}

            if st == "RECORDING":
                # 不做 preflight — 正在录的中途硬件掉线了也要能停; 不然数据丢了
                res = self.save(SaveRecordingReq(success=True, note="pedal", scene_tags=[]))
                return {"action": "saved", **res}

            # IDLE: 需要 (1) 已有 last_template/operator, (2) preflight 通过
            tpl_id = self.last_template_id
            op = self.last_operator
            if not tpl_id or not op:
                return {"action": "rejected",
                        "reason": "no template/operator remembered; click 开始 once from UI or set KAI0_DEFAULT_TEMPLATE/OPERATOR",
                        "state": st}

        # 把 snapshot+preflight 放到锁外做, 避免长时间持锁阻塞 capture_loop
        try:
            snap = snapshot_fn()
        except Exception as e:  # noqa: BLE001
            return {"action": "rejected", "reason": f"status snapshot failed: {e}"}
        fails = collect_failures(snap)
        if fails:
            return {"action": "rejected", "reason": "preflight failed",
                    "failures": fails, "state": "IDLE"}

        # 再拿锁启动 (期间状态若被人抢先变了也安全, start() 自己会再校验 IDLE)
        try:
            res = self.start(StartRecordingReq(template_id=tpl_id, operator=op))
        except (RuntimeError, ValueError) as e:
            return {"action": "rejected", "reason": str(e)}
        return {"action": "started", **res}

    # ---------- 内部 ----------
    def _capture_loop(self) -> None:
        period = 1.0 / FPS
        next_tick = time.time()
        writer = self._writer
        if writer is None:
            return
        while not self._stop_evt.is_set():
            now = time.time()
            if now < next_tick:
                time.sleep(min(period, next_tick - now))
                continue
            next_tick += period
            # 若已掉后多帧则直接追到当前时间，避免越追越远
            if now - next_tick > 5 * period:
                next_tick = now + period

            frames = {cam: bridge.get_frame_rgb(cam) for cam in CAMERAS}
            depth_frames = {
                cam: bridge.get_frame_depth(cam) if hasattr(bridge, "get_frame_depth") else None
                for cam in DEPTH_CAMERAS
            }
            try:
                state, action = bridge.get_state_action()
            except Exception as e:  # noqa: BLE001
                log.warning("get_state_action failed: %s", e)
                state, action = [0.0] * 14, [0.0] * 14
            try:
                writer.write_tick(frames, state, action, now, depth_frames=depth_frames)
            except Exception as e:  # noqa: BLE001
                log.exception("write_tick failed: %s", e)
                self._stop_evt.set()
                return

    def _stop_worker(self) -> None:
        self._stop_evt.set()
        w = self._worker
        if w and w.is_alive():
            w.join(timeout=3.0)
        self._worker = None

    def _reset(self) -> None:
        self.state = "IDLE"
        self.task_id = None
        self.subset = None
        self.prompt = None
        self.operator = None
        self.template_id = None
        self.started_at = None
        self.episode_id = None
        self.error = None

    # ---------- meta I/O (thin shims → dataset_writer) ----------
    def _write_meta(self, writer: _EpisodeWriter, duration: float, req: SaveRecordingReq) -> None:
        write_episode_meta(writer, duration,
                           success=req.success, note=req.note,
                           scene_tags=req.scene_tags)

    def _update_info_json(self, task: Optional[str], subset: Optional[str]) -> None:
        update_info_json(task, subset)

    @staticmethod
    def _features_block() -> dict:
        return features_block()


recorder = Recorder()
