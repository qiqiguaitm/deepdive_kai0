"""录制状态机：开始 / 保存 / 丢弃 三按钮。

后台 30Hz 采集线程从 ros_bridge 拉帧 + 关节快照，PyAV(libsvtav1) 编码为
480x640 AV1 mp4，pyarrow 写 LeRobot v2.1 parquet。discard 时删除半成品文件。
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import zarr  # depth 写入; 没装就关掉 depth 录制
    _HAS_ZARR = True
except ImportError:
    _HAS_ZARR = False

from .config import DATA_ROOT
from .models import RecState, SaveRecordingReq, StartRecordingReq
from .ros_bridge import bridge
from .stats_service import service as stats
from .templates import store as templates


CAMERAS = ("top_head", "hand_left", "hand_right")
FPS = 30
WIDTH = 640
HEIGHT = 480

log = logging.getLogger(__name__)


def dated_task_name(task: str) -> str:
    """裸 task 名 → 带日期的 on-disk 目录名: 'Task_A' → 'Task_A_2026-04-16'.
    所有需要"目录名而不是模板 task 名"的地方都该过这个函数, 避免 DB / 路径不匹配."""
    return f"{task}_{datetime.date.today().strftime('%Y-%m-%d')}"


def _task_subset_root(task: str, subset: str) -> Path:
    """新布局: <DATA_ROOT>/<task>_<YYYY-MM-DD>/<subset>/...

    日期取自 'now'. 同一天再录同一 (task,subset) 时落到同一目录, episode_id 自增;
    跨日则落到新目录, episode_id 从 0 重新开始 (因为 stats_service 按目录扫描)."""
    return DATA_ROOT / dated_task_name(task) / subset


def _pick_codec() -> tuple[str, str, dict]:
    """返回 (spec_name, codec_name, options)。

    KAI0_VIDEO_CODEC: h264(默认) | av1
      - h264: libx264，所有播放器/浏览器原生支持，双击即播
      - av1:  libsvtav1 > libaom-av1，匹配 LeRobot 原始规格但许多播放器无法解码
    """
    choice = os.environ.get("KAI0_VIDEO_CODEC", "h264").lower()
    avail = set(av.codecs_available)
    if choice == "av1":
        if "libsvtav1" in avail:
            return "av1", "libsvtav1", {"preset": "8", "crf": "32"}
        if "libaom-av1" in avail:
            return "av1", "libaom-av1", {"cpu-used": "8", "crf": "32", "b:v": "0"}
        log.warning("AV1 encoder not found, falling back to libx264")
    return "h264", "libx264", {"preset": "veryfast", "crf": "23"}


class _EpisodeWriter:
    """单条 episode 的磁盘写入封装：3 个 mp4 容器 + 1 个 parquet buffer。"""

    def __init__(self, task: str, subset: str, ep: int, prompt: str,
                 template_id: str, operator: str) -> None:
        self.task = task
        self.subset = subset
        self.ep = ep
        self.prompt = prompt
        self.template_id = template_id
        self.operator = operator

        self.root = _task_subset_root(task, subset)
        self.pq_path = self.root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        self.video_paths = {
            cam: self.root / "videos" / "chunk-000" / cam / f"episode_{ep:06d}.mp4"
            for cam in CAMERAS
        }
        # Depth 走 zarr DirectoryStore, 一个目录 = 一个 episode 的一个相机
        self.depth_paths = {
            cam: self.root / "videos" / "chunk-000" / f"{cam}_depth" / f"episode_{ep:06d}.zarr"
            for cam in CAMERAS
        }
        for p in [self.pq_path.parent, *(v.parent for v in self.video_paths.values()),
                  *(d.parent for d in self.depth_paths.values())]:
            p.mkdir(parents=True, exist_ok=True)

        spec_name, codec_name, codec_opts = _pick_codec()
        self._spec_name = spec_name
        self._codec_name = codec_name
        self._containers: dict[str, av.container.OutputContainer] = {}
        self._streams: dict[str, av.video.stream.VideoStream] = {}
        for cam, path in self.video_paths.items():
            container = av.open(str(path), mode="w")
            stream = container.add_stream(codec_name, rate=FPS)
            stream.width = WIDTH
            stream.height = HEIGHT
            stream.pix_fmt = "yuv420p"
            stream.options = dict(codec_opts)
            self._containers[cam] = container
            self._streams[cam] = stream

        # Depth zarr 数组: 形状 (T, H, W) uint16, chunk = (1, H, W) 方便按帧 mmap.
        # 用 blosc/zstd 压缩, 实测 RealSense 深度 ~5-10x 压缩率.
        self._depth_arrays: dict[str, object] = {}
        if _HAS_ZARR:
            try:
                compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
            except Exception:
                compressor = None
            for cam, path in self.depth_paths.items():
                # 先清掉残留 (上次同名 episode 中途崩了)
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
                z = zarr.open(
                    str(path), mode="w",
                    shape=(0, HEIGHT, WIDTH),
                    chunks=(1, HEIGHT, WIDTH),
                    dtype="uint16",
                    compressor=compressor,
                )
                self._depth_arrays[cam] = z
        else:
            log.warning("zarr not installed, depth recording disabled")

        # parquet 行缓冲（python 原生，保存时转 arrow）
        self._rows_state: list[list[float]] = []
        self._rows_action: list[list[float]] = []
        self._rows_ts: list[float] = []
        self._frame_idx = 0
        self._t0 = time.time()

    def write_tick(self, frames: dict[str, np.ndarray],
                   state: list[float], action: list[float], ts: float,
                   depth_frames: dict[str, np.ndarray] | None = None) -> None:
        for cam in CAMERAS:
            arr = frames.get(cam)
            if arr is None:
                # 用黑帧占位避免帧数不一致
                arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            else:
                arr = self._ensure_size(arr)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame.pts = self._frame_idx
            for packet in self._streams[cam].encode(frame):
                self._containers[cam].mux(packet)

        # 深度: 每相机 append 一帧到 zarr; 缺帧用 0 占位保持帧数一致
        if self._depth_arrays:
            depth_frames = depth_frames or {}
            for cam in CAMERAS:
                z = self._depth_arrays.get(cam)
                if z is None:
                    continue
                d = depth_frames.get(cam)
                if d is None or d.shape != (HEIGHT, WIDTH):
                    d = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
                else:
                    d = np.ascontiguousarray(d.astype(np.uint16, copy=False))
                z.append(d[None, :, :])  # 在 axis=0 (时间轴) 上 append 一帧

        # pad/trim 到 14 维 float32
        s = list(state)[:14] + [0.0] * max(0, 14 - len(state))
        a = list(action)[:14] + [0.0] * max(0, 14 - len(action))
        self._rows_state.append([float(x) for x in s])
        self._rows_action.append([float(x) for x in a])
        self._rows_ts.append(float(ts - self._t0))
        self._frame_idx += 1

    @staticmethod
    def _ensure_size(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == HEIGHT and arr.shape[1] == WIDTH and arr.shape[2] == 3:
            return np.ascontiguousarray(arr)
        # RealSense 配置为 640x480，正常不走这里；走到则简单中心 crop/resize
        from PIL import Image
        img = Image.fromarray(arr).resize((WIDTH, HEIGHT))
        return np.asarray(img, dtype=np.uint8)

    def finalize(self) -> None:
        for cam, stream in self._streams.items():
            for packet in stream.encode():
                self._containers[cam].mux(packet)
            self._containers[cam].close()
        self._containers.clear()
        self._streams.clear()
        self._write_parquet()

    def abort(self) -> None:
        """丢弃：关闭容器并删除输出文件。"""
        for container in self._containers.values():
            try:
                container.close()
            except Exception:  # noqa: BLE001
                pass
        self._containers.clear()
        self._streams.clear()
        self._depth_arrays.clear()
        for path in self.video_paths.values():
            path.unlink(missing_ok=True)
        for d in self.depth_paths.values():
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        self.pq_path.unlink(missing_ok=True)

    def _write_parquet(self) -> None:
        n = len(self._rows_state)
        table = pa.table({
            "observation.state": pa.array(self._rows_state, type=pa.list_(pa.float32())),
            "action": pa.array(self._rows_action, type=pa.list_(pa.float32())),
            "timestamp": pa.array(self._rows_ts, type=pa.float32()),
            "frame_index": pa.array(list(range(n)), type=pa.int64()),
            "episode_index": pa.array([self.ep] * n, type=pa.int64()),
            "index": pa.array(list(range(n)), type=pa.int64()),
            "task_index": pa.array([0] * n, type=pa.int64()),
        })
        pq.write_table(table, self.pq_path)

    @property
    def frame_count(self) -> int:
        return self._frame_idx


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
            stats.upsert_one(_task_subset_root(task, subset) / "data" / "chunk-000" / f"episode_{ep_id:06d}.parquet")
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
                for cam in CAMERAS
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

    # ---------- meta I/O ----------
    def _write_meta(self, writer: _EpisodeWriter, duration: float, req: SaveRecordingReq) -> None:
        meta_dir = writer.root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "episode_id": writer.ep,
            "length": writer.frame_count,
            "duration_s": round(duration, 3),
            "operator": writer.operator,
            "prompt": writer.prompt,
            "template_id": writer.template_id,
            "success": req.success,
            "note": req.note,
            "scene_tags": req.scene_tags,
            "created_at": time.time(),
        }
        with (meta_dir / "episodes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tasks_path = meta_dir / "tasks.jsonl"
        tasks_path.touch()
        existing_prompts = set()
        for ln in tasks_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                existing_prompts.add(json.loads(ln).get("task"))
            except json.JSONDecodeError:
                continue
        if writer.prompt not in existing_prompts:
            with tasks_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"task_index": len(existing_prompts), "task": writer.prompt},
                    ensure_ascii=False,
                ) + "\n")

    def _update_info_json(self, task: Optional[str], subset: Optional[str]) -> None:
        if not task or not subset:
            return
        root = _task_subset_root(task, subset)
        info_path = root / "meta" / "info.json"
        ep_log_path = root / "meta" / "episodes.jsonl"
        total_ep = 0
        total_frames = 0
        if ep_log_path.exists():
            for ln in ep_log_path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                total_ep += 1
                total_frames += int(d.get("length", 0))

        info = {
            "codebase_version": "v2.1",
            "robot_type": "agilex",
            "total_episodes": total_ep,
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": total_ep * len(CAMERAS),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": FPS,
            "splits": {"train": f"0:{total_ep}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            # depth_path 不在 LeRobot v2.1 标准里, 我们扩展一个: 用 zarr DirectoryStore,
            # 路径规则与 video 平行, 但相机 key 加 "_depth" 后缀 (top_head_depth/...).
            "depth_path": "videos/chunk-{episode_chunk:03d}/{video_key}_depth/episode_{episode_index:06d}.zarr",
            "features": self._features_block(),
        }
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _features_block() -> dict:
        spec_codec = _pick_codec()[0]
        img_feat = {
            "dtype": "video",
            "shape": [HEIGHT, WIDTH, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": HEIGHT,
                "video.width": WIDTH,
                "video.codec": spec_codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        }
        # 深度特征: zarr 数组, 一帧一行 chunk, blosc-zstd 压缩。
        # dtype "uint16_zarr" 是我们自定义标记 — LeRobot 默认 loader 不认识这个,
        # 但 openpi/自家训练 dataloader 可以据此选择 zarr.open 而不是 mp4 解码。
        depth_feat = {
            "dtype": "uint16_zarr",
            "shape": [HEIGHT, WIDTH],
            "names": ["height", "width"],
            "info": {
                "store": "zarr.DirectoryStore",
                "compressor": "blosc.zstd:level3:bitshuffle",
                "unit": "millimeter",
                "depth.height": HEIGHT,
                "depth.width": WIDTH,
                "depth.fps": FPS,
            },
        }
        return {
            **{f"observation.images.{cam}": img_feat for cam in CAMERAS},
            **{f"observation.depth.{cam}": depth_feat for cam in CAMERAS},
            "observation.state": {"dtype": "float32", "shape": [14], "names": None},
            "action": {"dtype": "float32", "shape": [14], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }


recorder = Recorder()
