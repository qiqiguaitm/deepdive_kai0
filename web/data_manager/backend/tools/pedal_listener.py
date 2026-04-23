#!/usr/bin/env python3
"""USB foot-pedal → /api/recorder/toggle bridge.

设计目标:
  * 通用性: 按 VID:PID 定位设备 (不绑 /dev/input/eventN, 换 USB 口不影响);
    PEDAL_VID/PEDAL_PID/PEDAL_KEY 可覆盖以适配不同型号踏板
  * 可靠性: 启动找不到设备 → 重试; 运行中设备断开 → 回到查找循环; 独占设备
    防止 F3 (或配置的 key) 漏到其他窗口
  * 零冲突: 只调 backend 的 /api/recorder/toggle, 真正的启停互斥由后端
    Recorder._lock 保证, 鼠标与踏板同一条代码路径 (内部状态机)

默认踏板识别到的是 HID 0483:5750 (STM32, 映射 F3). 不同型号只需改环境变量.

用法:
  python pedal_listener.py
环境变量:
  PEDAL_VID=0483         USB VID (hex, 默认 0483)
  PEDAL_PID=5750         USB PID (hex, 默认 5750)
  PEDAL_KEY=KEY_F3       evdev key name, 默认 KEY_F3
  PEDAL_EDGE=release     release(松开触发, 默认) | press(踩下触发)
  PEDAL_DEBOUNCE_MS=500  同方向连触最小间隔 ms, 默认 500
  BACKEND_URL=http://127.0.0.1:8787
  PEDAL_LOG_LEVEL=INFO
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import evdev

# ------------------------------- config -----------------------------------
VID = int(os.environ.get("PEDAL_VID", "0483"), 16)
PID = int(os.environ.get("PEDAL_PID", "5750"), 16)
KEY_NAME = os.environ.get("PEDAL_KEY", "KEY_F3")
EDGE = os.environ.get("PEDAL_EDGE", "release").lower()  # "release" | "press"
DEBOUNCE_S = float(os.environ.get("PEDAL_DEBOUNCE_MS", "500")) / 1000.0
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8787").rstrip("/")
LOG_LEVEL = os.environ.get("PEDAL_LOG_LEVEL", "INFO").upper()
RETRY_FIND_S = 2.0

TRIGGER_VALUE = 0 if EDGE == "release" else 1  # evdev: 1=down, 0=up

# key name → code
try:
    TRIGGER_KEY = evdev.ecodes.ecodes[KEY_NAME]
except KeyError:
    print(f"[pedal] unknown key name {KEY_NAME!r}; see evdev.ecodes", file=sys.stderr)
    sys.exit(2)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [pedal] %(levelname)s %(message)s",
)
log = logging.getLogger("pedal")

_running = True


def _handle_signal(signum, _frame):
    global _running
    log.info("received signal %d, shutting down", signum)
    _running = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------- device lookup -------------------------------
_warned_perm_paths: set[str] = set()


def find_pedal() -> Optional[evdev.InputDevice]:
    """Locate the pedal by VID:PID. Prefer the interface that reports the
    configured trigger key (some pedals expose both kbd + mouse HID interfaces)."""
    candidates: list[evdev.InputDevice] = []
    perm_denied: list[str] = []
    for path in sorted(Path("/dev/input").glob("event*")):
        try:
            d = evdev.InputDevice(str(path))
        except PermissionError:
            perm_denied.append(str(path))
            continue
        except OSError as e:
            log.debug("skip %s: %s", path, e)
            continue
        if d.info.vendor == VID and d.info.product == PID:
            candidates.append(d)
        else:
            d.close()

    # 关键 UX: 如果整轮扫描都没打开任何 event 节点 (全是 PermissionError), 这是一个
    # 强信号说明当前进程不在 'input' 组. 在 INFO 级别显式提示一次, 不要悄悄 retry.
    if not candidates and perm_denied:
        new_paths = [p for p in perm_denied if p not in _warned_perm_paths]
        if new_paths:
            log.warning(
                "no /dev/input/event* readable by this process (PermissionError on %d paths). "
                "Current process needs 'input' group membership. Fix: "
                "`sudo gpasswd -a %s input` then **log out and back in** "
                "(or install web/data_manager/config/99-kai0-pedal.rules + udevadm reload). "
                "Process groups: `cat /proc/self/status | grep Groups`",
                len(perm_denied), os.environ.get("USER", "$USER"),
            )
            _warned_perm_paths.update(perm_denied)

    if not candidates:
        return None

    # Prefer an interface that actually carries TRIGGER_KEY
    with_key: list[evdev.InputDevice] = []
    for d in candidates:
        caps = d.capabilities().get(evdev.ecodes.EV_KEY, [])
        if TRIGGER_KEY in caps:
            with_key.append(d)
        else:
            d.close()

    if not with_key:
        # 没有带 trigger key 的接口: 回退用第一个 candidate (可能是 generic-hid),
        # 这样至少日志里能看到事件, 方便诊断
        log.warning("VID:PID %04x:%04x found but no interface reports %s; "
                    "using first candidate (debug only)", VID, PID, KEY_NAME)
        first = candidates[0]
        for d in candidates[1:]:
            d.close()
        return first

    # 多个接口都带 trigger key — 不正常, 但兜底用第一个
    if len(with_key) > 1:
        log.warning("multiple VID:PID %04x:%04x interfaces carry %s, picking %s",
                    VID, PID, KEY_NAME, with_key[0].path)
    first = with_key[0]
    for d in with_key[1:]:
        d.close()
    return first


# ---------------------------- backend call --------------------------------
def call_toggle() -> None:
    url = f"{BACKEND_URL}/api/recorder/toggle"
    req = urllib.request.Request(url, data=b"", method="POST",
                                  headers={"Content-Type": "application/json",
                                           "X-Role": "collector"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            body = resp.read().decode("utf-8", "replace")
            log.info("toggle %d in %.0f ms: %s", resp.status, (time.time() - t0) * 1000, body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
        # 409 = 被后端拒绝 (preflight 失败 / SAVING / ERROR), 正常业务反馈
        level = logging.WARNING if e.code == 409 else logging.ERROR
        log.log(level, "toggle HTTP %d: %s", e.code, body)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.error("toggle connect failed: %s", e)


# ---------------------------- main loop -----------------------------------
def run_session(dev: evdev.InputDevice) -> None:
    """One device session: grab, read events, trigger on edge. Returns on
    device error or shutdown signal."""
    try:
        dev.grab()
        grabbed = True
        log.info("grabbed %s (%s)", dev.path, dev.name)
    except OSError as e:
        grabbed = False
        log.warning("grab failed on %s: %s — continuing without exclusive access "
                    "(key events will leak to focused window)", dev.path, e)

    last_fire = 0.0
    try:
        for ev in dev.read_loop():
            if not _running:
                break
            if ev.type != evdev.ecodes.EV_KEY or ev.code != TRIGGER_KEY:
                continue
            if ev.value != TRIGGER_VALUE:
                continue
            now = time.monotonic()
            if now - last_fire < DEBOUNCE_S:
                log.debug("debounced (%.0f ms < %.0f ms)",
                          (now - last_fire) * 1000, DEBOUNCE_S * 1000)
                continue
            last_fire = now
            log.info("pedal fired (code=%d edge=%s)", ev.code, EDGE)
            call_toggle()
    except OSError as e:
        # 常见: 设备拔掉 (ENODEV); Errno 19 — 回到外层循环重新查找
        log.warning("device read error: %s — reconnecting", e)
    finally:
        if grabbed:
            try:
                dev.ungrab()
            except OSError:
                pass
        try:
            dev.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    log.info("pedal listener starting — VID:PID=%04x:%04x key=%s edge=%s debounce=%.0fms backend=%s",
             VID, PID, KEY_NAME, EDGE, DEBOUNCE_S * 1000, BACKEND_URL)
    waiting_logged = False
    while _running:
        dev = find_pedal()
        if dev is None:
            if not waiting_logged:
                log.info("waiting for pedal VID:PID=%04x:%04x ...", VID, PID)
                waiting_logged = True
            time.sleep(RETRY_FIND_S)
            continue
        waiting_logged = False
        run_session(dev)
    log.info("pedal listener exited cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
