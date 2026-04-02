#!/usr/bin/env python3
"""
三相机同时 30fps 最大分辨率 RGB+Depth 压测脚本

测试内容:
  1. 三相机同时打开 RGB + Depth @ 30fps 最大分辨率
  2. 持续采集 10 秒，统计实际 FPS、丢帧、延迟
  3. 检查 USB 带宽是否足够

硬件:
  - D435 (SN: 254622070889): RGB 1920x1080 + Depth 1280x720 @ 30fps
  - D405 (SN: 409122273074): RGB 1280x720  + Depth 1280x720 @ 30fps
  - D405 (SN: 409122271568): RGB 1280x720  + Depth 1280x720 @ 30fps

USB 拓扑:
  - Bus 002 (10Gbps xHCI): D435 (Port1, 5Gbps) + D405-A (Port2, 5Gbps)
  - Bus 004 (10Gbps xHCI): D405-B (Port2, 5Gbps)
  → 两条独立 10Gbps 总线，带宽充足

用法:
  python3 scripts/test_realsense_3cam.py [--duration 10] [--save-sample]
"""

import argparse
import time
import threading
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not found. Install via: pip install pyrealsense2")
    exit(1)

# ── 相机配置 ──────────────────────────────────────────────────────────────────
CAMERAS = [
    {
        "name": "D435 (top)",
        "serial": "254622070889",
        "rgb_w": 1920, "rgb_h": 1080,
        "depth_w": 1280, "depth_h": 720,
        "fps": 30,
    },
    {
        "name": "D405-A (wrist)",
        "serial": "409122273074",
        "rgb_w": 1280, "rgb_h": 720,
        "depth_w": 1280, "depth_h": 720,
        "fps": 30,
    },
    {
        "name": "D405-B (wrist)",
        "serial": "409122271568",
        "rgb_w": 1280, "rgb_h": 720,
        "depth_w": 1280, "depth_h": 720,
        "fps": 30,
    },
]


class CameraThread:
    """单相机采集线程，记录帧统计"""

    def __init__(self, cam_cfg):
        self.cfg = cam_cfg
        self.name = cam_cfg["name"]
        self.serial = cam_cfg["serial"]

        # 统计
        self.frame_count = 0
        self.drop_count = 0
        self.latencies = []       # per-frame 采集耗时 (ms)
        self.rgb_shape = None
        self.depth_shape = None
        self.error = None
        self.actual_fps = 0.0

        self._pipeline = None
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass

    def _run(self):
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial)
            config.enable_stream(
                rs.stream.color,
                self.cfg["rgb_w"], self.cfg["rgb_h"],
                rs.format.bgr8, self.cfg["fps"],
            )
            config.enable_stream(
                rs.stream.depth,
                self.cfg["depth_w"], self.cfg["depth_h"],
                rs.format.z16, self.cfg["fps"],
            )

            profile = pipeline.start(config)
            self._pipeline = pipeline

            # 获取实际设备信息
            device = profile.get_device()
            usb_type = device.get_info(rs.camera_info.usb_type_descriptor)

            # 丢弃前几帧 (自动曝光稳定)
            for _ in range(15):
                pipeline.wait_for_frames(timeout_ms=3000)

            prev_frame_num = -1
            t_start = time.monotonic()

            while not self._stop.is_set():
                t0 = time.monotonic()
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                t1 = time.monotonic()

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # 记录帧形状 (只记一次)
                if self.rgb_shape is None:
                    color_img = np.asanyarray(color_frame.get_data())
                    depth_img = np.asanyarray(depth_frame.get_data())
                    self.rgb_shape = color_img.shape
                    self.depth_shape = depth_img.shape

                # 丢帧检测 (通过 frame number 间隔)
                fn = color_frame.get_frame_number()
                if prev_frame_num >= 0 and fn > prev_frame_num + 1:
                    self.drop_count += fn - prev_frame_num - 1
                prev_frame_num = fn

                self.frame_count += 1
                self.latencies.append((t1 - t0) * 1000)  # ms

            elapsed = time.monotonic() - t_start
            self.actual_fps = self.frame_count / elapsed if elapsed > 0 else 0

        except Exception as e:
            self.error = str(e)


def main():
    parser = argparse.ArgumentParser(description="三相机 30fps 最大分辨率压测")
    parser.add_argument("--duration", type=int, default=10, help="测试持续秒数 (default: 10)")
    parser.add_argument("--save-sample", action="store_true", help="保存一帧样图到 /tmp/")
    args = parser.parse_args()

    # ── 阶段 1: 枚举设备 ──────────────────────────────────────────────────
    print("=" * 70)
    print("三相机 RGB+Depth 30fps 最大分辨率压测")
    print("=" * 70)

    ctx = rs.context()
    connected = {
        dev.get_info(rs.camera_info.serial_number): dev.get_info(rs.camera_info.name)
        for dev in ctx.query_devices()
    }
    print(f"\n检测到 {len(connected)} 个 RealSense 设备:")
    for sn, name in connected.items():
        print(f"  {name} (SN: {sn})")

    missing = [c for c in CAMERAS if c["serial"] not in connected]
    if missing:
        print(f"\nERROR: 以下相机未连接:")
        for c in missing:
            print(f"  {c['name']} (SN: {c['serial']})")
        return 1

    # ── 阶段 2: 同时启动三相机 ─────────────────────────────────────────────
    print(f"\n配置:")
    for c in CAMERAS:
        print(f"  {c['name']}: RGB {c['rgb_w']}x{c['rgb_h']} + Depth {c['depth_w']}x{c['depth_h']} @ {c['fps']}fps")

    print(f"\n启动三相机...")
    threads = [CameraThread(c) for c in CAMERAS]

    # 按顺序启动，间隔 0.5s 避免 USB 冲突
    for t in threads:
        t.start()
        time.sleep(0.5)

    print(f"采集中 ({args.duration}s)...")
    time.sleep(args.duration)

    print("停止采集...")
    for t in threads:
        t.stop()

    # ── 阶段 3: 保存样图 (可选) ───────────────────────────────────────────
    if args.save_sample:
        print("\n保存样图...")
        for c in CAMERAS:
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(c["serial"])
                config.enable_stream(rs.stream.color, c["rgb_w"], c["rgb_h"], rs.format.bgr8, c["fps"])
                config.enable_stream(rs.stream.depth, c["depth_w"], c["depth_h"], rs.format.z16, c["fps"])
                pipeline.start(config)
                for _ in range(15):
                    pipeline.wait_for_frames(timeout_ms=3000)
                frames = pipeline.wait_for_frames(timeout_ms=3000)
                import cv2
                color_img = np.asanyarray(frames.get_color_frame().get_data())
                depth_img = np.asanyarray(frames.get_depth_frame().get_data())
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
                )
                tag = c["serial"]
                cv2.imwrite(f"/tmp/cam_{tag}_rgb.png", color_img)
                cv2.imwrite(f"/tmp/cam_{tag}_depth.png", depth_vis)
                print(f"  {c['name']}: /tmp/cam_{tag}_rgb.png, /tmp/cam_{tag}_depth.png")
                pipeline.stop()
            except Exception as e:
                print(f"  {c['name']}: save failed - {e}")

    # ── 阶段 4: 输出报告 ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试报告")
    print("=" * 70)

    all_ok = True
    for t in threads:
        print(f"\n  {t.name} (SN: {t.serial})")
        if t.error:
            print(f"    ERROR: {t.error}")
            all_ok = False
            continue

        lat = np.array(t.latencies) if t.latencies else np.array([0])
        print(f"    RGB:     {t.rgb_shape}")
        print(f"    Depth:   {t.depth_shape}")
        print(f"    帧数:    {t.frame_count}")
        print(f"    丢帧:    {t.drop_count}")
        print(f"    实际FPS: {t.actual_fps:.1f}")
        print(f"    延迟:    avg={lat.mean():.1f}ms  p50={np.median(lat):.1f}ms  p99={np.percentile(lat,99):.1f}ms  max={lat.max():.1f}ms")

        # 判定
        if t.actual_fps < 25:
            print(f"    [WARN] FPS 低于 25，可能存在 USB 带宽不足")
            all_ok = False
        if t.drop_count > t.frame_count * 0.05:
            print(f"    [WARN] 丢帧率 > 5%")
            all_ok = False

    print("\n" + "-" * 70)
    if all_ok:
        print("结论: PASS - 三相机同时 30fps 最大分辨率 RGB+Depth 正常运行")
    else:
        print("结论: WARN - 存在问题，见上方详情")
    print("-" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
