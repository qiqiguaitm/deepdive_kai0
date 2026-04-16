import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { EpisodeMeta, Role } from "../types";

interface Props {
  ep: EpisodeMeta | null;
  role: Role;
  onCloned: (taskKey: string) => void;
  onDeleted: () => void;
}

const CAMS = ["hand_left", "top_head", "hand_right"] as const;
type Cam = (typeof CAMS)[number];

export function ReplayPanel({ ep, role, onCloned, onDeleted }: Props) {
  const videoRefs = useRef<Record<Cam, HTMLVideoElement | null>>({} as any);
  const depthRefs = useRef<Record<Cam, HTMLImageElement | null>>({} as any);

  // 每相机的 depth zarr 元数据 (帧数, 用于把 video.currentTime 映射成 frame_index)
  const [depthInfo, setDepthInfo] = useState<Record<Cam, { frames: number } | null>>(
    { hand_left: null, top_head: null, hand_right: null });
  // 用户可调的 depth 窗位 (mm). 0.2-2m 适合桌面操作.
  const [minMm, setMinMm] = useState(200);
  const [maxMm, setMaxMm] = useState(2000);
  // 上一次拉过的 frame, 用于节流 (浏览器可能 timeupdate 60Hz, 服务端 PNG 5ms+ 也吃不消)
  const lastFrameRef = useRef<Record<Cam, number>>({ hand_left: -1, top_head: -1, hand_right: -1 });

  // episode 切换时, 重新拉每个相机的 depth info; 没有 depth 数据的 episode 自动隐藏深度面板
  useEffect(() => {
    if (!ep) return;
    let cancelled = false;
    setDepthInfo({ hand_left: null, top_head: null, hand_right: null });
    lastFrameRef.current = { hand_left: -1, top_head: -1, hand_right: -1 };
    Promise.all(CAMS.map(async cam => {
      try {
        const info = await api.depthInfo(ep.task_id, ep.subset, ep.episode_id, cam);
        return [cam, info] as const;
      } catch {
        return [cam, null] as const;
      }
    })).then(results => {
      if (cancelled) return;
      const next: any = {};
      for (const [cam, info] of results) next[cam] = info;
      setDepthInfo(next);
    });
    return () => { cancelled = true; };
  }, [ep?.task_id, ep?.subset, ep?.episode_id]);

  const hasAnyDepth = Object.values(depthInfo).some(d => d && d.frames > 0);

  // 同步 depth 帧到当前 video 时间。一个 timeupdate handler 同时更新所有相机.
  // 速率限制: 同一相机相邻 frame_index 相同就跳过 (避免重复 GET).
  const syncDepthFrame = (cam: Cam) => {
    const v = videoRefs.current[cam];
    const img = depthRefs.current[cam];
    const info = depthInfo[cam];
    if (!v || !img || !info || !ep) return;
    // duration_s 可能不准; 优先用 video.duration, 实在没有再回落
    const total = isFinite(v.duration) && v.duration > 0 ? v.duration : (ep.duration_s || 1);
    const idx = Math.min(info.frames - 1, Math.max(0, Math.floor((v.currentTime / total) * info.frames)));
    if (idx === lastFrameRef.current[cam]) return;
    lastFrameRef.current[cam] = idx;
    img.src = api.depthFrameUrl(ep.task_id, ep.subset, ep.episode_id, cam, idx, minMm, maxMm);
  };

  // 每次 minMm/maxMm 变化都强制重渲染当前帧
  useEffect(() => {
    for (const cam of CAMS) {
      lastFrameRef.current[cam] = -1;
      syncDepthFrame(cam);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [minMm, maxMm, depthInfo]);

  const playAll = () => Object.values(videoRefs.current).forEach(r => r && r.play());
  const pauseAll = () => Object.values(videoRefs.current).forEach(r => r && r.pause());

  if (!ep) return <div className="panel area-replay"><h3>回放</h3>从左侧选择一条 episode。</div>;

  return (
    <div className="panel area-replay">
      <h3>回放: {ep.task_id}/{ep.subset} #{ep.episode_id} · {ep.duration_s.toFixed(1)}s · {ep.success ? "成功" : "失败"}</h3>
      <div style={{ marginBottom: 6, color: "var(--muted)" }}>prompt: <b>{ep.prompt || "—"}</b></div>

      <div className="replay-vids">
        {CAMS.map(cam => (
          <video key={cam} ref={el => (videoRefs.current[cam] = el)}
                 src={api.videoUrl(ep.task_id, ep.subset, ep.episode_id, cam)}
                 onTimeUpdate={() => syncDepthFrame(cam)}
                 onSeeked={() => syncDepthFrame(cam)}
                 onLoadedMetadata={() => syncDepthFrame(cam)}
                 controls={false} muted preload="metadata" />
        ))}
      </div>

      {hasAnyDepth && (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "6px 0", color: "var(--muted)" }}>
            <span>深度 (JET)</span>
            <label>min mm <input type="number" value={minMm} step={50} min={0}
                                 onChange={e => setMinMm(parseInt(e.target.value || "0"))}
                                 style={{ width: 70 }} /></label>
            <label>max mm <input type="number" value={maxMm} step={100} min={1}
                                 onChange={e => setMaxMm(parseInt(e.target.value || "1"))}
                                 style={{ width: 80 }} /></label>
            <span style={{ fontSize: 11 }}>(0.2m – 2m 桌面默认)</span>
          </div>
          <div className="replay-vids">
            {CAMS.map(cam => (
              depthInfo[cam] && depthInfo[cam]!.frames > 0
                ? <img key={cam} ref={el => (depthRefs.current[cam] = el)}
                       alt={`${cam} depth`}
                       style={{ background: "#000", width: "100%", aspectRatio: "640/480" }} />
                : <div key={cam} style={{ background: "#222", color: "#888",
                       display: "flex", alignItems: "center", justifyContent: "center",
                       aspectRatio: "640/480" }}>无深度数据</div>
            ))}
          </div>
        </>
      )}

      <div className="controls" style={{ flexWrap: "wrap" }}>
        <button onClick={playAll}>▶ 播放</button>
        <button onClick={pauseAll}>⏸ 暂停</button>
        <button onClick={() => onCloned(`${ep.task_id}/${ep.subset}`)}>以此配置新建采集</button>
        {role === "admin" && (
          <button className="btn-discard"
            onClick={async () => {
              if (!confirm(`删除 ${ep.task_id}/${ep.subset} #${ep.episode_id}? 不可恢复。`)) return;
              await api.delEpisode(ep.task_id, ep.subset, ep.episode_id);
              onDeleted();
            }}>删除</button>
        )}
      </div>
      {ep.incomplete && <p style={{ color: "var(--bad)" }}>⚠ {ep.incomplete_reason}</p>}
    </div>
  );
}
