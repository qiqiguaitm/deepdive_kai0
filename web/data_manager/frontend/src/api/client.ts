import type {
  EpisodeMeta, JointState, RecorderSnap, Role, StatsResponse, Template,
} from "../types";

export function getRole(): Role {
  return (localStorage.getItem("role") as Role) || "collector";
}
export function setRole(r: Role) { localStorage.setItem("role", r); }
export function getOperator(): string { return localStorage.getItem("operator") || ""; }
export function setOperator(o: string) { localStorage.setItem("operator", o); }

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("X-Role", getRole());
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const r = await fetch(path, { ...init, headers });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const api = {
  templates: (only_enabled = false) =>
    req<Template[]>(`/api/templates?only_enabled=${only_enabled}`),
  upsertTemplate: (t: Template) =>
    req<Template>(`/api/templates/${t.id}`, { method: "PUT", body: JSON.stringify(t) }),
  delTemplate: (id: string) =>
    req<{ deleted: boolean }>(`/api/templates/${id}`, { method: "DELETE" }),

  recorder: () => req<RecorderSnap>(`/api/recorder`),
  startRec: (template_id: string, operator: string) =>
    req<RecorderSnap>(`/api/recorder/start`, { method: "POST", body: JSON.stringify({ template_id, operator }) }),
  saveRec: (success: boolean, note: string, scene_tags: string[]) =>
    req<{ saved_episode_id: number }>(`/api/recorder/save`, { method: "POST", body: JSON.stringify({ success, note, scene_tags }) }),
  discardRec: () => req<RecorderSnap>(`/api/recorder/discard`, { method: "POST" }),
  estop: () => req<{ ok: boolean }>(`/api/recorder/estop`, { method: "POST" }),

  stats: () => req<StatsResponse>(`/api/stats`),
  rescan: () => req<{ rescanned: number }>(`/api/stats/rescan`, { method: "POST" }),

  episodes: (q: Record<string, string | undefined>) => {
    const usp = new URLSearchParams();
    Object.entries(q).forEach(([k, v]) => v != null && v !== "" && usp.set(k, v));
    return req<EpisodeMeta[]>(`/api/episodes?${usp}`);
  },
  delEpisode: (task: string, subset: string, ep: number) =>
    req<{ deleted: boolean }>(`/api/episodes/${task}/${subset}/${ep}`, { method: "DELETE" }),
  videoUrl: (task: string, subset: string, ep: number, cam: string) =>
    `/api/episodes/${task}/${subset}/${ep}/video/${cam}`,
  // 深度: 一帧一张 PNG (后端 JET 上色), 前端按视频时间戳 → frame_index 拉
  depthFrameUrl: (task: string, subset: string, ep: number, cam: string,
                  frame: number, minMm = 200, maxMm = 2000) =>
    `/api/episodes/${task}/${subset}/${ep}/depth/${cam}/frame/${frame}?min_mm=${minMm}&max_mm=${maxMm}`,
  depthInfo: (task: string, subset: string, ep: number, cam: string) =>
    req<{ frames: number; height: number; width: number }>(
      `/api/episodes/${task}/${subset}/${ep}/depth/${cam}/info`),

  joints: () => req<JointState>(`/api/joints`),
};
