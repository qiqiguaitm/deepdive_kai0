import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { EpisodeMeta } from "../types";

interface Props {
  selected: EpisodeMeta | null;
  onSelect: (e: EpisodeMeta) => void;
  refreshKey: number;
}

export function EpisodeList({ selected, onSelect, refreshKey }: Props) {
  const [items, setItems] = useState<EpisodeMeta[]>([]);
  const [task, setTask] = useState("");
  const [subset, setSubset] = useState("");
  const [okFilter, setOkFilter] = useState("");
  const [kw, setKw] = useState("");

  const load = async () => {
    try {
      const r = await api.episodes({
        task_id: task || undefined,
        subset: subset || undefined,
        success: okFilter === "" ? undefined : okFilter,
        prompt_kw: kw || undefined,
      });
      setItems(r);
    } catch {}
  };
  useEffect(() => { load(); }, [task, subset, okFilter, kw, refreshKey]);

  return (
    <div className="panel area-list">
      <h3>历史 Episode</h3>
      <div style={{ display: "grid", gap: 4, marginBottom: 8 }}>
        <input placeholder="task_id (Task_A)" value={task} onChange={e => setTask(e.target.value)} />
        <select value={subset} onChange={e => setSubset(e.target.value)}>
          <option value="">所有 subset</option>
          <option value="base">base</option>
          <option value="dagger">dagger</option>
        </select>
        <select value={okFilter} onChange={e => setOkFilter(e.target.value)}>
          <option value="">所有结果</option>
          <option value="true">成功</option>
          <option value="false">失败</option>
        </select>
        <input placeholder="prompt 关键词" value={kw} onChange={e => setKw(e.target.value)} />
      </div>
      <div className="ep-list">
        {items.length === 0 && <div style={{ color: "var(--muted)" }}>暂无</div>}
        {items.map(e => (
          <div key={`${e.task_id}/${e.subset}/${e.episode_id}`}
               className={`ep-row ${selected?.episode_id === e.episode_id && selected?.task_id === e.task_id ? "active" : ""}`}
               onClick={() => onSelect(e)}>
            <div>
              <b>{e.task_id}/{e.subset}</b> #{e.episode_id.toString().padStart(6, "0")}
              {e.success ? " ✅" : " ❌"} {e.incomplete ? " ⚠残缺" : ""}
            </div>
            <div className="meta">
              {e.duration_s.toFixed(1)}s · {(e.size_bytes / 1024).toFixed(1)} KB · {e.operator || "—"}
            </div>
            <div className="meta" title={e.prompt} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {e.prompt}
            </div>
            <div className="meta">{new Date(e.created_at * 1000).toLocaleString()}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
