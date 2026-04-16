import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Role, StatsResponse } from "../types";

export function StatsCard({ role, refreshKey }: { role: Role; refreshKey: number }) {
  const [s, setS] = useState<StatsResponse | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = async () => {
    try { setS(await api.stats()); } catch {}
  };
  useEffect(() => { reload(); const id = setInterval(reload, 5000); return () => clearInterval(id); }, [refreshKey]);

  const rescan = async () => {
    setBusy(true);
    try { await api.rescan(); await reload(); } catch (e: any) { alert(e.message); }
    finally { setBusy(false); }
  };
  if (!s) return <div className="panel area-stats stats-card"><h3>统计</h3>加载中…</div>;

  return (
    <div className="panel area-stats stats-card">
      <h3>已录数据统计 (磁盘真实计数)</h3>
      <div className="num">{s.total.toLocaleString()}</div>
      <div className="grid">
        <div>今日: <b>{s.today}</b></div>
        <div>本周: <b>{s.this_week}</b></div>
        <div>残缺: <b style={{ color: s.incomplete ? "var(--bad)" : undefined }}>{s.incomplete}</b></div>
        <div>总时长: <b>{(s.total_duration_s / 3600).toFixed(2)} h</b></div>
        <div style={{ gridColumn: "span 2" }}>总大小: <b>{(s.total_size_bytes / 1e9).toFixed(2)} GB</b></div>
      </div>

      <h4>按 Task / Subset</h4>
      {s.by_task_subset.map(b => <div key={b.key} className="bucket"><span>{b.key || "(未知)"}</span><b>{b.count}</b></div>)}

      <h4>按操作员</h4>
      {s.by_operator.map(b => <div key={b.key} className="bucket"><span>{b.key || "(未知)"}</span><b>{b.count}</b></div>)}

      <h4>按 Prompt</h4>
      {s.by_prompt.slice(0, 6).map(b => <div key={b.key} className="bucket"><span title={b.key} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 200 }}>{b.key || "(未知)"}</span><b>{b.count}</b></div>)}

      <h4>成功 / 失败</h4>
      {s.by_success.map(b => <div key={b.key} className="bucket"><span>{b.key}</span><b>{b.count}</b></div>)}

      <div style={{ marginTop: 8, color: "var(--muted)", fontSize: 11 }}>
        最后扫描: {new Date(s.last_scan_at * 1000).toLocaleTimeString()}
      </div>
      {role === "admin" && (
        <button style={{ marginTop: 8 }} disabled={busy} onClick={rescan}>{busy ? "扫描中…" : "强制重扫"}</button>
      )}
    </div>
  );
}
