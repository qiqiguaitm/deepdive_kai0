import { useMemo } from "react";
import type { Role, Template } from "../types";
import { api } from "../api/client";

interface Props {
  role: Role;
  setRole: (r: Role) => void;
  operator: string;
  setOperator: (o: string) => void;
  templates: Template[];
  selectedTaskKey: string;            // "Task_A/base"
  setSelectedTaskKey: (k: string) => void;
  selectedTemplateId: string;
  setSelectedTemplateId: (id: string) => void;
  onOpenTemplates: () => void;
  disabled: boolean;
}

export function TopBar(p: Props) {
  const enabled = useMemo(() => p.templates.filter(t => t.enabled), [p.templates]);
  const tasks = useMemo(() => {
    const set = new Set<string>();
    enabled.forEach(t => set.add(`${t.task_id}/${t.subset}`));
    return Array.from(set).sort();
  }, [enabled]);
  const prompts = useMemo(
    () => enabled.filter(t => `${t.task_id}/${t.subset}` === p.selectedTaskKey),
    [enabled, p.selectedTaskKey],
  );

  return (
    <div className="topbar">
      <span>任务</span>
      <select value={p.selectedTaskKey} disabled={p.disabled}
        onChange={e => { p.setSelectedTaskKey(e.target.value); p.setSelectedTemplateId(""); }}>
        <option value="">选择任务…</option>
        {tasks.map(k => <option key={k} value={k}>{k}</option>)}
      </select>

      <span>Prompt</span>
      <select value={p.selectedTemplateId} disabled={p.disabled || !p.selectedTaskKey}
        onChange={e => p.setSelectedTemplateId(e.target.value)}>
        <option value="">选择 Prompt…</option>
        {prompts.map(t => <option key={t.id} value={t.id}>{t.prompt}</option>)}
      </select>

      <span>操作员</span>
      <input value={p.operator} onChange={e => p.setOperator(e.target.value)} placeholder="姓名" style={{ width: 110 }} />

      <div className="spacer"></div>

      {p.role === "admin" && (
        <button onClick={p.onOpenTemplates}>模板管理</button>
      )}
      <button onClick={() => p.setRole(p.role === "admin" ? "collector" : "admin")}>
        切换为{p.role === "admin" ? "采集员" : "管理员"}
      </button>
      <button className="estop" onClick={() => api.estop()}>🛑 急停</button>
    </div>
  );
}
