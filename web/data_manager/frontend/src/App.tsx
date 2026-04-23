import { useEffect, useRef, useState } from "react";
import { api, getOperator, getRole, setOperator, setRole } from "./api/client";
import { ArmsPanel } from "./components/ArmsPanel";
import { CameraGrid } from "./components/CameraGrid";
import { Controls } from "./components/Controls";
import { EpisodeList } from "./components/EpisodeList";
import { ReplayPanel } from "./components/ReplayPanel";
import { StatsCard } from "./components/StatsCard";
import { FloatingHealth, StatusBar } from "./components/StatusBar";
import { TemplateManager } from "./components/TemplateManager";
import { TopBar } from "./components/TopBar";
import type { EpisodeMeta, Role, StatusPayload, Template } from "./types";

export function App() {
  const [role, setRoleState] = useState<Role>(getRole());
  const [operator, setOperatorState] = useState(getOperator());
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [connected, setConnected] = useState(false);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [taskKey, setTaskKey] = useState("");
  const [tplId, setTplId] = useState("");
  const [selectedEp, setSelectedEp] = useState<EpisodeMeta | null>(null);
  const [showTplMgr, setShowTplMgr] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);

  // role / operator persistence
  const onRole = (r: Role) => { setRole(r); setRoleState(r); };
  const onOp = (o: string) => { setOperator(o); setOperatorState(o); };

  // load templates
  const reloadTpls = async () => {
    try { setTemplates(await api.templates()); } catch {}
  };
  useEffect(() => { reloadTpls(); }, []);

  // status WS
  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/status`);
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = ev => { try { setStatus(JSON.parse(ev.data)); } catch {} };
    return () => ws.close();
  }, []);

  // when recorder transitions to IDLE after save, bump refresh so list/stats refresh
  const lastState = useRef<string>("IDLE");
  useEffect(() => {
    const cur = status?.recorder.state || "IDLE";
    if (lastState.current !== "IDLE" && cur === "IDLE") {
      setRefreshKey(k => k + 1);
    }
    lastState.current = cur;
  }, [status?.recorder.state]);

  // 把 tplId / operator 实时同步给后端 /api/session, 这样踏板等外设第一次就能
  // 触发启动, 不用先点一次"开始". 300ms 去抖是为了敲名字时不逐键 PUT.
  useEffect(() => {
    if (!tplId && !operator) return;  // 都空: 没什么可同步
    const h = window.setTimeout(() => {
      api.setSession(tplId || null, operator || null).catch(() => { /* backend 暂时挂了, 下次再试 */ });
    }, 300);
    return () => window.clearTimeout(h);
  }, [tplId, operator]);

  return (
    <div className="app">
      <StatusBar status={status} role={role} operator={operator} connected={connected} />
      <TopBar
        role={role} setRole={onRole}
        operator={operator} setOperator={onOp}
        templates={templates}
        selectedTaskKey={taskKey} setSelectedTaskKey={setTaskKey}
        selectedTemplateId={tplId} setSelectedTemplateId={setTplId}
        onOpenTemplates={() => setShowTplMgr(true)}
        disabled={status?.recorder.state === "RECORDING"}
      />
      <div className="main">
        <EpisodeList
          selected={selectedEp}
          onSelect={setSelectedEp}
          refreshKey={refreshKey}
          role={role}
          onDeleted={(e) => {
            if (selectedEp
                && selectedEp.task_id === e.task_id
                && selectedEp.subset === e.subset
                && selectedEp.episode_id === e.episode_id) {
              setSelectedEp(null);
            }
            setRefreshKey(k => k + 1);
          }}
        />
        <CameraGrid cameras={status?.cameras || {}} />
        <ArmsPanel />
        <Controls
          rec={status?.recorder ?? null}
          status={status}
          connected={connected}
          templateId={tplId}
          operator={operator}
          onChanged={() => setRefreshKey(k => k + 1)}
        />
        <ReplayPanel
          ep={selectedEp}
          role={role}
          onCloned={(k) => { setTaskKey(k); setSelectedEp(null); }}
          onDeleted={() => { setSelectedEp(null); setRefreshKey(k => k + 1); }}
        />
        <StatsCard role={role} refreshKey={refreshKey} />
      </div>
      {showTplMgr && (
        <TemplateManager onClose={() => setShowTplMgr(false)} onChanged={reloadTpls} />
      )}
      <FloatingHealth status={status} connected={connected} />
    </div>
  );
}
