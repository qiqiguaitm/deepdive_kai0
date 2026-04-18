import { useState } from "react";
import { api } from "../api/client";
import type { RecorderSnap, StatusPayload } from "../types";
import { collectFailures } from "./StatusBar";

interface Props {
  rec: RecorderSnap | null;
  status: StatusPayload | null;
  connected: boolean;
  templateId: string;
  operator: string;
  onChanged: () => void;
}

export function Controls({ rec, status, connected, templateId, operator, onChanged }: Props) {
  const [success, setSuccess] = useState(true);
  const [note, setNote] = useState("");
  const [tags, setTags] = useState("");
  const [busy, setBusy] = useState(false);

  const state = rec?.state || "IDLE";
  const hasMeta = !!templateId && !!operator;
  const canStart = state === "IDLE" && hasMeta;
  const canEnd = state === "RECORDING";
  // 开始按钮保持可点，即便 state===ERROR / 缺 meta / 系统红灯，
  // 这样 onStart 里的弹窗校验才能触发（否则 disabled 按钮吃掉 click，看不到弹窗）。
  const startDisabled = state === "RECORDING" || state === "SAVING" || busy;

  const wrap = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try { await fn(); onChanged(); }
    catch (e: any) { alert(e.message || String(e)); }
    finally { setBusy(false); }
  };

  const onStart = () => {
    if (!hasMeta) {
      alert("请先选择任务 + Prompt 并填写操作员姓名。");
      return;
    }
    if (!connected || !status) {
      alert("系统异常：未连接到后端状态流，无法继续进行，请修复后再采集。");
      return;
    }
    const failures = collectFailures(status);
    if (failures.length > 0) {
      alert(`系统异常，无法继续进行，请修复后再采集：\n- ${failures.join("\n- ")}`);
      return;
    }
    if (state !== "IDLE") {
      alert(`当前录制状态为 ${state}，无法开始新的录制。请先丢弃当前会话。`);
      return;
    }
    wrap(() => api.startRec(templateId, operator));
  };

  return (
    <div className="panel area-ctrl">
      <h3>录制控制</h3>
      <div className="meta-form">
        <span>结果</span>
        <select value={success ? "ok" : "fail"} onChange={e => setSuccess(e.target.value === "ok")}>
          <option value="ok">成功</option>
          <option value="fail">失败</option>
        </select>
        <span>场景标签</span>
        <input value={tags} onChange={e => setTags(e.target.value)} placeholder="逗号分隔，如 light_dim,desk_a" />
        <span>备注</span>
        <textarea value={note} onChange={e => setNote(e.target.value)} rows={2} style={{ gridColumn: "span 3" }} />
      </div>
      <div className="controls">
        <button className="btn-start" disabled={startDisabled}
          onClick={onStart}>● 开始</button>
        <button className="btn-save" disabled={!canEnd || busy}
          onClick={() => wrap(() => api.saveRec(success, note, tags.split(",").map(s => s.trim()).filter(Boolean)))}>■ 保存</button>
        <button className="btn-discard" disabled={state === "IDLE" || busy}
          onClick={() => wrap(() => api.discardRec())}>✕ 丢弃</button>
      </div>
      {!canStart && state === "IDLE" && (
        <p style={{ color: "var(--muted)", marginTop: 6 }}>请先选择任务 + Prompt 并填写操作员姓名。</p>
      )}
    </div>
  );
}
