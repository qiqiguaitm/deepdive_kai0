import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Template } from "../types";

interface Props { onClose: () => void; onChanged: () => void; }

const empty: Template = { id: "", task_id: "Task_A", subset: "base", prompt: "", enabled: true, note: "" };

export function TemplateManager({ onClose, onChanged }: Props) {
  const [items, setItems] = useState<Template[]>([]);
  const [draft, setDraft] = useState<Template>(empty);

  const reload = async () => setItems(await api.templates());
  useEffect(() => { reload(); }, []);

  const save = async (t: Template) => {
    if (!t.id || !t.prompt) { alert("id / prompt 必填"); return; }
    await api.upsertTemplate(t);
    await reload(); onChanged();
  };
  const del = async (id: string) => {
    if (!confirm(`删除模板 ${id}?`)) return;
    await api.delTemplate(id); await reload(); onChanged();
  };

  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <h2>模板管理（管理员）</h2>
        <table className="tpl-table">
          <thead><tr><th>id</th><th>task</th><th>subset</th><th>prompt</th><th>启用</th><th>备注</th><th></th></tr></thead>
          <tbody>
            {items.map(t => (
              <Row key={t.id} t={t} onSave={save} onDel={() => del(t.id)} />
            ))}
            <tr><td colSpan={7} style={{ paddingTop: 14 }}><b>新增</b></td></tr>
            <Row t={draft} onSave={async tt => { await save(tt); setDraft(empty); }} onDel={() => setDraft(empty)} isNew />
          </tbody>
        </table>
        <div style={{ marginTop: 12, textAlign: "right" }}>
          <button onClick={onClose}>关闭</button>
        </div>
      </div>
    </div>
  );
}

function Row({ t, onSave, onDel, isNew }: { t: Template; onSave: (t: Template) => void; onDel: () => void; isNew?: boolean }) {
  const [d, setD] = useState<Template>(t);
  useEffect(() => setD(t), [t]);
  return (
    <tr>
      <td><input value={d.id} disabled={!isNew} onChange={e => setD({ ...d, id: e.target.value })} /></td>
      <td><input value={d.task_id} onChange={e => setD({ ...d, task_id: e.target.value })} /></td>
      <td>
        <select value={d.subset} onChange={e => setD({ ...d, subset: e.target.value as any })}>
          <option value="base">base</option><option value="dagger">dagger</option>
        </select>
      </td>
      <td><input value={d.prompt} onChange={e => setD({ ...d, prompt: e.target.value })} /></td>
      <td style={{ textAlign: "center" }}>
        <input type="checkbox" checked={d.enabled} onChange={e => setD({ ...d, enabled: e.target.checked })} />
      </td>
      <td><input value={d.note} onChange={e => setD({ ...d, note: e.target.value })} /></td>
      <td>
        <button onClick={() => onSave(d)}>{isNew ? "新增" : "保存"}</button>
        {!isNew && <button onClick={onDel} style={{ color: "var(--bad)" }}>删</button>}
      </td>
    </tr>
  );
}
