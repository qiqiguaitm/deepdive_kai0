import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { JointState } from "../types";

function JointBar({ name, val, max = 1.5 }: { name: string; val: number; max?: number }) {
  const pct = Math.min(100, (Math.abs(val) / max) * 100);
  return (
    <div className="joint-row">
      <span className="name">{name}</span>
      <div className="bar"><div style={{ width: `${pct}%`, left: val < 0 ? `${50 - pct / 2}%` : "50%" }} /></div>
      <span className="val">{val.toFixed(3)}</span>
    </div>
  );
}

export function ArmsPanel() {
  const [j, setJ] = useState<JointState | null>(null);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try { const v = await api.joints(); if (alive) setJ(v); } catch {}
    };
    tick();
    const id = setInterval(tick, 200);
    return () => { alive = false; clearInterval(id); };
  }, []);
  if (!j) return <div className="panel area-arms"><h3>双臂状态</h3>—</div>;
  return (
    <div className="panel area-arms">
      <h3>双臂状态 (14 维 obs)</h3>
      <div className="arms">
        <div>
          <b>左臂</b>
          {j.left_joints.map((v, i) => <JointBar key={i} name={`J${i + 1}`} val={v} />)}
          <JointBar name="夹爪" val={j.left_gripper} max={1} />
        </div>
        <div>
          <b>右臂</b>
          {j.right_joints.map((v, i) => <JointBar key={i} name={`J${i + 1}`} val={v} />)}
          <JointBar name="夹爪" val={j.right_gripper} max={1} />
        </div>
      </div>
    </div>
  );
}
