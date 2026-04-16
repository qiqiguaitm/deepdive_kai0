export type Role = "collector" | "admin";
export type RecState = "IDLE" | "RECORDING" | "SAVING" | "ERROR";

export interface Template {
  id: string;
  task_id: string;
  subset: "base" | "dagger";
  prompt: string;
  enabled: boolean;
  note: string;
}

export interface RecorderSnap {
  state: RecState;
  task_id: string | null;
  subset: string | null;
  prompt: string | null;
  operator: string | null;
  template_id: string | null;
  episode_id: number | null;
  elapsed_s: number;
  error: string | null;
}

export interface CameraHealth {
  fps: number;
  target_fps?: number;
  dropped: number;
  latency_ms: number;
}

export interface StatusPayload {
  ts: number;
  health: { ros2: boolean; can_left: boolean; can_right: boolean; teleop: boolean };
  cameras: Record<string, CameraHealth>;
  recorder: RecorderSnap;
  next_episode_id: number | null;
  stats_total: number;
  stats_today: number;
  disk_free_gb: number;
  write_mbps: number;
  warnings: string[];
}

export interface JointState {
  left_joints: number[];
  right_joints: number[];
  left_gripper: number;
  right_gripper: number;
  left_temp: number[];
  right_temp: number[];
  left_torque: number[];
  right_torque: number[];
}

export interface StatsBucket { key: string; count: number; }

export interface StatsResponse {
  total: number;
  today: number;
  this_week: number;
  incomplete: number;
  total_duration_s: number;
  total_size_bytes: number;
  by_task_subset: StatsBucket[];
  by_operator: StatsBucket[];
  by_prompt: StatsBucket[];
  by_success: StatsBucket[];
  last_scan_at: number;
}

export interface EpisodeMeta {
  episode_id: number;
  task_id: string;
  subset: string;
  prompt: string;
  operator: string;
  success: boolean;
  note: string;
  duration_s: number;
  size_bytes: number;
  created_at: number;
  parquet_path: string;
  video_paths: Record<string, string>;
  incomplete: boolean;
  incomplete_reason: string | null;
}
