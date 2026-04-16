from typing import Literal, Optional
from pydantic import BaseModel, Field

Role = Literal["collector", "admin"]
RecState = Literal["IDLE", "RECORDING", "SAVING", "ERROR"]


class Template(BaseModel):
    id: str
    task_id: str
    subset: Literal["base", "dagger"]
    prompt: str
    enabled: bool = True
    note: str = ""


class StartRecordingReq(BaseModel):
    template_id: str
    operator: str


class SaveRecordingReq(BaseModel):
    success: bool = True
    note: str = ""
    scene_tags: list[str] = Field(default_factory=list)


class EpisodeMeta(BaseModel):
    episode_id: int
    task_id: str
    subset: str
    prompt: str
    operator: str
    success: bool
    note: str
    duration_s: float
    size_bytes: int
    created_at: float
    parquet_path: str
    video_paths: dict[str, str]
    incomplete: bool = False
    incomplete_reason: Optional[str] = None


class StatsBucket(BaseModel):
    key: str
    count: int


class StatsResponse(BaseModel):
    total: int
    today: int
    this_week: int
    incomplete: int
    total_duration_s: float
    total_size_bytes: int
    by_task_subset: list[StatsBucket]
    by_operator: list[StatsBucket]
    by_prompt: list[StatsBucket]
    by_success: list[StatsBucket]
    last_scan_at: float
