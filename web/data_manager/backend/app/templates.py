import threading
from pathlib import Path
import yaml
from .models import Template
from .config import TEMPLATES_PATH


class TemplateStore:
    def __init__(self, path: Path = TEMPLATES_PATH) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._items: dict[str, Template] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("templates: []\n", encoding="utf-8")
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        items = [Template(**t) for t in (raw.get("templates") or [])]
        self._items = {t.id: t for t in items}

    def _flush(self) -> None:
        data = {"templates": [t.model_dump() for t in self._items.values()]}
        tmp = self.path.with_suffix(".yml.tmp")
        tmp.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        tmp.replace(self.path)

    def list(self, only_enabled: bool = False) -> list[Template]:
        with self._lock:
            items = list(self._items.values())
        return [t for t in items if t.enabled] if only_enabled else items

    def get(self, tid: str) -> Template | None:
        with self._lock:
            return self._items.get(tid)

    def upsert(self, t: Template) -> Template:
        with self._lock:
            self._items[t.id] = t
            self._flush()
        return t

    def delete(self, tid: str) -> bool:
        with self._lock:
            if tid in self._items:
                del self._items[tid]
                self._flush()
                return True
        return False


store = TemplateStore()
