from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class Config:
    raw: dict

    @staticmethod
    def load(path: str | Path) -> "Config":
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return Config(raw=data)

    def get(self, *keys, default=None):
        cur = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur
