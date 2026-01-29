# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 22:46:19 2026

@author: Elian PC
"""

from __future__ import annotations
from pathlib import Path
import json
import time
import yaml

def make_run_dir(base: str = "results") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

def write_yaml(path: Path, obj) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")

