from __future__ import annotations

from pathlib import Path


def resolve_detector_model_spec(project_root: Path, detector_model_path: str) -> str:
    candidate = project_root / detector_model_path
    if candidate.exists():
        return str(candidate)
    return detector_model_path
