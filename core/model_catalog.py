from __future__ import annotations

from pathlib import Path

from core.config_models import normalize_config_path


DETECTION_MODEL_EXCLUDE_SUFFIXES = ("-seg", "-pose", "-obb", "-cls")
REMOTE_DETECTION_PRESETS: list[tuple[str, str]] = [
    ("YOLO26 balanced", "models/yolo26m.pt"),
    ("YOLO26 fast", "models/yolo26s.pt"),
]


def available_detection_models(project_root: Path) -> list[tuple[str, str]]:
    models_dir = project_root / "models"
    items: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    if models_dir.exists():
        for model_path in sorted(models_dir.glob("*.pt")):
            if not is_detection_model_file(model_path):
                continue
            relative = str(model_path.relative_to(project_root))
            items.append((prettify_model_label(model_path.stem), relative))
            seen_paths.add(relative)
    for label, spec in REMOTE_DETECTION_PRESETS:
        if spec not in seen_paths:
            items.append((label, spec))
    if not items:
        items.append(("YOLO26 balanced", "models/yolo26m.pt"))
    return items


def is_detection_model_file(model_path: Path) -> bool:
    stem = model_path.stem.lower()
    return not any(stem.endswith(suffix) for suffix in DETECTION_MODEL_EXCLUDE_SUFFIXES)


def prettify_model_label(stem: str) -> str:
    lower = stem.lower()
    if lower.startswith("rtdetr"):
        normalized = stem.replace("rtdetr", "RT-DETR")
        return normalized
    label = stem.replace("yolo", "YOLO")
    quality_map = {
        "n": "fast",
        "s": "balanced",
        "m": "high accuracy",
        "l": "very high accuracy",
        "x": "max accuracy",
    }
    descriptor = quality_map.get(stem[-1].lower()) if stem else None
    if descriptor:
        return f"{label} {descriptor}"
    return label


def resolve_detector_model_spec(project_root: Path, detector_model_path: str) -> str:
    normalized_model_path = normalize_config_path(detector_model_path)
    candidate = project_root / normalized_model_path
    if candidate.exists():
        return str(candidate)
    return normalized_model_path
