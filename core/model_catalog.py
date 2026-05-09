from __future__ import annotations

from pathlib import Path


DETECTION_MODEL_EXCLUDE_SUFFIXES = ("-seg", "-pose", "-obb", "-cls")


def available_detection_models(project_root: Path) -> list[tuple[str, str]]:
    models_dir = project_root / "models"
    items: list[tuple[str, str]] = []
    if models_dir.exists():
        for model_path in sorted(models_dir.glob("*.pt")):
            if not is_detection_model_file(model_path):
                continue
            items.append((prettify_model_label(model_path.stem), str(model_path.relative_to(project_root))))
    if not items:
        items.append(("No local models found", "models/yolo26m.pt"))
    return items


def is_detection_model_file(model_path: Path) -> bool:
    stem = model_path.stem.lower()
    return not any(stem.endswith(suffix) for suffix in DETECTION_MODEL_EXCLUDE_SUFFIXES)


def prettify_model_label(stem: str) -> str:
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
