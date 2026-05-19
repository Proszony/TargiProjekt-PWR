from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Protocol
import warnings

import cv2
import numpy as np

from core.model_downloads import ensure_model_file
from core.models import ReIDConfig


class ReIDBackend(Protocol):
    def load(self) -> None: ...

    def is_available(self) -> bool: ...

    def embed_person_crop(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> list[float]: ...

    def embed_batch(
        self,
        items: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    ) -> list[list[float]]: ...


@dataclass(slots=True)
class ReIDStatus:
    backend_name: str
    available: bool
    degraded_reason: str | None = None


class DisabledReIDBackend:
    def __init__(self, reason: str = "disabled") -> None:
        self.reason = reason

    def load(self) -> None:
        return

    def is_available(self) -> bool:
        return False

    def embed_person_crop(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> list[float]:
        return _fallback_descriptor(frame_bgr, bbox_xyxy)

    def embed_batch(
        self,
        items: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    ) -> list[list[float]]:
        return [self.embed_person_crop(frame_bgr, bbox) for frame_bgr, bbox in items]


class TorchreidEmbeddingBackend:
    def __init__(self, config: ReIDConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self._extractor = None
        self._load_error: str | None = None

    def load(self) -> None:
        if self._extractor is not None:
            return
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Cython evaluation .* unavailable.*",
                    category=UserWarning,
                )
                from torchreid.reid.utils import FeatureExtractor
                import torch
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._load_error = f"torchreid unavailable: {exc}"
            return

        weights_path = (self.project_root / self.config.weights_path).resolve()
        if not weights_path.exists():
            if self.config.download_if_missing and self.config.weights_url:
                try:
                    ensure_model_file(weights_path, self.config.weights_url)
                except Exception as exc:  # pragma: no cover - network path
                    self._load_error = f"weights download failed: {exc}"
                    return
            else:
                self._load_error = f"weights missing: {weights_path}"
                return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self._extractor = FeatureExtractor(
                    model_name=self.config.model_name,
                    model_path=str(weights_path),
                    image_size=(self.config.input_height, self.config.input_width),
                    device=device,
                    verbose=False,
                )
        except Exception as exc:  # pragma: no cover - backend specific path
            self._load_error = f"extractor init failed: {exc}"
            self._extractor = None

    def is_available(self) -> bool:
        return self._extractor is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def embed_person_crop(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> list[float]:
        results = self.embed_batch([(frame_bgr, bbox_xyxy)])
        return results[0] if results else []

    def embed_batch(
        self,
        items: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    ) -> list[list[float]]:
        if self._extractor is None:
            return [_fallback_descriptor(frame_bgr, bbox) for frame_bgr, bbox in items]
        crops = [_crop_person(frame_bgr, bbox) for frame_bgr, bbox in items]
        valid_crops = [crop for crop in crops if crop is not None]
        if not valid_crops:
            return [[] for _ in items]
        import torch

        features = self._extractor(valid_crops)
        if isinstance(features, torch.Tensor):
            feature_rows = features.detach().cpu().numpy()
        else:
            feature_rows = np.asarray(features)
        normalized: list[list[float]] = []
        feature_iter = iter(feature_rows.tolist())
        for crop in crops:
            if crop is None:
                normalized.append([])
                continue
            vector = next(feature_iter)
            normalized.append(_normalize_vector(vector))
        return normalized


class OnnxEmbeddingBackend:
    def __init__(self, config: ReIDConfig, project_root: Path, backend_name: str) -> None:
        self.config = config
        self.project_root = project_root
        self.backend_name = backend_name
        self._session = None
        self._input_name: str | None = None
        self._load_error: str | None = None

    def load(self) -> None:
        if self._session is not None:
            return
        weights_path = (self.project_root / self.config.weights_path).resolve()
        if not weights_path.exists():
            if self.config.download_if_missing and self.config.weights_url:
                try:
                    ensure_model_file(weights_path, self.config.weights_url)
                except Exception as exc:  # pragma: no cover - network path
                    self._load_error = f"weights download failed: {exc}"
                    return
            else:
                self._load_error = f"weights missing: {weights_path}"
                return
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._load_error = f"onnxruntime unavailable: {exc}"
            return
        providers = ["CPUExecutionProvider"]
        try:
            available = ort.get_available_providers()
        except Exception:  # pragma: no cover - defensive
            available = []
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(str(weights_path), providers=providers)
        except Exception as exc:  # pragma: no cover - backend specific path
            self._load_error = f"onnx session init failed: {exc}"
            return
        inputs = session.get_inputs()
        if not inputs:
            self._load_error = "onnx model has no inputs"
            return
        self._session = session
        self._input_name = inputs[0].name

    def is_available(self) -> bool:
        return self._session is not None and self._input_name is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def embed_person_crop(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> list[float]:
        results = self.embed_batch([(frame_bgr, bbox_xyxy)])
        return results[0] if results else []

    def embed_batch(
        self,
        items: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    ) -> list[list[float]]:
        if self._session is None or self._input_name is None:
            return [_fallback_descriptor(frame_bgr, bbox) for frame_bgr, bbox in items]
        prepared: list[np.ndarray] = []
        valid_mask: list[bool] = []
        for frame_bgr, bbox in items:
            tensor = _prepare_onnx_tensor(frame_bgr, bbox, self.config)
            if tensor is None:
                valid_mask.append(False)
                continue
            valid_mask.append(True)
            prepared.append(tensor)
        if not prepared:
            return [[] for _ in items]
        batch = np.concatenate(prepared, axis=0).astype(np.float32)
        try:
            outputs = self._session.run(None, {self._input_name: batch})
        except Exception as exc:  # pragma: no cover - runtime path
            self._load_error = f"onnx inference failed: {exc}"
            return [_fallback_descriptor(frame_bgr, bbox) for frame_bgr, bbox in items]
        if not outputs:
            return [[] for _ in items]
        features = np.asarray(outputs[0])
        if features.ndim == 1:
            features = features.reshape(1, -1)
        rows = iter(features.tolist())
        normalized: list[list[float]] = []
        for is_valid in valid_mask:
            if not is_valid:
                normalized.append([])
                continue
            normalized.append(_normalize_vector(next(rows)))
        return normalized


def create_reid_backend(config: ReIDConfig, project_root: Path) -> ReIDBackend:
    if not config.enabled:
        return DisabledReIDBackend("disabled by config")
    if config.backend in {"torchreid_osnet", "torchreid_model"}:
        return TorchreidEmbeddingBackend(config, project_root)
    if config.backend == "fastreid":
        return OnnxEmbeddingBackend(config, project_root, "fastreid")
    if config.backend == "tao_reid":
        return OnnxEmbeddingBackend(config, project_root, "tao_reid")
    return DisabledReIDBackend(f"unsupported backend: {config.backend}")


def _crop_person(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
) -> np.ndarray | None:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _prepare_onnx_tensor(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    config: ReIDConfig,
) -> np.ndarray | None:
    crop = _crop_person(frame_bgr, bbox_xyxy)
    if crop is None:
        return None
    resized = cv2.resize(crop, (config.input_width, config.input_height), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    mean = np.asarray(config.input_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(config.input_std, dtype=np.float32).reshape(1, 1, 3)
    std = np.where(std <= 1e-6, 1.0, std)
    tensor = (tensor - mean) / std
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.expand_dims(tensor, axis=0)


def _fallback_descriptor(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
) -> list[float]:
    crop = _crop_person(frame_bgr, bbox_xyxy)
    if crop is None:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, histogram, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return histogram.flatten().astype(np.float32).tolist()


def _normalize_vector(values: list[float] | np.ndarray) -> list[float]:
    vector = np.asarray(values, dtype=np.float32)
    if vector.size == 0:
        return []
    norm = np.linalg.norm(vector)
    if norm <= 1e-6:
        return vector.tolist()
    return (vector / norm).tolist()
