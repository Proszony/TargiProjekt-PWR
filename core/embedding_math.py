from __future__ import annotations

import math


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-6 or right_norm <= 1e-6:
        return 0.0
    score = sum(a * b for a, b in zip(left, right)) / max(left_norm * right_norm, 1e-6)
    return max(0.0, min(score, 1.0))


def blend_embedding(base: list[float], new: list[float], *, momentum: float) -> list[float]:
    if not base:
        return list(new)
    if len(base) != len(new):
        return list(new)
    blended = [(momentum * previous) + ((1.0 - momentum) * current) for previous, current in zip(base, new)]
    norm = math.sqrt(sum(value * value for value in blended))
    if norm <= 1e-6:
        return blended
    return [value / norm for value in blended]
