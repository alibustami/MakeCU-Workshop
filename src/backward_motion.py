"""Backward walking controller built on top of the forward gait."""

from __future__ import annotations

from typing import Optional

try:
    from forward_motion import ForwardMotionController, ForwardMotionSummary
except ModuleNotFoundError:  # pragma: no cover - package-relative import fallback
    from .forward_motion import ForwardMotionController, ForwardMotionSummary


class BackwardMotionController(ForwardMotionController):
    """Reuses the forward gait but drives the base along the negative heading."""

    def __init__(
        self,
        robot_id: int,
        *,
        speed: float,
        mode: str,
        amount: float,
        lock_base: bool = False,
    ) -> None:
        super().__init__(robot_id=robot_id, speed=speed, mode=mode, amount=amount, lock_base=lock_base)

    def start(self, start_time: Optional[float] = None) -> None:  # noqa: D401 - inherit docstring
        super().start(start_time)
        forward_dir = getattr(self, "_forward_dir", None)
        if forward_dir is not None:
            self._forward_dir = (-forward_dir[0], -forward_dir[1], -forward_dir[2])  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def travelled_distance(self) -> float:
        return -super().travelled_distance


__all__ = ["BackwardMotionController", "ForwardMotionSummary"]
