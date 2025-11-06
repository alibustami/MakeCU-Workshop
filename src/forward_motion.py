"""Simple forward-walk controller with a scripted joint-space gait."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import pybullet as pb


def _rotate_vector(
    orientation: Tuple[float, float, float, float],
    vector: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Rotate a vector by the quaternion `orientation`."""
    rotation = pb.getMatrixFromQuaternion(orientation)
    return (
        rotation[0] * vector[0] + rotation[1] * vector[1] + rotation[2] * vector[2],
        rotation[3] * vector[0] + rotation[4] * vector[1] + rotation[5] * vector[2],
        rotation[6] * vector[0] + rotation[7] * vector[1] + rotation[8] * vector[2],
    )


@dataclass(slots=True)
class ForwardMotionSummary:
    """Captures basic telemetry about a completed forward-walk command."""

    duration_s: float
    distance_m: float


class ForwardMotionController:
    """Scripted joint-space gait that walks the robot forward at a constant speed."""

    _VALID_MODES = {"time", "distance"}
    _LEG_JOINT_ORDER: Sequence[str] = (
        "hip_pitch_joint",
        "hip_roll_joint",
        "thigh_joint",
        "calf_joint",
        "ankle_pitch_joint",
        "ankle_roll_joint",
    )
    _MAX_FORCE = 150.0
    _STEP_FREQUENCY = 1.0  # steps per second for each leg
    _HIP_SWING = 0.30
    _HIP_ROLL_SWING = 0.12
    _YAW_SWING = 0.05
    _KNEE_BASE = 0.55
    _KNEE_SWING = 0.30
    _ANKLE_BASE = -0.50
    _ANKLE_SWING = 0.20
    _ANKLE_ROLL_SWING = -0.08

    def __init__(self, robot_id: int, *, speed: float, mode: str, amount: float, lock_base: bool = False) -> None:
        if speed <= 0.0:
            raise ValueError("speed must be positive")
        if amount <= 0.0:
            raise ValueError("amount must be positive")
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self._VALID_MODES)}")

        self._robot_id = robot_id
        self._speed = speed
        self._mode = mode
        self._target_amount = amount
        self._start_time: Optional[float] = None
        self._start_position: Optional[Tuple[float, float, float]] = None
        self._start_orientation: Optional[Tuple[float, float, float, float]] = None
        self._forward_dir: Optional[Tuple[float, float, float]] = None
        self._target_duration, self._target_distance = self._derive_targets(speed, mode, amount)
        self._summary: Optional[ForwardMotionSummary] = None
        self._finished = False
        self._joint_indices: Dict[str, int] = {}
        self._rest_pose: Dict[str, float] = {}
        self._last_distance = 0.0
        self._lock_base = lock_base

    @staticmethod
    def _derive_targets(speed: float, mode: str, amount: float) -> Tuple[float, float]:
        if mode == "time":
            duration = amount
            distance = speed * duration
        else:
            distance = amount
            duration = distance / speed
        return duration, distance

    def _gather_joint_indices(self) -> None:
        expected = []
        for prefix in ("l_", "r_"):
            for suffix in self._LEG_JOINT_ORDER:
                expected.append(f"{prefix}{suffix}")

        name_to_index: Dict[str, int] = {}
        for joint_index in range(pb.getNumJoints(self._robot_id)):
            joint_info = pb.getJointInfo(self._robot_id, joint_index)
            name = joint_info[1].decode("utf-8")
            name_to_index[name] = joint_index

        missing = [name for name in expected if name not in name_to_index]
        if missing:
            raise RuntimeError(f"Missing joints in URDF: {', '.join(missing)}")

        for name in expected:
            self._joint_indices[name] = name_to_index[name]
            position = pb.getJointState(self._robot_id, name_to_index[name])[0]
            self._rest_pose[name] = position
            pb.setJointMotorControl2(
                self._robot_id,
                name_to_index[name],
                pb.POSITION_CONTROL,
                targetPosition=position,
                force=self._MAX_FORCE,
            )

    def start(self, start_time: Optional[float] = None) -> None:
        """Latch initial pose, orientation, and joint references."""
        if self._finished or self._start_time is not None:
            return

        self._gather_joint_indices()
        self._start_time = start_time if start_time is not None else time.perf_counter()
        position, orientation = pb.getBasePositionAndOrientation(self._robot_id)
        self._start_position = position
        self._start_orientation = orientation
        forward = _rotate_vector(orientation, (1.0, 0.0, 0.0))
        horizontal = (forward[0], forward[1], 0.0)
        norm = math.hypot(horizontal[0], horizontal[1])
        if norm < 1e-6:
            horizontal = (1.0, 0.0, 0.0)
        else:
            horizontal = (horizontal[0] / norm, horizontal[1] / norm, 0.0)
        self._forward_dir = horizontal
        self._last_distance = 0.0

    def stop(self, now: Optional[float] = None) -> None:
        if self._finished:
            return
        if now is None:
            now = time.perf_counter()
        pb.resetBaseVelocity(self._robot_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        if (
            self._start_position is not None
            and self._forward_dir is not None
            and self._start_orientation is not None
        ):
            final_position = (
                self._start_position[0] + self._forward_dir[0] * self._last_distance,
                self._start_position[1] + self._forward_dir[1] * self._last_distance,
                self._start_position[2],
            )
            pb.resetBasePositionAndOrientation(self._robot_id, final_position, self._start_orientation)
        if self._joint_indices and not self._lock_base:
            neutral_offsets = {
                "hip_pitch_joint": 0.0,
                "hip_roll_joint": 0.0,
                "thigh_joint": 0.0,
                "calf_joint": self._KNEE_BASE,
                "ankle_pitch_joint": self._ANKLE_BASE,
                "ankle_roll_joint": 0.0,
            }
            for leg_prefix in ("l_", "r_"):
                for suffix, offset in neutral_offsets.items():
                    joint_name = f"{leg_prefix}{suffix}"
                    joint_idx = self._joint_indices[joint_name]
                    target = self._rest_pose[joint_name] + offset
                    pb.setJointMotorControl2(
                        self._robot_id,
                        joint_idx,
                        pb.POSITION_CONTROL,
                        targetPosition=target,
                        force=self._MAX_FORCE,
                    )
        duration = 0.0
        if self._start_time is not None:
            duration = max(0.0, now - self._start_time)
        self._summary = ForwardMotionSummary(duration_s=duration, distance_m=self._last_distance)
        self._finished = True

    def update(self, now: Optional[float] = None) -> None:
        if self._finished:
            return
        if self._start_time is None:
            self.start(now)

        assert self._start_time is not None
        assert self._start_position is not None
        assert self._start_orientation is not None
        assert self._forward_dir is not None

        current_time = now if now is not None else time.perf_counter()
        elapsed = max(0.0, current_time - self._start_time)

        if self._target_duration <= 0.0:
            self.stop(current_time)
            return

        progress = min(elapsed / self._target_duration, 1.0)
        distance = progress * self._target_distance
        self._apply_base_motion(distance)
        self._apply_gait(elapsed)

        if progress >= 1.0:
            self.stop(current_time)

    def _apply_base_motion(self, travelled: float) -> None:
        assert self._start_position is not None
        assert self._start_orientation is not None
        assert self._forward_dir is not None

        new_position = (
            self._start_position[0] + self._forward_dir[0] * travelled,
            self._start_position[1] + self._forward_dir[1] * travelled,
            self._start_position[2],
        )
        pb.resetBasePositionAndOrientation(self._robot_id, new_position, self._start_orientation)
        if self._lock_base:
            pb.resetBaseVelocity(self._robot_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        else:
            forward_velocity = (
                self._forward_dir[0] * self._speed,
                self._forward_dir[1] * self._speed,
                0.0,
            )
            pb.resetBaseVelocity(self._robot_id, forward_velocity, (0.0, 0.0, 0.0))
        self._last_distance = travelled

    def _apply_gait(self, elapsed: float) -> None:
        if self._lock_base:
            return
        phase = elapsed * self._STEP_FREQUENCY * 2.0 * math.pi
        leg_configs = {"l": phase, "r": phase + math.pi}

        for leg_prefix, leg_phase in leg_configs.items():
            hip_pitch_target = self._rest_pose[f"{leg_prefix}_hip_pitch_joint"] + self._HIP_SWING * math.sin(leg_phase)
            hip_roll_target = self._rest_pose[f"{leg_prefix}_hip_roll_joint"] + self._HIP_ROLL_SWING * math.sin(leg_phase)
            thigh_target = self._rest_pose[f"{leg_prefix}_thigh_joint"] + self._YAW_SWING * math.sin(leg_phase)

            knee_target = (
                self._rest_pose[f"{leg_prefix}_calf_joint"]
                + self._KNEE_BASE
                + self._KNEE_SWING * math.sin(leg_phase - math.pi / 2.0)
            )
            ankle_pitch_target = (
                self._rest_pose[f"{leg_prefix}_ankle_pitch_joint"]
                + self._ANKLE_BASE
                + self._ANKLE_SWING * math.sin(leg_phase + math.pi / 2.0)
            )
            ankle_roll_target = (
                self._rest_pose[f"{leg_prefix}_ankle_roll_joint"] + self._ANKLE_ROLL_SWING * math.sin(leg_phase)
            )

            self._set_joint_target(f"{leg_prefix}_hip_pitch_joint", hip_pitch_target)
            self._set_joint_target(f"{leg_prefix}_hip_roll_joint", hip_roll_target)
            self._set_joint_target(f"{leg_prefix}_thigh_joint", thigh_target)
            self._set_joint_target(f"{leg_prefix}_calf_joint", knee_target)
            self._set_joint_target(f"{leg_prefix}_ankle_pitch_joint", ankle_pitch_target)
            self._set_joint_target(f"{leg_prefix}_ankle_roll_joint", ankle_roll_target)

    def _set_joint_target(self, joint_name: str, target: float) -> None:
        if self._lock_base:
            return
        index = self._joint_indices[joint_name]
        pb.setJointMotorControl2(
            self._robot_id,
            index,
            pb.POSITION_CONTROL,
            targetPosition=target,
            force=self._MAX_FORCE,
        )

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def summary(self) -> Optional[ForwardMotionSummary]:
        return self._summary

    @property
    def travelled_distance(self) -> float:
        return self._last_distance
