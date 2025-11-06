"""In-place yaw rotation controller that mimics the teleop cmd_vel mapping."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import pybullet as pb


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Return a quaternion representing a rotation about +Z by `yaw` radians."""
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _quat_multiply(
    qa: Tuple[float, float, float, float],
    qb: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Quaternion product qa * qb using Bullet's (x, y, z, w) ordering."""
    ax, ay, az, aw = qa
    bx, by, bz, bw = qb
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


@dataclass(slots=True)
class RotationMotionSummary:
    """Telemetry captured after completing a rotation command."""

    duration_s: float
    angle_rad: float


class RotationMotionController:
    """Rotate the robot about its vertical axis at a constant angular speed."""

    _VALID_MODES = {"time", "angle"}
    _LEG_JOINT_ORDER: Sequence[str] = (
        "hip_pitch_joint",
        "hip_roll_joint",
        "thigh_joint",
        "calf_joint",
        "ankle_pitch_joint",
        "ankle_roll_joint",
    )
    _MAX_FORCE = 150.0
    _STEP_FREQUENCY = 0.6  # steps per second for each leg (slower for stability)
    _HIP_SWING = 0.06
    _HIP_ROLL_SWING = 0.10
    _THIGH_SWING = 0.14
    _KNEE_SWING = 0.10
    _ANKLE_SWING = 0.06
    _ANKLE_ROLL_SWING = -0.04

    def __init__(
        self,
        robot_id: int,
        *,
        angular_speed: float,
        mode: str,
        amount: float,
        lock_base: bool = False,
    ) -> None:
        if angular_speed <= 0.0:
            raise ValueError("angular_speed must be positive")
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self._VALID_MODES)}")
        if mode == "time" and amount <= 0.0:
            raise ValueError("amount must be positive")

        self._robot_id = robot_id
        self._angular_speed = angular_speed
        self._mode = mode
        if mode == "angle":
            if amount == 0.0:
                raise ValueError("amount must be non-zero for angle mode")
            signed_amount = amount
            magnitude = abs(amount)
        else:
            signed_amount = amount
            magnitude = amount

        self._target_amount = signed_amount
        self._target_duration, angle_magnitude = self._derive_targets(angular_speed, mode, magnitude)
        self._target_angle = angle_magnitude if signed_amount >= 0.0 else -angle_magnitude
        self._start_time: Optional[float] = None
        self._start_position: Optional[Tuple[float, float, float]] = None
        self._start_orientation: Optional[Tuple[float, float, float, float]] = None
        self._summary: Optional[RotationMotionSummary] = None
        self._finished = False
        self._joint_indices: Dict[str, int] = {}
        self._rest_pose: Dict[str, float] = {}
        self._last_angle = 0.0
        self._direction = math.copysign(1.0, self._target_angle)
        self._lock_base = lock_base

    @staticmethod
    def _derive_targets(angular_speed: float, mode: str, amount: float) -> Tuple[float, float]:
        if mode == "time":
            duration = amount
            angle = angular_speed * duration
        else:
            angle = amount
            duration = angle / angular_speed
        return duration, angle

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
            joint_idx = name_to_index[name]
            self._joint_indices[name] = joint_idx
            position = pb.getJointState(self._robot_id, joint_idx)[0]
            self._rest_pose[name] = position
            pb.setJointMotorControl2(
                self._robot_id,
                joint_idx,
                pb.POSITION_CONTROL,
                targetPosition=position,
                force=self._MAX_FORCE,
            )

    def start(self, start_time: Optional[float] = None) -> None:
        """Capture the initial pose and configure joint references."""
        if self._finished or self._start_time is not None:
            return

        self._gather_joint_indices()
        self._start_time = start_time if start_time is not None else time.perf_counter()
        position, orientation = pb.getBasePositionAndOrientation(self._robot_id)
        self._start_position = position
        self._start_orientation = orientation
        self._last_angle = 0.0

    def stop(self, now: Optional[float] = None) -> None:
        if self._finished:
            return
        if now is None:
            now = time.perf_counter()
        pb.resetBaseVelocity(self._robot_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        duration = 0.0
        if self._start_time is not None:
            duration = max(0.0, now - self._start_time)
        self._summary = RotationMotionSummary(duration_s=duration, angle_rad=self._last_angle)
        if self._joint_indices:
            for joint_name, joint_idx in self._joint_indices.items():
                target = self._rest_pose[joint_name]
                pb.setJointMotorControl2(
                    self._robot_id,
                    joint_idx,
                    pb.POSITION_CONTROL,
                    targetPosition=target,
                    force=self._MAX_FORCE,
                )
        self._finished = True

    def update(self, now: Optional[float] = None) -> None:
        if self._finished:
            return
        if self._start_time is None:
            self.start(now)

        assert self._start_time is not None
        assert self._start_position is not None
        assert self._start_orientation is not None

        current_time = now if now is not None else time.perf_counter()
        elapsed = max(0.0, current_time - self._start_time)
        if self._target_duration <= 0.0:
            self.stop(current_time)
            return

        progress = min(elapsed / self._target_duration, 1.0)
        angle = progress * self._target_angle
        self._apply_base_rotation(angle)
        self._apply_gait(elapsed, progress)

        if progress >= 1.0:
            self.stop(current_time)

    def _apply_base_rotation(self, angle: float) -> None:
        assert self._start_position is not None
        assert self._start_orientation is not None

        delta_quat = _quat_from_yaw(angle)
        new_orientation = _quat_multiply(delta_quat, self._start_orientation)
        pb.resetBasePositionAndOrientation(self._robot_id, self._start_position, new_orientation)
        if self._lock_base:
            pb.resetBaseVelocity(self._robot_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        else:
            angular_velocity = (0.0, 0.0, self._angular_speed * self._direction)
            pb.resetBaseVelocity(self._robot_id, (0.0, 0.0, 0.0), angular_velocity)
        self._last_angle = angle

    def _apply_gait(self, elapsed: float, progress: float) -> None:
        if self._lock_base:
            return
        phase = elapsed * self._STEP_FREQUENCY * 2.0 * math.pi
        leg_configs = {"l": phase, "r": phase + math.pi}
        amplitude_scale = math.sin(min(progress, 1.0) * math.pi)

        for leg_prefix, leg_phase in leg_configs.items():
            direction_sign = self._direction if leg_prefix == "l" else -self._direction
            hip_pitch_target = (
                self._rest_pose[f"{leg_prefix}_hip_pitch_joint"]
                + amplitude_scale * self._HIP_SWING * math.sin(leg_phase) * 0.5
            )
            hip_roll_target = (
                self._rest_pose[f"{leg_prefix}_hip_roll_joint"]
                + amplitude_scale * direction_sign * self._HIP_ROLL_SWING * math.sin(leg_phase + math.pi / 2.0)
            )
            thigh_target = (
                self._rest_pose[f"{leg_prefix}_thigh_joint"]
                + amplitude_scale * direction_sign * self._THIGH_SWING * math.sin(leg_phase)
            )

            knee_target = (
                self._rest_pose[f"{leg_prefix}_calf_joint"]
                + amplitude_scale * self._KNEE_SWING * math.sin(leg_phase - math.pi / 2.0)
            )
            ankle_pitch_target = (
                self._rest_pose[f"{leg_prefix}_ankle_pitch_joint"]
                + amplitude_scale * self._ANKLE_SWING * math.sin(leg_phase + math.pi / 2.0)
            )
            ankle_roll_target = (
                self._rest_pose[f"{leg_prefix}_ankle_roll_joint"]
                + amplitude_scale * direction_sign * self._ANKLE_ROLL_SWING * math.sin(leg_phase)
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
    def summary(self) -> Optional[RotationMotionSummary]:
        return self._summary

    @property
    def rotated_angle(self) -> float:
        return self._last_angle
