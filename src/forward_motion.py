"""Simple forward-walk controller with a scripted gait for the 12-DOF robot."""

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
    _STEP_FREQUENCY = 1.2  # steps per second for each leg
    _HIP_SWING = 0.30
    _HIP_ROLL_SWING = 0.12
    _YAW_SWING = 0.05
    _KNEE_BASE = 0.55
    _KNEE_SWING = 0.30
    _ANKLE_BASE = -0.50
    _ANKLE_SWING = 0.20
    _ANKLE_ROLL_SWING = -0.08

    def __init__(self, robot_id: int, *, speed: float, mode: str, amount: float) -> None:
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
        self._last_update_time: Optional[float] = None
        self._debug_last_report_time: float = -math.inf
        self._debug_update_count: int = 0

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
        self._last_update_time = self._start_time
        print(
            "[ForwardMotion] start",
            f"speed={self._speed:.3f} m/s",
            f"mode={self._mode}",
            f"target_distance={self._target_distance:.3f} m",
            f"target_duration={self._target_duration:.3f} s",
            f"forward_dir=({self._forward_dir[0]:.3f},{self._forward_dir[1]:.3f},{self._forward_dir[2]:.3f})",
            flush=True,
        )

    def stop(self, now: Optional[float] = None) -> None:
        if self._finished:
            return
        if now is None:
            now = time.perf_counter()
        pb.resetBaseVelocity(self._robot_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        duration = 0.0
        if self._start_time is not None:
            duration = max(0.0, now - self._start_time)
        self._summary = ForwardMotionSummary(duration_s=duration, distance_m=self._last_distance)
        self._finished = True
        if self._summary is not None:
            print(
                "[ForwardMotion] stop",
                f"duration={self._summary.duration_s:.3f} s",
                f"distance={self._summary.distance_m:.3f} m",
                flush=True,
            )

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

        dt = 0.0
        if self._last_update_time is not None:
            dt = max(0.0, current_time - self._last_update_time)
        self._last_update_time = current_time

        if self._target_duration <= 0.0:
            self.stop(current_time)
            return

        travelled = self._distance_travelled()
        self._last_distance = travelled

        commanded_speed = 0.0
        if self._mode == "distance":
            remaining_distance = self._target_distance - travelled
            if remaining_distance <= 1e-4:
                self.stop(current_time)
                return
            commanded_speed = self._apply_base_motion(
                dt,
                remaining_distance=remaining_distance,
                remaining_duration=None,
            )
        else:
            remaining_duration = self._target_duration - elapsed
            if remaining_duration <= 0.0:
                self.stop(current_time)
                return
            commanded_speed = self._apply_base_motion(
                dt,
                remaining_distance=None,
                remaining_duration=remaining_duration,
            )

        self._apply_gait(elapsed)

        travelled = self._distance_travelled()
        self._last_distance = travelled

        if self._mode == "distance":
            if (self._target_distance - travelled) <= 1e-4:
                self.stop(current_time)
        else:
            if (self._target_duration - elapsed) <= 0.0:
                self.stop(current_time)

        self._debug_update_count += 1
        if current_time - self._debug_last_report_time >= 0.5:
            base_pos, base_orn = pb.getBasePositionAndOrientation(self._robot_id)
            base_vel, base_angular = pb.getBaseVelocity(self._robot_id)
            remaining_distance = self._target_distance - travelled if self._mode == "distance" else float("nan")
            remaining_duration = self._target_duration - elapsed if self._mode == "time" else float("nan")
            roll, pitch, yaw = pb.getEulerFromQuaternion(base_orn)
            print(
                "[ForwardMotion] update",
                f"count={self._debug_update_count}",
                f"elapsed={elapsed:.3f} s",
                f"dt={dt:.4f} s",
                f"travelled={travelled:.3f} m",
                f"remaining_distance={remaining_distance:.3f}",
                f"remaining_duration={remaining_duration:.3f}",
                f"cmd_speed={commanded_speed:.3f} m/s",
                f"base_pos=({base_pos[0]:.3f},{base_pos[1]:.3f},{base_pos[2]:.3f})",
                f"base_rpy=({math.degrees(roll):.2f},{math.degrees(pitch):.2f},{math.degrees(yaw):.2f}) deg",
                f"base_vel=({base_vel[0]:.3f},{base_vel[1]:.3f},{base_vel[2]:.3f})",
                flush=True,
            )
            self._debug_last_report_time = current_time

    def _distance_travelled(self) -> float:
        assert self._start_position is not None
        assert self._forward_dir is not None

        current_position, _ = pb.getBasePositionAndOrientation(self._robot_id)
        delta = (
            current_position[0] - self._start_position[0],
            current_position[1] - self._start_position[1],
            0.0,
        )
        travelled = delta[0] * self._forward_dir[0] + delta[1] * self._forward_dir[1]
        return max(0.0, travelled)

    def _apply_base_motion(
        self,
        dt: float,
        *,
        remaining_distance: Optional[float],
        remaining_duration: Optional[float],
    ) -> float:
        assert self._forward_dir is not None

        commanded_speed = self._speed
        if remaining_distance is not None:
            if dt > 1e-6:
                max_distance_this_step = self._speed * dt
                if remaining_distance < max_distance_this_step:
                    commanded_speed = max(0.0, remaining_distance / dt)
        elif remaining_duration is not None:
            if remaining_duration <= 0.0:
                commanded_speed = 0.0

        linear_velocity = (
            self._forward_dir[0] * commanded_speed,
            self._forward_dir[1] * commanded_speed,
            0.0,
        )
        pb.resetBaseVelocity(self._robot_id, linear_velocity, (0.0, 0.0, 0.0))
        return commanded_speed

    def _apply_gait(self, elapsed: float) -> None:
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
