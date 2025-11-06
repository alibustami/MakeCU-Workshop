"""Interactive keyboard teleoperation for the clpai_12dof_0905 robot in PyBullet."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import pybullet as pb
import pybullet_data

_ESCAPE_KEYCODES = {ord("q"), ord("Q"), 27}
try:
    _ESCAPE_KEYCODES.add(getattr(pb, "B3G_ESCAPE"))
except AttributeError:
    pass

if __package__ in (None, ""):
    from backward_motion import BackwardMotionController
    from forward_motion import ForwardMotionController, ForwardMotionSummary
    from rotation_motion import RotationMotionController, RotationMotionSummary
    from main import (  # type: ignore[attr-defined]  # noqa: PLC2701
        _SensorVisualizer,
        _ensure_above_ground,
        _link_name_map,
        _resolve_urdf,
        _workspace_root,
    )
else:
    from .backward_motion import BackwardMotionController
    from .forward_motion import ForwardMotionController, ForwardMotionSummary
    from .rotation_motion import RotationMotionController, RotationMotionSummary
    from .main import (  # type: ignore[attr-defined]  # noqa: PLC2701
        _SensorVisualizer,
        _ensure_above_ground,
        _link_name_map,
        _resolve_urdf,
        _workspace_root,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the PyBullet GUI (required for keyboard teleoperation).",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1.0 / 240.0,
        help="Simulation timestep in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=60,
        help="Number of zero-gravity settle steps before enabling gravity.",
    )
    parser.add_argument(
        "--sensors",
        dest="sensors",
        action="store_true",
        default=None,
        help="Enable camera and IMU overlays in the GUI (default: on when GUI).",
    )
    parser.add_argument(
        "--no-sensors",
        dest="sensors",
        action="store_false",
        help="Disable camera and IMU overlays.",
    )
    parser.add_argument(
        "--forward-distance",
        type=float,
        default=0.4,
        help="Meters to walk for each Up/Down arrow press (default: %(default)s).",
    )
    parser.add_argument(
        "--forward-speed",
        type=float,
        default=0.6,
        help="Forward walking speed in m/s (default: %(default)s).",
    )
    parser.add_argument(
        "--turn-angle",
        type=float,
        default=15.0,
        help="Degrees to rotate for each Left/Right arrow press (default: %(default)s).",
    )
    parser.add_argument(
        "--turn-speed",
        type=float,
        default=45.0,
        help="Yaw rotation speed in deg/s (default: %(default)s).",
    )
    parser.add_argument(
        "--queue-limit",
        type=int,
        default=4,
        help="Maximum number of queued motion commands (default: %(default)s).",
    )
    parser.add_argument(
        "--base",
        choices=("free", "fixed"),
        default="free",
        help="Set to 'fixed' to rigidly attach the base to the world (default: %(default)s).",
    )

    args = parser.parse_args(argv)

    if args.timestep <= 0.0:
        parser.error("--timestep must be positive")
    if args.forward_distance <= 0.0:
        parser.error("--forward-distance must be positive")
    if args.forward_speed <= 0.0:
        parser.error("--forward-speed must be positive")
    if args.turn_angle <= 0.0:
        parser.error("--turn-angle must be positive")
    if args.turn_speed <= 0.0:
        parser.error("--turn-speed must be positive")
    if args.queue_limit <= 0:
        parser.error("--queue-limit must be at least 1")

    if args.sensors is None:
        args.sensors = args.gui

    return args


def _print_instructions(forward_distance: float, turn_angle: float) -> None:
    print(
        "[Keyboard] Ready for teleoperation:\n"
        f"  Up Arrow   -> walk forward {forward_distance:.2f} m\n"
        f"  Down Arrow -> walk backward {forward_distance:.2f} m\n"
        f"  Left Arrow -> rotate +{turn_angle:.1f} deg\n"
        f"  Right Arrow-> rotate -{turn_angle:.1f} deg\n"
        "  Space      -> stop current motion and clear queue\n"
        "  Esc / q    -> exit simulation",
        flush=True,
    )


def _create_forward_command(
    robot_id: int,
    *,
    distance: float,
    speed: float,
    lock_base: bool,
) -> Tuple[str, ForwardMotionController]:
    controller = ForwardMotionController(
        robot_id=robot_id,
        speed=speed,
        mode="distance",
        amount=distance,
        lock_base=lock_base,
    )
    return "forward", controller


def _create_backward_command(
    robot_id: int,
    *,
    distance: float,
    speed: float,
    lock_base: bool,
) -> Tuple[str, BackwardMotionController]:
    controller = BackwardMotionController(
        robot_id=robot_id,
        speed=speed,
        mode="distance",
        amount=distance,
        lock_base=lock_base,
    )
    return "backward", controller


def _create_rotation_command(
    robot_id: int,
    *,
    turn_speed_deg: float,
    turn_angle_deg: float,
    direction: float,
    lock_base: bool,
) -> Tuple[str, RotationMotionController]:
    label = "turn_left" if direction >= 0.0 else "turn_right"
    controller = RotationMotionController(
        robot_id=robot_id,
        angular_speed=math.radians(turn_speed_deg),
        mode="angle",
        amount=math.radians(turn_angle_deg) * direction,
        lock_base=lock_base,
    )
    return label, controller


def _handle_summary(label: str, summary: Optional[object]) -> None:
    if summary is None:
        return
    if isinstance(summary, ForwardMotionSummary):
        if label == "backward":
            direction = "Backward walk"
            distance = abs(summary.distance_m)
        else:
            direction = "Forward walk"
            distance = summary.distance_m
        print(
            f"[Keyboard] {direction} finished in {summary.duration_s:.2f}s "
            f"covering {distance:.2f} m.",
            flush=True,
        )
    elif isinstance(summary, RotationMotionSummary):
        print(
            f"[Keyboard] {label} finished in {summary.duration_s:.2f}s "
            f"turning {math.degrees(summary.angle_rad):.1f} deg.",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not args.gui:
        print("[Keyboard] The GUI is required for keyboard control. Re-run with --gui.", file=sys.stderr)
        return 2
    base_locked = args.base == "fixed"

    connection_mode = pb.GUI
    print("[Keyboard] connecting to PyBullet (GUI)", flush=True)
    cid = pb.connect(connection_mode)
    print("[Keyboard] connected", f"cid={cid}", flush=True)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet.")

    sensor_visualizer: Optional[_SensorVisualizer] = None
    active_label: Optional[str] = None
    active_controller: Optional[object] = None
    command_queue: Deque[Tuple[str, object]] = deque()
    queue_limit = max(1, args.queue_limit)

    try:
        pb.setAdditionalSearchPath(str(_workspace_root() / "robot"))
        pb.setGravity(0.0, 0.0, 0.0)
        print("[Keyboard] gravity disabled for spawn", flush=True)

        plane_urdf = Path(pybullet_data.getDataPath()) / "plane.urdf"
        plane_id = pb.loadURDF(str(plane_urdf))
        print("[Keyboard] plane loaded", f"plane_id={plane_id}", flush=True)
        robot_id = pb.loadURDF(str(_resolve_urdf()), useFixedBase=False)
        print("[Keyboard] robot URDF loaded", f"robot_id={robot_id}", flush=True)

        link_map = _link_name_map(robot_id)
        if args.sensors:
            try:
                print("[Keyboard] enabling sensor visualizer overlays", flush=True)
                sensor_visualizer = _SensorVisualizer(
                    robot_id=robot_id,
                    camera_link_index=link_map.get("camera_optical_frame"),
                    imu_link_index=link_map.get("imu_link"),
                )
                sensor_visualizer.enable_gui_elements()
            except Exception as exc:
                sensor_visualizer = None
                print("[Keyboard] sensor visualizer init failed; disabling overlays.", f"error={exc}", flush=True)
        else:
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)

        _ensure_above_ground(robot_id, clearance=2e-3, plane_z=0.0)

        pb.setTimeStep(args.timestep)
        print("[Keyboard] starting settle loop", f"steps={max(args.settle_steps, 0)}", flush=True)
        for idx in range(max(args.settle_steps, 0)):
            pb.stepSimulation()
            if sensor_visualizer is not None:
                sensor_visualizer.update(time.perf_counter())
            time.sleep(args.timestep)

        pb.setGravity(0.0, 0.0, -9.81)
        base_pos, base_orn = pb.getBasePositionAndOrientation(robot_id)
        if base_locked:
            print("[Keyboard] base operating in rigid (kinematic) mode", flush=True)
        print(
            "[Keyboard] robot ready",
            f"base_pos=({base_pos[0]:.3f},{base_pos[1]:.3f},{base_pos[2]:.3f})",
            f"base_quat=({base_orn[0]:.3f},{base_orn[1]:.3f},{base_orn[2]:.3f},{base_orn[3]:.3f})",
            flush=True,
        )

        _print_instructions(args.forward_distance, args.turn_angle)

        loop_start = time.perf_counter()
        last_status = loop_start
        print("[Keyboard] entering teleop loop (press Esc or q to exit)", flush=True)

        exit_requested = False
        while pb.isConnected(cid) and not exit_requested:
            now = time.perf_counter()

            events = pb.getKeyboardEvents()
            for key, state in events.items():
                if not (state & pb.KEY_WAS_TRIGGERED):
                    continue

                if key == pb.B3G_UP_ARROW:
                    label, controller = _create_forward_command(
                        robot_id,
                        distance=args.forward_distance,
                        speed=args.forward_speed,
                        lock_base=base_locked,
                    )
                elif key == pb.B3G_DOWN_ARROW:
                    label, controller = _create_backward_command(
                        robot_id,
                        distance=args.forward_distance,
                        speed=args.forward_speed,
                        lock_base=base_locked,
                    )
                elif key == pb.B3G_LEFT_ARROW:
                    label, controller = _create_rotation_command(
                        robot_id,
                        turn_speed_deg=args.turn_speed,
                        turn_angle_deg=args.turn_angle,
                        direction=1.0,
                        lock_base=base_locked,
                    )
                elif key == pb.B3G_RIGHT_ARROW:
                    label, controller = _create_rotation_command(
                        robot_id,
                        turn_speed_deg=args.turn_speed,
                        turn_angle_deg=args.turn_angle,
                        direction=-1.0,
                        lock_base=base_locked,
                    )
                elif key in (pb.B3G_SPACE, ord(" ")):
                    if active_controller is not None:
                        active_controller.stop(now)  # type: ignore[attr-defined]
                        _handle_summary(active_label or "", getattr(active_controller, "summary", None))
                    active_label = None
                    active_controller = None
                    command_queue.clear()
                    print("[Keyboard] motions halted; queue cleared.", flush=True)
                    continue
                elif key in _ESCAPE_KEYCODES:
                    exit_requested = True
                    break
                else:
                    continue

                if len(command_queue) >= queue_limit:
                    dropped_label, _ = command_queue.popleft()
                    print(f"[Keyboard] queue full; dropping oldest command '{dropped_label}'.", flush=True)
                command_queue.append((label, controller))
                print(f"[Keyboard] queued '{label}' command.", flush=True)

            if active_controller is None and command_queue:
                active_label, active_controller = command_queue.popleft()
                active_controller.start(now)  # type: ignore[attr-defined]
                print(f"[Keyboard] started '{active_label}' command.", flush=True)

            if active_controller is not None:
                active_controller.update(now)  # type: ignore[attr-defined]
                if active_controller.is_finished:  # type: ignore[attr-defined]
                    _handle_summary(active_label or "", getattr(active_controller, "summary", None))
                    active_label = None
                    active_controller = None

            try:
                pb.stepSimulation()
            except Exception as exc:
                print("[Keyboard] stepSimulation failed", f"error={exc}", flush=True)
                break

            if sensor_visualizer is not None:
                sensor_visualizer.update(now)

            if (now - last_status) >= 0.5:
                base_pos, base_orn = pb.getBasePositionAndOrientation(robot_id)
                lin_vel, ang_vel = pb.getBaseVelocity(robot_id)
                roll, pitch, yaw = pb.getEulerFromQuaternion(base_orn)
                print(
                    "[Keyboard] status",
                    f"sim_time={now - loop_start:.2f}s",
                    f"base_pos=({base_pos[0]:.3f},{base_pos[1]:.3f},{base_pos[2]:.3f})",
                    f"base_rpy=({math.degrees(roll):.1f},{math.degrees(pitch):.1f},{math.degrees(yaw):.1f}) deg",
                    f"lin_vel=({lin_vel[0]:.3f},{lin_vel[1]:.3f},{lin_vel[2]:.3f})",
                    f"ang_vel=({ang_vel[0]:.3f},{ang_vel[1]:.3f},{ang_vel[2]:.3f})",
                    f"active={active_label or 'idle'}",
                    f"queue_len={len(command_queue)}",
                    flush=True,
                )
                last_status = now

            time.sleep(args.timestep)

    finally:
        if sensor_visualizer is not None:
            sensor_visualizer.shutdown()
        if pb.isConnected(cid):
            pb.disconnect(cid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
