"""Load the clpai_12dof_0905 URDF into a PyBullet simulation (stable spawn)."""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import time
from pathlib import Path
from typing import Optional

import multiprocessing as mp

import pybullet as pb
import pybullet_data

if __package__ is None or __package__ == "":
    from forward_motion import ForwardMotionController, ForwardMotionSummary
    from rotation_motion import RotationMotionController, RotationMotionSummary
else:
    from .forward_motion import ForwardMotionController, ForwardMotionSummary
    from .rotation_motion import RotationMotionController, RotationMotionSummary

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is optional
    plt = None  # type: ignore[assignment]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Connect to PyBullet using the GUI backend (default is DIRECT).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2400,
        help="Number of simulation steps to run before exiting (default: %(default)s).",
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
        help="Steps to run with zero gravity to settle constraints before enabling gravity.",
    )
    parser.add_argument(
        "--walk-mode",
        choices=("time", "distance"),
        default="time",
        help="Interpret walk-amount as seconds (time) or meters (distance).",
    )
    parser.add_argument(
        "--walk-amount",
        type=float,
        default=3.0,
        help="Duration in seconds if walk-mode=time, or distance in meters if walk-mode=distance.",
    )
    parser.add_argument(
        "--walk-speed",
        type=float,
        default=0.6,
        help="Forward speed in meters per second while walking.",
    )
    parser.add_argument(
        "--turn-angle",
        type=float,
        default=90.0,
        help="Target yaw rotation in degrees (positive rotates counter-clockwise, 0 disables the turn).",
    )
    parser.add_argument(
        "--turn-speed",
        type=float,
        default=45.0,
        help="Yaw rotation speed in degrees per second (must be positive).",
    )
    parser.add_argument(
        "--base",
        choices=("free", "fixed"),
        default="free",
        help="Set to 'fixed' to rigidly attach the base to the world (default: %(default)s).",
    )
    parser.add_argument(
        "--sensors",
        dest="sensors",
        action="store_true",
        default=None,
        help="Enable camera and IMU debug overlays in the GUI (default: on when GUI).",
    )
    parser.add_argument(
        "--no-sensors",
        dest="sensors",
        action="store_false",
        help="Disable camera and IMU debug overlays in the GUI.",
    )
    args = parser.parse_args(argv)

    if args.walk_amount < 0.0:
        parser.error("--walk-amount must be non-negative")
    if args.walk_speed <= 0.0:
        parser.error("--walk-speed must be positive")
    if args.turn_speed <= 0.0:
        parser.error("--turn-speed must be positive")

    if args.sensors is None:
        args.sensors = args.gui

    return args


def _workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_urdf() -> Path:
    workspace_root = _workspace_root()
    urdf_path = workspace_root / "robot" / "clpai_12dof_0905.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found at {urdf_path}")
    return urdf_path


def _link_name_map(body_id: int) -> dict[str, int]:
    """Create a map from link name to joint index for quick lookup."""
    link_map: dict[str, int] = {}
    for joint_index in range(pb.getNumJoints(body_id)):
        joint_info = pb.getJointInfo(body_id, joint_index)
        link_name = joint_info[12].decode("utf-8")
        if link_name:
            link_map[link_name] = joint_index
    return link_map


def _rotate_vector(
    orientation: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a vector using the provided quaternion without NumPy."""
    rot = pb.getMatrixFromQuaternion(orientation)
    return (
        rot[0] * vector[0] + rot[1] * vector[1] + rot[2] * vector[2],
        rot[3] * vector[0] + rot[4] * vector[1] + rot[5] * vector[2],
        rot[6] * vector[0] + rot[7] * vector[1] + rot[8] * vector[2],
    )


def _imu_plot_worker(data_queue: "mp.Queue[tuple[float, float, float, float] | None]", window: float) -> None:
    """Child-process IMU plotter to avoid blocking the PyBullet GUI thread."""
    import time as _time
    from collections import deque as _deque

    try:
        import matplotlib.pyplot as _plt
    except Exception:
        return

    _plt.ion()
    fig, axes = _plt.subplots(3, 1, sharex=True, figsize=(5, 6))
    axes = axes.tolist()
    fig.suptitle("IMU Orientation (deg)")
    colors = [("roll", "tab:red"), ("pitch", "tab:orange"), ("yaw", "tab:blue")]
    lines = {}
    for ax, (key, color) in zip(axes, colors):
        line, = ax.plot([], [], color=color, linewidth=1.5)
        ax.set_ylabel(f"{key.capitalize()} (deg)")
        ax.set_ylim(-180.0, 180.0)
        ax.grid(True, linestyle="--", alpha=0.4)
        lines[key] = line
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    try:
        fig.canvas.manager.set_window_title("IMU Orientation (Roll/Pitch/Yaw)")
    except Exception:
        pass

    start_time: Optional[float] = None
    history = {
        "t": _deque(maxlen=int(window * 240)),
        "roll": _deque(maxlen=int(window * 240)),
        "pitch": _deque(maxlen=int(window * 240)),
        "yaw": _deque(maxlen=int(window * 240)),
    }

    last_draw = _time.perf_counter()
    running = True
    while running:
        try:
            sample = data_queue.get(timeout=0.1)
        except queue.Empty:
            sample = "__EMPTY__"

        if sample == "__EMPTY__":
            if not _plt.fignum_exists(fig.number):
                break
        elif sample is None:
            break
        else:
            ts, roll, pitch, yaw = sample
            if start_time is None:
                start_time = ts
            rel_time = ts - start_time
            history["t"].append(rel_time)
            history["roll"].append(roll)
            history["pitch"].append(pitch)
            history["yaw"].append(yaw)

        now = _time.perf_counter()
        if now - last_draw >= 0.05:
            if history["t"]:
                t_vals = list(history["t"])
                lines["roll"].set_data(t_vals, list(history["roll"]))
                lines["pitch"].set_data(t_vals, list(history["pitch"]))
                lines["yaw"].set_data(t_vals, list(history["yaw"]))
                x_max = t_vals[-1]
                x_min = max(0.0, x_max - window)
                for ax in axes:
                    ax.set_xlim(x_min, max(x_max, window))
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            try:
                _plt.pause(0.001)
            except Exception:
                break
            last_draw = now

    try:
        _plt.close(fig)
    except Exception:
        pass


class _IMUPlotter:
    """Async IMU plot hosted in a separate process."""

    def __init__(self, window: float = 10.0) -> None:
        self._window = window
        ctx = mp.get_context("spawn")
        self._queue: mp.Queue = ctx.Queue(maxsize=8)
        self._process = ctx.Process(target=_imu_plot_worker, args=(self._queue, window), daemon=True)
        self._process.start()

    def push(self, timestamp: float, roll: float, pitch: float, yaw: float) -> None:
        if not self._process.is_alive():
            return
        try:
            self._queue.put_nowait((timestamp, roll, pitch, yaw))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((timestamp, roll, pitch, yaw))
            except queue.Full:
                pass

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)


class _SensorVisualizer:
    """Render camera previews and IMU attitude into the PyBullet GUI."""

    def __init__(
        self,
        robot_id: int,
        camera_link_index: Optional[int],
        imu_link_index: Optional[int],
        camera_update_rate: float = 30.0,
        imu_update_rate: float = 20.0,
    ) -> None:
        self._robot_id = robot_id
        self._camera_link_index = camera_link_index
        self._imu_link_index = imu_link_index
        self._camera_period = 1.0 / camera_update_rate if camera_update_rate > 0.0 else 0.0
        self._imu_period = 1.0 / imu_update_rate if imu_update_rate > 0.0 else 0.0
        self._last_camera_sample = 0.0
        self._last_imu_sample = 0.0
        self._imu_text_uid: Optional[int] = None
        self._camera_resolution = (640, 360)
        self._camera_fov = 1.047  # ~60 degrees, matches URDF sensor configuration.
        self._camera_near = 0.1
        self._camera_far = 50.0
        self._camera_renderer = pb.ER_TINY_RENDERER
        self._imu_plot_enabled = plt is not None
        self._imu_plotter: Optional[_IMUPlotter] = None
        if self._imu_plot_enabled:
            try:
                self._imu_plotter = _IMUPlotter(window=10.0)
            except Exception as exc:
                print("[SensorVisualizer] Failed to start IMU plotter, falling back to text.", f"error={exc}", flush=True)
                self._imu_plot_enabled = False

    def enable_gui_elements(self) -> None:
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
        if self._camera_link_index is not None:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    def update(self, now: float) -> None:
        if self._camera_link_index is not None and (now - self._last_camera_sample) >= self._camera_period:
            self._render_camera_preview()
            self._last_camera_sample = now

        if self._imu_link_index is not None and (now - self._last_imu_sample) >= self._imu_period:
            self._update_imu(now)
            self._last_imu_sample = now

    def _render_camera_preview(self) -> None:
        link_state = pb.getLinkState(self._robot_id, self._camera_link_index, computeForwardKinematics=True)
        if not link_state:
            return

        position = link_state[4]
        orientation = link_state[5]
        forward = _rotate_vector(orientation, (0.0, 0.0, 1.0))
        up = _rotate_vector(orientation, (0.0, -1.0, 0.0))
        target = (
            position[0] + forward[0],
            position[1] + forward[1],
            position[2] + forward[2],
        )
        view_matrix = pb.computeViewMatrix(position, target, up)
        aspect = self._camera_resolution[0] / self._camera_resolution[1]
        projection_matrix = pb.computeProjectionMatrixFOV(
            fov=math.degrees(self._camera_fov),
            aspect=aspect,
            nearVal=self._camera_near,
            farVal=self._camera_far,
        )
        try:
            pb.getCameraImage(
                width=self._camera_resolution[0],
                height=self._camera_resolution[1],
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=self._camera_renderer,
            )
        except Exception:
            # Fall back to the software renderer if hardware OpenGL is unavailable (headless runs).
            self._camera_renderer = pb.ER_TINY_RENDERER
            pb.getCameraImage(
                width=self._camera_resolution[0],
                height=self._camera_resolution[1],
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=self._camera_renderer,
            )

    def _update_imu(self, now: float) -> None:
        link_state = pb.getLinkState(self._robot_id, self._imu_link_index, computeForwardKinematics=True)
        if not link_state:
            return

        position = link_state[4]
        orientation = link_state[5]
        roll, pitch, yaw = pb.getEulerFromQuaternion(orientation)
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)

        self._record_imu_sample(now, roll_deg, pitch_deg, yaw_deg)

        if self._imu_plot_enabled and self._imu_plotter is not None:
            if self._imu_text_uid is not None:
                pb.removeUserDebugItem(self._imu_text_uid)
                self._imu_text_uid = None
        else:
            text = (
                "IMU orientation (deg)\n"
                f"Roll : {roll_deg:6.2f}\n"
                f"Pitch: {pitch_deg:6.2f}\n"
                f"Yaw  : {yaw_deg:6.2f}"
            )
            text_position = (position[0], position[1], position[2] + 0.2)
            self._imu_text_uid = pb.addUserDebugText(
                text,
                text_position,
                textColorRGB=[1.0, 0.9, 0.1],
                textSize=1.4,
                lifeTime=0.0,
                replaceItemUniqueId=self._imu_text_uid if self._imu_text_uid is not None else -1,
            )

    def _record_imu_sample(self, now: float, roll_deg: float, pitch_deg: float, yaw_deg: float) -> None:
        if self._imu_plot_enabled and self._imu_plotter is not None:
            self._imu_plotter.push(now, roll_deg, pitch_deg, yaw_deg)

    def shutdown(self) -> None:
        if self._imu_plotter is not None:
            self._imu_plotter.close()


def _ensure_above_ground(body_id: int, clearance: float = 2e-3, plane_z: float = 0.0) -> None:
    """
    Raise the base just enough so the *lowest point of any link* sits a hair above the ground.
    This prevents initial interpenetration with the plane (which causes the pop/hop).
    """
    # Collect AABBs for base (-1) and all links.
    link_indices = [-1] + list(range(pb.getNumJoints(body_id)))
    lowest_z = None
    for li in link_indices:
        lower, _ = pb.getAABB(body_id, li)
        if lower is None:
            continue
        z = lower[2]
        lowest_z = z if lowest_z is None else min(lowest_z, z)

    if lowest_z is None:
        return

    needed = plane_z + clearance - lowest_z
    if needed <= 0.0:
        return

    base_pos, base_orn = pb.getBasePositionAndOrientation(body_id)
    new_pos = (base_pos[0], base_pos[1], base_pos[2] + needed)
    pb.resetBasePositionAndOrientation(body_id, new_pos, base_orn)
    pb.resetBaseVelocity(body_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    base_locked = args.base == "fixed"

    connection_mode = pb.GUI if args.gui else pb.DIRECT
    print(
        "[Main] connecting to PyBullet",
        f"mode={'GUI' if connection_mode == pb.GUI else 'DIRECT'}",
        flush=True,
    )
    cid = pb.connect(connection_mode)
    print("[Main] connected", f"cid={cid}", flush=True)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet.")

    sensor_visualizer: Optional[_SensorVisualizer] = None
    try:
        # Allow URDFs to load relative meshes from your repo path if needed.
        pb.setAdditionalSearchPath(str(_workspace_root() / "robot"))

        # --- Stable spawn sequence ---
        # 1) Start with zero gravity so nothing "pops" while we place the robot.
        pb.setGravity(0.0, 0.0, 0.0)
        print("[Main] gravity disabled for spawn", flush=True)

        # 2) Load the plane and the robot.
        plane_urdf = Path(pybullet_data.getDataPath()) / "plane.urdf"
        plane_id = pb.loadURDF(str(plane_urdf))
        print("[Main] plane loaded", f"plane_id={plane_id}", flush=True)
        robot_id = pb.loadURDF(str(_resolve_urdf()), useFixedBase=False)
        print("[Main] robot URDF loaded", f"robot_id={robot_id}", flush=True)

        link_map = _link_name_map(robot_id)
        if connection_mode == pb.GUI and args.sensors:
            try:
                print("[Main] enabling sensor visualizer overlays", flush=True)
                sensor_visualizer = _SensorVisualizer(
                    robot_id=robot_id,
                    camera_link_index=link_map.get("camera_optical_frame"),
                    imu_link_index=link_map.get("imu_link"),
                )
                sensor_visualizer.enable_gui_elements()
            except Exception as exc:
                sensor_visualizer = None
                print("[Main] sensor visualizer init failed; disabling overlays.", f"error={exc}", flush=True)
        elif connection_mode == pb.GUI:
            print("[Main] GUI sensor overlays disabled (use --sensors to enable).", flush=True)

        # 3) Lift the robot so the very lowest link point is just above z=0.
        _ensure_above_ground(robot_id, clearance=2e-3, plane_z=0.0)

        # 4) Let joint constraints settle a bit while gravity is zero.
        pb.setTimeStep(args.timestep)
        print("[Main] starting settle loop", f"steps={max(args.settle_steps, 0)}", flush=True)
        for idx in range(max(args.settle_steps, 0)):
            try:
                pb.stepSimulation()
            except Exception as exc:
                print("[Main] settle loop stepSimulation failed", f"idx={idx}", f"error={exc}", flush=True)
                raise

            if idx < 5 or idx % 10 == 0:
                base_pos, _ = pb.getBasePositionAndOrientation(robot_id)
                print(
                    "[Main] settle loop",
                    f"idx={idx}",
                    f"base_pos=({base_pos[0]:.4f},{base_pos[1]:.4f},{base_pos[2]:.4f})",
                    flush=True,
                )

            if sensor_visualizer is not None:
                try:
                    sensor_visualizer.update(time.perf_counter())
                except Exception as exc:
                    print("[Main] sensor_visualizer.update error", f"idx={idx}", f"error={exc}", flush=True)

            # Sleeping makes the GUI smoother; harmless in DIRECT.
            time.sleep(args.timestep)

        # 5) Enable normal gravity and run.
        pb.setGravity(0.0, 0.0, -9.81)
        print("[Main] gravity enabled", flush=True)

        base_pos, base_orn = pb.getBasePositionAndOrientation(robot_id)
        print(
            "[Main] robot loaded",
            f"base_pos=({base_pos[0]:.4f},{base_pos[1]:.4f},{base_pos[2]:.4f})",
            f"base_quat=({base_orn[0]:.4f},{base_orn[1]:.4f},{base_orn[2]:.4f},{base_orn[3]:.4f})",
            flush=True,
        )
        if base_locked:
            print("[Main] base operating in rigid (kinematic) mode", flush=True)

        controller_queue: list[tuple[str, object]] = []
        controller_summaries: list[tuple[str, object]] = []

        if args.walk_amount > 0.0:
            print(
                "[Main] creating ForwardMotionController",
                f"mode={args.walk_mode}",
                f"amount={args.walk_amount}",
                f"speed={args.walk_speed}",
                flush=True,
            )
            forward_controller = ForwardMotionController(
                robot_id=robot_id,
                speed=args.walk_speed,
                mode=args.walk_mode,
                amount=args.walk_amount,
                lock_base=base_locked,
            )
            controller_queue.append(("forward", forward_controller))

        if abs(args.turn_angle) > 1e-6:
            print(
                "[Main] creating RotationMotionController",
                f"angle={args.turn_angle} deg",
                f"speed={args.turn_speed} deg/s",
                flush=True,
            )
            rotation_controller = RotationMotionController(
                robot_id=robot_id,
                angular_speed=math.radians(args.turn_speed),
                mode="angle",
                amount=math.radians(args.turn_angle),
                lock_base=base_locked,
            )
            controller_queue.append(("rotation", rotation_controller))

        active_label: Optional[str] = None
        active_controller: Optional[object] = None
        loop_start_time = time.perf_counter()
        last_loop_debug = loop_start_time

        if controller_queue:
            active_label, active_controller = controller_queue.pop(0)
            active_controller.start(time.perf_counter())  # type: ignore[attr-defined]

        for step in range(max(args.steps, 0)):
            now = time.perf_counter()
            if active_controller is not None and not active_controller.is_finished:  # type: ignore[union-attr]
                active_controller.update(now)  # type: ignore[attr-defined]

            pb.stepSimulation()
            if sensor_visualizer is not None:
                sensor_visualizer.update(now)
            time.sleep(args.timestep)

            if active_controller is not None and (now - last_loop_debug) >= 0.5:
                base_pos, base_orn = pb.getBasePositionAndOrientation(robot_id)
                lin_vel, ang_vel = pb.getBaseVelocity(robot_id)
                roll, pitch, yaw = pb.getEulerFromQuaternion(base_orn)
                travelled = getattr(active_controller, "travelled_distance", None)
                rotated = getattr(active_controller, "rotated_angle", None)
                metric_parts = []
                if travelled is not None:
                    metric_parts.append(f"travelled={travelled:.3f} m")
                if rotated is not None:
                    metric_parts.append(f"rotated={math.degrees(rotated):.2f} deg")
                metric_str = ", ".join(metric_parts)
                print(
                    "[Main] loop",
                    f"step={step}",
                    f"sim_time={now - loop_start_time:.3f} s",
                    f"base_pos=({base_pos[0]:.3f},{base_pos[1]:.3f},{base_pos[2]:.3f})",
                    f"base_rpy=({math.degrees(roll):.2f},{math.degrees(pitch):.2f},{math.degrees(yaw):.2f}) deg",
                    f"base_lin_vel=({lin_vel[0]:.3f},{lin_vel[1]:.3f},{lin_vel[2]:.3f})",
                    f"base_ang_vel=({ang_vel[0]:.3f},{ang_vel[1]:.3f},{ang_vel[2]:.3f})",
                    f"controller={active_label}",
                    f"controller_finished={active_controller.is_finished}",  # type: ignore[union-attr]
                    metric_str,
                    flush=True,
                )
                last_loop_debug = now

            if active_controller is not None and active_controller.is_finished:  # type: ignore[union-attr]
                summary = getattr(active_controller, "summary", None)
                if summary is not None:
                    controller_summaries.append((active_label or "", summary))
                active_label = None
                active_controller = None
                if controller_queue:
                    active_label, active_controller = controller_queue.pop(0)
                    active_controller.start(now)  # type: ignore[attr-defined]
                else:
                    break

        if active_controller is not None and not active_controller.is_finished:  # type: ignore[union-attr]
            active_controller.stop()  # type: ignore[attr-defined]
            summary = getattr(active_controller, "summary", None)
            if summary is not None:
                controller_summaries.append((active_label or "", summary))

        for label, summary in controller_summaries:
            if isinstance(summary, ForwardMotionSummary):
                print(
                    f"Forward walk finished in {summary.duration_s:.2f}s "
                    f"covering {summary.distance_m:.2f} m.",
                )
            elif isinstance(summary, RotationMotionSummary):
                print(
                    f"Rotation finished in {summary.duration_s:.2f}s "
                    f"turning {math.degrees(summary.angle_rad):.1f} deg.",
                )

        print(f"Loaded robot with body unique ID: {robot_id} (plane id: {plane_id})")

        if connection_mode == pb.GUI:
            print(
                "[Main] controllers finished; keeping GUI open until you close the window or press Ctrl+C.",
                flush=True,
            )
            idle_sleep = max(args.timestep, 1.0 / 240.0)
            try:
                while pb.isConnected(cid):
                    now = time.perf_counter()
                    try:
                        pb.stepSimulation()
                    except Exception as exc:
                        print("[Main] idle loop stepSimulation failed", f"error={exc}", flush=True)
                        break
                    if sensor_visualizer is not None:
                        try:
                            sensor_visualizer.update(now)
                        except Exception as exc:
                            print("[Main] sensor_visualizer.update error", f"error={exc}", flush=True)
                    time.sleep(idle_sleep)
            except KeyboardInterrupt:
                print("[Main] interrupt received; exiting idle loop.", flush=True)
    finally:
        if sensor_visualizer is not None:
            sensor_visualizer.shutdown()
        if pb.isConnected(cid):
            pb.disconnect(cid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
_DEFAULT_MPL_DIR = Path(__file__).resolve().parent.parent / ".mpl_cache"
if "MPLCONFIGDIR" not in os.environ:
    try:
        _DEFAULT_MPL_DIR.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_DEFAULT_MPL_DIR)
    except Exception:
        pass
