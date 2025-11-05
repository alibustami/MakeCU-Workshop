"""Load the clpai_12dof_0905 URDF into a PyBullet simulation (stable spawn)."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import pybullet as pb
import pybullet_data

if __package__ is None or __package__ == "":
    from forward_motion import ForwardMotionController
else:
    from .forward_motion import ForwardMotionController

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
        default=1.0,
        help="Forward speed in meters per second while walking.",
    )
    parser.add_argument(
        "--sensors",
        action="store_true",
        help="Enable camera and IMU debug overlays in the GUI (may require writable Matplotlib cache).",
    )
    args = parser.parse_args(argv)

    if args.walk_amount < 0.0:
        parser.error("--walk-amount must be non-negative")
    if args.walk_speed <= 0.0:
        parser.error("--walk-speed must be positive")

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
        self._camera_renderer = pb.ER_BULLET_HARDWARE_OPENGL
        self._imu_plot_enabled = plt is not None
        self._imu_times: deque[float] = deque(maxlen=600)
        self._imu_series = {
            "roll": deque(maxlen=600),
            "pitch": deque(maxlen=600),
            "yaw": deque(maxlen=600),
        }
        self._imu_plot_window = 10.0
        self._imu_plot_start: Optional[float] = None
        self._imu_fig = None
        self._imu_axes = None
        self._imu_lines: dict[str, object] = {}

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

        if self._imu_plot_enabled:
            self._ensure_imu_plot()
            self._refresh_imu_plot()
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
        if not self._imu_plot_enabled:
            return

        if self._imu_plot_start is None:
            self._imu_plot_start = now

        rel_time = now - self._imu_plot_start
        self._imu_times.append(rel_time)
        self._imu_series["roll"].append(roll_deg)
        self._imu_series["pitch"].append(pitch_deg)
        self._imu_series["yaw"].append(yaw_deg)

    def _ensure_imu_plot(self) -> None:
        if not self._imu_plot_enabled or self._imu_fig is not None:
            return

        plt.ion()
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(5, 6))
        axes = axes.tolist()
        fig.suptitle("IMU Orientation (deg)")
        colors = [("roll", "tab:red"), ("pitch", "tab:orange"), ("yaw", "tab:blue")]
        for ax, (key, color) in zip(axes, colors):
            line, = ax.plot([], [], color=color, linewidth=1.5)
            ax.set_ylabel(f"{key.capitalize()} (deg)")
            ax.set_ylim(-180.0, 180.0)
            ax.grid(True, linestyle="--", alpha=0.4)
            self._imu_lines[key] = line
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        try:
            fig.canvas.manager.set_window_title("IMU Orientation (Roll/Pitch/Yaw)")
        except Exception:
            pass
        self._imu_fig = fig
        self._imu_axes = axes

    def _refresh_imu_plot(self) -> None:
        if not self._imu_plot_enabled or self._imu_fig is None or self._imu_axes is None:
            return

        times = list(self._imu_times)
        if not times:
            return

        end_time = times[-1]
        start_time = max(0.0, end_time - self._imu_plot_window)
        x_max = max(end_time, self._imu_plot_window)

        for idx, (key, line) in enumerate(self._imu_lines.items()):
            data = list(self._imu_series[key])
            line.set_data(times, data)
            ax = self._imu_axes[idx]
            ax.set_xlim(start_time, x_max)

        self._imu_fig.canvas.draw_idle()
        self._imu_fig.canvas.flush_events()
        plt.pause(0.001)


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
        sensor_visualizer: Optional[_SensorVisualizer] = None
        if connection_mode == pb.GUI and args.sensors:
            print("[Main] enabling sensor visualizer overlays", flush=True)
            sensor_visualizer = _SensorVisualizer(
                robot_id=robot_id,
                camera_link_index=link_map.get("camera_optical_frame"),
                imu_link_index=link_map.get("imu_link"),
            )
            sensor_visualizer.enable_gui_elements()
        elif connection_mode == pb.GUI:
            print("[Main] GUI sensor overlays disabled (pass --sensors to enable).", flush=True)

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

        motion_controller: Optional[ForwardMotionController] = None
        loop_start_time = time.perf_counter()
        last_loop_debug = loop_start_time
        if args.walk_amount > 0.0:
            print(
                "[Main] creating ForwardMotionController",
                f"mode={args.walk_mode}",
                f"amount={args.walk_amount}",
                f"speed={args.walk_speed}",
                flush=True,
            )
            motion_controller = ForwardMotionController(
                robot_id=robot_id,
                speed=args.walk_speed,
                mode=args.walk_mode,
                amount=args.walk_amount,
            )
            motion_controller.start(time.perf_counter())

        for step in range(max(args.steps, 0)):
            now = time.perf_counter()
            if motion_controller is not None and not motion_controller.is_finished:
                motion_controller.update(now)

            pb.stepSimulation()
            if sensor_visualizer is not None:
                sensor_visualizer.update(now)
            time.sleep(args.timestep)

            if motion_controller is not None and (now - last_loop_debug) >= 0.5:
                base_pos, base_orn = pb.getBasePositionAndOrientation(robot_id)
                lin_vel, ang_vel = pb.getBaseVelocity(robot_id)
                roll, pitch, yaw = pb.getEulerFromQuaternion(base_orn)
                travelled = motion_controller.travelled_distance
                print(
                    "[Main] loop",
                    f"step={step}",
                    f"sim_time={now - loop_start_time:.3f} s",
                    f"base_pos=({base_pos[0]:.3f},{base_pos[1]:.3f},{base_pos[2]:.3f})",
                    f"base_rpy=({math.degrees(roll):.2f},{math.degrees(pitch):.2f},{math.degrees(yaw):.2f}) deg",
                    f"base_lin_vel=({lin_vel[0]:.3f},{lin_vel[1]:.3f},{lin_vel[2]:.3f})",
                    f"base_ang_vel=({ang_vel[0]:.3f},{ang_vel[1]:.3f},{ang_vel[2]:.3f})",
                    f"controller_finished={motion_controller.is_finished}",
                    f"controller_travelled={travelled:.3f} m",
                    flush=True,
                )
                last_loop_debug = now

            if motion_controller is not None and motion_controller.is_finished:
                break

        if motion_controller is not None and not motion_controller.is_finished:
            motion_controller.stop()

        if motion_controller is not None and motion_controller.summary is not None:
            summary = motion_controller.summary
            print(
                f"Forward walk finished in {summary.duration_s:.2f}s "
                f"covering {summary.distance_m:.2f} m.",
            )

        print(f"Loaded robot with body unique ID: {robot_id} (plane id: {plane_id})")
    finally:
        if pb.isConnected(cid):
            pb.disconnect(cid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
