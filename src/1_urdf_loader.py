"""Minimal URDF loading utility for the MakeCU PyBullet workshop."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import pybullet as pb
import pybullet_data

if __package__ in (None, ""):
    from main import (  # type: ignore[attr-defined]
        _ensure_above_ground,
        _resolve_urdf,
        _workspace_root,
    )
else:
    from .main import _ensure_above_ground, _resolve_urdf, _workspace_root  # type: ignore[attr-defined]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the PyBullet GUI instead of running headless (DIRECT).",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to the URDF to load (defaults to the workshop robot).",
    )
    parser.add_argument(
        "--fixed-base",
        action="store_true",
        help="Load the robot with a fixed (kinematic) base.",
    )
    parser.add_argument(
        "--no-plane",
        dest="plane",
        action="store_false",
        help="Skip loading the default PyBullet plane.",
    )
    parser.add_argument(
        "--plane",
        dest="plane",
        action="store_true",
        default=True,
        help="Load the default PyBullet plane (default: on).",
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
        help="Number of zero-gravity steps to run before enabling gravity.",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=5.0,
        help="How long to keep stepping the simulation after spawn (seconds).",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=-9.81,
        help="Gravity acceleration along -Z (m/s^2).",
    )
    parser.add_argument(
        "--clearance",
        type=float,
        default=2e-3,
        help="Minimum clearance above the ground plane when spawning the robot.",
    )
    parser.add_argument(
        "--resource-root",
        type=str,
        default=None,
        help="Directory used for relative meshes referenced by the URDF.",
    )

    args = parser.parse_args(argv)

    if args.timestep <= 0.0:
        parser.error("--timestep must be positive")
    if args.settle_steps < 0:
        parser.error("--settle-steps must be non-negative")
    if args.run_seconds < 0.0:
        parser.error("--run-seconds must be non-negative")
    if args.clearance < 0.0:
        parser.error("--clearance must be non-negative")

    return args


def _resolve_requested_urdf(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = (_workspace_root() / path).resolve()
        else:
            path = path.resolve()
    else:
        path = _resolve_urdf()

    if not path.exists():
        raise FileNotFoundError(f"URDF not found at {path}")

    return path


def _resolve_resource_root(explicit: str | None, urdf_path: Path) -> Path:
    if explicit:
        root = Path(explicit).expanduser()
        if not root.is_absolute():
            root = (_workspace_root() / root).resolve()
        else:
            root = root.resolve()
    else:
        root = urdf_path.parent

    if not root.exists():
        raise FileNotFoundError(f"Resource root does not exist: {root}")

    return root


def _load_plane() -> int:
    plane_path = Path(pybullet_data.getDataPath()) / "plane.urdf"
    return pb.loadURDF(str(plane_path))


def _run_steps(step_count: int, timestep: float) -> None:
    for _ in range(step_count):
        pb.stepSimulation()
        time.sleep(timestep)


def _run_for(duration: float, timestep: float) -> None:
    if duration <= 0.0:
        return
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        pb.stepSimulation()
        time.sleep(timestep)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    urdf_path = _resolve_requested_urdf(args.urdf)
    resource_root = _resolve_resource_root(args.resource_root, urdf_path)

    connection_mode = pb.GUI if args.gui else pb.DIRECT
    print(
        "[URDF Loader] connecting to PyBullet",
        f"mode={'GUI' if connection_mode == pb.GUI else 'DIRECT'}",
        flush=True,
    )
    cid = pb.connect(connection_mode)
    print("[URDF Loader] connected", f"cid={cid}", flush=True)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet.")

    try:
        pb.setAdditionalSearchPath(str(resource_root))
        pb.setTimeStep(args.timestep)
        pb.setGravity(0.0, 0.0, 0.0)
        print(
            "[URDF Loader] spawn parameters",
            f"timestep={args.timestep}",
            f"settle_steps={args.settle_steps}",
            f"gravity={args.gravity}",
            flush=True,
        )

        plane_id = None
        if args.plane:
            plane_id = _load_plane()
            print("[URDF Loader] plane loaded", f"plane_id={plane_id}", flush=True)

        robot_id = pb.loadURDF(str(urdf_path), useFixedBase=args.fixed_base)
        print(
            "[URDF Loader] robot loaded",
            f"robot_id={robot_id}",
            f"fixed_base={args.fixed_base}",
            f"urdf={urdf_path}",
            flush=True,
        )

        _ensure_above_ground(robot_id, clearance=args.clearance, plane_z=0.0)
        print(
            "[URDF Loader] clearance applied",
            f"clearance={args.clearance}",
            flush=True,
        )

        if args.settle_steps > 0:
            print("[URDF Loader] running settle loop", f"steps={args.settle_steps}", flush=True)
            _run_steps(args.settle_steps, args.timestep)

        pb.setGravity(0.0, 0.0, args.gravity)
        base_pos, base_orn = pb.getBasePositionAndOrientation(robot_id)
        print(
            "[URDF Loader] ready",
            f"base_pos=({base_pos[0]:.3f},{base_pos[1]:.3f},{base_pos[2]:.3f})",
            f"base_quat=({base_orn[0]:.3f},{base_orn[1]:.3f},{base_orn[2]:.3f},{base_orn[3]:.3f})",
            flush=True,
        )

        if args.run_seconds > 0.0:
            print(
                "[URDF Loader] stepping simulation",
                f"duration={args.run_seconds} s",
                flush=True,
            )
            _run_for(args.run_seconds, args.timestep)

        if args.gui:
            print("[URDF Loader] GUI active - close the window or press Ctrl+C to exit.", flush=True)
            while pb.isConnected(cid):
                pb.stepSimulation()
                time.sleep(args.timestep)

    finally:
        if pb.isConnected(cid):
            pb.disconnect(cid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
