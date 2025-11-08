import logging
import math
import time
from typing import List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

logger = logging.getLogger(__name__)

class Humanoid12BulletEnv(gym.Env):
    """
    Minimal Gymnasium wrapper for a 12-DoF humanoid in PyBullet.
    Control: PD position targets (stable) or raw torque.
    Observation: base state + joint pos/vel + foot contacts.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        urdf_path: str,
        controlled_joint_names: Optional[List[str]] = None,
        foot_link_names: Optional[List[str]] = None,
        render: bool = False,
        follow_camera: bool = True,
        camera_distance: float = 2.5,
        camera_yaw: float = 50.0,
        camera_pitch: float = -35.0,
        actuation: str = "pos",           # "pos" or "torque"
        frame_skip: int = 4,
        timestep: float = 1.0 / 240.0,
        delta_pos_scale: float = 0.5,     # rad per action unit when actuation="pos"
        torque_scale: float = 40.0,       # Nm per action unit when actuation="torque"
        alive_bonus: float = 1.0,
    ):
        super().__init__()
        assert actuation in ("pos", "torque")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.urdf_path = urdf_path
        self.render_gui = render
        self.actuation = actuation
        self.frame_skip = frame_skip
        self.dt = timestep
        self.delta_pos_scale = delta_pos_scale
        self.torque_scale = torque_scale
        self.alive_bonus = alive_bonus
        self.follow_camera = bool(render and follow_camera)
        self.camera_distance = camera_distance
        self.camera_yaw = camera_yaw
        self.camera_pitch = camera_pitch
        self.logger.info(
            "Initializing environment (actuation=%s, render=%s, frame_skip=%d, dt=%.5f)",
            self.actuation, self.render_gui, self.frame_skip, self.dt
        )

        # Bullet connection and basic world
        connection_mode = p.GUI if self.render_gui else p.DIRECT
        self._p = bullet_client.BulletClient(connection_mode=connection_mode)
        self.cid = self._p._client
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setTimeStep(self.dt)
        self._p.setGravity(0, 0, -9.81)

        # Placeholders filled in _load_world()
        self.plane_id = None
        self.robot_id = None
        self.joint_ids: List[int] = []
        self.joint_name_to_id = {}
        self.foot_link_ids: List[int] = []

        # Build world once to infer spaces
        self._load_world(controlled_joint_names, foot_link_names)

        n = len(self.joint_ids)  # should be 12
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

        # obs: [base_h, base_rpy(3), base_lin_vel(3), base_ang_vel(3),
        #       joint_pos(n), joint_vel(n), foot_contacts(len(foot_links))]
        obs_dim = 1 + 3 + 3 + 3 + n + n + len(self.foot_link_ids)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # Disable default motors if we plan to use raw torque
        if self.actuation == "torque":
            for j in self.joint_ids:
                p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0.0)

    # --- Bullet helpers -------------------------------------------------------
    def _load_world(self, controlled_joint_names, foot_link_names):
        if self.plane_id is not None:
            self._p.resetSimulation()

        self.plane_id = self._p.loadURDF("plane.urdf")
        flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
        self.robot_id = self._p.loadURDF(
            self.urdf_path, [0, 0, 1.0], p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False, flags=flags
        )
        self._ensure_above_ground()
        self._update_debug_camera()

        # Collect all movable joints
        movable_types = {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, p.JOINT_SPHERICAL, p.JOINT_PLANAR}
        all_movable = [j for j in range(self._p.getNumJoints(self.robot_id))
                       if self._p.getJointInfo(self.robot_id, j)[2] in movable_types]

        # Map names
        self.joint_name_to_id = {
            self._p.getJointInfo(self.robot_id, j)[1].decode("utf-8"): j for j in all_movable
        }

        # Choose controlled joints (prefer provided names; otherwise take first 12 movable)
        if controlled_joint_names:
            self.joint_ids = [self.joint_name_to_id[name] for name in controlled_joint_names]
        else:
            self.joint_ids = all_movable[:12]

        # Foot link ids for contact (pass names to be precise)
        if foot_link_names:
            link_name_to_id = {
                self._p.getJointInfo(self.robot_id, j)[12].decode("utf-8"): j for j in range(self._p.getNumJoints(self.robot_id))
            }
            self.foot_link_ids = [link_name_to_id[name] for name in foot_link_names]
        else:
            # Fallback: guess links containing "foot" / "ankle"
            link_name_to_id = {
                self._p.getJointInfo(self.robot_id, j)[12].decode("utf-8"): j for j in range(self._p.getNumJoints(self.robot_id))
            }
            self.foot_link_ids = [idx for name, idx in link_name_to_id.items()
                                  if ("foot" in name.lower() or "ankle" in name.lower())][:2]
        self.logger.info(
            "Loaded URDF '%s' with %d controlled joints and %d foot links",
            self.urdf_path, len(self.joint_ids), len(self.foot_link_ids)
        )

    def _get_obs(self) -> np.ndarray:
        pos, quat = self._p.getBasePositionAndOrientation(self.robot_id)
        rpy = p.getEulerFromQuaternion(quat)
        lin_vel, ang_vel = self._p.getBaseVelocity(self.robot_id)

        js = self._p.getJointStates(self.robot_id, self.joint_ids)
        q = np.array([s[0] for s in js], dtype=np.float32)
        dq = np.array([s[1] for s in js], dtype=np.float32)

        # Foot contacts with the plane
        contacts = []
        for link_id in self.foot_link_ids:
            cps = self._p.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id, linkIndexA=link_id)
            contacts.append(1.0 if len(cps) > 0 else 0.0)

        obs = np.concatenate([
            np.array([pos[2]], dtype=np.float32),
            np.array(rpy, dtype=np.float32),
            np.array(lin_vel, dtype=np.float32),
            np.array(ang_vel, dtype=np.float32),
            q, dq,
            np.array(contacts, dtype=np.float32)
        ])
        return obs

    def _apply_action(self, a: np.ndarray):
        a = np.asarray(a, dtype=np.float32)
        if self.actuation == "pos":
            # Position targets are relative deltas from current q
            cur_q = np.array([s[0] for s in self._p.getJointStates(self.robot_id, self.joint_ids)], dtype=np.float32)
            targets = cur_q + self.delta_pos_scale * a
            self._p.setJointMotorControlArray(
                self.robot_id, self.joint_ids, controlMode=p.POSITION_CONTROL,
                targetPositions=targets.tolist(),
                positionGains=[0.6] * len(self.joint_ids),
                velocityGains=[0.1] * len(self.joint_ids),
                forces=[150.0] * len(self.joint_ids)  # max servo effort per joint
            )
        else:
            torques = (self.torque_scale * a).tolist()
            self._p.setJointMotorControlArray(
                self.robot_id, self.joint_ids, controlMode=p.TORQUE_CONTROL,
                forces=torques
            )

    # --- Gym API --------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._p.resetSimulation()
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setTimeStep(self.dt)
        self._p.setGravity(0, 0, -9.81)
        self.plane_id = self._p.loadURDF("plane.urdf")
        flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
        self.robot_id = self._p.loadURDF(
            self.urdf_path, [0, 0, 1.0], p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False, flags=flags
        )

        # Re-apply joint/foot selections (same ids as before)
        if self.actuation == "torque":
            for j in self.joint_ids:
                self._p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0.0)

        # Small random pose/joints
        for j in self.joint_ids:
            self._p.resetJointState(self.robot_id, j, targetValue=self.np_random.uniform(-0.05, 0.05))

        # Domain randomization (friction, mass scale)
        fric = float(self.np_random.uniform(0.7, 1.3))
        self._p.changeDynamics(self.plane_id, -1, lateralFriction=fric)
        mass_scale = float(self.np_random.uniform(0.8, 1.2))
        for link in range(-1, self._p.getNumJoints(self.robot_id)):
            try:
                props = self._p.getDynamicsInfo(self.robot_id, link)
                self._p.changeDynamics(self.robot_id, link, mass=props[0]*mass_scale)
            except Exception:
                pass

        self._ensure_above_ground()
        self._update_debug_camera()
        self.logger.info(
            "Environment reset complete (friction=%.2f, mass_scale=%.2f)",
            fric, mass_scale
        )
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._apply_action(action)
        for _ in range(self.frame_skip):
            self._p.stepSimulation()
        self._update_debug_camera()

        obs = self._get_obs()
        # Reward shaping
        base_lin_vel, base_ang_vel = self._p.getBaseVelocity(self.robot_id)
        forward = float(base_lin_vel[0])              # x-forward velocity
        lateral = float(base_lin_vel[1])
        yaw_rate = float(base_ang_vel[2])

        act_pen = 0.001 * float(np.square(action).sum())
        reward = self.alive_bonus + 1.5 * forward - 0.25 * (lateral**2) - 0.05 * (yaw_rate**2) - act_pen

        # Termination: fall or large tilt
        pos, quat = self._p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(quat)
        fell = (pos[2] < 0.35) or (abs(roll) > 1.0) or (abs(pitch) > 1.0)
        terminated = bool(fell)
        truncated = False

        info = {"reward_forward": forward, "reward_act_pen": -act_pen}
        if terminated:
            self.logger.info(
                "Episode terminated: height=%.2f roll=%.2f pitch=%.2f reward=%.3f forward_vel=%.3f",
                pos[2], roll, pitch, reward, forward
            )
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_gui:
            time.sleep(self.dt * self.frame_skip)

    def close(self):
        if self._p.isConnected():
            self._p.disconnect()
            self.logger.info("Disconnected Bullet client.")

    # --- Spawn / visualization helpers --------------------------------------
    def _ensure_above_ground(self, clearance: float = 2e-3, plane_z: float = 0.0):
        if self.robot_id is None:
            return
        link_indices = [-1] + list(range(self._p.getNumJoints(self.robot_id)))
        lowest_z = None
        for link_id in link_indices:
            lower, _ = self._p.getAABB(self.robot_id, link_id)
            if lower is None:
                continue
            lowest_z = lower[2] if lowest_z is None else min(lowest_z, lower[2])

        if lowest_z is None:
            return

        offset = plane_z + clearance - lowest_z
        if abs(offset) <= 1e-6:
            return

        base_pos, base_orn = self._p.getBasePositionAndOrientation(self.robot_id)
        new_pos = (base_pos[0], base_pos[1], base_pos[2] + offset)
        self._p.resetBasePositionAndOrientation(self.robot_id, new_pos, base_orn)
        self._p.resetBaseVelocity(self.robot_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def _update_debug_camera(self):
        if not (self.follow_camera and self.robot_id is not None):
            return
        pos, _ = self._p.getBasePositionAndOrientation(self.robot_id)
        self._p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_distance,
            cameraYaw=self.camera_yaw,
            cameraPitch=self.camera_pitch,
            cameraTargetPosition=list(pos)
        )
