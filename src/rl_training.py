# train.py
import argparse
import os
from functools import partial

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from rl_base import Humanoid12BulletEnv

URDF = "robot/clpai_12dof_0905_rl.urdf"
JOINTS = [
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_thigh_joint",
    "r_calf_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_thigh_joint",
    "l_calf_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint"
]
FEET = ["r_ankle_pitch_link", "l_ankle_pitch_link"]  # or the appropriate link names


def parse_args():
    parser = argparse.ArgumentParser(description="Train the 12-DoF humanoid with Stable-Baselines3.")
    parser.add_argument("--algo", choices=["PPO", "SAC"], default="PPO",
                        help="On-policy/off-policy algorithm to use.")
    parser.add_argument("--total-timesteps", type=int, default=3_000_000,
                        help="Training horizon per run.")
    parser.add_argument("--num-agents", type=int, default=8,
                        help="Number of vectorized environments (agents) to train in parallel.")
    parser.add_argument("--gui", action="store_true",
                        help="Enable the PyBullet GUI to visualize training (forces num_agents=1).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed forwarded to Stable-Baselines3.")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory where the policy and VecNormalize statistics are stored.")
    return parser.parse_args()


def make_env(render_gui: bool = False):
    return Humanoid12BulletEnv(
        urdf_path=URDF,
        controlled_joint_names=JOINTS if JOINTS else None,
        foot_link_names=FEET,
        render=render_gui,
        actuation="pos",      # try "torque" later
        frame_skip=4,
        timestep=1/240
    )


if __name__ == "__main__":
    args = parse_args()
    if args.num_agents < 1:
        raise ValueError("num_agents must be >= 1")

    num_envs = args.num_agents
    if args.gui and num_envs > 1:
        print("PyBullet GUI requested; forcing num_agents=1 so the visualizer stays stable.")
        num_envs = 1

    env_fn = partial(make_env, render_gui=args.gui)
    vec_env = make_vec_env(env_fn, n_envs=num_envs, seed=args.seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if args.algo == "PPO":
        model = PPO(
            "MlpPolicy", vec_env, device="auto",
            n_steps=2048, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, learning_rate=3e-4,
            policy_kwargs=dict(net_arch=[256, 256]), verbose=1, seed=args.seed
        )
    else:
        model = SAC(
            "MlpPolicy", vec_env, device="auto",
            buffer_size=500_000, batch_size=256, gamma=0.99, tau=0.02,
            learning_rate=3e-4, train_freq=64, gradient_steps=64,
            policy_kwargs=dict(net_arch=[256, 256]), verbose=1, seed=args.seed
        )

    model.learn(total_timesteps=args.total_timesteps)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_prefix = os.path.join(args.checkpoint_dir, f"{args.algo.lower()}_humanoid12")
    model.save(f"{ckpt_prefix}_model")
    vec_env.save(f"{ckpt_prefix}_vecnorm.pkl")
