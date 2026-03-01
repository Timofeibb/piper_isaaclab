"""Train a Piper arm reach policy using RSL-RL PPO.

Usage (from workspace/manipulator/, inside conda env env_isaaclab):

    # Headless training (fastest):
    python scripts/rsl_rl/train.py --task Isaac-Reach-Piper-v0 --num_envs 4096 --headless

    # With rendering:
    python scripts/rsl_rl/train.py --task Isaac-Reach-Piper-v0 --num_envs 1024

    # Resume from checkpoint:
    python scripts/rsl_rl/train.py --task Isaac-Reach-Piper-v0 --resume --load_run <run_dir>
"""

import argparse
import os
import sys

# ── Ensure the manipulator package is importable ──────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "source", "manipulator")
sys.path.insert(0, _PACKAGE_DIR)

from isaaclab.app import AppLauncher

# -- CLI ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train Piper reach with RSL-RL PPO.")
parser.add_argument("--task", type=str, default="Isaac-Reach-Piper-v0", help="Gym env id")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max training iterations")

# RSL-RL specific
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
parser.add_argument("--load_run", type=str, default=None, help="Run name to load")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint file to load")

# AppLauncher adds --headless, --device, etc.
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Imports after sim launch ──────────────────────────────────────────────
import gymnasium as gym
import torch
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

# Register Piper envs
import manipulator  # noqa: F401


def main():
    # ── Resolve configs from registry (entry point strings → actual objects) ──
    env_cfg: ManagerBasedRLEnvCfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    # Override from CLI
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    env_cfg.seed = args.seed

    # ── Create environment ──────────────────────────────────────────────────
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── Logging ─────────────────────────────────────────────────────────────
    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root = os.path.abspath(log_root)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # ── Create runner ───────────────────────────────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Resume if requested
    if args.resume:
        resume_path = os.path.join(log_root, args.load_run) if args.load_run else log_dir
        if args.load_checkpoint:
            resume_path = os.path.join(resume_path, args.load_checkpoint)
        else:
            model_files = [f for f in os.listdir(resume_path) if f.startswith("model_")]
            if model_files:
                latest = sorted(model_files)[-1]
                resume_path = os.path.join(resume_path, latest)
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(resume_path)

    # ── Train ───────────────────────────────────────────────────────────────
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
