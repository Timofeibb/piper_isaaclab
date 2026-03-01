"""Play (evaluate) a trained Piper arm policy and export deployment models.

Runs the trained policy in the simulator for visual evaluation, then
exports the model as:
  - TorchScript JIT  (.pt)  — for C++ / LibTorch deployment
  - ONNX             (.onnx) — for Python / cross-platform inference

Exported models are saved to:
  logs/rsl_rl/<experiment>/<run>/exported/policy.pt
  logs/rsl_rl/<experiment>/<run>/exported/policy.onnx

Usage (from workspace/manipulator/, inside conda env env_isaaclab):

    # Play latest checkpoint + export:
    python scripts/rsl_rl/play.py --task Isaac-Place-Piper-Play-v0

    # Export only (skip simulation loop):
    python scripts/rsl_rl/play.py --task Isaac-Place-Piper-Play-v0 --export_only

    # Specific run directory:
    python scripts/rsl_rl/play.py --load_run 2026-02-22_16-43-40

    # Specific checkpoint in a specific run:
    python scripts/rsl_rl/play.py --load_run 2026-02-22_16-43-40 --load_checkpoint model_500.pt
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "source", "manipulator")
sys.path.insert(0, _PACKAGE_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Piper policy and export deployment models.")
parser.add_argument("--task", type=str, default="Isaac-Reach-Piper-Play-v0", help="Gym env id")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments")
parser.add_argument("--load_run", type=str, default=None, help="Run directory name (default: latest)")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint file (default: latest)")
parser.add_argument("--export_only", action="store_true", default=False, help="Export model and exit (skip sim loop)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import manipulator  # noqa: F401


def _find_latest_run(log_root: str) -> str:
    """Find the most recent run directory by sorting alphabetically (timestamp-based names)."""
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"Log root does not exist: {log_root}")
    runs = [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]
    if not runs:
        raise FileNotFoundError(f"No run directories found in: {log_root}")
    return sorted(runs)[-1]


def _find_latest_checkpoint(run_dir: str) -> str:
    """Find the latest model_*.pt checkpoint in a run directory."""
    model_files = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not model_files:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in: {run_dir}")
    # Sort numerically by iteration number
    model_files.sort(key=lambda f: int(f.replace("model_", "").replace(".pt", "")))
    return model_files[-1]


def main():
    env_cfg: ManagerBasedRLEnvCfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── Resolve checkpoint path ─────────────────────────────────────────────
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    # Auto-find latest run if not specified
    if args.load_run is None:
        run_name = _find_latest_run(log_root)
        print(f"[INFO] Auto-selected latest run: {run_name}")
    else:
        run_name = args.load_run

    run_dir = os.path.join(log_root, run_name)

    # Auto-find latest checkpoint if not specified
    if args.load_checkpoint is None:
        ckpt_name = _find_latest_checkpoint(run_dir)
        print(f"[INFO] Auto-selected latest checkpoint: {ckpt_name}")
    else:
        ckpt_name = args.load_checkpoint

    resume_path = os.path.join(run_dir, ckpt_name)
    print(f"[INFO] Loading: {resume_path}")

    # ── Load policy ─────────────────────────────────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ── Export models for deployment ────────────────────────────────────────
    # Extract the neural network module
    try:
        policy_nn = runner.alg.policy  # rsl_rl >= 2.3
    except AttributeError:
        policy_nn = runner.alg.actor_critic  # rsl_rl < 2.3

    # Extract the observation normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_dir = os.path.join(run_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    # TorchScript JIT — for C++ / LibTorch deployment
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    print(f"[INFO] Exported JIT model  → {os.path.join(export_dir, 'policy.pt')}")

    # ONNX — for Python / cross-platform inference
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    print(f"[INFO] Exported ONNX model → {os.path.join(export_dir, 'policy.onnx')}")

    if args.export_only:
        print("[INFO] --export_only: skipping simulation loop.")
        env.close()
        return

    # ── Run policy in simulator ─────────────────────────────────────────────
    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
