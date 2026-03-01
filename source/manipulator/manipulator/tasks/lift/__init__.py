"""Lift task for the Piper 6-DOF arm with gripper.

Registers Gym environments:
  - Isaac-Lift-Piper-v0         (joint position control, training)
  - Isaac-Lift-Piper-Play-v0    (joint position control, evaluation)
"""

import gymnasium as gym

from . import agents

##
# Joint Position Control
##

gym.register(
    id="Isaac-Lift-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_env_cfg:PiperLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiperLiftPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lift-Piper-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_env_cfg:PiperLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiperLiftPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
