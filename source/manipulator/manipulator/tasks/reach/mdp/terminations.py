"""Custom termination functions for Piper reach-to-object task.

Terminations:
  - height_limit_exceeded: Any arm body above a hard ceiling → episode ends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def height_limit_exceeded(
    env: ManagerBasedRLEnv,
    max_height: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate the episode if ANY checked arm body exceeds *max_height*.

    This is the hard termination companion to the soft
    ``height_violation_penalty`` reward. Set *max_height* slightly above
    the reward's ``max_height`` to give the penalty room to teach before
    the hard cut.

    Heights are measured **relative to the robot root** (arm_base).

    Args:
        max_height: Height above robot root in metres. Default 0.42 m.
        asset_cfg:  SceneEntityCfg with body_ids covering all arm links.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Z of all checked bodies: (N, num_bodies)
    body_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # Robot root Z: (N, 1)
    root_z = robot.data.root_pos_w[:, 2:3]
    # Height above arm_base
    relative_z = body_z - root_z  # (N, num_bodies)

    # True if ANY body exceeds the ceiling
    exceeded = (relative_z > max_height).any(dim=-1)  # (N,)
    return exceeded
