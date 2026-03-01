"""Custom termination functions for Piper lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate if the object falls below minimum_height (world frame).

    This catches objects that have been knocked off the surface or fallen
    through the ground.

    Args:
        minimum_height: World-frame Z below which the episode terminates.
        object_cfg: SceneEntityCfg for the object.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, 2] < minimum_height
