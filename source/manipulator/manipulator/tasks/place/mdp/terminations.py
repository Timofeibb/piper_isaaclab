"""Custom termination functions for Piper place task.

Reuses lift terminations and adds place-specific ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Re-export lift termination
from manipulator.tasks.lift.mdp.terminations import object_dropped

__all__ = ["object_dropped"]
