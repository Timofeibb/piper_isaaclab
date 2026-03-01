"""Custom reward functions for Piper lift task.

Rewards:
  - object_ee_distance: tanh-kernel reaching reward (reused from reach)
  - object_is_lifted: binary reward when object is above threshold height
  - object_goal_distance: tanh-kernel tracking reward gated on lift height
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_body_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tanh-kernel reward for EE-to-object distance.

    Returns 1 when EE is at object centre, decays with distance.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[ee_body_cfg.name]

    ee_pos_w = robot.data.body_pos_w[:, ee_body_cfg.body_ids[0], :]
    obj_pos_w = obj.data.root_pos_w[:, :3]

    dist = torch.norm(ee_pos_w - obj_pos_w, dim=-1)
    return 1.0 - torch.tanh(dist / std)


def object_ee_distance_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_body_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Raw L2 distance from EE to object."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[ee_body_cfg.name]

    ee_pos_w = robot.data.body_pos_w[:, ee_body_cfg.body_ids[0], :]
    obj_pos_w = obj.data.root_pos_w[:, :3]

    return torch.norm(ee_pos_w - obj_pos_w, dim=-1)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary reward: 1.0 if object is above minimal_height, else 0.0.

    This is the key lift reward — the policy must learn to reach, grasp,
    and lift the object to get this reward. No explicit grasp reward needed.

    Args:
        minimal_height: Height threshold (world frame) above which the object
            is considered "lifted". Should be above the table/spawn surface.
        object_cfg: SceneEntityCfg for the object.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2]
    return (obj_z > minimal_height).float()


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Tanh-kernel reward for object-to-goal distance, gated on lift height.

    Only gives reward when the object is above minimal_height. This teaches
    the policy to first lift, then move toward the goal.

    The command provides the goal position in robot body frame; we convert
    to world frame for comparison with the object.

    Args:
        std: Tanh kernel standard deviation.
        minimal_height: Object must be above this height to get reward.
        command_name: Name of the command providing goal pose.
        robot_cfg: SceneEntityCfg for the robot.
        object_cfg: SceneEntityCfg for the object.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Goal position from command (in robot body frame)
    command = env.command_manager.get_command(command_name)
    # command shape: (N, 7) — [pos_x, pos_y, pos_z, qw, qx, qy, qz]
    goal_pos_b = command[:, :3]

    # Convert goal from body frame to world frame
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )

    # Object position in world frame
    obj_pos_w = obj.data.root_pos_w[:, :3]

    # Distance to goal
    dist = torch.norm(goal_pos_w - obj_pos_w, dim=-1)

    # Gate on height — only reward tracking when object is lifted
    is_lifted = (obj_pos_w[:, 2] > minimal_height).float()

    return is_lifted * (1.0 - torch.tanh(dist / std))
