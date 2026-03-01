"""Custom observation functions for Piper lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position expressed in the robot's root frame. Shape: (N, 3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w
    )
    return obj_pos_b


def ee_object_vector(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_body_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vector from EE to object in robot frame. Shape: (N, 3)."""
    robot: Articulation = env.scene[ee_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    ee_pos_w = robot.data.body_pos_w[:, ee_body_cfg.body_ids[0], :]
    obj_pos_w = obj.data.root_pos_w[:, :3]

    # Transform both to robot body frame
    delta_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w
    )
    ee_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w
    )
    return delta_b - ee_b
