"""Custom observation functions for Piper reach-to-object task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_mul,
    quat_conjugate,
    euler_xyz_from_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Sentinel detection ───────────────────────────────────────────────────────
# Objects placed underground (z < -1 m relative to robot root) are treated as
# "no target present". This is set by the reset_object_in_workspace event
# when no_target_prob > 0.

_NO_TARGET_Z_THRESHOLD = -1.0


def _has_target_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (N,) bool tensor: True if the object is above the sentinel threshold."""
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object"]
    obj_z = obj.data.root_pos_w[:, 2]
    root_z = robot.data.root_pos_w[:, 2]
    return (obj_z - root_z) > _NO_TARGET_Z_THRESHOLD


def has_target(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Binary flag: 1.0 if there is a target object, 0.0 if idle. Shape: (N, 1).

    The policy uses this to decide between reaching behaviour and holding
    the default (home) pose. During sim2sim deployment, set to 0.0 when
    the perception system detects no valid target in the workspace.
    """
    mask = _has_target_mask(env)
    return mask.unsqueeze(-1).float()


def object_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position expressed in the robot's root frame. Shape: (N, 3).

    Returns zeros when no target is present (object underground).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w
    )
    # Zero-out when no target
    mask = _has_target_mask(env).unsqueeze(-1)  # (N, 1)
    return obj_pos_b * mask


def ee_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    ee_body_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector position in the robot's root frame. Shape: (N, 3)."""
    robot: Articulation = env.scene[ee_body_cfg.name]

    ee_pos_w = robot.data.body_pos_w[:, ee_body_cfg.body_ids[0], :]
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w
    )
    return ee_pos_b


def gripper_finger_positions_in_robot_frame(
    env: ManagerBasedRLEnv,
    finger_l_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_r_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of both gripper fingers in robot frame. Shape: (N, 6)."""
    robot: Articulation = env.scene[finger_l_cfg.name]

    fl_pos_w = robot.data.body_pos_w[:, finger_l_cfg.body_ids[0], :]
    fr_pos_w = robot.data.body_pos_w[:, finger_r_cfg.body_ids[0], :]

    fl_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, fl_pos_w
    )
    fr_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, fr_pos_w
    )
    return torch.cat([fl_pos_b, fr_pos_b], dim=-1)


def ee_object_vector(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_body_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vector from EE to object in robot frame. Shape: (N, 3).

    Gives the policy a direct signal about the direction to move.
    Returns zeros when no target is present.
    """
    robot: Articulation = env.scene[ee_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    ee_pos_w = robot.data.body_pos_w[:, ee_body_cfg.body_ids[0], :]
    obj_pos_w = obj.data.root_pos_w[:, :3]

    # Rotate into robot body frame
    delta_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w
    )
    ee_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w
    )
    vec = delta_b - ee_b
    # Zero-out when no target
    mask = _has_target_mask(env).unsqueeze(-1)  # (N, 1)
    return vec * mask


# ── Object orientation ───────────────────────────────────────────────────────


def object_orientation_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object orientation as Euler angles (roll, pitch, yaw) in robot frame.

    Shape: (N, 3). Returns zeros when no target is present.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    obj_quat_w = obj.data.root_quat_w  # (N, 4) wxyz
    robot_quat_w = robot.data.root_quat_w  # (N, 4) wxyz

    # Object orientation relative to robot: q_rel = q_robot^-1 * q_obj
    robot_quat_inv = quat_conjugate(robot_quat_w)
    obj_quat_b = quat_mul(robot_quat_inv, obj_quat_w)

    # Convert to Euler RPY
    roll, pitch, yaw = euler_xyz_from_quat(obj_quat_b)
    rpy = torch.stack([roll, pitch, yaw], dim=-1)
    # Zero-out when no target
    mask = _has_target_mask(env).unsqueeze(-1)
    return rpy * mask


# ── Object properties (shape ID + dimensions) ───────────────────────────────


def object_properties(
    env: ManagerBasedRLEnv,
    shape_id: float = 0.0,
    dim_x: float = 0.05,
    dim_y: float = 0.05,
    dim_z: float = 0.05,
) -> torch.Tensor:
    """Static object property vector: [shape_id, dim_x, dim_y, dim_z].

    Shape: (N, 4).

    During **training** with a single known shape, pass the correct values
    as params. During **sim2sim deployment**, override these with values
    from the perception pipeline.

    Shape ID convention:
      1 = cube_small, 2 = cube_medium, 3 = sphere, 4 = cylinder, 5 = stick

    Dimensions are the bounding-box extents (metres) in the object's
    local X, Y, Z axes.
    """
    props = torch.tensor(
        [shape_id, dim_x, dim_y, dim_z],
        device=env.device, dtype=torch.float32,
    )
    return props.unsqueeze(0).expand(env.num_envs, -1)
