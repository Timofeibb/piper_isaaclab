"""Custom event functions for Piper lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_object_on_surface(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (0.15, 0.45),
    y_range: tuple[float, float] = (-0.20, 0.20),
    z_height: float = 0.22,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset object to a random position on the virtual surface in front of the arm.

    For the Piper arm on a rover deck (base at z=0.2), objects should spawn
    at deck height. Uses rectangular sampling in the reachable front area.

    Args:
        x_range: (min, max) X position in env-local frame.
        y_range: (min, max) Y position in env-local frame.
        z_height: Height of the surface (world frame = env_origin_z + z_height).
        object_cfg: SceneEntityCfg for the object.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    num = len(env_ids)
    device = env.device

    x = torch.empty(num, device=device).uniform_(*x_range)
    y = torch.empty(num, device=device).uniform_(*y_range)
    z = torch.full((num,), z_height, device=device)
    pos_local = torch.stack([x, y, z], dim=-1)

    # Convert to world frame
    pos_world = pos_local + env.scene.env_origins[env_ids]

    # Identity quaternion (object upright)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(num, -1).clone()

    # Zero velocity
    zeros = torch.zeros(num, 3, device=device)

    obj.write_root_pose_to_sim(torch.cat([pos_world, quat], dim=-1), env_ids)
    obj.write_root_velocity_to_sim(torch.cat([zeros, zeros], dim=-1), env_ids)


def randomize_base_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    z_range: tuple[float, float] = (-0.02, 0.02),
    roll_range: tuple[float, float] = (-0.05, 0.05),
    pitch_range: tuple[float, float] = (-0.05, 0.05),
    yaw_range: tuple[float, float] = (-0.1, 0.1),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Randomize the robot base height and orientation on reset.

    Same as the reach task version — adds robustness to deployment.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    num = len(env_ids)
    device = env.device

    # Start from default root state
    default_state = robot.data.default_root_state[env_ids].clone()

    # Randomize Z offset
    dz = torch.empty(num, device=device).uniform_(*z_range)
    default_state[:, 2] += dz

    # Randomize orientation (small roll/pitch/yaw)
    roll = torch.empty(num, device=device).uniform_(*roll_range)
    pitch = torch.empty(num, device=device).uniform_(*pitch_range)
    yaw = torch.empty(num, device=device).uniform_(*yaw_range)
    delta_quat = quat_from_euler_xyz(roll, pitch, yaw)

    # Compose with default quaternion
    default_quat = default_state[:, 3:7]
    new_quat = quat_mul(delta_quat, default_quat)
    default_state[:, 3:7] = new_quat

    # Add env origins for world frame
    default_state[:, :3] += env.scene.env_origins[env_ids]

    robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
