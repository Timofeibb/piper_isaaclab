"""Custom event functions for Piper reach-to-object task.

Events:
  - reset_object_in_workspace: Randomise object position within workspace on reset
  - randomize_base_tilt: Apply small roll/pitch to arm base (uneven terrain)

IMPORTANT: ``write_root_pose_to_sim`` expects **world-frame** positions.
``default_root_state`` and locally-sampled positions are **env-local**, so
``env.scene.env_origins[env_ids]`` must be added before writing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_object_in_workspace(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    r_range: tuple[float, float] = (0.10, 0.60),
    theta_range: tuple[float, float] = (-2.53, 2.08),
    z_range: tuple[float, float] = (0.02, 0.08),
    no_target_prob: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Teleport the target object to a random position within the arm workspace.

    Samples in **polar coordinates** (r, theta) to cover the full reachable
    annulus around the arm base, minus the dead zone behind the arm where
    joint 1 cannot reach.

    Joint 1 limits: [-2.618, +2.168] rad = [-150°, +124°].
    Dead zone: ~86° arc behind the arm. Default theta_range adds ~5° safety
    margin inside each joint limit.

    Radial sampling uses sqrt(uniform) to get uniform area density in the
    annulus (avoids over-sampling near the centre).

    Positions are sampled in env-local frame and converted to world frame.
    The Z coordinate is absolute (relative to env origin / ground plane).

    When *no_target_prob* > 0, a fraction of environments will have the object
    placed underground (z = -5 m) to simulate "no target present".

    Args:
        r_range: (min, max) radial distance in metres.
        theta_range: (min, max) azimuth angle in radians (0 = X+ / forward).
                     Defaults to ~5° inside joint 1 limits.
        z_range: (min, max) height above env origin in metres.
        no_target_prob: Fraction of envs with no target (object underground).
        object_cfg: SceneEntityCfg for the target object.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    num = len(env_ids)
    device = env.device

    # Polar sampling with uniform area density: r = sqrt(U) * (r_max - r_min) + r_min
    # More precisely: r = sqrt(U * (r_max^2 - r_min^2) + r_min^2)
    r_min, r_max = r_range
    u = torch.rand(num, device=device)
    r = torch.sqrt(u * (r_max**2 - r_min**2) + r_min**2)
    theta = torch.empty(num, device=device).uniform_(*theta_range)

    # Convert to Cartesian (env-local frame, arm base at origin)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = torch.empty(num, device=device).uniform_(*z_range)
    pos_local = torch.stack([x, y, z], dim=-1)

    # With probability no_target_prob, hide the object underground
    if no_target_prob > 0.0:
        no_target_mask = torch.rand(num, device=device) < no_target_prob
        pos_local[no_target_mask, 0] = 0.0
        pos_local[no_target_mask, 1] = 0.0
        pos_local[no_target_mask, 2] = -5.0

    # Convert to world frame
    pos_world = pos_local + env.scene.env_origins[env_ids]

    # Identity quaternion (object upright)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(num, -1).clone()

    # Zero velocity
    zeros = torch.zeros(num, 3, device=device)

    # Write to object (world frame)
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
    """Randomise the arm base height and orientation on each reset.

    Simulates the arm being mounted on a rover at varying deck heights
    and on uneven terrain, without needing the full rover geometry.

    Args:
        z_range: Offset to add to the default Z position (metres).
                 E.g. (-0.02, 0.02) → ±20 mm around the default 200 mm.
        roll_range: Roll in radians. ±0.05 ≈ ±3°.
        pitch_range: Pitch in radians. ±0.05 ≈ ±3°.
        yaw_range: Yaw in radians. ±0.1 ≈ ±6° (rover heading variation).
        asset_cfg: SceneEntityCfg pointing to the robot articulation.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    num = len(env_ids)
    device = env.device

    # ── Position: default + random Z offset ──────────────────────────────
    default_pos = robot.data.default_root_state[env_ids, :3].clone()
    z_offset = torch.empty(num, device=device).uniform_(*z_range)
    default_pos[:, 2] += z_offset

    # Convert to world frame
    pos_world = default_pos + env.scene.env_origins[env_ids]

    # ── Orientation: random roll / pitch / yaw ───────────────────────────
    roll = torch.empty(num, device=device).uniform_(*roll_range)
    pitch = torch.empty(num, device=device).uniform_(*pitch_range)
    yaw = torch.empty(num, device=device).uniform_(*yaw_range)

    delta_quat = quat_from_euler_xyz(roll, pitch, yaw)
    default_quat = robot.data.default_root_state[env_ids, 3:7].clone()
    new_quat = quat_mul(delta_quat, default_quat)

    # ── Write to sim (world frame) ───────────────────────────────────────
    root_pose = torch.cat([pos_world, new_quat], dim=-1)
    robot.write_root_pose_to_sim(root_pose, env_ids)

    # Zero velocity
    zeros = torch.zeros(num, 6, device=device)
    robot.write_root_velocity_to_sim(zeros, env_ids)
