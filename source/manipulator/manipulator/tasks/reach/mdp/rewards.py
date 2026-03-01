"""Custom reward functions for Piper reach-to-object task.

Rewards:
  - EE → object distance (tanh-kernel, primary reaching signal)
  - Gripper alignment (EE pointing toward object)
  - Object within gripper width (bonus when approach vector is good)
  - Default pose tracking (penalise deviation when far from any object)
  - Idle pose holding (strong bonus for holding default pose when no target)
  - Height violation penalty (arm bodies above workspace ceiling)
  - Smoothness penalties (action rate, joint velocity, joint acceleration)
  - Joint limit proximity penalty

All reaching/grasping rewards are automatically masked to zero when the
target object is placed underground (no-target sentinel), so the policy
learns to remain idle when ``has_target == 0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Sentinel detection (shared with observations.py) ────────────────────────

_NO_TARGET_Z_THRESHOLD = -1.0


def _has_target_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (N,) bool: True if the object is above the sentinel threshold."""
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object"]
    obj_z = obj.data.root_pos_w[:, 2]
    root_z = robot.data.root_pos_w[:, 2]
    return (obj_z - root_z) > _NO_TARGET_Z_THRESHOLD


# ── Reaching ─────────────────────────────────────────────────────────────────


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_body_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tanh-kernel reward for EE-to-object distance.

    Returns 1 when EE is at object centre, decays with distance.
    Masked to 0 when no target is present.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[ee_body_cfg.name]

    ee_pos_w = robot.data.body_pos_w[:, ee_body_cfg.body_ids[0], :]
    obj_pos_w = obj.data.root_pos_w[:, :3]

    dist = torch.norm(ee_pos_w - obj_pos_w, dim=-1)
    reward = 1.0 - torch.tanh(dist / std)
    return reward * _has_target_mask(env).float()


def object_ee_distance_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_body_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Raw L2 distance from EE to object. Masked to 0 when no target."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[ee_body_cfg.name]

    ee_pos_w = robot.data.body_pos_w[:, ee_body_cfg.body_ids[0], :]
    obj_pos_w = obj.data.root_pos_w[:, :3]

    dist = torch.norm(ee_pos_w - obj_pos_w, dim=-1)
    return dist * _has_target_mask(env).float()


# ── Gripper alignment ───────────────────────────────────────────────────────


def gripper_object_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    finger_l_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_r_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for aligning the gripper midpoint with the object.

    Masked to 0 when no target is present.
    """
    robot: Articulation = env.scene[finger_l_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    finger_l_pos = robot.data.body_pos_w[:, finger_l_cfg.body_ids[0], :]
    finger_r_pos = robot.data.body_pos_w[:, finger_r_cfg.body_ids[0], :]
    grip_midpoint = 0.5 * (finger_l_pos + finger_r_pos)

    obj_pos_w = obj.data.root_pos_w[:, :3]
    dist = torch.norm(grip_midpoint - obj_pos_w, dim=-1)
    reward = 1.0 - torch.tanh(dist / std)
    return reward * _has_target_mask(env).float()


# ── Default pose ─────────────────────────────────────────────────────────────


def default_pose_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 deviation of joint positions from the asset's default (home) pose.

    Used as a penalty (negative weight) to encourage the arm to stay near
    its rest configuration when not actively reaching.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    default_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(current_pos - default_pos), dim=-1)


# ── Joint limits proximity ──────────────────────────────────────────────────


def joint_limit_proximity(
    env: ManagerBasedRLEnv,
    threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty when joints are within *threshold* (fraction) of their limits.

    Returns sum of per-joint penalties. Each joint contributes when it's
    within threshold*range of either limit.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    lo = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    hi = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    rng = hi - lo

    margin = threshold * rng
    # distance INTO the margin zone (0 when safe, >0 when close to limit)
    lower_pen = torch.clamp(margin - (pos - lo), min=0.0) / (margin + 1e-6)
    upper_pen = torch.clamp(margin - (hi - pos), min=0.0) / (margin + 1e-6)

    return torch.sum(lower_pen + upper_pen, dim=-1)


# ── Object in gripper zone ──────────────────────────────────────────────────


def object_in_gripper_zone(
    env: ManagerBasedRLEnv,
    threshold: float = 0.03,
    finger_l_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_r_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary reward: 1 when object centre is within *threshold* of gripper midpoint.

    Masked to 0 when no target is present.
    """
    robot: Articulation = env.scene[finger_l_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    finger_l_pos = robot.data.body_pos_w[:, finger_l_cfg.body_ids[0], :]
    finger_r_pos = robot.data.body_pos_w[:, finger_r_cfg.body_ids[0], :]
    grip_midpoint = 0.5 * (finger_l_pos + finger_r_pos)

    obj_pos_w = obj.data.root_pos_w[:, :3]
    dist = torch.norm(grip_midpoint - obj_pos_w, dim=-1)
    reward = torch.where(dist < threshold, 1.0, 0.0)
    return reward * _has_target_mask(env).float()


# ── Height violation ─────────────────────────────────────────────────────────


def height_violation_penalty(
    env: ManagerBasedRLEnv,
    max_height: float = 0.40,
    soft_start: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for arm bodies exceeding the workspace height ceiling.

    Returns 0 when all bodies are below *soft_start*, then ramps linearly
    up to 1.0 at *max_height* and beyond. The check runs over ALL body_ids
    specified in asset_cfg (should include link1–link6, gripper_base, EE).

    The penalty is the **maximum** violation across all checked bodies,
    so a single body going high is enough to trigger it.

    Heights are measured **relative to the robot root** (arm_base), so the
    limit is invariant to environment placement and terrain tilt.

    Args:
        max_height: Height above robot root in metres. Default 0.40 m.
        soft_start: Height at which the penalty begins ramping.
        asset_cfg:  SceneEntityCfg with body_ids set to the arm bodies.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Z positions of all checked bodies: (N, num_bodies)
    body_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # Robot root Z: (N, 1) — arm_base height in world frame
    root_z = robot.data.root_pos_w[:, 2:3]
    # Height above arm_base
    relative_z = body_z - root_z  # (N, num_bodies)

    # Worst (highest) body per environment
    max_z, _ = relative_z.max(dim=-1)  # (N,)

    # Ramp: 0 below soft_start, linear to 1 at max_height, clamped at 1
    margin = max_height - soft_start
    penalty = torch.clamp((max_z - soft_start) / (margin + 1e-6), min=0.0, max=1.0)
    return penalty


# ── Idle pose holding (no-target environments) ──────────────────────────────


def idle_pose_holding(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positive reward for staying near the default pose when no target is present.

    Analogous to the "stand still" reward in legged locomotion: when the
    command is zero (no target), the best action is to hold the home pose.

    Returns a reward in [0, 1]:
      - 1.0 when all arm joints are exactly at default pose
      - Decays via exp(-deviation / threshold)
      - Only active when ``has_target == False``; returns 0 for target envs.

    Args:
        threshold: Controls sensitivity. Lower = stricter. Default 0.05 rad.
        asset_cfg: SceneEntityCfg with joint_ids set to the arm joints.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    default_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]

    deviation = torch.sum(torch.square(current_pos - default_pos), dim=-1)
    reward = torch.exp(-deviation / (threshold**2))

    # Only active when no target
    no_target = (~_has_target_mask(env)).float()
    return reward * no_target


# ── Joint acceleration (smoothness) ─────────────────────────────────────────


def joint_acceleration_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 norm of joint accelerations (finite-differenced from velocity).

    Penalises rapid changes in joint velocity, encouraging smooth motion.
    Uses ``data.joint_acc`` which is the finite-differenced acceleration
    already computed by the physics engine.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_acc[:, asset_cfg.joint_ids]), dim=-1)


def joint_jerk_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 norm of joint jerk (change in acceleration between steps).

    Even stronger smoothness signal than acceleration — penalises sudden
    changes in acceleration. Requires storing previous-step acceleration
    in a buffer; on first call the jerk is zero.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    acc = robot.data.joint_acc[:, asset_cfg.joint_ids]

    # Store previous acceleration in the environment extras dict
    key = f"_prev_joint_acc_{asset_cfg.name}"
    if key not in env.extras:
        env.extras[key] = torch.zeros_like(acc)

    prev_acc = env.extras[key]
    jerk = acc - prev_acc
    env.extras[key] = acc.clone()

    return torch.sum(torch.square(jerk), dim=-1)
