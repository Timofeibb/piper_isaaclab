"""Custom reward functions for Piper place task.

Extends the lift rewards with placement-specific rewards:
  - object_ee_distance: (from lift) tanh reaching reward
  - object_is_lifted: (from lift) binary lift gate
  - object_goal_distance: (from lift) goal tracking gated on lift
  - object_placed_at_goal: NEW — reward for placing object at goal position
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Re-export all lift rewards (reaching, lift, goal tracking)
from manipulator.tasks.lift.mdp.rewards import (
    object_ee_distance,
    object_ee_distance_l2,
    object_is_lifted,
    object_goal_distance,
)

__all__ = [
    "object_ee_distance",
    "object_ee_distance_l2",
    "object_is_lifted",
    "object_goal_distance",
    "object_goal_distance_xy",
    "object_transport_to_goal_xy",
    "object_placed_at_goal",
    "gripper_release_near_goal",
]


def object_goal_distance_xy(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Tanh-kernel reward for object-to-goal XY distance, gated on lift.

    Same as ``object_goal_distance`` but uses **XY distance only**, ignoring
    the Z axis.  This is critical for the place task where the goal is at
    ground level — using 3D distance would create an impossible optimization
    (object must be lifted AND close to a ground-level goal).

    Args:
        std: Tanh kernel standard deviation.
        minimal_height: Object must be above this height to get reward.
        command_name: Name of the command providing goal pose.
        robot_cfg: SceneEntityCfg for the robot.
        object_cfg: SceneEntityCfg for the object.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )

    obj_xy = obj.data.root_pos_w[:, :2]
    goal_xy = goal_pos_w[:, :2]
    dist_xy = torch.norm(obj_xy - goal_xy, dim=-1)

    obj_z = obj.data.root_pos_w[:, 2]
    is_lifted = (obj_z > minimal_height).float()

    return is_lifted * (1.0 - torch.tanh(dist_xy / std))


def object_transport_to_goal_xy(
    env: ManagerBasedRLEnv,
    std: float,
    lift_threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Tanh-kernel reward for object-to-goal XY distance, gated on **was_lifted**.

    Unlike ``object_goal_distance_xy`` which requires the object to be
    *currently* lifted (``is_lifted``), this requires only that the object
    *was* lifted at some point during the current episode (``was_lifted``).

    This is critical for the place task: the transport reward stays active
    while the policy lowers the object toward the ground, creating a smooth
    reward gradient from “hovering above goal” through to “placed at goal”.
    Without this, there is a reward desert between the lift zone and the
    placement zone where the policy gets zero reward.

    Args:
        std: Tanh kernel standard deviation.
        lift_threshold: Object must have exceeded this z at least once.
        command_name: Name of the command providing goal pose.
        robot_cfg: SceneEntityCfg for the robot.
        object_cfg: SceneEntityCfg for the object.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2]

    # Reuse the shared was_lifted buffer
    if not hasattr(env, "_place__was_lifted"):
        env._place__was_lifted = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    new_eps = env.episode_length_buf <= 1
    env._place__was_lifted[new_eps] = False
    env._place__was_lifted |= obj_z > lift_threshold

    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )

    obj_xy = obj.data.root_pos_w[:, :2]
    goal_xy = goal_pos_w[:, :2]
    dist_xy = torch.norm(obj_xy - goal_xy, dim=-1)

    was_lifted = env._place__was_lifted.float()
    return was_lifted * (1.0 - torch.tanh(dist_xy / std))


def object_placed_at_goal(
    env: ManagerBasedRLEnv,
    std: float,
    max_object_height: float,
    lift_threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Tanh-kernel reward for placing object at the goal position.

    **Triple-gated** to prevent "do nothing" exploit:
      1. Object must have been lifted above ``lift_threshold`` at some point
         during the current episode (tracked via ``_place__was_lifted`` buffer).
      2. Object must currently be back on the surface (z < ``max_object_height``).
      3. Object must be near the goal in the XY plane.

    Without gate #1 the reward fires at episode start (object spawns on the
    ground and may be near the goal by chance), teaching the policy to never
    move.

    Args:
        std: Tanh kernel standard deviation.
        max_object_height: Object z must be below this to get reward.
        lift_threshold: Object must have exceeded this z at least once.
        command_name: Name of the command providing goal pose.
        robot_cfg: SceneEntityCfg for the robot.
        object_cfg: SceneEntityCfg for the object.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2]

    # ── Per-env "was lifted" tracking ──────────────────────────────────
    if not hasattr(env, "_place__was_lifted"):
        env._place__was_lifted = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    # Reset flag for freshly-started episodes (episode_length_buf == 1
    # is the first step after a reset)
    new_eps = env.episode_length_buf <= 1
    env._place__was_lifted[new_eps] = False
    # Mark as lifted when object exceeds threshold
    env._place__was_lifted |= obj_z > lift_threshold

    # ── Goal XY proximity ──────────────────────────────────────────────
    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )

    obj_xy = obj.data.root_pos_w[:, :2]
    goal_xy = goal_pos_w[:, :2]
    dist_xy = torch.norm(obj_xy - goal_xy, dim=-1)
    xy_reward = 1.0 - torch.tanh(dist_xy / std)

    # ── Triple gate ────────────────────────────────────────────────────
    was_lifted = env._place__was_lifted.float()
    on_surface = (obj_z < max_object_height).float()

    return was_lifted * on_surface * xy_reward


def gripper_release_near_goal(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    command_name: str,
    open_threshold: float = 0.03,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for opening the gripper when the object is near the goal XY.

    Provides an **immediate** learning signal for the release action,
    bridging the temporal credit-assignment gap between "holding above goal"
    and "object resting at goal".  Without this, the policy never discovers
    that releasing is beneficial because the one-step loss from dropping
    lift+transport reward is immediate, while placement reward is delayed
    until the object settles on the surface.

    Checks **both** gripper fingers:
      - joint7 opens to +0.05  → open when > +open_threshold
      - joint8 opens to -0.05  → open when < -open_threshold

    Fires when ALL of:
      - Object was lifted this episode (prevents "never pick up" exploit)
      - Object XY is within ``xy_threshold`` of goal XY
      - Both gripper joints are in the open position

    Args:
        xy_threshold: Max XY distance from goal for the reward to fire.
        command_name: Name of the command providing goal pose.
        open_threshold: Absolute joint displacement beyond which a finger
            is considered open (default 0.03 for a 0.05 range).
        robot_cfg: SceneEntityCfg for the robot.
        object_cfg: SceneEntityCfg for the object.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # ── Was-lifted gate (reuse existing buffer) ────────────────────────
    if not hasattr(env, "_place__was_lifted"):
        env._place__was_lifted = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    was_lifted = env._place__was_lifted.float()

    # ── Object near goal XY ────────────────────────────────────────────
    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )
    obj_xy = obj.data.root_pos_w[:, :2]
    goal_xy = goal_pos_w[:, :2]
    dist_xy = torch.norm(obj_xy - goal_xy, dim=-1)
    near_goal = (dist_xy < xy_threshold).float()

    # ── Both gripper fingers open ──────────────────────────────────────
    # joint7: open = +0.05, closed = 0.0  → open when > +threshold
    # joint8: open = -0.05, closed = 0.0  → open when < -threshold
    j7_idx = robot.find_joints("joint7")[0][0]
    j8_idx = robot.find_joints("joint8")[0][0]
    j7_open = (robot.data.joint_pos[:, j7_idx] > open_threshold).float()
    j8_open = (robot.data.joint_pos[:, j8_idx] < -open_threshold).float()
    gripper_open = j7_open * j8_open  # both must be open

    return was_lifted * near_goal * gripper_open
