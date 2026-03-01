"""Reach-to-object environment for the Piper 6-DOF arm.

Clean reach task matching the proven IsaacLab pattern:
  - Arm-only control (no gripper action)
  - Minimal observations: joint state + object position + actions
  - Simple reward: tanh reach + L2 reach + smoothness penalties
  - Physical target object spawned in polar workspace
  - Base pose randomization (height + tilt)

This is Step 1 of the manipulation progression:
  Step 1: Reach  — learn arm control to reach target positions
  Step 2: Lift   — add binary gripper, learn grasp from lift reward
  Step 3: Place  — full pick-and-place with staged rewards
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from manipulator.tasks.reach import mdp
from manipulator.assets.piper import PIPER_ARM_CFG
from manipulator.assets.target_objects import TARGET_OBJECT_CFG


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

_EE_BODY = "link_end_effector"
_ARM_JOINTS = ["joint[1-6]"]


# ═════════════════════════════════════════════════════════════════════════════
# Scene
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class PiperReachSceneCfg(InteractiveSceneCfg):
    """Scene: Piper arm (fixed base) + physical target object."""

    robot: ArticulationCfg = PIPER_ARM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    object: RigidObjectCfg = TARGET_OBJECT_CFG.copy()

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Actions — arm only, NO gripper
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class ActionsCfg:
    """6 arm joints (position control). No gripper action."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_ARM_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Commands — null (object position IS the target)
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class CommandsCfg:
    null = mdp.NullCommandCfg()


# ═════════════════════════════════════════════════════════════════════════════
# Observations — minimal, matching IsaacLab reach pattern
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Minimal obs: joint state + object pos + ee-to-object + actions.

        joint_pos_rel (8D) + joint_vel_rel (8D) +
        object_pos (3D) + ee_to_object (3D) + last_action (6D) = 28D
        """

        # Joint positions relative to default (8D: 6 arm + 2 gripper)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # Joint velocities (8D)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # Object position in robot root frame (3D)
        object_pos = ObsTerm(
            func=mdp.object_position_in_robot_frame,
        )

        # Vector from EE to object in robot frame (3D)
        ee_to_object = ObsTerm(
            func=mdp.ee_object_vector,
            params={
                "ee_body_cfg": SceneEntityCfg("robot", body_names=[_EE_BODY]),
            },
        )

        # Previous actions (6D — arm only)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ═════════════════════════════════════════════════════════════════════════════
# Events
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class EventCfg:

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Polar sampling: full reachable annulus minus dead zone behind arm
    reset_object = EventTerm(
        func=mdp.reset_object_in_workspace,
        mode="reset",
        params={
            "r_range": (0.10, 0.60),
            "theta_range": (-2.53, 2.08),
            "z_range": (0.02, 0.08),
            "no_target_prob": 0.0,
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    # Base height + tilt randomization
    randomize_base = EventTerm(
        func=mdp.randomize_base_pose,
        mode="reset",
        params={
            "z_range": (-0.02, 0.02),
            "roll_range": (-0.05, 0.05),
            "pitch_range": (-0.05, 0.05),
            "yaw_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# Rewards — clean, minimal, matching IsaacLab reach
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class RewardsCfg:
    """4 reward terms — same structure as the working IsaacLab reach task."""

    # Dense tanh reward: 1 when at object, decays with distance
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={
            "std": 0.1,
            "ee_body_cfg": SceneEntityCfg("robot", body_names=[_EE_BODY]),
        },
    )

    # L2 penalty: linear push toward object
    reaching_object_coarse = RewTerm(
        func=mdp.object_ee_distance_l2,
        weight=-0.2,
        params={
            "ee_body_cfg": SceneEntityCfg("robot", body_names=[_EE_BODY]),
        },
    )

    # Smoothness penalties (start small, ramp via curriculum)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Terminations — time-out only
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ═════════════════════════════════════════════════════════════════════════════
# Curriculum — ramp smoothness penalties
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class CurriculumCfg:

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Top-level environment config
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class PiperReachEnvCfg(ManagerBasedRLEnvCfg):
    """Piper arm reach-to-object — Step 1 of manipulation progression."""

    scene: PiperReachSceneCfg = PiperReachSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.sim.dt = 1.0 / 60.0
        self.decimation = 2
        self.episode_length_s = 12.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.3)


@configclass
class PiperReachEnvCfg_PLAY(PiperReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
