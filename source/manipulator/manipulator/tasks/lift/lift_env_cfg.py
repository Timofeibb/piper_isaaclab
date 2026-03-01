"""Lift environment for the Piper 6-DOF arm with gripper.

Step 2 of the manipulation progression:
  Step 1: Reach  — learn arm control to reach target positions
  Step 2: Lift   — add binary gripper, learn grasp from lift reward
  Step 3: Place  — full pick-and-place with staged rewards

The policy learns to reach, grasp, and lift a rigid object to a
commanded goal position. Grasping emerges naturally from the lift
reward — no explicit grasp reward is needed.

Key differences from reach:
  - 7D action: 6 arm joints + 1 binary gripper (open/close)
  - UniformPoseCommand provides a random goal position above the deck
  - Rewards: reaching + lift (binary) + goal tracking (gated on lift)
  - Termination: time-out + object dropped below surface
  - Shorter episodes (5s vs 12s)
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

from manipulator.tasks.lift import mdp
from manipulator.assets.piper import PIPER_ARM_CFG
from manipulator.assets.target_objects import TARGET_OBJECT_CFG


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

_EE_BODY = "link_end_effector"
_ARM_JOINTS = ["joint[1-6]"]
_GRIPPER_JOINTS = ["joint[7-8]"]

# No physical table — objects rest on the ground plane (z=0).
# Spawn slightly above ground so object settles naturally.
_OBJECT_SPAWN_Z = 0.02
# Object is "lifted" when it's above this height (ground + margin) 0.10 m is enough to clear the gripper fingers, but we add some margin to be robust to physics imperfections.
_LIFT_HEIGHT = 0.25


# ═════════════════════════════════════════════════════════════════════════════
# Scene
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class PiperLiftSceneCfg(InteractiveSceneCfg):
    """Scene: Piper arm (fixed base) + graspable rigid object."""

    robot: ArticulationCfg = PIPER_ARM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Randomized objects (5 shapes: cubes, sphere, cylinder, stick)
    object: RigidObjectCfg = TARGET_OBJECT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.30, 0.0, _OBJECT_SPAWN_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

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
# Actions — arm + binary gripper
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class ActionsCfg:
    """6 arm joints (position) + 1 binary gripper action (open/close)."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_ARM_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=_GRIPPER_JOINTS,
        # Piper gripper: joint7 range [0, 0.05], joint8 range [-0.05, 0]
        # Open = fingers fully apart, Close = fingers together
        open_command_expr={"joint7": 0.05, "joint8": -0.05},
        close_command_expr={"joint7": 0.0, "joint8": 0.0},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Commands — goal position for the lifted object
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class CommandsCfg:
    """Goal pose for the object — sampled uniformly above the deck."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=_EE_BODY,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Goal area: in front of the arm, above ground
            pos_x=(0.15, 0.45),
            pos_y=(-0.20, 0.20),
            pos_z=(0.15, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Observations
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for lift policy.

        joint_pos_rel (8D) + joint_vel_rel (8D) +
        object_pos (3D) + goal_pos (7D) + last_action (7D) = 33D
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
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_frame,
        )

        # Goal position from command (7D: pos + quat)
        target_object_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"},
        )

        # Previous actions (7D: 6 arm + 1 gripper binary)
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

    # Object spawns on the deck surface in front of the arm
    reset_object = EventTerm(
        func=mdp.reset_object_on_surface,
        mode="reset",
        params={
            "x_range": (0.15, 0.45),
            "y_range": (-0.20, 0.20),
            "z_height": _OBJECT_SPAWN_Z,
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    # Base height + tilt randomization (same as reach)
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
# Rewards — reaching + lifting + goal tracking + smoothness
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class RewardsCfg:
    """6 reward terms following IsaacLab lift pattern.

    The reaching reward provides initial signal. The lift reward (binary,
    high weight) is the main driver — the policy must learn to close the
    gripper at the right time to get this reward. Goal tracking rewards
    (gated on lift) teach precise placement after lifting.
    """

    # Dense tanh reaching reward — same as reach task
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={
            "std": 0.1,
            "ee_body_cfg": SceneEntityCfg("robot", body_names=[_EE_BODY]),
        },
    )

    # Binary lift reward — main training signal
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=15.0,
        params={
            "minimal_height": _LIFT_HEIGHT,
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    # Goal tracking — coarse (gated on lift)
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        weight=16.0,
        params={
            "std": 0.3,
            "minimal_height": _LIFT_HEIGHT,
            "command_name": "object_pose",
        },
    )

    # Goal tracking — fine-grained (gated on lift)
    object_goal_tracking_fine = RewTerm(
        func=mdp.object_goal_distance,
        weight=5.0,
        params={
            "std": 0.05,
            "minimal_height": _LIFT_HEIGHT,
            "command_name": "object_pose",
        },
    )

    # Smoothness penalties (start small, ramp via curriculum)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Terminations
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Object fell off the surface
    object_dropping = DoneTerm(
        func=mdp.object_dropped,
        params={
            "minimum_height": -0.05,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# Curriculum — ramp smoothness penalties
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class CurriculumCfg:

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.1, "num_steps": 10000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.1, "num_steps": 10000},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Top-level environment config
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class PiperLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Piper arm lift task — Step 2 of manipulation progression."""

    scene: PiperLiftSceneCfg = PiperLiftSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.sim.dt = 0.01  # 100 Hz physics
        self.decimation = 2  # 50 Hz control
        self.episode_length_s = 5.0  # shorter than reach
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.3)

        # PhysX settings for object interaction
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class PiperLiftEnvCfg_PLAY(PiperLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
