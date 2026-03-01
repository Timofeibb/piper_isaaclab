"""Place environment for the Piper 6-DOF arm with gripper.

Step 3 of the manipulation progression:
  Step 1: Reach  — learn arm control to reach target positions
  Step 2: Lift   — add binary gripper, learn grasp + lift
  Step 3: Place  — full pick-and-place with staged rewards

The policy must reach an object, grasp it, lift it, carry it to a
commanded goal location, and place it down.  A green disc on the ground
visualises the placement target for human observers (no effect on the
policy, observations, or rewards).

Key differences from lift:
  - PlaceGoalCommand instead of UniformPoseCommand (adds green disc)
  - Goal z is at ground level (placement ON the surface)
  - Longer episodes (8 s) — more time needed to complete full sequence
  - Additional placement reward (XY proximity, gated on object on surface)
  - Randomized objects (5 shapes via MultiAssetSpawner)
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

from manipulator.tasks.place import mdp
from manipulator.assets.piper import PIPER_ARM_CFG
from manipulator.assets.target_objects import TARGET_OBJECT_CFG
from manipulator.tasks.place.place_goal_command import PlaceGoalCommandCfg


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

_EE_BODY = "link_end_effector"
_ARM_JOINTS = ["joint[1-6]"]
_GRIPPER_JOINTS = ["joint[7-8]"]

_OBJECT_SPAWN_Z = 0.02
# Lift gate — object must be above this to get lift reward / set was_lifted
_LIFT_HEIGHT = 0.10
# Placement gate — overlaps with lift zone to eliminate reward dead zone
# (transport persists via was_lifted, placement activates here)
_PLACED_HEIGHT = 0.12


# ═════════════════════════════════════════════════════════════════════════════
# Scene
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class PiperPlaceSceneCfg(InteractiveSceneCfg):
    """Scene: Piper arm (fixed base) + randomized graspable object."""

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
# Actions — arm + binary gripper (same as lift)
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
        open_command_expr={"joint7": 0.05, "joint8": -0.05},
        close_command_expr={"joint7": 0.0, "joint8": 0.0},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Commands — placement goal with green disc visualisation
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class CommandsCfg:
    """Goal pose for object placement — sampled on the ground plane.

    Uses PlaceGoalCommandCfg which draws a green disc at the target
    location for human observation (purely visual, not in policy obs).
    """

    object_pose = PlaceGoalCommandCfg(
        asset_name="robot",
        body_name=_EE_BODY,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        min_distance_from_object=0.15,
        ranges=PlaceGoalCommandCfg.Ranges(
            # Wider placement area — full reachable ground zone
            pos_x=(0.10, 0.50),
            pos_y=(-0.30, 0.30),
            # Goal height: ground level to 30 mm above ground
            pos_z=(0.0, 0.03),
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
        """Observations for place policy.

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

    # Object spawns on the ground surface in front of the arm
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
# Rewards — reaching + lifting + transport + placement + smoothness
# ═════════════════════════════════════════════════════════════════════════════


@configclass
class RewardsCfg:
    """Staged reward structure for pick-and-place.

    Stage 1 — Reach:  EE→object tanh reward (always active)
    Stage 2 — Lift:   binary reward when object above _LIFT_HEIGHT
    Stage 3 — Transport: object→goal tracking gated on lift
    Stage 4 — Place:  object near goal XY, gated on object back on surface
    Smoothness penalties ramp via curriculum.
    """

    # Stage 1: Dense reaching reward
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={
            "std": 0.1,
            "ee_body_cfg": SceneEntityCfg("robot", body_names=[_EE_BODY]),
        },
    )

    # Stage 2: Binary lift reward (moderate weight — transport + place
    # must clearly dominate so the policy eventually releases)
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=5.0,
        params={
            "minimal_height": _LIFT_HEIGHT,
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    # Stage 3: Transport — XY-only, gated on WAS_LIFTED (not is_lifted)
    # Persists during descent so there’s no reward desert between the
    # lift zone (z>0.10) and placement zone (z<0.12).
    object_goal_tracking = RewTerm(
        func=mdp.object_transport_to_goal_xy,
        weight=8.0,
        params={
            "std": 0.3,
            "lift_threshold": _LIFT_HEIGHT,
            "command_name": "object_pose",
        },
    )

    object_goal_tracking_fine = RewTerm(
        func=mdp.object_transport_to_goal_xy,
        weight=3.0,
        params={
            "std": 0.05,
            "lift_threshold": _LIFT_HEIGHT,
            "command_name": "object_pose",
        },
    )

    # Stage 4a: Release bonus — immediate signal for opening gripper at goal
    gripper_release = RewTerm(
        func=mdp.gripper_release_near_goal,
        weight=15.0,
        params={
            "xy_threshold": 0.08,
            "command_name": "object_pose",
            "open_threshold": 0.03,
        },
    )

    # Stage 4b: Placement — triple-gated: was_lifted + on_surface + near_goal
    object_placement = RewTerm(
        func=mdp.object_placed_at_goal,
        weight=25.0,
        params={
            "std": 0.05,
            "max_object_height": _PLACED_HEIGHT,
            "lift_threshold": _LIFT_HEIGHT,
            "command_name": "object_pose",
        },
    )

    object_placement_fine = RewTerm(
        func=mdp.object_placed_at_goal,
        weight=10.0,
        params={
            "std": 0.02,
            "max_object_height": _PLACED_HEIGHT,
            "lift_threshold": _LIFT_HEIGHT,
            "command_name": "object_pose",
        },
    )

    # Smoothness penalties
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
class PiperPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Piper arm place task — Step 3 of manipulation progression."""

    scene: PiperPlaceSceneCfg = PiperPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
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
        self.episode_length_s = 8.0  # longer for full pick-and-place
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.3)

        # PhysX settings for object interaction
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class PiperPlaceEnvCfg_PLAY(PiperPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
