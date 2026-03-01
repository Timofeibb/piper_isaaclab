"""Configuration for the Piper 6-DOF robotic arm with gripper.

The Piper arm is a 6-DOF manipulator with a parallel-jaw gripper.
Rover platform has been stripped — arm_base is fixed directly to the world.
Working radius: 626.7 mm (without EE).

Joint chain (arm-only URDF):
  arm_base (root, fixed) -> link1 -> link2 -> link3 ->
  link4 -> link5 -> link6 -> gripper_base -> link_end_effector
                                          -> link7 (gripper finger L, prismatic)
                                          -> link8 (gripper finger R, prismatic)

URDF is converted to USD at runtime via sim_utils.UrdfFileCfg.
"""

import os
import shutil

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# ── Paths ────────────────────────────────────────────────────────────────────
# piper.py lives at:  source/manipulator/manipulator/assets/piper.py
# URDF lives at:      source/manipulator/assets/piper_description/urdf/piper_description_v100.urdf
# So we go up 2 levels (assets/ -> manipulator/ -> manipulator/) then into assets/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPER_ASSETS_DIR = os.path.join(_THIS_DIR, "..", "..", "assets", "piper_description")
_PIPER_URDF_PATH = os.path.join(_PIPER_ASSETS_DIR, "urdf", "piper_description_v100.urdf")
_PIPER_MESHES_DIR = os.path.join(_PIPER_ASSETS_DIR, "meshes")

# ── Prepare URDF for the converter ───────────────────────────────────────────
# The URDF references meshes via `package://piper_description/meshes/...`
# We create a symlink structure in /tmp so the converter resolves them.
_TMP_ROOT = "/tmp/IsaacLab/manipulator"
_TMP_PKG = os.path.join(_TMP_ROOT, "piper_description")
_TMP_MESHES = os.path.join(_TMP_PKG, "meshes")
_TMP_URDF = os.path.join(_TMP_PKG, "urdf", "piper_description_v100.urdf")

os.makedirs(os.path.join(_TMP_PKG, "urdf"), exist_ok=True)

# Symlink meshes directory
if os.path.islink(_TMP_MESHES):
    os.remove(_TMP_MESHES)
elif os.path.isdir(_TMP_MESHES):
    shutil.rmtree(_TMP_MESHES)
elif os.path.exists(_TMP_MESHES):
    os.remove(_TMP_MESHES)
os.symlink(os.path.realpath(_PIPER_MESHES_DIR), _TMP_MESHES)

# Symlink URDF file
if os.path.islink(_TMP_URDF):
    os.remove(_TMP_URDF)
elif os.path.exists(_TMP_URDF):
    os.remove(_TMP_URDF)
os.symlink(os.path.realpath(_PIPER_URDF_PATH), _TMP_URDF)

# ── URDF spawn config ────────────────────────────────────────────────────────

_PIPER_URDF_SPAWN_CFG = sim_utils.UrdfFileCfg(
    asset_path=_TMP_URDF,
    fix_base=True,  # arm_base fixed directly to world
    merge_fixed_joints=False,  # keep gripper_base, EE frames
    replace_cylinders_with_capsules=True,
    self_collision=True,  # arm self-collision avoidance
    activate_contact_sensors=False,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0, damping=0  # overridden by actuator configs
        )
    ),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    ),
)

# ── Joint limits from URDF (for reference) ──────────────────────────────────
# joint1: revolute, [-2.618, 2.168]   (base rotation)
# joint2: revolute, [0, 3.14]         (shoulder)
# joint3: revolute, [-2.967, 0]       (elbow)
# joint4: revolute, [-1.745, 1.745]   (wrist 1)
# joint5: revolute, [-1.22, 1.22]     (wrist 2)
# joint6: revolute, [-2.0944, 2.0944] (wrist 3)
# joint7: prismatic, [0, 0.045]       (gripper finger L)
# joint8: prismatic, [-0.045, 0]      (gripper finger R)

# ── Standard PD ArticulationCfg ─────────────────────────────────────────────

PIPER_ARM_CFG = ArticulationCfg(
    spawn=_PIPER_URDF_SPAWN_CFG,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),  # 200 mm above ground (simulates rover deck height)
        joint_pos={
            # Arm joints — default (rest) pose: arm folded down, low profile
            "joint1": 0.0,       # base centered
            "joint2": 0.07,      # shoulder nearly flat
            "joint3": 0.0,       # elbow straight
            "joint4": 0.0,       # wrist 1 neutral
            "joint5": 0.5,       # wrist 2 slightly angled
            "joint6": 0.0,       # wrist 3 neutral
            # Gripper — closed at rest
            "joint7": 0.0,
            "joint8": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Shoulder + elbow (joints 1-3): higher torque, slower for accuracy
        "piper_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=2.0,
            stiffness=80.0,
            damping=20.0,
        ),
        # Wrist (joints 4-6): lower torque, finer control, slower
        "piper_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            effort_limit_sim=30.0,
            velocity_limit_sim=2.0,
            stiffness=40.0,
            damping=10.0,
        ),
        # Gripper (joints 7-8): prismatic, high stiffness for firm grasp
        "piper_gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint[7-8]"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=2000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# ── High PD variant (for IK / OSC controllers) ──────────────────────────────
# Higher gains for tighter position tracking when using task-space controllers

PIPER_ARM_HIGH_PD_CFG = PIPER_ARM_CFG.copy()
PIPER_ARM_HIGH_PD_CFG.actuators["piper_shoulder"] = ImplicitActuatorCfg(
    joint_names_expr=["joint[1-3]"],
    effort_limit_sim=100.0,
    velocity_limit_sim=5.0,
    stiffness=400.0,
    damping=80.0,
)
PIPER_ARM_HIGH_PD_CFG.actuators["piper_wrist"] = ImplicitActuatorCfg(
    joint_names_expr=["joint[4-6]"],
    effort_limit_sim=30.0,
    velocity_limit_sim=5.0,
    stiffness=200.0,
    damping=40.0,
)
