"""Physical target object configurations for the Piper gripper reach-and-grasp task.

All objects are sized to fit within the Piper parallel-jaw gripper:
  - Max gripper opening: 90 mm (joint7: 0→0.045 m, joint8: 0→-0.045 m)
  - Finger depth: ~80 mm
  - Practical graspable range: 20–70 mm diameter/width

Objects are spawned as rigid bodies with physics materials configured
for realistic grasping (moderate friction, no bounce).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass

# ── Shared physics properties for graspable objects ──────────────────────────

_GRASP_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=4,
    max_angular_velocity=500.0,
    max_linear_velocity=500.0,
    max_depenetration_velocity=1.0,
    disable_gravity=False,
)

_GRASP_COLLISION_PROPS = CollisionPropertiesCfg(
    collision_enabled=True,
    contact_offset=0.005,
    rest_offset=0.0,
)

_GRASP_MATERIAL = RigidBodyMaterialCfg(
    static_friction=0.8,
    dynamic_friction=0.6,
    restitution=0.0,
    friction_combine_mode="average",
)

_GRASP_MASS = MassPropertiesCfg(mass=0.05)  # 50 g — light enough for the arm


# ── Individual shape spawner configs ─────────────────────────────────────────

# Small cube — 4×4×4 cm
CUBE_SMALL_CFG = sim_utils.CuboidCfg(
    size=(0.04, 0.04, 0.04),
    rigid_props=_GRASP_RIGID_PROPS,
    collision_props=_GRASP_COLLISION_PROPS,
    mass_props=_GRASP_MASS,
    physics_material=_GRASP_MATERIAL,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),  # red
)

# Medium cube — 5×5×5 cm
CUBE_MEDIUM_CFG = sim_utils.CuboidCfg(
    size=(0.05, 0.05, 0.05),
    rigid_props=_GRASP_RIGID_PROPS,
    collision_props=_GRASP_COLLISION_PROPS,
    mass_props=_GRASP_MASS,
    physics_material=_GRASP_MATERIAL,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),  # green
)

# Sphere — 3 cm radius (6 cm diameter)
SPHERE_CFG = sim_utils.SphereCfg(
    radius=0.03,
    rigid_props=_GRASP_RIGID_PROPS,
    collision_props=_GRASP_COLLISION_PROPS,
    mass_props=_GRASP_MASS,
    physics_material=_GRASP_MATERIAL,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.9)),  # blue
)

# Cylinder — 2 cm radius, 8 cm tall (stick-like)
CYLINDER_CFG = sim_utils.CylinderCfg(
    radius=0.02,
    height=0.08,
    axis="Z",
    rigid_props=_GRASP_RIGID_PROPS,
    collision_props=_GRASP_COLLISION_PROPS,
    mass_props=_GRASP_MASS,
    physics_material=_GRASP_MATERIAL,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.7, 0.1)),  # yellow
)

# Thin cylinder / stick — 1.5 cm radius, 12 cm long
STICK_CFG = sim_utils.CylinderCfg(
    radius=0.015,
    height=0.12,
    axis="Z",
    rigid_props=_GRASP_RIGID_PROPS,
    collision_props=_GRASP_COLLISION_PROPS,
    mass_props=_GRASP_MASS,
    physics_material=_GRASP_MATERIAL,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.3, 0.1)),  # brown
)


# ── Multi-asset spawner (randomised shape per env) ───────────────────────────

RANDOM_OBJECT_CFG = sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=[
        CUBE_SMALL_CFG,
        CUBE_MEDIUM_CFG,
        SPHERE_CFG,
        CYLINDER_CFG,
        STICK_CFG,
    ],
    random_choice=True,
    # Global overrides (applied on top of individual configs)
    rigid_props=_GRASP_RIGID_PROPS,
    collision_props=_GRASP_COLLISION_PROPS,
    mass_props=_GRASP_MASS,
)


# ── Pre-built RigidObjectCfg for the target object ──────────────────────────
# Use RANDOM_OBJECT_CFG for varied shapes, or swap with a single shape cfg.

TARGET_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/TargetObject",
    spawn=RANDOM_OBJECT_CFG,
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.30, 0.0, 0.04),   # 30 cm in front of arm, on ground level + half-height
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

# ── Single-shape variants (for debugging / deterministic training) ───────────

TARGET_CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/TargetObject",
    spawn=CUBE_MEDIUM_CFG,
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.30, 0.0, 0.04),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

TARGET_SPHERE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/TargetObject",
    spawn=SPHERE_CFG,
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.30, 0.0, 0.04),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)
