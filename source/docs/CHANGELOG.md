# Changelog

All notable changes to the Piper Manipulator RL project.

## [0.2.1] - 2026-02-23

### Fixed — Critical env_origins bug + height reference
- **events.py**: `reset_object_in_workspace` and `randomize_base_tilt` now
  add `env.scene.env_origins[env_ids]` before writing poses. Previously all
  objects/robots were teleported to world origin, breaking all but env 0.
- **rewards.py / terminations.py**: Height checks now measure **relative to
  robot root** (arm_base) instead of absolute world Z. The 400mm limit is
  invariant to env placement and terrain tilt.
- Height penalty weight reduced -5.0 → -2.0, soft_start widened 0.35 → 0.38
  to prevent the penalty from freezing the arm.

### Added — Idle / no-target behaviour ("stand still")
- **events.py**: `reset_object_in_workspace` accepts `no_target_prob` param.
  When triggered (~20% of envs), the object is placed underground (z=-5m) to
  signal "no target present".
- **observations.py**: `has_target()` returns a binary flag (1D) — 1.0 when
  a target object is in the workspace, 0.0 when idle.
- **rewards.py**: All reaching/grasping rewards (`object_ee_distance`,
  `object_ee_distance_l2`, `gripper_object_alignment`, `object_in_gripper_zone`)
  are multiplied by the `has_target` mask — they return 0 in idle envs.
- **rewards.py**: `idle_pose_holding()` — positive reward (exp-decay) for
  staying near default pose, only active when `has_target == 0`.
  Analogous to the "stand still" reward in legged locomotion.
- **observations.py**: Object-related observations (`object_position`,
  `object_orientation`, `ee_object_vector`) return zeros when no target,
  giving the policy a clean idle signal.

### Added — Extra observations for sim2sim
- `object_orientation_in_robot_frame` (3D RPY)
- `object_properties` (4D: shape_id, dim_x, dim_y, dim_z)
- `has_target` (1D binary flag)
- Total observation space: ~49D

---

## [0.2.0] - 2025-07-27

### Changed — Reach-to-Object Upgrade
The task has been upgraded from abstract EE-pose-tracking to a physical
reach-to-object task with gripper interaction.

#### Physical target objects (`assets/target_objects.py`) — NEW
- 5 shape variants sized for the 90mm parallel-jaw gripper:
  - `CUBE_SMALL` (40mm), `CUBE_MEDIUM` (50mm), `SPHERE` (30mm radius),
    `CYLINDER` (20mm × 80mm), `STICK` (15mm × 120mm)
- All objects: 50g mass, friction 0.8/0.6, disabled self-collision
- `RANDOM_OBJECT_CFG` — `MultiAssetSpawnerCfg` spawns a random shape per env
- `TARGET_OBJECT_CFG` — `RigidObjectCfg` using random spawner
- Deterministic variants: `TARGET_CUBE_CFG`, `TARGET_SPHERE_CFG`

#### Gripper control
- Added `BinaryJointPositionActionCfg` for open/close gripper
  - Open: joint7=0.04, joint8=−0.04 (full 90mm)
  - Close: joint7=0.0, joint8=0.0
- Total action space: 6 (arm joints) + 1 (gripper binary) = 7D

#### Custom MDP modules (`tasks/reach/mdp/`) — NEW
- **rewards.py** — 7 reward functions:
  - `object_ee_distance`: tanh kernel (σ=0.1) for smooth reaching
  - `object_ee_distance_l2`: raw L2 coarse signal
  - `gripper_object_alignment`: gripper midpoint→object distance
  - `default_pose_deviation`: L2 from home joints (always-on, low weight)
  - `joint_limit_proximity`: penalty within 10% of joint limits
  - `object_in_gripper_zone`: binary success when EE < 3cm from object
  - `height_violation_penalty`: soft ramp 350→400mm across all arm bodies
- **observations.py** — 4 observation functions:
  - `object_position_in_robot_frame`, `ee_position_in_robot_frame`,
    `gripper_finger_positions_in_robot_frame` (6D),
    `ee_object_vector` (3D direction)
- **events.py** — 2 event functions:
  - `reset_object_in_workspace`: teleport object to random position
    (x: 0.15–0.45, y: ±0.20, z: 0.02–0.10)
  - `randomize_base_tilt`: ±0.05 rad roll/pitch per reset (rover terrain sim)
- **terminations.py** — 1 termination function:
  - `height_limit_exceeded`: terminate if ANY arm body exceeds 0.42m

#### Environment config rewrite (`reach_env_cfg.py`)
- Now standalone `ManagerBasedRLEnvCfg` (no longer inherits base `ReachEnvCfg`)
- `NullCommandCfg` — the physical object IS the target
- Observation space ~38D: joint pos/vel, object pos, EE pos,
  EE→object vector, finger positions, last action
- 9 reward terms with tuned weights (reaching w=2.0, grasped w=5.0,
  height penalty w=−5.0, etc.)
- Dual height enforcement: soft reward ramp + hard termination at 420mm
- Curriculum: ramp `action_rate` and `joint_vel` penalty weights over time

#### Default pose update (`assets/piper.py`)
- Changed from (0, 1.0, −1.5, 0, 0, 0, 0.04, −0.04) to
  (0, 0.07, 0, 0, 0.5, 0, 0, 0) — compact low-profile resting pose

#### Agent config updates
- RSL-RL PPO: network widened [128,128,64]→[256,128,64],
  observation normalization enabled, max_iterations 1500→3000,
  experiment renamed `piper_reach_v1`
- SKRL PPO: matching network widening, timesteps 36000→72000,
  directory renamed `reach_piper_v1`

### Fixed
- **train.py**: removed duplicate `simulation_app.close()` call
- **piper.py**: symlink creation now handles stale directories
  (`shutil.rmtree` for dirs, `os.unlink` for links/files)

---

## [0.1.0] - 2026-02-22

### Added
- Initial project structure following IsaacLab extension pattern
- **Asset definition** (`manipulator/assets/piper.py`):
  - `PIPER_ARM_CFG` — standard PD gains for joint position control
  - `PIPER_ARM_HIGH_PD_CFG` — high PD gains for IK/OSC controllers
  - Runtime URDF→USD conversion (symlinks to `/tmp/IsaacLab/manipulator/`)
  - Two actuator groups: shoulder (joints 1-3, stiffness=80) and wrist (joints 4-6, stiffness=40)
  - Gripper actuator (joints 7-8, prismatic, stiffness=2000)
  - Self-collision enabled for rover platform collision avoidance
- **Reach task** (`manipulator/tasks/reach/`):
  - `PiperReachEnvCfg` — EE pose tracking with joint position control
  - `PiperReachEnvCfg_PLAY` — small-scale evaluation config (50 envs)
  - Gym registration: `Isaac-Reach-Piper-v0`, `Isaac-Reach-Piper-Play-v0`
  - Asymmetric workspace bounds accounting for rover-mounted arm:
    - Front (X+): min 0.10m, max 0.55m
    - Lateral (Y±): ±0.20m to ±0.30m (20cm min clearance from rover body)
    - Height (Z): 0.05m to 0.50m
    - Back (X−): excluded — rover body occupies this zone
  - Reuses IsaacLab reach MDP (position/orientation tracking rewards, action rate penalty, curriculum)
- **Agent configs**:
  - RSL-RL PPO (`agents/rsl_rl_ppo_cfg.py`): [128,128,64] network, lr=3e-4, 1500 iterations
  - SKRL PPO (`agents/skrl_ppo_cfg.yaml`): matching architecture
- **Training scripts** (`scripts/rsl_rl/train.py`, `scripts/rsl_rl/play.py`)
- Package installable via `pip install -e .`

### Workspace Geometry Notes
The arm is mounted at the **front edge** of the rover platform (arm_base at origin).
Rover body extends rearward (X−). Coordinate frame relative to `arm_base`:

```
                    Arm base (0,0,0)
                         |
     Front (X+)          |          Back (X−)
   ← picking zone →      |    ← rover body (no pick) →
                         |
   min 10cm from base    |    platform: x = -0.50 to +0.10
                         |    upper deck: x = -0.45 to -0.05
                         |    tracks: y = ±0.215
```

Min working radius by direction:
- **Front (X+)**: ~10cm (arm self-collision limit)
- **Left/Right (Y±)**: ~20cm (rover body width ~38cm, tracks at ±21.5cm)
- **Back (X−)**: not usable for picking (rover body); reserved for placing only
- **Max radius**: 626.7mm (without end-effector)
