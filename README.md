# Manipulator RL — Piper 6-DOF Arm

Reinforcement learning environments for the **Piper 6-DOF robotic arm** with parallel-jaw gripper, built on [IsaacLab](https://github.com/isaac-sim/IsaacLab).

The package implements a 3-stage manipulation curriculum progressing from reaching to full pick-and-place.

## Robot

**Piper arm** — 6-DOF manipulator with a prismatic parallel-jaw gripper.

| Property | Value |
|----------|-------|
| DOF | 6 revolute (arm) + 2 prismatic (gripper) |
| Working radius | 626.7 mm (without end-effector) |
| Max gripper opening | 90 mm |
| Base height | 200 mm (simulates rover deck) |
| URDF source | `source/manipulator/assets/piper_description/` |

Joint chain:
```
arm_base (fixed) → link1 → link2 → link3 → link4 → link5 → link6
  → gripper_base → link_end_effector
                 → link7 (finger L, prismatic, 0–45mm)
                 → link8 (finger R, prismatic, -45mm–0)
```

Two actuator configurations are provided:

- **`PIPER_ARM_CFG`** — Standard PD gains for joint position control (shoulder stiffness=80, wrist=40, gripper=2000)
- **`PIPER_ARM_HIGH_PD_CFG`** — High PD gains for IK/OSC controllers (shoulder stiffness=400, wrist=200)

## Tasks

### Step 1: Reach (`Isaac-Reach-Piper-v0`)

<video src="data/Reach.mp4" width="600" autoplay loop muted></video>

Move the end-effector to a physical target object spawned randomly in the workspace.

| Property | Value |
|----------|-------|
| Actions | 6D — arm joint positions (no gripper) |
| Observations | 28D — joint pos/vel, object pos, EE→object vector, last action |
| Episode length | 12 s |
| Control freq | 30 Hz (sim 60 Hz, decimation 2) |
| Reward | tanh reach + L2 coarse reach + smoothness penalties |
| Termination | Time-out only |

The target object is sampled in **polar coordinates** across the reachable annulus (r: 10–60 cm, theta: -145° to +119°), avoiding the dead zone behind the arm.

### Step 2: Lift (`Isaac-Lift-Piper-v0`)

<video src="data/Lift.mp4" width="600" autoplay loop muted></video>

Reach a rigid object, grasp it, and lift it to a commanded goal position.

| Property | Value |
|----------|-------|
| Actions | 7D — 6 arm joints + 1 binary gripper (open/close) |
| Observations | 33D — joint pos/vel, object pos, goal pose (7D), last action |
| Episode length | 5 s |
| Control freq | 50 Hz (sim 100 Hz, decimation 2) |
| Reward | tanh reach + binary lift (w=15) + goal tracking (gated on lift, w=16) |
| Termination | Time-out, object dropped below -5 cm |

Grasping **emerges** from the lift reward — no explicit grasp reward is used. The policy must learn to close the gripper at the right time to receive the lift bonus.

### Step 3: Place (`Isaac-Place-Piper-v0`)

<video src="data/Place.mp4" width="600" autoplay loop muted></video>

Full pick-and-place: reach, grasp, lift, transport, and place at a commanded ground location.

| Property | Value |
|----------|-------|
| Actions | 7D — 6 arm joints + 1 binary gripper (open/close) |
| Observations | 33D — joint pos/vel, object pos, goal pose (7D), last action |
| Episode length | 8 s |
| Control freq | 50 Hz (sim 100 Hz, decimation 2) |
| Reward | 4-stage: reach → lift → transport (XY) → placement |
| Termination | Time-out, object dropped below -5 cm |

Key reward design features:
- **`was_lifted` flag**: Transport reward persists during descent, eliminating the reward desert between the lift zone and placement zone.
- **Triple-gated placement**: Requires was_lifted + on_surface + near_goal to prevent the "do nothing" exploit.
- **Gripper release bonus**: Immediate signal for opening the gripper at the goal, bridging the temporal credit-assignment gap.
- **`PlaceGoalCommand`**: Green disc visualization at the placement target with min-distance rejection sampling.

## Target Objects

5 shapes sized for the 90 mm gripper, spawned randomly per environment via `MultiAssetSpawner`:

| Shape | Dimensions | Color |
|-------|-----------|-------|
| Cube (small) | 40×40×40 mm | Red |
| Cube (medium) | 50×50×50 mm | Green |
| Sphere | 30 mm radius | Blue |
| Cylinder | 20 mm radius × 80 mm | Yellow |
| Stick | 15 mm radius × 120 mm | Brown |

All objects: 50 g mass, friction 0.8/0.6 static/dynamic, zero restitution.

## Domain Randomization

Applied on every episode reset:

- **Object position**: Polar sampling (reach) or rectangular (lift/place) across the reachable workspace
- **Base pose**: Height ±20 mm, roll/pitch ±3°, yaw ±6° (simulates uneven rover terrain)
- **Joint positions**: Scaled from default pose (0.5×–1.5×)
- **Object shape**: Random from 5 variants (via `MultiAssetSpawner`)

## Project Structure

```
manipulator/
├── scripts/
│   └── rsl_rl/
│       ├── train.py              # Training script (RSL-RL PPO)
│       └── play.py               # Evaluation + JIT/ONNX export
├── source/
│   ├── config/
│   │   └── extension.toml        # Package metadata
│   ├── docs/
│   │   └── CHANGELOG.md
│   └── manipulator/
│       ├── __init__.py            # Triggers Gym registration on import
│       ├── assets/
│       │   ├── piper.py           # ArticulationCfg (URDF→USD, actuators)
│       │   └── target_objects.py  # RigidObjectCfg (5 shapes + random spawner)
│       ├── tasks/
│       │   ├── reach/
│       │   │   ├── __init__.py    # gym.register (v0 + Play)
│       │   │   ├── reach_env_cfg.py
│       │   │   ├── mdp/           # rewards, observations, events, terminations
│       │   │   └── agents/        # rsl_rl_ppo_cfg.py, skrl_ppo_cfg.yaml
│       │   ├── lift/
│       │   │   ├── __init__.py
│       │   │   ├── lift_env_cfg.py
│       │   │   ├── mdp/
│       │   │   └── agents/
│       │   └── place/
│       │       ├── __init__.py
│       │       ├── place_env_cfg.py
│       │       ├── place_goal_command.py  # Custom command + green disc marker
│       │       ├── mdp/
│       │       └── agents/
│       └── assets/
│           └── piper_description/
│               ├── urdf/          # piper_description_v100.urdf
│               └── meshes/        # STL mesh files
└── logs/
    └── rsl_rl/                    # Training logs (tensorboard)
```

## Installation

From the workspace root, with the IsaacLab conda environment active:

```bash
cd workspace/manipulator/source/manipulator
pip install -e .
```

## Training

```bash
cd workspace/manipulator

# Reach — headless, 4096 envs
python scripts/rsl_rl/train.py --task Isaac-Reach-Piper-v0 --num_envs 4096 --headless

# Lift
python scripts/rsl_rl/train.py --task Isaac-Lift-Piper-v0 --num_envs 4096 --headless

# Place
python scripts/rsl_rl/train.py --task Isaac-Place-Piper-v0 --num_envs 4096 --headless

# With rendering (slower)
python scripts/rsl_rl/train.py --task Isaac-Reach-Piper-v0 --num_envs 1024

# Resume from checkpoint
python scripts/rsl_rl/train.py --task Isaac-Reach-Piper-v0 --resume --load_run <run_dir>
```

Logs are saved to `logs/rsl_rl/<experiment_name>/<timestamp>/`.

## Evaluation & Export

```bash
# Play latest checkpoint (auto-finds latest run + checkpoint)
python scripts/rsl_rl/play.py --task Isaac-Reach-Piper-Play-v0

# Specific run and checkpoint
python scripts/rsl_rl/play.py --task Isaac-Reach-Piper-Play-v0 \
    --load_run 2026-02-22_16-43-40 --load_checkpoint model_1500.pt

# Export only (JIT + ONNX, skip simulation loop)
python scripts/rsl_rl/play.py --task Isaac-Place-Piper-Play-v0 --export_only
```

Exported models are saved to `logs/rsl_rl/<experiment>/<run>/exported/`:
- `policy.pt` — TorchScript JIT (for C++ / LibTorch)
- `policy.onnx` — ONNX (for cross-platform inference)

## Gym Environment IDs

| Environment | ID | Purpose |
|------------|-----|---------|
| Reach (train) | `Isaac-Reach-Piper-v0` | 4096 envs, obs noise ON |
| Reach (eval) | `Isaac-Reach-Piper-Play-v0` | 50 envs, obs noise OFF |
| Lift (train) | `Isaac-Lift-Piper-v0` | 4096 envs, obs noise ON |
| Lift (eval) | `Isaac-Lift-Piper-Play-v0` | 50 envs, obs noise OFF |
| Place (train) | `Isaac-Place-Piper-v0` | 4096 envs, obs noise ON |
| Place (eval) | `Isaac-Place-Piper-Play-v0` | 50 envs, obs noise OFF |

## Agent Configurations

Both RSL-RL and SKRL configs are provided for each task.

### RSL-RL PPO

| Parameter | Reach | Lift | Place |
|-----------|-------|------|-------|
| Network | [64, 64] | [256, 128, 64] | [256, 128, 64] |
| Obs normalization | OFF | ON | ON |
| Learning rate | 1e-3 | 1e-4 | 1e-4 |
| Entropy coef | 0.001 | 0.006 | 0.006 |
| Gamma | 0.99 | 0.98 | 0.99 |
| Max iterations | 1500 | 1500 | 3000 |
| Steps/env | 24 | 24 | 24 |

## Version

Current: **0.2.0** — See [CHANGELOG](source/docs/CHANGELOG.md) for full history.

## License

Apache-2.0
