"""Custom command term for the place task — adds green disc visualization.

Extends ``UniformPoseCommand`` with a flat green disc that shows the
placement target on the ground.  The disc is purely visual — it does not
affect observations, rewards, or physics.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass


class PlaceGoalCommand(UniformPoseCommand):
    """Uniform pose command with an additional green disc on the ground.

    Features on top of ``UniformPoseCommand``:
      - Green disc drawn at the goal (x, y) on the ground plane.
      - Minimum-distance rejection: goals closer than
        ``cfg.min_distance_from_object`` to the current object position
        are resampled (prevents trivially easy episodes).
    """

    # ------------------------------------------------------------------
    # Command resampling with min-distance rejection
    # ------------------------------------------------------------------
    def _resample_command(self, env_ids: Sequence[int]):
        """Sample goal poses, rejecting any within min distance of the object."""
        min_dist = self.cfg.min_distance_from_object
        if min_dist <= 0.0:
            super()._resample_command(env_ids)
            return

        # Get current object XY in robot body frame
        obj: RigidObject = self._env.scene["object"]
        from isaaclab.utils.math import subtract_frame_transforms

        obj_pos_w = obj.data.root_pos_w[env_ids, :3]
        obj_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            obj_pos_w,
        )
        obj_xy = obj_pos_b[:, :2]  # shape (N, 2)

        # Sample with rejection (max 10 attempts, then accept whatever we have)
        remaining = torch.arange(len(env_ids), device=self.device)
        for _ in range(10):
            if len(remaining) == 0:
                break
            actual_ids = [env_ids[i] for i in remaining.tolist()] if isinstance(env_ids, list) else env_ids[remaining]
            # Call parent to fill pose_command_b for these envs
            super()._resample_command(actual_ids)
            # Check XY distance from object
            goal_xy = self.pose_command_b[actual_ids, :2]
            dist = torch.norm(goal_xy - obj_xy[remaining], dim=-1)
            too_close = dist < min_dist
            remaining = remaining[too_close]
            if len(remaining) > 0:
                obj_xy = obj_xy  # keep full array, indexing handles the rest

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        # Call parent to create/toggle the standard frame markers
        super()._set_debug_vis_impl(debug_vis)

        if debug_vis:
            if not hasattr(self, "place_disc_visualizer"):
                disc_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/place_target",
                    markers={
                        "disc": sim_utils.CylinderCfg(
                            radius=0.08,
                            height=0.002,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 1.0, 0.0),
                                opacity=0.7,
                            ),
                        ),
                    },
                )
                self.place_disc_visualizer = VisualizationMarkers(disc_cfg)
            self.place_disc_visualizer.set_visibility(True)
        else:
            if hasattr(self, "place_disc_visualizer"):
                self.place_disc_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Draw the standard frame markers (goal + current EE)
        super()._debug_vis_callback(event)

        # Draw the green disc at the goal XY, projected onto the ground
        disc_pos = self.pose_command_w[:, :3].clone()
        # Set z to ground level (env origin z + small offset to avoid z-fighting)
        disc_pos[:, 2] = self._env.scene.env_origins[:, 2] + 0.001
        self.place_disc_visualizer.visualize(translations=disc_pos)


# ═══════════════════════════════════════════════════════════════════════
# Config — drop-in replacement for UniformPoseCommandCfg
# ═══════════════════════════════════════════════════════════════════════


@configclass
class PlaceGoalCommandCfg(UniformPoseCommandCfg):
    """Same as ``UniformPoseCommandCfg`` but spawns a green disc marker.

    Set ``min_distance_from_object`` > 0 to reject goals that are too
    close to the current object position (prevents trivially easy episodes).
    """

    class_type: type = PlaceGoalCommand
    min_distance_from_object: float = 0.15
