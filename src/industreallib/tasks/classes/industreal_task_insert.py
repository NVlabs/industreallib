# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Class definition for IndustRealTaskInsert.

This script defines the IndustRealTaskInsert class. The class gets the
observations that are specific to the Insert task and can also contain
other task-specific methods.
"""

# Third Party
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# NVIDIA
from industreallib.tasks.classes.industreal_task_base import IndustRealTaskBase


class IndustRealTaskInsert(IndustRealTaskBase):
    """Defines class for Insert task."""

    def __init__(self, args, task_instance_config, in_sequence):
        """Initializes the superclass."""
        super().__init__(
            args=args, task_instance_config=task_instance_config, in_sequence=in_sequence
        )

    def _get_observations(self, goal_pos, goal_ori_mat, franka_arm):
        """Gets the robot state from frankapy. Extracts the observations."""
        # NOTE: The position and orientation observations should be in the same
        # coordinate frame as during training, and the orientation observations
        # should have the same representations as during training. In Factory and
        # IndustReal, the position and orientation observations are typically in
        # the robot base frame, and the orientation observations are typically
        # represented as quaternions (x, y, z, w).

        curr_state = franka_arm.get_robot_state()
        # For list of all keys in state dict, see
        # https://github.com/iamlab-cmu/frankapy/blob/master/frankapy/franka_arm_state_client.py

        curr_joint_angles = curr_state["joints"]
        curr_pos = curr_state["pose"].translation
        curr_ori_mat = curr_state["pose"].rotation

        # NOTE: For IndustReal insertion policies, default goal z-position was 3 mm lower
        goal_pos_offset = goal_pos.copy()
        goal_pos_offset[2] -= 0.003

        # NOTE: For IndustReal insertion policies, default observed goal roll angle during training
        # was 0.0 (instead of pi)
        goal_ori_mat_offset = goal_ori_mat.copy()
        if self.task_instance_config.rl.override_obs_goal_roll_angle is not None:
            goal_ori_euler = Rotation.from_matrix(goal_ori_mat).as_euler("xyz")
            goal_ori_euler[0] = 0.0
            goal_ori_euler[1] = 0.0
            goal_ori_mat_offset = Rotation.from_euler("xyz", goal_ori_euler).as_matrix()

        observations = (
            torch.from_numpy(
                np.hstack(
                    [
                        curr_joint_angles,
                        curr_pos,
                        Rotation.from_matrix(curr_ori_mat).as_quat(),  # (x, y, z, w)
                        goal_pos_offset,
                        Rotation.from_matrix(goal_ori_mat_offset).as_quat(),
                        goal_pos_offset - curr_pos,
                    ]
                )  # (x, y, z, w)
            )
            .to(torch.float32)
            .to(self._device)
        )

        return observations, curr_state
