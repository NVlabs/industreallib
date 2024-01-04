# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""Class definition for IndustRealSequenceBase.

This script defines the IndustRealSequenceBase base class. The script defines
the variables and methods that are common for all sequences.
"""

# Third Party
from frankapy import FrankaArm

# NVIDIA
import industreallib.control.scripts.control_utils as control_utils
import industreallib.perception.scripts.detect_objects as detect_objects
import industreallib.perception.scripts.map_workspace as map_workspace
import industreallib.tasks.scripts.task_utils as task_utils


class IndustRealSequenceBase:
    """Defines base class for all sequences."""

    def __init__(self, args, sequence_instance_config):
        """Initializes the task instances for the sequence."""
        self._args = args
        self._sequence_instance_config = sequence_instance_config
        self.task_instances = []

        self._get_task_instances()
        self.franka_arm = FrankaArm()
        control_utils.set_sigint_response(franka_arm=self.franka_arm)
        self._get_goals_from_perception()

    def _get_task_instances(self):
        """Gets the task instances for the sequence."""
        for (
            task_instance_config_name
        ) in self._sequence_instance_config.sequence.task_instance_config_names:
            # Get task instance configuration
            task_instance_config = task_utils.get_task_instance_config(
                task_instance_config_name=task_instance_config_name,
                task_instance_config_subdir=(
                    self._sequence_instance_config.sequence.task_instance_config_subdir
                ),
            )

            # Get task instance
            # NOTE: Since in_sequence is True, instantiating the task instance will not
            # instantiate FrankaArm or get goals.
            task_instance = task_utils.get_task_instance(
                args=self._args, task_instance_config=task_instance_config, in_sequence=True
            )
            self.task_instances.append(task_instance)

    def _get_goals_from_perception(self):
        """Gets goals for the tasks in the sequence based on perception."""
        object_coords, object_labels = self._get_object_coords_from_perception()
        self._get_goals_for_each_task(object_coords=object_coords, object_labels=object_labels)

    def _get_object_coords_from_perception(self):
        """Gets object coordinates (x, y, theta) from perception."""
        # Map workspace (saves to JSON)
        map_workspace.main(
            perception_config_file_name=self._sequence_instance_config.goals.perception.config,
            franka_arm=self.franka_arm,
        )

        # Detect objects and get object coordinates (x, y, theta)
        object_coords, object_labels = detect_objects.main(
            perception_config_file_name=self._sequence_instance_config.goals.perception.config
        )

        return object_coords, object_labels

    def _get_goals_for_each_task(self, object_coords, object_labels):
        """Gets goals (x, y, z, roll, pitch, yaw) for each task."""
        # NOTE: Consider a scene with a round peg, round hole, rectangular peg, and
        # rectangular hole. A Pick-Place-Insert policy must pick up a peg, place it
        # above its corresponding hole, insert it, pick up the second peg, place it
        # above its corresponding hole, and insert it. Thus, the object coordinates
        # of the *pegs* must be used to derive the goals for the Pick policy, and the
        # object coordinates of the *holes* must be used to derive the goals for the
        # Place and Insert policies (with different goal heights). This method
        # determines the appropriate object for each task using selectors (specified
        # in the task instance configuration) and derives goals using the object
        # coordinates and goal heights (also specified in the task instance
        # configuration).

        for task_instance in self.task_instances:
            task_instance.convert_object_coords_to_goals(
                object_coords=object_coords,
                object_labels=object_labels,
                perception_config_file_name=self._sequence_instance_config.goals.perception.config,
            )

            if self._args.debug_mode:
                print(f'\nTask: {task_instance.task_instance_config["task"]["class"]}')
                print("\nGoals:")
                for label, goal in zip(task_instance.goal_labels, task_instance.goal_coords):
                    print(f"{label}: {goal}")
