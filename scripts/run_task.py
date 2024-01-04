# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Run task script.

This script allows a user to quickly run a single task instance on the
Franka. The specified task instance must have a corresponding
src/industreallib/tasks/instance_configs/<task_name>.yaml file.

Typical usage example:

  python run_task.py -t reach
"""

# Standard Library
import argparse

# NVIDIA
import industreallib.tasks.scripts.task_utils as task_utils


def get_args():
    """Gets arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--task_instance_config_name",
        required=True,
        help="Task instance configuration to run",
    )
    parser.add_argument(
        "-d",
        "--debug_mode",
        action="store_true",
        required=False,
        help="Enable output for debugging",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    """Gets the task instance. Runs the task instance."""
    args = get_args()

    task_instance_config = task_utils.get_task_instance_config(
        task_instance_config_name=args.task_instance_config_name, task_instance_config_subdir=None
    )
    task_instance = task_utils.get_task_instance(
        args=args, task_instance_config=task_instance_config, in_sequence=False
    )

    task_instance.go_to_goals()
