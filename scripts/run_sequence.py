# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Run sequence script.

This script allows a user to quickly run a sequence instance (itself
consisting of task instances) on the Franka. The tasks instances are
executed in an interleaved fashion (e.g., pick place insert for goal 1,
pick place insert for goal 2, and so on). The specified sequence must
have a corresponding src/industreallib/sequences/instance_configs/
<sequence_name>.yaml file.

Typical usage example:

  python run_sequence.py -s pick_place_insert_pegs
"""

# Standard Library
import argparse

# NVIDIA
import industreallib.sequences.scripts.sequence_utils as sequence_utils


def _get_args():
    """Gets arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--sequence_instance_config_name",
        required=True,
        help="Sequence instance configuration to run",
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
    """Gets the sequence instance. Runs the sequence instance on each assembly."""
    args = _get_args()

    # Get sequence instance
    sequence_instance_config = sequence_utils.get_sequence_instance_config(args=args)
    sequence_instance = sequence_utils.get_sequence_instance(
        args=args, sequence_instance_config=sequence_instance_config
    )

    # Get number of assemblies to assemble
    num_assemblies = sequence_utils.get_num_assemblies(
        sequence_instance_config=sequence_instance_config
    )

    # For each assembly (e.g., round-peg-and-round-hole), execute each task instance
    # in sequence instance (e.g., pick pegs, place pegs, insert pegs), and repeat
    visited_goals = []
    for i in range(num_assemblies):
        print(f"\nRunning {args.sequence_instance_config_name} sequence...")
        for task_num, task_instance in enumerate(sequence_instance.task_instances):
            # Get name of current task instance
            task_instance_config_name = (
                sequence_instance_config.sequence.task_instance_config_names[task_num]
            )

            # Get name of part (e.g., round_peg) assigned to current task instance
            part_name = sequence_instance_config.sequence[
                f"{task_instance_config_name}"
            ].part_order[i]

            print(f"\nRunning {task_instance_config_name} task on {part_name} part...")

            # Get goal coordinates for current part
            # NOTE: If there is more than one detection label that has the desired part
            # name (e.g., round_peg), the goal coordinates will be retrieved for the
            # *first* detection label with that part name, unless those goal coordinates
            # have already been visited. To prevent unexpected behavior, it is
            # recommended to have a unique detection label for each part, or to restrict
            # the scene to unique parts.
            for label, goal in zip(task_instance.goal_labels, task_instance.goal_coords):
                if label == part_name and goal not in visited_goals:
                    visited_goals.append(goal)
                    break
            else:
                raise ValueError(f"Desired part {part_name} not found.")

            # Do pre-task procedure, go to goal coordinates, and do post-task procedure
            task_instance.do_simple_procedure(
                procedure=sequence_instance_config.sequence[
                    f"{task_instance_config_name}"
                ].do_before,
                franka_arm=sequence_instance.franka_arm,
            )
            task_instance.go_to_goal(goal=goal, franka_arm=sequence_instance.franka_arm)
            task_instance.do_simple_procedure(
                procedure=sequence_instance_config.sequence[
                    f"{task_instance_config_name}"
                ].do_after,
                franka_arm=sequence_instance.franka_arm,
            )

            print(f"\nFinished running {task_instance_config_name} task on {part_name} part.")

        print(f"\nFinished running {args.sequence_instance_config_name} sequence.")
