# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Simple action script.

This script allows a user to quickly run a simple action with the Franka
using the frankapy library. The actions currently include get pose,
get angles, open gripper, close gripper, go home, go to position,
go upward, go downward, perturb yaw, and guide.

Typical usage example:

  python run_simple_action.py -a get_pose
"""

# Standard Library
import argparse

# Third Party
from frankapy import FrankaArm

# NVIDIA
import industreallib.control.scripts.control_utils as control_utils


def get_args():
    """Gets arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--action",
        type=str,
        required=True,
        help=(
            "Options: get_pose, get_angles, open_gripper, close_gripper, go_home, go_to_pos,"
            " go_upward, go_downward, perturb yaw, guide"
        ),
    )
    parser.add_argument(
        "-p",
        "--position",
        nargs="+",
        help=(
            "Position in robot base frame (e.g., FrankaPy home position is 0.3069 0.0 0.4867). Used"
            " for go_to_pos"
        ),
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    franka_arm = FrankaArm()

    if args.action == "get_pose":
        control_utils.print_pose(franka_arm=franka_arm)
    elif args.action == "get_angles":
        control_utils.print_joint_angles(franka_arm=franka_arm)
    elif args.action == "open_gripper":
        control_utils.open_gripper(franka_arm=franka_arm)
    elif args.action == "close_gripper":
        control_utils.close_gripper(franka_arm=franka_arm)
    elif args.action == "go_home":
        control_utils.go_home(franka_arm=franka_arm, duration=5.0)
    elif args.action == "go_to_pos":
        control_utils.go_to_pos(franka_arm=franka_arm, pos=args.position, duration=5.0)
    elif args.action == "go_upward":
        control_utils.go_upward(franka_arm=franka_arm, dist=0.1, duration=5.0)
    elif args.action == "go_downward":
        control_utils.go_downward(franka_arm=franka_arm, dist=0.1, duration=5.0)
    elif args.action == "perturb_yaw":
        control_utils.perturb_yaw(franka_arm=franka_arm, bounds=[-0.5236, 0.5236])  # 30 deg
    elif args.action == "guide":
        control_utils.get_pose_from_guide_mode(franka_arm=franka_arm, max_duration=60.0)
    else:
        raise ValueError("Action is invalid.")
