# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Calibrate extrinsics script.

This script is a standalone script that calibrates the extrinsics of a wrist-
mounted Intel RealSense camera on a Franka robot. The script loads parameters
for the calibration procedure from a specified YAML file, runs a calibration
procedure where multiple robot and tag poses are recorded, computes the
extrinsics, and writes them to a JSON file.

Typical usage example:
python calibrate_extrinsics.py -p perception.yaml
"""

# Standard Library
import argparse
import json
import os
import time

# Third Party
import cv2
import numpy as np
import pupil_apriltags as apriltag
from frankapy import FrankaArm

# NVIDIA
import industreallib.control.scripts.control_utils as control_utils
import industreallib.perception.scripts.perception_utils as perception_utils


def get_args():
    """Gets arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--perception_config_file_name",
        required=True,
        help="Perception configuration to load",
    )

    args = parser.parse_args()

    return args


def _get_robot_goals(config):
    """Generates robot goal poses.

    The position is random, and the orientation points at a specified position.
    """
    # Generate random goal positions
    rand_pos_x = np.random.uniform(
        config.robot.goal_pos_bounds.x[0],
        config.robot.goal_pos_bounds.x[1],
        size=(config.robot.num_goals, 1),
    )
    rand_pos_y = np.random.uniform(
        config.robot.goal_pos_bounds.y[0],
        config.robot.goal_pos_bounds.y[1],
        size=(config.robot.num_goals, 1),
    )
    rand_pos_z = np.random.uniform(
        config.robot.goal_pos_bounds.z[0],
        config.robot.goal_pos_bounds.z[1],
        size=(config.robot.num_goals, 1),
    )
    goal_pos = np.concatenate((rand_pos_x, rand_pos_y, rand_pos_z), axis=1)  # (num_goals, 3)

    # Compute unit goal orientation vectors (i.e., unit vectors pointing from goal positions to
    # specified position). These vectors will be desired z-axis for end effector
    goal_ori_vec = config.robot.pos_to_point_at - goal_pos  # (num_goals, 3)
    goal_ori_vec_norm = np.expand_dims(
        np.linalg.norm(goal_ori_vec, axis=1), axis=1
    )  # (num_goals, 1)
    unit_goal_ori_vec = goal_ori_vec / goal_ori_vec_norm  # (num_goals, 3)

    # Compute arbitrary perpendicular unit vectors
    # These vectors will be desired x- and y-axes for end effector
    perp_vec_a = np.cross(unit_goal_ori_vec, np.array([0.0, 1.0, 0.0]))  # (num_goals, 3)
    perp_vec_a_norm = np.expand_dims(np.linalg.norm(perp_vec_a, axis=1), axis=1)  # (num_goals, 1)
    unit_perp_vec_a = perp_vec_a / perp_vec_a_norm  # (num_goals, 3)
    perp_vec_b = np.cross(unit_goal_ori_vec, unit_perp_vec_a)  # (num_goals, 3)
    perp_vec_b_norm = np.expand_dims(np.linalg.norm(perp_vec_b, axis=1), axis=1)  # (num_goals, 1)
    unit_perp_vec_b = perp_vec_b / perp_vec_b_norm  # (num_goals, 3)

    # Construct goal orientation matrices
    goal_ori_mat = np.transpose(
        np.concatenate(
            (unit_perp_vec_a, unit_perp_vec_b, unit_goal_ori_vec), axis=1
        ).reshape(
            (config.robot.num_goals, 3, 3)
        ), axes=(0, 2, 1)
    )  # (num_goals, 3, 3)

    return goal_pos, goal_ori_mat


def _run_calibration(config, franka_arm, goal_posits, goal_ori_mats, detector, intrinsics):
    """Gets robot and tag poses from multiple viewpoints. Gets the extrinsics."""
    robot_poses_t, robot_poses_r, tag_poses_t, tag_poses_r = _collect_robot_and_tag_poses(
        config, franka_arm, goal_posits, goal_ori_mats, detector, intrinsics
    )

    camera_pose_t, camera_pose_r = _get_extrinsics(
        robot_poses_t=robot_poses_t,
        robot_poses_r=robot_poses_r,
        tag_poses_t=tag_poses_t,
        tag_poses_r=tag_poses_r,
    )

    return camera_pose_t, camera_pose_r


def _collect_robot_and_tag_poses(
    config, franka_arm, goal_posits, goal_ori_mats, detector, intrinsics
):
    """Moves robot to each goal pose. Detects the AprilTag. Gets the robot pose and tag pose."""
    _go_home(franka_arm=franka_arm, home_joint_angles=config.robot.home_joint_angles)

    num_tag_detections = 0
    robot_poses_t, robot_poses_r, tag_poses_t, tag_poses_r = [], [], [], []

    for goal_pos, goal_ori_mat in zip(goal_posits, goal_ori_mats):
        # If position-to-point-at is directly below goal, reject goal
        # Reference: https://github.com/IFL-CAMP/easy_handeye/blob/master/docs/troubleshooting.md
        if (
            np.abs(goal_pos[0] - config.robot.pos_to_point_at[0]) < 0.05
            and np.abs(goal_pos[1] - config.robot.pos_to_point_at[1]) < 0.05
        ):
            print("\nOverhead sample rejected.")
            continue

        robot_pose_t, robot_pose_r = _move_robot_to_goal(
            franka_arm=franka_arm, goal_pos=goal_pos, goal_ori_mat=goal_ori_mat
        )

        image = perception_utils.get_image(
            pipeline=pipeline, display_images=config.tag_detection.display_images
        )
        (
            is_tag_detected,
            tag_pose_t,
            tag_pose_r,
            _,
            tag_corner_pixels,
            tag_family,
        ) = perception_utils.get_tag_pose_in_camera_frame(
            detector=detector,
            image=image,
            intrinsics=intrinsics,
            tag_length=config.tag.length,
            tag_active_pixel_ratio=config.tag.active_pixel_ratio,
        )

        if is_tag_detected:
            print("\nTag detected.")
            num_tag_detections += 1
            print(f"Collected {num_tag_detections} tag detections.")

            robot_poses_t.append(robot_pose_t.tolist())
            robot_poses_r.append(robot_pose_r.tolist())
            tag_poses_t.append(tag_pose_t.tolist())
            tag_poses_r.append(tag_pose_r.tolist())

            # Draw labels on image
            image_labeled = perception_utils.label_tag_detection(
                image=image, tag_corner_pixels=tag_corner_pixels, tag_family=tag_family
            )

            if config.tag_detection.display_images:
                cv2.imshow("Tag Detection", image_labeled)
                cv2.waitKey(delay=2000)
                cv2.destroyAllWindows()

        else:
            print("\nTag not detected.")

        # Return home (otherwise elbow will drift, as no nullspace regularization)
        _go_home(franka_arm=franka_arm, home_joint_angles=config.robot.home_joint_angles)

        if num_tag_detections == config.tag_detection.num_detections:
            break

    return robot_poses_t, robot_poses_r, tag_poses_t, tag_poses_r


def _move_robot_to_goal(franka_arm, goal_pos, goal_ori_mat):
    """Moves the robot to a goal pose."""
    # First use frankapy controller (better IK) to go to pose,
    # then use libfranka controller (better accuracy) to go to same pose
    control_utils.go_to_pose(
        franka_arm=franka_arm, pos=goal_pos, ori_mat=goal_ori_mat, duration=5.0, use_impedance=True
    )
    control_utils.go_to_pose(
        franka_arm=franka_arm, pos=goal_pos, ori_mat=goal_ori_mat, duration=5.0, use_impedance=False
    )

    _spin_robot_end_effector(franka_arm=franka_arm)

    print("\nAllowing vibrations to decay...")
    time.sleep(5.0)
    print("Allowed vibrations to decay.")

    curr_pose = franka_arm.get_pose()
    curr_pos = curr_pose.translation.copy()
    curr_ori_mat = curr_pose.rotation.copy()

    return curr_pos, curr_ori_mat


def _spin_robot_end_effector(franka_arm):
    """Spins the robot end effector."""
    control_utils.perturb_yaw(franka_arm=franka_arm, bounds=[-np.pi / 4, np.pi / 4])


def _go_home(franka_arm, home_joint_angles):
    """Moves the robot to the home joint configuration."""
    control_utils.go_to_joint_angles(
        franka_arm=franka_arm, joint_angles=home_joint_angles, duration=5.0
    )


def _get_extrinsics(robot_poses_t, robot_poses_r, tag_poses_t, tag_poses_r):
    """Gets the extrinsics (i.e., the camera pose in the end-effector frame)."""
    # NOTE: Manually-calculated extrinsics from CAD are
    # translation: [[0.03625], [-0.03277814], [-0.04985334]]
    # rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    # Z-translation calculated from (top of sensor board - focal length)

    camera_pose_r, camera_pose_t = cv2.calibrateHandEye(
        R_gripper2base=np.asarray(robot_poses_r),
        t_gripper2base=np.asarray(robot_poses_t),
        R_target2cam=np.asarray(tag_poses_r),
        t_target2cam=np.asarray(tag_poses_t),
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    # Algorithm selected based on https://www.youtube.com/watch?v=xQ79ysnrzUk

    print("\nComputed extrinsics.")

    return camera_pose_t, camera_pose_r


def _save_extrinsics(config, camera_pose_t, camera_pose_r):
    """Saves the intrinsics to a JSON file."""
    with open(os.path.join(os.path.dirname(__file__), '..', 'io', config.output.file_name), "w") as f:
        extrinsics = {"position": camera_pose_t.tolist(), "orientation": camera_pose_r.tolist()}
        json.dump(extrinsics, f)
    print("\nSaved extrinsics to file.")


if __name__ == "__main__":
    """Initializes the robot and camera. Generates goals. Runs calibration. Saves the extrinsics."""

    args = get_args()
    config = perception_utils.get_perception_config(
        file_name=args.perception_config_file_name, module_name="calibrate_extrinsics"
    )

    # Initialize robot, camera, and AprilTag detector
    franka_arm = FrankaArm()
    pipeline = perception_utils.get_camera_pipeline(
        width=config.camera.image_width, height=config.camera.image_height
    )
    intrinsics = perception_utils.get_intrinsics(pipeline=pipeline)
    detector = apriltag.Detector(
        families=config.tag.type, quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25
    )

    goal_posits, goal_ori_mats = _get_robot_goals(config=config)

    camera_pose_t, camera_pose_r = _run_calibration(
        config=config,
        franka_arm=franka_arm,
        goal_posits=goal_posits,
        goal_ori_mats=goal_ori_mats,
        detector=detector,
        intrinsics=intrinsics,
    )

    _save_extrinsics(config=config, camera_pose_t=camera_pose_t, camera_pose_r=camera_pose_r)
