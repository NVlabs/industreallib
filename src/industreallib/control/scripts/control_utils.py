# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Control utilities module.

This module defines utility functions for controlling a Franka robot with
the frankapy library.
"""

# Standard Library
import random
import signal

# Third Party
import numpy as np
import rospy
from autolab_core import RigidTransform
from frankapy import SensorDataMessageType
from frankapy.proto import CartesianImpedanceSensorMessage, PosePositionSensorMessage
from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg
from scipy.spatial.transform import Rotation


def open_gripper(franka_arm):
    """Opens the gripper."""
    print("\nOpening gripper...")
    franka_arm.open_gripper()
    print("Opened gripper.")


def close_gripper(franka_arm):
    """Closes the gripper."""
    print("\nClosing gripper...")
    franka_arm.close_gripper()
    print("Closed gripper.")


def go_to_joint_angles(franka_arm, joint_angles, duration):
    """Goes to a specified set of joint angles."""
    print("\nGoing to goal joint angles...")
    franka_arm.goto_joints(
        joints=joint_angles, duration=duration, use_impedance=False, ignore_virtual_walls=True
    )
    print("Finished going to goal joint angles.")

    print_joint_angles(franka_arm=franka_arm)


def go_to_pos(franka_arm, pos, duration):
    """Goes to a specified position, with gripper pointing downward."""
    # Compose goal transform
    transform = RigidTransform(
        translation=pos,
        rotation=[[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        from_frame="franka_tool",
        to_frame="world",
    )

    print("\nGoing to goal position...")
    franka_arm.goto_pose(
        tool_pose=transform, duration=duration, use_impedance=False, ignore_virtual_walls=True
    )
    print("Finished going to goal position.")

    curr_pose = franka_arm.get_pose()
    print("\nCurrent position:", curr_pose.translation)


def go_to_pose(franka_arm, pos, ori_mat, duration, use_impedance):
    """Goes to a specified pose."""
    # Compose goal transform
    transform = RigidTransform(
        translation=pos, rotation=ori_mat.T, from_frame="franka_tool", to_frame="world"
    )

    print("\nGoing to goal pose...")
    franka_arm.goto_pose(
        tool_pose=transform,
        duration=duration,
        use_impedance=use_impedance,
        ignore_virtual_walls=True,
    )
    print("Finished going to goal pose.")

    print_pose(franka_arm=franka_arm)


def go_home(franka_arm, duration):
    """Goes to a hard-coded home configuration."""
    # NOTE: from frankapy/frankapy/franka_arm.py,
    # FrankaPy home pose defined as
    #   joint angles: [0.0, -pi / 4, 0.0, -3 * pi / 4, 0.0, pi / 2, pi / 4]
    # or
    #   end-effector pose:
    #     translation: [0.3069, 0.0, 0.4867]
    #     rotation: [1  0  0]
    #               [0 -1  0]
    #               [0  0 -1]
    # IndustRealLib home pose defined as relative offset of
    #   translation: [0.2, 0.0, 0.0]
    # which corresponds to
    #   joint angles (measured): [0.0, -1.76076077e-01, 0.0, -1.86691416e+00,
    #                             0.0, 1.69344379e+00, pi / 4]

    print("\nGoing to home configuration...")
    go_to_joint_angles(
        franka_arm=franka_arm,
        joint_angles=[0.0, -1.76076077e-01, 0.0, -1.86691416e00, 0.0, 1.69344379e00, np.pi / 4],
        duration=duration,
    )
    print("Reached home configuration.")


def go_upward(franka_arm, dist, duration):
    """Goes upward by a specified distance while maintaining gripper orientation."""
    # Compose delta transform to goal
    transform = RigidTransform(
        translation=[0.0, 0.0, dist],
        rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        from_frame="world",
        to_frame="world",
    )

    print("\nGoing upward...")
    franka_arm.goto_pose_delta(
        delta_tool_pose=transform, duration=duration, use_impedance=False, ignore_virtual_walls=True
    )
    print("Finished going upward.")


def go_downward(franka_arm, dist, duration):
    """Goes downward by a specified distance while maintaining gripper orientation."""
    # Compose delta transform to goal
    transform = RigidTransform(
        translation=[0.0, 0.0, -dist],
        rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        from_frame="world",
        to_frame="world",
    )

    print("\nGoing downward...")
    franka_arm.goto_pose_delta(
        delta_tool_pose=transform, duration=duration, use_impedance=False, ignore_virtual_walls=True
    )
    print("Finished going downward.")


def get_pose_from_guide_mode(franka_arm, max_duration):
    """Activates guide mode. When complete, gets the gripper pose."""
    print("\nStarted guide mode.")

    input("Caution: Robot may drop. Press Enter to continue...")
    franka_arm.run_guide_mode(duration=max_duration, block=False)

    input(
        f"Robot is in guide mode for max duration of {max_duration} seconds. Press Enter to"
        " terminate..."
    )
    franka_arm.stop_skill()

    print("Stopped guide mode.")

    curr_pose = franka_arm.get_pose()
    print_pose(franka_arm=franka_arm)

    return curr_pose


def print_joint_angles(franka_arm):
    """Prints the current joint angles."""
    curr_ang = franka_arm.get_joints()
    print("\nCurrent joint angles:\n", curr_ang)


def print_pose(franka_arm):
    """Prints the current end-effector pose."""
    curr_pose = franka_arm.get_pose()
    print("\nCurrent pose:")
    print("-------------")
    print("Position:", curr_pose.translation)
    print("Orientation:\n", curr_pose.rotation)
    # print('Orientation:\n', Rotation.from_matrix(curr_pose.rotation).as_euler('xyz'))


def get_pose_error(curr_pos, curr_ori_mat, targ_pos, targ_ori_mat):
    """Gets the error between a current pose and a target pose."""
    # Compute position error
    pos_err = np.linalg.norm(targ_pos - curr_pos)

    # Compute orientation error in radians
    ori_err_rad = (
        Rotation.from_matrix(targ_ori_mat) * Rotation.from_matrix(curr_ori_mat).inv()
    ).magnitude()

    return pos_err, ori_err_rad


def print_pose_error(curr_pos, curr_ori_mat, targ_pos, targ_ori_mat):
    """Prints the current pose, the target pose, and the error between the two poses."""
    print("\nCurrent pose:")
    print("-------------")
    print("Position:", curr_pos)
    print("Orientation:\n", curr_ori_mat)

    print("\nTarget pose:")
    print("-------------")
    print("Position:", targ_pos)
    print("Orientation:\n", targ_ori_mat)

    pos_err, ori_err_rad = get_pose_error(
        curr_pos=curr_pos, curr_ori_mat=curr_ori_mat, targ_pos=targ_pos, targ_ori_mat=targ_ori_mat
    )
    print("\nPose error:")
    print("-------------")
    print("Position:", pos_err)
    print("Orientation:", ori_err_rad)


def perturb_xy_pos(franka_arm, radial_bound):
    """Randomly perturbs the xy-position within a specified radius."""
    # Use rejection sampling to randomly sample delta_x and delta_y within circle
    curr_dist = np.inf
    while curr_dist > radial_bound:
        # Sample delta_x and delta_y within bounding square
        delta_x = random.uniform(-radial_bound, radial_bound)
        delta_y = random.uniform(-radial_bound, radial_bound)
        curr_dist = np.linalg.norm([delta_x, delta_y])

    # Compose delta transform to goal
    transform = RigidTransform(
        translation=[delta_x, delta_y, 0.0],
        rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        from_frame="world",
        to_frame="world",
    )

    print("\nPerturbing xy-position...")
    franka_arm.goto_pose_delta(
        delta_tool_pose=transform, use_impedance=False, ignore_virtual_walls=True
    )
    print("Finished perturbing xy-position.")


def perturb_z_pos(franka_arm, bounds):
    """Randomly perturbs the z-position within a specified range."""
    delta_z = random.uniform(bounds[0], bounds[1])

    # Compose delta transform to goal
    transform = RigidTransform(
        translation=[0.0, 0.0, delta_z],
        rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        from_frame="world",
        to_frame="world",
    )

    print("\nPerturbing z-position...")
    franka_arm.goto_pose_delta(
        delta_tool_pose=transform, use_impedance=False, ignore_virtual_walls=True
    )
    print("Finished perturbing z-position.")


def perturb_yaw(franka_arm, bounds):
    """Randomly perturbs the gripper yaw angle within a specified range."""
    delta_yaw = random.uniform(bounds[0], bounds[1])

    # Define goal configuration
    curr_joint_angles = franka_arm.get_joints()
    targ_joint_angles = curr_joint_angles.copy()
    targ_joint_angles[-1] += delta_yaw

    print("\nPerturbing yaw...")
    franka_arm.goto_joints(joints=targ_joint_angles, use_impedance=False, ignore_virtual_walls=True)
    print("Finished perturbing yaw.")


def get_vec_rot_mat(unit_vec_a, unit_vec_b):
    """Gets rotation matrices that rotate one set of unit vectors to another set of unit vectors."""
    # Compute rotation axes (axis = u cross v / norm(u cross v))
    cross_prod = np.cross(unit_vec_a, unit_vec_b)  # (num_vecs, 3)
    cross_prod_norm = np.expand_dims(np.linalg.norm(cross_prod, axis=1), axis=1)  # (num_vecs, 1)
    rot_axis = cross_prod / cross_prod_norm  # (num_vecs, 3)

    # Compute rotation angles (theta = arccos(u dot v))
    rot_angle = np.expand_dims(
        np.arccos(np.einsum("ij,ij->i", unit_vec_a, unit_vec_b)), axis=1
    )  # (num_vecs, 1)

    # Compute axis-angle representation of rotation
    rot_axis_angle = rot_axis * rot_angle  # (num_vecs, 3)

    # Compute rotation matrix
    rot_mat = Rotation.from_rotvec(rot_axis_angle).as_matrix()  # (num_vecs, 3, 3)

    return rot_mat


def compose_ros_msg(targ_pos, targ_ori_quat, prop_gains, msg_count):
    """Composes a ROS message to send to franka-interface for task-space impedance control."""
    # NOTE: Closely adapted from
    # https://github.com/iamlab-cmu/frankapy/blob/master/examples/run_dynamic_pose.py
    # NOTE: The sensor message classes expect the input quaternions to be represented as
    # (w, x, y, z).

    curr_time = rospy.Time.now().to_time()
    proto_msg_pose = PosePositionSensorMessage(
        id=msg_count, timestamp=curr_time, position=targ_pos, quaternion=targ_ori_quat
    )
    proto_msg_impedance = CartesianImpedanceSensorMessage(
        id=msg_count,
        timestamp=curr_time,
        translational_stiffnesses=prop_gains[:3],
        rotational_stiffnesses=prop_gains[3:6],
    )
    ros_msg = make_sensor_group_msg(
        trajectory_generator_sensor_msg=sensor_proto2ros_msg(
            sensor_proto_msg=proto_msg_pose, sensor_data_type=SensorDataMessageType.POSE_POSITION
        ),
        feedback_controller_sensor_msg=sensor_proto2ros_msg(
            sensor_proto_msg=proto_msg_impedance,
            sensor_data_type=SensorDataMessageType.CARTESIAN_IMPEDANCE,
        ),
    )

    return ros_msg


def set_sigint_response(franka_arm):
    """Sets a custom response to a SIGINT signal, which is executed on Ctrl + C."""

    def handler(signum, frame):
        """Defines a custom handler that stops a FrankaPy skill."""
        # NOTE: Without this code block, if Ctrl + C is executed during a FrankaPy dynamic skill,
        # the robot may exhibit unexpected behavior (e.g., sudden motions) when a dynamic skill
        # is executed in a subsequent experiment.
        franka_arm.stop_skill()
        raise KeyboardInterrupt  # default behavior

    signal.signal(signal.SIGINT, handler)
