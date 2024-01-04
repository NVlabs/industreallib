# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Class definition for IndustRealTaskBase.

This script defines the IndustRealTaskBase base class. The class defines
the variables and methods that are common for all tasks.
"""

# Standard Library
import json
import os
import random

# Third Party
import numpy as np
import rospy
import torch
import yaml
from autolab_core import RigidTransform
from franka_interface_msgs.msg import SensorDataGroup
from frankapy import FrankaArm, FrankaConstants
from gym.spaces import Box
from rl_games.algos_torch.players import PpoPlayerContinuous
from scipy.spatial.transform import Rotation

# NVIDIA
import industreallib.control.scripts.control_utils as control_utils
import industreallib.perception.scripts.detect_objects as detect_objects
import industreallib.perception.scripts.map_workspace as map_workspace
import industreallib.perception.scripts.perception_utils as perception_utils


class IndustRealTaskBase:
    """Defines base class for all tasks."""

    def __init__(self, args, task_instance_config, in_sequence):
        """Initializes the configuration, goals, and robot for the task."""
        self._args = args
        self.task_instance_config = task_instance_config
        self._home_joint_angles = [
            0.0,
            -1.76076077e-01,
            0.0,
            -1.86691416e00,
            0.0,
            1.69344379e00,
            np.pi / 4,
        ]
        self._policy = None
        self.goal_coords = []
        self.goal_labels = []
        self._ros_publisher = None
        self._ros_rate = None
        self._ros_msg_count = 0
        self._device = "cuda"

        # For PLAI
        self._prev_targ_pos = None
        self._prev_targ_ori_mat = None

        # If task is not part of sequence, instantiate FrankaArm and get goals
        if not in_sequence:
            self.franka_arm = FrankaArm()
            control_utils.set_sigint_response(franka_arm=self.franka_arm)
            self._get_goals()

    def _get_goals(self):
        """Gets goals for the task."""
        # NOTE: All goals are assumed to be in the form of [goal_position; goal_orientation],
        # where the position and orientation goals are in the robot base frame, and the
        # orientation goals are Euler angles (intrinsic).

        if self.task_instance_config.goals.source == "random":
            self._get_goals_randomly()
        elif self.task_instance_config.goals.source == "perception":
            self._get_goals_from_perception()
        elif self.task_instance_config.goals.source == "guide":
            self._get_goals_from_guide_mode()
        elif self.task_instance_config.goals.source == "file":
            self._get_goals_from_file()
        else:
            raise ValueError("Invalid goal source.")

        if self.task_instance_config.rl.observation_noise_pos is not None:
            self._apply_noise_to_goals()

    def _get_goals_randomly(self):
        """Gets goals for the task by randomly generating them within specified bounds."""
        for _ in range(self.task_instance_config.goals.random.num_goals):
            goal = [
                random.uniform(bound[0], bound[1])
                for bound in self.task_instance_config.goals.random.bounds
            ]
            self.goal_coords.append(goal)

        if self._args.debug_mode:
            print("\nGoals:")
            for goal in self.goal_coords:
                print(goal)

    def _get_goals_from_perception(self):
        """Gets goals for the task based on perception."""
        object_coords, object_labels = self._get_object_coords_from_perception(
            perception_config_file_name=self.task_instance_config.goals.perception.config
        )
        self.convert_object_coords_to_goals(
            object_coords=object_coords,
            object_labels=object_labels,
            perception_config_file_name=self.task_instance_config.goals.perception.config,
        )

        if self._args.debug_mode:
            print("\nGoals:")
            for label, goal in zip(self.goal_labels, self.goal_coords):
                print(f"{label}: {goal}")

    def _get_goals_from_guide_mode(self):
        """Gets goals for the task by allowing the user to guide the robot to the goal."""
        # NOTE: This procedure is useful for insertion-only experiments. Currently,
        # getting multiple goals is not supported.

        pose = control_utils.get_pose_from_guide_mode(franka_arm=self.franka_arm, max_duration=60.0)
        goal = np.asarray([*pose.translation, *Rotation.from_matrix(pose.rotation).as_euler("XYZ")])
        self.goal_coords = [goal]

        # Perturb position and yaw angle
        if self.task_instance_config.goals.guide.z_perturbation is not None:
            control_utils.close_gripper(franka_arm=self.franka_arm)
            control_utils.perturb_z_pos(
                franka_arm=self.franka_arm,
                bounds=self.task_instance_config.goals.guide.z_perturbation,
            )
        if self.task_instance_config.goals.guide.xy_perturbation is not None:
            control_utils.perturb_xy_pos(
                franka_arm=self.franka_arm,
                radial_bound=self.task_instance_config.goals.guide.xy_perturbation,
            )
        if self.task_instance_config.goals.guide.yaw_perturbation is not None:
            control_utils.perturb_yaw(
                franka_arm=self.franka_arm,
                bounds=self.task_instance_config.goals.guide.yaw_perturbation,
            )

        if self._args.debug_mode:
            print("\nGoals:")
            for goal in self.goal_coords:
                print(goal)

    def _get_goals_from_file(self):
        """Gets goals for the task from a JSON file."""
        with open(self.task_instance_config.goals.file.path) as f:
            json_obj = f.read()
        self.goal_coords = json.loads(json_obj)

        if self._args.debug_mode:
            print("\nGoals:")
            for goal in self.goal_coords:
                print(goal)

    def _get_object_coords_from_perception(self, perception_config_file_name):
        """Gets object coordinates (x, y, theta) from perception."""
        # Map workspace (saves to JSON)
        map_workspace.main(
            perception_config_file_name=perception_config_file_name, franka_arm=self.franka_arm
        )

        # Detect objects and get object coordinates (x, y, theta)
        object_coords, object_labels = detect_objects.main(
            perception_config_file_name=perception_config_file_name
        )
        if not object_coords:
            raise ValueError(
                "No objects detected. Make sure checkpoint and scene are specified                 "
                "             correctly in perception configuration file."
            )

        return object_coords, object_labels

    def convert_object_coords_to_goals(
        self, object_coords, object_labels, perception_config_file_name
    ):
        """Converts object coordinates (x, y, theta) to goals (x, y, z, roll, pitch, yaw)."""
        perception_config = perception_utils.get_perception_config(
            file_name=perception_config_file_name, module_name="generate_goals"
        )

        # For each detection
        for coords, label in zip(object_coords, object_labels):
            # For each desired detection
            for selector in self.task_instance_config.goals.perception.selectors:
                # If current detection is desired detection
                if label == selector:
                    # Define default goal
                    goal = np.asarray(
                        [
                            coords[0],
                            coords[1],
                            self.task_instance_config.goals.perception.goal_heights[label],
                            np.pi,
                            0.0,
                            coords[2],
                        ]
                    )

                    # Ignore yaw angle for round objects
                    if "round" in selector:
                        goal[5] = 0.0

                    # Apply lateral goal offset for asymmetric parts (e.g., gear base)
                    if self.task_instance_config.goals.perception.goal_lateral_offsets is not None:
                        lateral_offset_local = np.expand_dims(
                            np.asarray(
                                self.task_instance_config.goals.perception.goal_lateral_offsets[
                                    label
                                ]
                            ),
                            axis=1,
                        )
                        from_local_to_global_r = np.array(
                            [
                                [np.cos(goal[5]), -np.sin(goal[5])],
                                [np.sin(goal[5]), np.cos(goal[5])],
                            ]
                        )
                        lateral_offset_global = np.squeeze(
                            from_local_to_global_r @ lateral_offset_local
                        )
                        goal[:2] += lateral_offset_global

                    # Apply one-time perception offset
                    goal += np.asarray([*perception_config.one_time_offset, 0.0, 0.0, 0.0])

                    self.goal_coords.append(goal.tolist())
                    self.goal_labels.append(label)

    def _apply_noise_to_goals(self):
        """Applies observation noise to goals."""
        x_noise = random.uniform(
            -self.task_instance_config.rl.observation_noise_pos[0],
            self.task_instance_config.rl.observation_noise_pos[0],
        )
        y_noise = random.uniform(
            -self.task_instance_config.rl.observation_noise_pos[1],
            self.task_instance_config.rl.observation_noise_pos[1],
        )
        z_noise = random.uniform(
            -self.task_instance_config.rl.observation_noise_pos[2],
            self.task_instance_config.rl.observation_noise_pos[2],
        )

        goal_coords = np.asarray(self.goal_coords)
        goal_coords[:, :3] += np.asarray([x_noise, y_noise, z_noise])
        self.goal_coords = goal_coords.tolist()

    def go_to_goals(self):
        """Goes to each goal in a list of goals. Performs actions before and after each goal."""
        for goal in self.goal_coords:
            self.do_simple_procedure(
                procedure=self.task_instance_config.motion.do_before, franka_arm=self.franka_arm
            )
            self.go_to_goal(goal=goal, franka_arm=self.franka_arm)
            self.do_simple_procedure(
                procedure=self.task_instance_config.motion.do_after, franka_arm=self.franka_arm
            )

    def go_to_goal(self, goal, franka_arm):
        """Goes to a goal."""
        if self.task_instance_config.motion.source == "rl":
            if self._policy is None:
                self._policy = self._get_policy()
            self._go_to_goal_with_rl(goal=goal, franka_arm=franka_arm)
        elif self.task_instance_config.motion.source in [
            "frankapy",
            "libfranka",
            "frankapy_and_libfranka",
        ]:
            self._go_to_goal_with_baseline(goal=goal, franka_arm=franka_arm)
        else:
            raise ValueError("Invalid motion source.")

    def _go_to_goal_with_rl(self, goal, franka_arm):
        """Goes to a goal using an RL policy."""
        goal_pos = goal[:3]
        goal_ori_mat = Rotation.from_euler("XYZ", goal[3:]).as_matrix()  # intrinsic rotations

        self._start_target_stream(franka_arm=franka_arm)
        print("\nStarted streaming targets.")

        print("\nGoing to goal pose with RL...")
        # Get observations, get actions, send targets, and repeat
        initial_time = rospy.get_time()
        while rospy.get_time() - initial_time < self.task_instance_config.motion.duration:
            observations, curr_state = self._get_observations(
                goal_pos=goal_pos, goal_ori_mat=goal_ori_mat, franka_arm=franka_arm
            )
            actions = self._get_actions(observations=observations)
            self._send_targets(
                actions=actions,
                curr_pos=curr_state["pose"].translation,
                curr_ori_mat=curr_state["pose"].rotation,
            )

            # If current pose is close enough to goal pose, terminate early
            pos_err, ori_err_rad = control_utils.get_pose_error(
                curr_pos=curr_state["pose"].translation,
                curr_ori_mat=curr_state["pose"].rotation,
                targ_pos=goal_pos,
                targ_ori_mat=goal_ori_mat,
            )
            if (
                pos_err < self.task_instance_config.rl.pos_err_thresh
                and ori_err_rad < self.task_instance_config.rl.ori_err_rad_thresh
            ):
                print("Terminated early due to error below threshold.")
                break

            self._ros_rate.sleep()
        print("Finished going to goal pose with RL.")

        franka_arm.stop_skill()
        print("\nStopped streaming targets.")

        self._prev_targ_pos, self._prev_targ_ori_mat = None, None

        if self._args.debug_mode:
            control_utils.print_pose_error(
                curr_pos=curr_state["pose"].translation,
                curr_ori_mat=curr_state["pose"].rotation,
                targ_pos=goal_pos,
                targ_ori_mat=goal_ori_mat,
            )

    def _go_to_goal_with_baseline(self, goal, franka_arm):
        """Goes to a goal using a baseline method (frankapy or libfranka)."""
        goal_pos = goal[:3]
        goal_ori_mat = Rotation.from_euler("XYZ", goal[3:]).as_matrix()  # intrinsic rotations
        transform = RigidTransform(
            translation=goal_pos, rotation=goal_ori_mat, from_frame="franka_tool", to_frame="world"
        )

        print(f"\nGoing to goal pose with {self.task_instance_config.motion.source}...")
        if "frankapy" in self.task_instance_config.motion.source:
            franka_arm.goto_pose(
                tool_pose=transform,
                duration=self.task_instance_config.motion.duration,
                use_impedance=True,
                ignore_virtual_walls=True,
            )
        if "libfranka" in self.task_instance_config.motion.source:
            franka_arm.goto_pose(
                tool_pose=transform,
                duration=self.task_instance_config.motion.duration,
                use_impedance=False,
                ignore_virtual_walls=True,
            )
        print(f"Finished going to goal pose with {self.task_instance_config.motion.source}.")

        if self._args.debug_mode:
            curr_pose = franka_arm.get_pose()
            control_utils.print_pose_error(
                curr_pos=curr_pose.translation,
                curr_ori_mat=curr_pose.rotation,
                targ_pos=goal_pos,
                targ_ori_mat=goal_ori_mat,
            )

    def _get_policy(self):
        """Gets an RL policy from rl-games."""
        # NOTE: Only PPO policies are currently supported.

        print("\nLoading an RL policy...")

        # Load config.yaml used in training
        with open(
            os.path.join(os.path.dirname(__file__), '..', '..', 'rl', 'checkpoints',
            self.task_instance_config.rl.checkpoint_name, 'config.yaml'),
            "r",
        ) as f:
            sim_config = yaml.safe_load(f)

        # Define env_info dict
        # NOTE: If not defined, rl-games will call rl_games.common.player.create_env() and
        # rl_games.common.env_configurations.get_env_info(). Afterward, rl-games will query
        # num_observations and num_actions. We only need to support those queries.
        # See rl_games.common.player.__init__() for more details.
        env_info = {
            "observation_space": Box(
                low=-np.Inf,
                high=np.Inf,
                shape=(sim_config["task"]["env"]["numObservations"],),
                dtype=np.float32,
            ),
            "action_space": Box(
                low=-1.0,
                high=1.0,
                shape=(sim_config["task"]["env"]["numActions"],),
                dtype=np.float32,
            ),
        }
        sim_config["train"]["params"]["config"]["env_info"] = env_info

        # Select device
        sim_config["train"]["params"]["config"]["device_name"] = self._device

        # Create rl-games agent
        policy = PpoPlayerContinuous(params=sim_config["train"]["params"])

        # Restore policy from checkpoint
        policy.restore(
            fn=(
                os.path.join(os.path.dirname(__file__), '..', '..', 'rl', 'checkpoints',
                self.task_instance_config.rl.checkpoint_name, 'nn',
                f"{self.task_instance_config.rl.checkpoint_name}.pth")
            )
        )

        # If RNN policy, reset RNN states
        policy.reset()

        print("Finished loading an RL policy.")

        return policy

    def _start_target_stream(self, franka_arm):
        """Starts streaming targets to franka-interface via frankapy."""
        self._ros_rate = rospy.Rate(self.task_instance_config.rl.policy_eval_freq)
        self._ros_publisher = rospy.Publisher(
            FrankaConstants.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000
        )

        # Initiate streaming with dummy command to go to current pose
        # NOTE: Closely adapted from
        # https://github.com/iamlab-cmu/frankapy/blob/master/examples/run_dynamic_pose.py
        franka_arm.goto_pose(
            tool_pose=franka_arm.get_pose(),
            duration=1.0,
            use_impedance=True,
            dynamic=True,
            buffer_time=self.task_instance_config.motion.duration * 10.0,
            cartesian_impedances=list(self.task_instance_config.control.prop_gains),
            ignore_virtual_walls=True,
        )

    def _get_observations(self):
        """Gets observations from frankapy. Should be defined in a task-specific subclass."""
        raise NotImplementedError

    def _get_actions(self, observations):
        """Gets actions from the policy. Applies action scaling factors."""
        actions = self._policy.get_action(obs=observations, is_deterministic=True)
        actions *= torch.tensor(
            self.task_instance_config.control.mode[
                self.task_instance_config.control.mode.type
            ].action_scale,
            dtype=torch.float32,
            device=self._device,
        )
        actions = actions.detach().cpu().numpy()

        return actions

    def _send_targets(self, actions, curr_pos, curr_ori_mat):
        """Sends pose targets to franka-interface via frankapy."""
        # NOTE: All actions are assumed to be in the form of [delta_position; delta_orientation],
        # where delta position is in the robot base frame, delta orientation is in the end-effector
        # frame, and delta orientation is an Euler vector (i.e., 3-element axis-angle
        # representation).

        if self.task_instance_config.control.mode.type == "nominal":
            targ_pos = curr_pos + actions[:3]
            targ_ori_mat = Rotation.from_rotvec(actions[3:6]).as_matrix() @ curr_ori_mat

        elif self.task_instance_config.control.mode.type in ["plai", "leaky_plai"]:
            if self._prev_targ_pos is None:
                self._prev_targ_pos = curr_pos.copy()
            if self._prev_targ_ori_mat is None:
                self._prev_targ_ori_mat = curr_ori_mat.copy()

            targ_pos = self._prev_targ_pos + actions[:3]
            targ_ori_mat = Rotation.from_rotvec(actions[3:6]).as_matrix() @ self._prev_targ_ori_mat

            if self.task_instance_config.control.mode.type == "leaky_plai":
                pos_err = targ_pos - curr_pos
                pos_err_clip = np.clip(
                    pos_err,
                    a_min=-np.asarray(
                        self.task_instance_config.control.mode.leaky_plai.pos_err_thresh
                    ),
                    a_max=np.asarray(
                        self.task_instance_config.control.mode.leaky_plai.pos_err_thresh
                    ),
                )
                targ_pos = curr_pos + pos_err_clip

                # TODO: Implement leaky PLAI for rotation

            self._prev_targ_pos = targ_pos.copy()
            self._prev_targ_ori_mat = targ_ori_mat.copy()

        else:
            raise ValueError("Invalid control mode.")

        ros_msg = control_utils.compose_ros_msg(
            targ_pos=targ_pos,
            targ_ori_quat=np.roll(
                Rotation.from_matrix(targ_ori_mat).as_quat(), shift=1
            ),  # (w, x, y, z)
            prop_gains=self.task_instance_config.control.prop_gains,
            msg_count=self._ros_msg_count,
        )

        self._ros_publisher.publish(ros_msg)
        self._ros_msg_count += 1

    def do_simple_procedure(self, procedure, franka_arm):
        """Does specified procedure."""
        # NOTE: Currently used to perform a set of simple actions before or after going to a goal

        if procedure:
            for step in procedure:
                if step == "open_gripper":
                    control_utils.open_gripper(franka_arm=franka_arm)
                elif step == "close_gripper":
                    control_utils.close_gripper(franka_arm=franka_arm)
                elif step == "go_upward":
                    control_utils.go_upward(franka_arm=franka_arm, dist=0.2, duration=2.0)
                elif step == "go_downward":
                    control_utils.go_downward(franka_arm=franka_arm, dist=0.2, duration=2.0)
                elif step == "go_home":
                    control_utils.go_home(franka_arm=franka_arm, duration=5.0)
                else:
                    raise ValueError(f"Invalid step {step} in motion procedure.")
