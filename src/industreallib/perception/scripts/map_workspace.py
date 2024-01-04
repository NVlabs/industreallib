# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Map workspace script.

This script maps the workspace (i.e., determines a mapping from image space
to world space) of a wrist-mounted Intel RealSense camera on a Franka robot.
The script loads parameters for the mapping procedure from a specified YAML
file, captures an image, runs the mapping procedure, and writes the
real-world xy-bounds of the image to a JSON file.

Typical usage examples:
- Standalone: python map_workspace.py -p perception.yaml
- Imported: map_workspace.main(perception_config_file_name=perception.yaml,
                               franka_arm=franka_arm)
"""

# Standard Library
import argparse
import json
import os

# Third Party
import cv2
import numpy as np
import pupil_apriltags as apriltag
from frankapy import FrankaArm

# NVIDIA
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


def main(perception_config_file_name, franka_arm):
    """Runs the main workspace mapping pipeline.

    Initializes the camera. Gets an image and tag pose. Gets and saves the workspace mapping.
    """
    config = perception_utils.get_perception_config(
        file_name=perception_config_file_name, module_name="map_workspace"
    )

    # Initialize camera and AprilTag detector
    pipeline = perception_utils.get_camera_pipeline(
        width=config.camera.image_width, height=config.camera.image_height
    )
    intrinsics = perception_utils.get_intrinsics(pipeline=pipeline)
    detector = apriltag.Detector(
        families=config.tag.type, quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25
    )
    # NOTE: Instantiation of this detector brightens the image

    image = perception_utils.get_image(
        pipeline=pipeline, display_images=config.tag_detection.display_images
    )
    (
        is_tag_detected,
        tag_pose_t,
        tag_pose_r,
        tag_center_pixel,
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
        workspace_bounds, pixel_length = _get_workspace_mapping(
            config=config,
            franka_arm=franka_arm,
            tag_center_pixel=tag_center_pixel,
            tag_corner_pixels=tag_corner_pixels,
            tag_pose_t=tag_pose_t,
            tag_pose_r=tag_pose_r,
        )
        _save_workspace_mapping(
            config=config, workspace_bounds=workspace_bounds, pixel_length=pixel_length
        )

        # Draw labels on image and save to file
        image_labeled = perception_utils.label_tag_detection(
            image=image, tag_corner_pixels=tag_corner_pixels, tag_family=tag_family
        )
        perception_utils.save_image(image=image_labeled, file_name=config.output.image_file_name)

        if config.tag_detection.display_images:
            cv2.imshow("Tag Detection", image_labeled)
            cv2.waitKey(delay=2000)
            cv2.destroyAllWindows()
    else:
        raise RuntimeError("Tag not detected.")

    return workspace_bounds, pixel_length


def _get_workspace_mapping(
    config, franka_arm, tag_center_pixel, tag_corner_pixels, tag_pose_t, tag_pose_r
):
    """Computes the workspace mapping (i.e., the mapping from image space to world space)."""
    # NOTE: Loosely based on code from
    # https://github.com/andyzeng/visual-pushing-grasping/blob/master/main.py

    # Get tag pose in robot frame
    extrinsics = perception_utils.get_extrinsics(file_name=config.input.extrinsics_file_name)
    curr_pose = franka_arm.get_pose()
    from_tag_to_robot_t, _ = perception_utils.convert_tag_pose_to_robot_frame(
        tag_pose_t=tag_pose_t,
        tag_pose_r=tag_pose_r,
        extrinsics=extrinsics,
        robot_pose_t=curr_pose.translation,
        robot_pose_r=curr_pose.rotation,
    )

    # Get real-world length of each pixel
    pixel_length = _get_pixel_length(config=config, tag_corner_pixels=tag_corner_pixels)

    # Compute real-world xy-coordinates of image corners in robot frame
    # NOTE: From point-of-view of wrist-mounted camera in default robot pose,
    # robot x-axis is up, robot y-axis is left, image x-axis is right, and image y-axis is down
    real_x_min = (
        from_tag_to_robot_t[0] - (config.camera.image_height - tag_center_pixel[1]) * pixel_length
    )
    real_x_max = from_tag_to_robot_t[0] + tag_center_pixel[1] * pixel_length
    real_y_min = (
        from_tag_to_robot_t[1] - (config.camera.image_width - tag_center_pixel[0]) * pixel_length
    )
    real_y_max = from_tag_to_robot_t[1] + tag_center_pixel[0] * pixel_length
    workspace_bounds = np.asarray([[real_x_min, real_x_max], [real_y_min, real_y_max]])

    print("\nComputed workspace mapping.")

    return workspace_bounds, pixel_length


def _get_pixel_length(config, tag_corner_pixels):
    """Gets the length of each pixel in an image, based on the known size of an AprilTag."""
    # Get mean length of side of tag in pixels
    edge_lengths_pixels = []
    for i, curr_corner_pixel in enumerate(tag_corner_pixels):
        next_corner_pixel = tag_corner_pixels[np.mod(i + 1, 4)]
        edge_length_pixels = np.linalg.norm(next_corner_pixel - curr_corner_pixel)
        edge_lengths_pixels.append(edge_length_pixels)
    mean_edge_length_pixels = np.mean(edge_lengths_pixels)

    # Compute length in meters of side of tag
    edge_length_meters = config.tag.length * 0.0254 * config.tag.active_pixel_ratio

    # Compute length of each pixel
    pixel_length = edge_length_meters / mean_edge_length_pixels

    return pixel_length


def _save_workspace_mapping(config, workspace_bounds, pixel_length):
    """Saves the workspace mapping to a JSON file."""
    with open(os.path.join(os.path.dirname(__file__), '..', 'io', config.output.file_name), "w") as f:
        mapping = {"workspace_bounds": workspace_bounds.tolist(), "pixel_length": pixel_length}
        json.dump(mapping, f)
    print("\nSaved workspace mapping to file.")


if __name__ == "__main__":
    """Gets arguments. Initializes the robot. Runs the script."""
    args = get_args()
    franka_arm = FrankaArm()

    main(perception_config_file_name=args.perception_config_file_name, franka_arm=franka_arm)
