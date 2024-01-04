# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Perception utilities module.

This module defines utility functions for perceiving a scene with an Intel
RealSense camera.
"""

# Standard Library
import json
import os

# Third Party
import cv2
import numpy as np
import pyrealsense2 as rs
from omegaconf import OmegaConf


def get_perception_config(file_name, module_name):
    """Gets an IndustRealLib perception configuration from a YAML file."""
    config = OmegaConf.load(os.path.join(os.path.dirname(__file__), '..', 'configs', file_name))[module_name]

    return config


def get_camera_pipeline(width, height):
    """Starts an RGB image stream from the RealSense. Gets pipeline object."""
    # NOTE: Spliced from
    # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        stream_type=rs.stream.color, width=width, height=height, format=rs.format.bgr8, framerate=30
    )

    pipeline.start(config)

    return pipeline


def load_realsense_params(file_name, realsense_config):
    """Loads a JSON file exported from the Intel RealSense Viewer."""
    with open(file_name) as f:
        json_obj = f.read()

    json_str = str(json_obj).replace("'", '"')
    rs.rs400_advanced_mode(device=realsense_config.get_device()).load_json(json_str)


def get_intrinsics(pipeline):
    """Gets the intrinsics for the RGB camera from the RealSense."""
    profile = pipeline.get_active_profile()

    color_profile = rs.video_stream_profile(profile.get_stream(stream_type=rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()

    color_intrinsics_dict = {
        "cx": color_intrinsics.ppx,
        "cy": color_intrinsics.ppy,
        "fx": color_intrinsics.fx,
        "fy": color_intrinsics.fy,
    }

    return color_intrinsics_dict


def get_image(pipeline, display_images):
    """Gets an RGB image from the RealSense."""
    print("\nAcquiring image...")
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    print("Acquired image.")

    if display_images:
        cv2.namedWindow(winname="RGB Output", flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(winname="RGB Output", mat=color_image)
        cv2.waitKey(delay=2000)
        cv2.destroyAllWindows()

    return color_image


def label_tag_detection(image, tag_corner_pixels, tag_family):
    """Labels a tag detection on an image."""
    image_labeled = image.copy()

    corner_a = (int(tag_corner_pixels[0][0]), int(tag_corner_pixels[0][1]))
    corner_b = (int(tag_corner_pixels[1][0]), int(tag_corner_pixels[1][1]))
    corner_c = (int(tag_corner_pixels[2][0]), int(tag_corner_pixels[2][1]))
    corner_d = (int(tag_corner_pixels[3][0]), int(tag_corner_pixels[3][1]))

    # Draw oriented box on image
    cv2.line(img=image_labeled, pt1=corner_a, pt2=corner_b, color=(0, 255, 0), thickness=2)
    cv2.line(img=image_labeled, pt1=corner_b, pt2=corner_c, color=(0, 255, 0), thickness=2)
    cv2.line(img=image_labeled, pt1=corner_c, pt2=corner_d, color=(0, 255, 0), thickness=2)
    cv2.line(img=image_labeled, pt1=corner_d, pt2=corner_a, color=(0, 255, 0), thickness=2)

    # Draw tag family on image
    cv2.putText(
        img=image_labeled,
        text=tag_family.decode("utf-8"),
        org=(corner_a[0], corner_c[1] - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(255, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    return image_labeled


def save_image(image, file_name):
    """Saves an image to file."""
    print("\nSaving image...")
    cv2.imwrite(filename=os.path.join(os.path.dirname(__file__), '..', 'io', file_name), img=image)
    print("Saved image.")


def get_tag_pose_in_camera_frame(detector, image, intrinsics, tag_length, tag_active_pixel_ratio):
    """Detects an AprilTag in an image. Gets the pose of the tag in the camera frame."""
    gray_image = cv2.cvtColor(src=image.astype(np.uint8), code=cv2.COLOR_BGR2GRAY)
    tag_active_length = tag_length * 0.0254 * tag_active_pixel_ratio
    detection = detector.detect(
        img=gray_image,
        estimate_tag_pose=True,
        camera_params=[intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
        tag_size=tag_active_length,
    )

    if detection:
        is_detected = True
        pos = detection[0].pose_t.copy().squeeze()  # (3, )
        ori_mat = detection[0].pose_R.copy()
        center_pixel = detection[0].center
        corner_pixels = detection[0].corners
        family = detection[0].tag_family

    else:
        is_detected = False
        pos, ori_mat, center_pixel, corner_pixels, family = None, None, None, None, None

    return is_detected, pos, ori_mat, center_pixel, corner_pixels, family


def convert_tag_pose_to_robot_frame(tag_pose_t, tag_pose_r, extrinsics, robot_pose_t, robot_pose_r):
    """Converts the tag pose in the camera frame to the tag pose in the robot frame."""
    # Define tag pose in camera frame
    from_tag_to_cam_htm = np.block(
        [[tag_pose_r, np.reshape(tag_pose_t, (3, 1))], [np.zeros((1, 3)), 1.0]]
    )

    # Get camera pose in EE frame
    from_cam_to_ee_t = np.asarray(extrinsics["position"])
    from_cam_to_ee_r = np.asarray(extrinsics["orientation"])
    from_cam_to_ee_htm = np.block(
        [[from_cam_to_ee_r, np.reshape(from_cam_to_ee_t, (3, 1))], [np.zeros((1, 3)), 1.0]]
    )

    # Define EE pose in robot frame
    from_ee_to_robot_t = robot_pose_t.copy()
    from_ee_to_robot_r = robot_pose_r.copy()
    from_ee_to_robot_htm = np.block(
        [[from_ee_to_robot_r, np.reshape(from_ee_to_robot_t, (3, 1))], [np.zeros((1, 3)), 1.0]]
    )

    # Compute tag pose in robot frame
    from_tag_to_robot_htm = from_ee_to_robot_htm @ from_cam_to_ee_htm @ from_tag_to_cam_htm
    from_tag_to_robot_t = from_tag_to_robot_htm[:3, 3]
    from_tag_to_robot_r = from_tag_to_robot_htm[:3, :3]

    return from_tag_to_robot_t, from_tag_to_robot_r


def get_extrinsics(file_name):
    """Loads the extrinsics from a JSON file."""
    with open(os.path.join(os.path.dirname(__file__), '..', 'io', file_name)) as f:
        json_obj = f.read()

    extrinsics_dict = json.loads(json_obj)

    return extrinsics_dict
