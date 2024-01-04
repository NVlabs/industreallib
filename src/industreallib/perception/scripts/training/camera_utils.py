# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Perception training scripts: Camera utilities.

This module defines utility functions for perceiving a scene with an Intel
RealSense camera.
"""

# Standard Library
import json

# Third Party
import cv2
import numpy as np
import pyrealsense2 as rs


def start_image_stream(load_viewer_params, viewer_params_json="viewer_params.json"):
    """Starts image stream from RealSense. Returns pipeline object."""
    # NOTE: Spliced from
    # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        stream_type=rs.stream.depth, width=1280, height=720, format=rs.format.z16, framerate=30
    )
    config.enable_stream(
        stream_type=rs.stream.color, width=1280, height=720, format=rs.format.bgr8, framerate=30
    )

    cfg = pipeline.start(config)

    if load_viewer_params:
        json_obj = json.load(open(viewer_params_json))
        json_string = str(json_obj).replace("'", '"')

        dev = cfg.get_device()
        advnc_mode = rs.rs400_advanced_mode(dev)
        advnc_mode.load_json(json_string)

    return pipeline


def get_image(pipeline, display_image=False):
    """Gets RGB and depth images from RealSense."""
    frames = pipeline.wait_for_frames()

    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    depth_frame = frames.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_color_map = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET
    )

    if display_image:
        images = np.hstack((color_image, depth_color_map))
        cv2.namedWindow("RGB-D Output", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RGB-D Output", images)
        cv2.waitKey(1000)

    return color_image, depth_image
