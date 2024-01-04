# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Detect objects script.

This script detects objects in an image from an Intel RealSense camera on a
Franka robot. The script loads parameters for the detection procedure from a
specified YAML file, captures an image, loads a trained detection model, runs
the detection procedure, writes object detection information to a YAML file,
and saves the labeled image.

Typical usage examples:
- Standalone: python detect_objects.py -p perception.yaml
- Imported: detect_objects.main(perception_config_file_name=perception.yaml)
"""

# Standard Library
import argparse
import json

# Third Party
import cv2
import numpy as np
import os
import torch
from kornia.augmentation import ColorJiggle
from torchvision.transforms import functional as tf

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


def main(perception_config_file_name):
    """Runs the detection pipeline.

    Captures an image. Loads a perception model. Detects objects. Gets their real-world coordinates.
    Saves the information to file.
    """
    config = perception_utils.get_perception_config(
        file_name=perception_config_file_name, module_name="detect_objects"
    )

    # Capture image
    pipeline = perception_utils.get_camera_pipeline(
        width=config.camera.image_width, height=config.camera.image_height
    )
    image = perception_utils.get_image(
        pipeline=pipeline, display_images=config.object_detection.display_images
    )  # np.ndarray (height, width, 3)
    image_tensor = tf.to_tensor(image).unsqueeze(0).to("cuda")  # torch.Tensor (1, 3, height, width)

    # Load perception model
    model = torch.load(
        os.path.join(os.path.dirname(__file__), '..', 'checkpoints', config.input.checkpoint_file_name),
        map_location="cuda",
    )

    # Get detections
    if config.augmentation.augment_image:
        # Augment image and get detections for image with highest confidence scores
        boxes, labels, scores, masks = _get_augmented_detections(
            config=config,
            model=model,
            image=image_tensor,
            num_augmentations=config.augmentation.num_augmentations,
        )
    else:
        boxes, labels, scores, masks = _get_detections(
            config=config, model=model, image=image_tensor
        )

    # Sort detections by scores
    combined_data = list(zip(boxes, labels, scores, masks))
    sorted_data = sorted(combined_data, key=lambda x: x[2], reverse=True)
    boxes, labels, scores, masks = zip(*sorted_data)

    # Get real-world (x, y, theta) coordinates of detected objects
    box_real_coords, _ = _get_real_object_coords(config=config, boxes=boxes, masks=masks)

    # Save object detection information to file
    label_names = config.object_detection.scene[config.object_detection.scene.type].label_names
    labels_text = [label_names[label.item()] for label in labels]
    _save_object_detection_info(
        config=config, object_coords=box_real_coords, labels_text=labels_text
    )

    # Draw labels on image and save to file
    image_labeled = _label_object_detections(
        image=image, boxes=boxes, labels_text=labels_text, scores=scores
    )
    perception_utils.save_image(image=image_labeled, file_name=config.output.image_file_name)

    if config.object_detection.display_images:
        cv2.imshow("Object Detections", image_labeled)
        cv2.waitKey(delay=2000)
        cv2.destroyAllWindows()

    return box_real_coords, labels_text


def _get_detections(config, model, image):
    """Gets bounding boxes, labels, scores, and masks from an image."""
    # Detect objects
    print("\nRunning detection...")
    model.eval()
    with torch.no_grad():
        detections = model(image)[0]
    print("Finished running detection.")

    # Extract detection data
    boxes = detections["boxes"].detach().to("cpu").numpy()
    labels = detections["labels"].detach().to("cpu").numpy()
    scores = detections["scores"].detach().to("cpu").numpy()
    masks = detections["masks"].detach().to("cpu").numpy()

    # Keep only high-confidence detections
    indices_to_keep = []
    for i, score in enumerate(scores):
        if score > config.object_detection.confidence_thresh:
            indices_to_keep.append(i)
    boxes = boxes[indices_to_keep]
    labels = labels[indices_to_keep]
    scores = scores[indices_to_keep]
    masks = masks[indices_to_keep]

    return boxes, labels, scores, masks


def _get_real_object_coords(config, boxes, masks):
    """Gets the real-world (x, y, theta) coordinates of detected objects."""
    with open(os.path.join(os.path.dirname(__file__), '..', 'io', config.input.workspace_mapping_file_name)) as f:
        json_obj = f.read()
    workspace_mapping = json.loads(json_obj)
    real_x_max = workspace_mapping["workspace_bounds"][0][1]
    real_y_max = workspace_mapping["workspace_bounds"][1][1]
    pixel_length = workspace_mapping["pixel_length"]

    angles = []
    for mask in masks:
        angle = _get_mask_angle(mask=mask)
        angles.append(angle)
    angles = np.asarray(angles)

    box_pixel_centers, box_real_coords = [], []
    for box, angle in zip(boxes, angles):
        box_center_pixel_x = (box[0] + box[2]) / 2.0
        box_center_pixel_y = (box[1] + box[3]) / 2.0
        box_center_real_x = real_x_max - box_center_pixel_y * pixel_length
        box_center_real_y = real_y_max - box_center_pixel_x * pixel_length
        # Alias angle to prevent large gripper rotations
        if angle < np.pi / 4:
            box_real_angle = angle
        elif angle > np.pi / 4:
            box_real_angle = angle - np.pi / 2
        box_pixel_centers.append([box_center_pixel_x, box_center_pixel_y])
        box_real_coords.append([box_center_real_x, box_center_real_y, box_real_angle])

    return box_real_coords, box_pixel_centers


def _get_mask_angle(mask):
    """Processes a mask. Gets its contours. Fits a minimum-area rectangle. Gets the angle."""
    mask_processed = mask.squeeze(0) * 255.0  # (height, width)

    contours, _ = cv2.findContours(
        image=mask_processed.astype(np.uint8),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )  # (1, num points, 1, 2)

    min_area_rect = cv2.minAreaRect(points=contours[0])
    angle = min_area_rect[2] * np.pi / 180.0  # [0, pi / 2]

    return angle


def _save_object_detection_info(config, object_coords, labels_text):
    """Saves object detection information to a JSON file."""
    with open(os.path.join(os.path.dirname(__file__), '..', 'io', config.output.json_file_name), "w") as f:
        mapping = {"object_coords": object_coords, "labels_text": labels_text}
        json.dump(mapping, f)
    print("\nSaved object detections to file.")


def _label_object_detections(image, boxes, labels_text, scores):
    """Labels object detection information on an image."""
    image_labeled = image.copy()

    # For each detection, draw box, label, and score
    for box, label_text, score in zip(boxes, labels_text, scores):
        # Draw box on image
        cv2.rectangle(
            img=image_labeled,
            pt1=(int(box[0]), int(box[1])),
            pt2=(int(box[2]), int(box[3])),
            color=(0, 255, 0),
            thickness=2,
        )

        # Draw label and score on image
        cv2.putText(
            img=image_labeled,
            text=f"{label_text} ({str(score)[:4]})",
            org=(int(box[0]), int(box[1] - 10.0)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return image_labeled


def _get_augmented_detections(config, model, image, num_augmentations):
    """Runs test-time augmentation.

    Augments an image a specified number of times. Detects objects in each augmentation.
    Keeps highest-confidence detections.
    """
    # Define augmentation that will apply random brightness, contrast, saturation, and hue
    # perturbation
    color_augmentation = ColorJiggle(
        p=1.0, brightness=(0.5, 2.0), contrast=(0.5, 2.0), saturation=(0.5, 2.0), hue=(-0.5, 0.5)
    )

    print("\nAugmenting image and running detection...")
    score_sum_best = 0.0
    for _ in range(num_augmentations):
        image_augmented = color_augmentation(image)
        boxes, labels, scores, masks = _get_detections(
            config=config, model=model, image=image_augmented
        )
        if scores.any():
            score_sum = np.sum(scores)
            if score_sum > score_sum_best:
                score_sum_best = score_sum
                boxes_best = boxes
                labels_best = labels
                scores_best = scores
                masks_best = masks
    print("\nFinished augmenting image and running detection.")

    return boxes_best, labels_best, scores_best, masks_best


if __name__ == "__main__":
    """Gets arguments. Runs the script."""
    args = get_args()

    main(perception_config_file_name=args.perception_config_file_name)
