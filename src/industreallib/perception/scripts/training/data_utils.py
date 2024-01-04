# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Perception training scripts: Data utilities.

This module defines utility functions for generating training data.
"""

# Standard Library
import os

# Third Party
import cv2
import numpy as np
import torch
from kornia.augmentation import AugmentationSequential, ColorJiggle
from PIL import Image
from torchvision.transforms import functional


def layer_img(background_img, overlay_img, label):
    """Randomly overlay asset images on top of a background image. Generate segmentation masks."""
    # find the object bounding box for the overlay image
    overlay_col_min, overlay_row_min, overlay_col_max, overlay_row_max = find_bbox(overlay_img)

    # calculate the size of the bounding box
    bbox_col, bbox_row = overlay_col_max - overlay_col_min, overlay_row_max - overlay_row_min

    # randomly pick a place in the background for the overlay.
    # NOTE: only overlay where the object patch (e.g., usb) is.
    background_cols, background_rows, background_channels = background_img.shape
    replace_rows_max, replace_cols_max = background_rows - bbox_row, background_cols - bbox_col
    replace_row, replace_col = np.random.choice(replace_rows_max), np.random.choice(
        replace_cols_max
    )

    # get the indices of pixels that need to be replaced
    replace_pixels = np.argwhere(overlay_img[:, :, 3] != 0)
    segmentation_mask = np.squeeze(np.zeros_like(background_img)[:, :, 0])

    # place the overlay on top of the background (by replacing pixels, only non-transparent pixels)
    for i in range(replace_pixels.shape[0]):
        pixel = replace_pixels[i, :]
        offset_col, offset_row = pixel[0] - overlay_col_min, pixel[1] - overlay_row_min

        background_img[replace_col + offset_col, replace_row + offset_row] = overlay_img[
            pixel[0], pixel[1], :3
        ]
        segmentation_mask[replace_col + offset_col, replace_row + offset_row] = label

    return background_img, segmentation_mask


def find_bbox(overlay_img):
    """Find bounding box for an asset in a background-removed image."""
    # get indices of pixels with non-transparent values
    pixel = np.argwhere(overlay_img[:, :, 3] != 0)

    # get indices of bounding box corners
    col_min = np.min(pixel[:, 0])
    col_max = np.max(pixel[:, 0])
    row_min = np.min(pixel[:, 1])
    row_max = np.max(pixel[:, 1])

    assert col_min < col_max and row_min < row_max

    return col_min, row_min, col_max, row_max


def generate_data(
    train_dir, mask_dir, tabletop_dir, asset_root_dir, asset_dir_list, num_train_imgs
):
    """Generate training data by overlaying asset images on top of tabletop background images."""
    print("Generating data...")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    for i in range(len(asset_dir_list)):
        curr_asset_dir = asset_root_dir + asset_dir_list[i]
        label = i + 1
        generate_data_one_class(
            tabletop_dir, curr_asset_dir, train_dir, mask_dir, num_train_imgs, label
        )

    print(
        f"Finished generating data. Check {train_dir} for images and {mask_dir} for segmentation"
        " masks."
    )


def generate_data_one_class(
    background_dir, overlay_dir, output_train_dir, output_mask_dir, num_img, label
):
    """Generate training data for one class of asset."""
    # get all image filenames
    overlay_imgs = get_image_paths(overlay_dir)
    background_imgs = get_image_paths(background_dir)

    for i in range(num_img):
        # randomly choose a background image
        background_idx = np.random.choice(len(background_imgs))
        background_path = background_imgs[background_idx]
        background_img = cv2.imread(background_path, 1)

        # randomly choose an overlay image
        overlay_idx = np.random.choice(len(overlay_imgs))
        overlay_path = overlay_imgs[overlay_idx]
        overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        # add alpha channel for transparent pixel checking later
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)

        # randomly place overlay image on background image
        background_img, segmentation_mask = layer_img(background_img, overlay_img, label)

        # save background image and segmentation mask to file
        cv2.imwrite(
            os.path.join(output_train_dir, str(label) + "_" + str(i) + ".png"), background_img
        )
        cv2.imwrite(
            os.path.join(output_mask_dir, str(label) + "_" + str(i) + ".png"), segmentation_mask
        )


def get_image_paths(img_dir):
    """Get all image paths in a given directory."""
    img_list = []

    for file in os.listdir(img_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_list.append(os.path.join(img_dir, filename))

    return img_list


class AssetDetectionDataset(torch.utils.data.Dataset):
    """Define dataset class for object detection model."""

    def __init__(self, root, transforms, img_dir, mask_dir):
        """Initializes the root directory, image augmentation transforms, and output directories."""
        self.root = root
        self.transforms = transforms
        self.aug_forms = AugmentationSequential(
            ColorJiggle(
                p=1.0,
                brightness=(0.5, 2.0),
                contrast=(0.5, 2.0),
                saturation=(0.5, 2.0),
                hue=(-0.5, 0.5),
            ),
            same_on_batch=False,
        )
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.imgs = sorted(os.listdir(os.path.join(root, img_dir)))
        self.masks = sorted(os.listdir(os.path.join(root, mask_dir)))

    def __getitem__(self, idx):
        """Gets training data."""
        # load images and masks
        img_path = os.path.join(self.root, self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.root, self.mask_dir, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        labels = torch.ones((num_objs,), dtype=torch.int64)
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

            labels[i] *= obj_ids[i]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        if boxes.shape[0] > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = 0.0
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None and boxes.shape[0] > 0:
            img, target = self.transforms(img, target)
        else:
            img = functional.to_tensor(img)

        if self.aug_forms is not None and boxes.shape[0] > 0:
            img = self.aug_forms(img).squeeze()

        return img, target

    def __len__(self):
        """Gets the size of training dataset."""
        return len(self.imgs)
