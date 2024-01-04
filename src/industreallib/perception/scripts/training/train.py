# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Perception training scripts: Training code.

This script allows a user to train a detection model.

Typical usage example:

  python train.py --asset [asset group] --num_classes [number of classes]
"""

# Standard Library
import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve().parent))

# Third Party
import cv2
import numpy as np
import torch
import torchvision
import transforms as trans
import utils
from data_utils import AssetDetectionDataset, generate_data
from engine import train_one_epoch
from perception_utils import get_camera_pipeline, get_image
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional


class ToTensor(object):
    """
    Class for converting image and label data to tensors.
    Created as a class in order to compose with other image transforms.
    """

    def __call__(self, image, target):
        """Convert image and label data (target) to tensors."""
        image = functional.to_tensor(image)
        return image, target


def get_transform(train):
    """Apply random horizontal flip and scaling to training images."""
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(trans.RandomHorizontalFlip(0.5))
        transforms.append(trans.ScaleJitter())
    return trans.Compose(transforms)


def get_model_instance_segmentation(num_classes):
    """Load instance segmentation model."""
    # load an instance segmentation model pre-trained on COCO
    # set weights=None if want to train from scratch
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get the number of input features for the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the box predictor with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # get the number of input channels for the mask predictor
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, num_classes)

    return model


def train(train_dir, mask_dir, num_classes, load_ckpt, ckpt_path, num_epochs, num_test):
    """Train object detection model."""
    if load_ckpt:
        if os.path.exists(ckpt_path):
            print(f"Loading model from {ckpt_path} and resuming training.")
        else:
            raise OSError(2, "No such file: ", ckpt_path)
    else:
        print("Start training from scratch.")

    # train on the GPU or, if a GPU is not available, on the CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # use our dataset and defined transformations
    dataset = AssetDetectionDataset(".", get_transform(train=True), train_dir, mask_dir)
    dataset_test = AssetDetectionDataset(".", get_transform(train=False), train_dir, mask_dir)

    # split the dataset into a training set and a test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-num_test])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-num_test:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=8, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=8, collate_fn=utils.collate_fn
    )

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    if load_ckpt and os.path.exists(ckpt_path):
        model = torch.load(ckpt_path)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        torch.save(model, ckpt_path)

    print(f"Finished training. Model saved at {ckpt_path}.")


def test_multi_imgs(test_dir, output_dir, num_classes, ckpt_path, num_objects=0, text_labels=None):
    """Test model with multiple images stored in test_dir."""
    print(f"Loading model from {ckpt_path} and testing with images from {test_dir}.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load model weights
    if os.path.exists(ckpt_path):
        model = torch.load(ckpt_path)
    else:
        raise OSError(2, "No such file: ", ckpt_path)

    model.eval()

    test_imgs = os.listdir(test_dir)
    for img in test_imgs:
        # load image as RGB and transform to tensor format
        rgb_img = Image.open(os.path.join(test_dir, img)).convert("RGB")
        rgb_img_tensor = functional.to_tensor(rgb_img).unsqueeze(0).to(device)

        # forward pass
        predictions = model(rgb_img_tensor)

        bboxes = predictions[0]["boxes"]
        masks = predictions[0]["masks"]
        labels = predictions[0]["labels"]
        scores = predictions[0]["scores"]

        # create output directory for each test image
        img_filename = Path(img).stem
        img_output_dir = os.path.join(output_dir, img_filename + "/")
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)

        # save prediction data
        np.savetxt(img_output_dir + "bboxes.out", bboxes.cpu().detach().numpy().copy())
        np.savetxt(img_output_dir + "labels.out", labels.cpu().detach().numpy().copy())
        np.savetxt(img_output_dir + "scores.out", scores.cpu().detach().numpy().copy())

        # save prediction visualization
        if num_objects == 0:
            # save all detected results
            num_objects_in_img = len(labels)
        else:
            # only save top N results
            num_objects_in_img = min(num_objects, len(labels))

        original_img = cv2.imread(os.path.join(test_dir, img), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(num_objects_in_img):
            bbox = bboxes[i].cpu().detach().numpy()
            label = labels[i].cpu().detach().numpy()

            mask = masks[i, 0, :, :].unsqueeze(-1).cpu().detach().numpy() * 255.0
            cv2.imwrite(img_output_dir + str(i) + "_mask.png", mask * 255.0)

            if text_labels is not None:
                cv2.rectangle(
                    original_img,
                    [int(bbox[0]), int(bbox[1])],
                    [int(bbox[2]), int(bbox[3])],
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    original_img,
                    text_labels[label - 1],
                    (int(bbox[0]), int(bbox[1])),
                    font,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                original_img = cv2.imread(os.path.join(test_dir, img), 1)
                cv2.rectangle(
                    original_img,
                    [int(bbox[0]), int(bbox[1])],
                    [int(bbox[2]), int(bbox[3])],
                    (0, 0, 255),
                    2,
                )
                cv2.imwrite(img_output_dir + str(i) + "_bbox.png", original_img)

        if text_labels is not None:
            cv2.imwrite(img_output_dir + "bbox_with_labels.png", original_img)

    print(f"Finished testing. Check {output_dir} for results.")


def test_from_camera(output_dir, num_classes, ckpt_path, num_object=0, text_labels=None):
    """Test model on camera image."""
    print(f"Load model from {ckpt_path}, and start testing with images from camera stream.")

    # load model weights
    if os.path.exists(ckpt_path):
        model = torch.load(ckpt_path)
    else:
        raise OSError(2, "No such file: ", ckpt_path)

    model.eval()

    # get RGB image from camera
    pipeline = get_camera_pipeline(width=1280, height=720)
    rgb_img = get_image(pipeline)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rgb_img_tensor = functional.to_tensor(rgb_img).unsqueeze(0).to(device)

    # forward pass
    predictions = model(rgb_img_tensor)

    bboxes = predictions[0]["boxes"]
    masks = predictions[0]["masks"]
    labels = predictions[0]["labels"]
    scores = predictions[0]["scores"]

    # create output directory for each test image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save prediction data
    np.savetxt(output_dir + "bboxes.out", bboxes.cpu().detach().numpy().copy())
    np.savetxt(output_dir + "labels.out", labels.cpu().detach().numpy().copy())
    np.savetxt(output_dir + "scores.out", scores.cpu().detach().numpy().copy())

    # save prediction visualization
    if num_object == 0:
        # save all detected results
        num_objects_in_img = len(labels)
    else:
        # only save top n results
        num_objects_in_img = num_object

    original_img = rgb_img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(num_objects_in_img):
        bbox = bboxes[i].cpu().detach().numpy()
        label = labels[i].cpu().detach().numpy()
        mask = masks[i, 0, :, :].unsqueeze(-1).cpu().detach().numpy() * 255.0
        cv2.imwrite(output_dir + str(i) + "_mask.png", mask * 255.0)

        if text_labels is not None:
            cv2.rectangle(
                original_img,
                [int(bbox[0]), int(bbox[1])],
                [int(bbox[2]), int(bbox[3])],
                (0, 0, 255),
                2,
            )
            cv2.putText(
                original_img,
                text_labels[label - 1],
                (int(bbox[0]), int(bbox[1])),
                font,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            original_img = rgb_img.copy()
            cv2.rectangle(
                original_img,
                [int(bbox[0]), int(bbox[1])],
                [int(bbox[2]), int(bbox[3])],
                (0, 0, 255),
                2,
            )
            cv2.imwrite(output_dir + str(i) + "_bbox.png", original_img)

    if text_labels is not None:
        cv2.imwrite(output_dir + "bbox_with_labels.png", original_img)

    print(f"Finished testing. Check {output_dir} for results.")


def main():
    """Train a detection model based on MaskRCNN for IndustReal assets."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset",
        type=str,
        help="asset group to detect; available groups: peg, gear, connector",
    )
    parser.add_argument("--num_classes", type=int, default=5, help="number of classes in group")
    parser.add_argument(
        "--generate_data",
        action="store_true",
        default=False,
        help=(
            "if True, generate dataset for training by overlaying asset images in ./asset_imgs/ on"
            " ./tabletop_background/"
        ),
    )
    parser.add_argument(
        "--num_train_imgs", type=int, default=1000, help="number of training images to generate"
    )
    parser.add_argument(
        "--num_test_imgs",
        type=int,
        default=100,
        help="number of test images to split from training images",
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument(
        "--load_ckpt",
        action="store_true",
        default=False,
        help="if True, load checkpoint for training or testing; if False, train from scratch",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoint.pt",
        help="path to existing checkpoint",
    )
    parser.add_argument(
        "--test_from_dir",
        action="store_true",
        default=False,
        help="if True, test model on local images",
    )
    parser.add_argument(
        "--test_from_camera",
        action="store_true",
        default=False,
        help="if True, test model on image from camera stream",
    )
    parser.add_argument(
        "--num_obj_in_test_img",
        type=int,
        default=0,
        help=(
            "number of objects in test image; setting to 0 will get all predicted bounding boxes;"
            " setting to N will get top-N"
        ),
    )
    parser.add_argument(
        "--train_dir", type=str, default="./train/", help="directory containing all training images"
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="./mask/",
        help="directory containing all segmentation masks for training",
    )
    parser.add_argument(
        "--asset_dir",
        type=str,
        default="./asset_imgs/",
        help="directory containing all asset images",
    )
    parser.add_argument(
        "--tabletop_dir",
        type=str,
        default="./asset_imgs/tabletop_background/",
        help="directory containing all tabletop background images",
    )
    parser.add_argument(
        "--test_dir", type=str, default="./test_imgs/", help="directory containing all test images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/",
        help="directory containing all detection outputs from testing",
    )

    args = parser.parse_args()

    if args.asset == "peg":
        asset_dir_list = ["round_pegs", "rectangular_pegs", "round_holes", "rectangular_holes"]
    elif args.asset == "gear":
        asset_dir_list = ["green_gear_base", "large_gears", "medium_gears", "small_gears"]
    elif args.asset == "connector":
        asset_dir_list = [
            "nema_2_prong_plug",
            "nema_2_prong_socket",
            "nema_3_prong_plug",
            "nema_3_prong_socket",
        ]
    else:
        raise ValueError(f"Undefined asset group: {args.asset}")

    if args.generate_data:
        generate_data(
            args.train_dir,
            args.mask_dir,
            args.tabletop_dir,
            args.asset_dir,
            asset_dir_list,
            args.num_train_imgs,
        )

    if args.test_from_camera:
        test_from_camera(
            args.output_dir,
            args.num_classes,
            args.ckpt_path,
            args.num_obj_in_test_img,
            asset_dir_list,
        )

    elif args.test_from_dir:
        test_multi_imgs(
            args.test_dir,
            args.output_dir,
            args.num_classes,
            args.ckpt_path,
            args.num_obj_in_test_img,
            asset_dir_list,
        )

    else:
        train(
            args.train_dir,
            args.mask_dir,
            args.num_classes,
            args.load_ckpt,
            args.ckpt_path,
            args.num_epochs,
            args.num_test_imgs,
        )


if __name__ == "__main__":
    main()
