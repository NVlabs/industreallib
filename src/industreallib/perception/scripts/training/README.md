# Asset Detection 

This folder contains the training and testing code for the detection model described in [Tang and Lin, et al., "IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to Reality," Robotics: Science and Systems (RSS), 2023](https://arxiv.org/abs/2305.17110). This code is based on the ``torchvision`` [object detection tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

# Files

- `train.py`: main Python script containing training and testing code
- `data_utils.py`: utility functions used for data generation
- `camera_utils.py`: utility functions used for getting images from Intel RealSense camera
- `coco_eval.py`, `coco_utils.py`, `engine.py`, `transforms.py`, `utils.py`: borrowed from [pytorch/vision](https://github.com/pytorch/vision/tree/main/references/detection). According to [this github issue](https://github.com/pytorch/vision/issues/2254#issuecomment-634062558), it is not possible to import these files from torchvision; it is instead recommended to copy-paste (or locally clone) these files when writing custom applications
- `viewer_params.json`: Intel RealSense camera configuration file

# Train Asset Detection Model

## Preparation

Put the asset images in `./asset_imgs/` and background images in `./tabletop_background/`. Here is an example of the directory structure for the [IndustReal](https://arxiv.org/abs/2305.17110) **connector** assets:
```  
industreallib/src/industreallib/perception/scripts/training/asset_imgs:
- nema_2_prong_plug/
	- nema_2_prong_plug_1.png
	- ...
- nema_2_prong_socket/
	- nema_2_prong_socket_1.png
	- ...
- nema_3_prong_plug/
	- nema_3_prong_plug_1.png
	- ...
- nema_3_prong_socket/
	- nema_3_prong_socket_1.png
	- ...
``` 
You can download the images used in [IndustReal](https://arxiv.org/abs/2305.17110) from [Google Drive](https://drive.google.com/drive/folders/1Z3z6U7HAAlpNT5VyqI9KUuz0BeQyAvZX?usp=sharing).

## Training and Testing

- To train a detection model **from scratch** (add `--generate_data` to generate data for training/testing):
```
python train.py --asset [asset group] --num_classes [number of classes] --generate_data
```
 where the default asset groups and corresponding numbers of classes are listed below:

 | Asset Group | Asset Included  | Num. of Classes [^1] |
|:-:|:-:|:-:|
| peg | round/rectangular pegs, round/rectangular holes | 5 |
| connector | 2-prong plug/socket, 3-prong plug/socket | 5 |
| gear | gear base, small/medium/large gear | 5 |
[^1]: We consider the background as a class, so the number of classes = the number of assets + 1.

- To train a detection model **from a pre-trained checkpoint**:
```
python train.py --asset [asset group] --num_classes [number of classes] --load_ckpt --ckpt_path [path to checkpoint file]
```
- To test a detection model on images **in a local directory**:
```
python train.py --asset [asset group] --num_classes [number of classes] --load_ckpt --ckpt_path [path to checkpoint file] --test_from_dir
```
A non-default directory can be specified with `--test_dir`.
- To test a detection model on an image **from the Intel RealSense camera stream**:
```
python train.py --asset [asset group] --num_classes [number of classes] --load_ckpt --ckpt_path [path to checkpoint file] --test_from_camera
```
- To get the **top-`N` predictions** on an image from the Intel RealSense camera stream:
```
python train.py --asset [asset group] --num_classes [number of classes] --load_ckpt --ckpt_path [path to checkpoint file] --test_from_camera --num_obj_in_test_img [N]
```

| Command-Line Argument | Default Value  | Description |
|:-:|:-:|:-:|
| asset | peg | asset group to detect; available groups: peg, gear, connector |
| num_classes | 5 | number of classes in group |
| generate_data | False |if True, generate dataset for training by overlaying asset images in `./asset_imgs/` on `./tabletop_background/` |
| num_train_imgs | 1000 | number of training images to generate |
| num_epochs | 10 | number of training epochs |
| load_ckpt | False | if True, load checkpoint for training or testing; if False, train from scratch |
| ckpt_path | checkpoint.pt | path to existing checkpoint |
| test_from_dir | False | if True, test model on local images |
| test_from_camera | False | if True, test model on image from camera stream |
| num_obj_in_test_img | 0 | number of objects in test image; setting to 0 will get all predicted bounding boxes; setting to `N` will get top-`N` |
| load_viewer_params | False | if True, load camera settings from ``viewer_params.json`` |
| train_dir | ./train/ | directory containing all training images |
| mask_dir | ./mask/ | directory containing all segmentation masks for training |
| asset_dir | ./asset_imgs/ | directory containing all asset images |
| tabletop_dir | ./tabletop_background/ | directory containing all tabletop background images |
| test_dir | ./test/ | directory containing all test images |
| output_dir | ./output/ | directory containing all detection outputs from testing |


## Adding a New Asset Group

1. Prepare data:
- Collect 10 images (with background removed) for each asset class.
- Organize the asset images: Put 10 images for each asset class in the same folder, and put all folders in `./asset_imgs/`
- If you would like to use your own tabletop image as the background for a synthetic dataset, collect empty tabletop images and put them in `./tabletop_background/`
2. Add a new asset group in the training code:
- Add another `elif` after [this line](train.py?ref_type=heads#L417) in [train.py](train.py?ref_type=heads) with `args.asset == <new group name>` and `asset_dir_list = [<all asset image folder names>]`.
- Use the following command-line arguments: 
  - `--asset <new group name>`, e.g., `nut_bolt`.
  - `--num_classes <number of asset classes + 1>` (We assume `background` is a separate class, so the total number of classes = the number of asset classes + 1)
  - `--generate_data` (You need to generate data for the newly-added group.)
3. Other tips:
- Make sure to remove the `train` and `mask` folders before generating new data. This will prevent images generated previously for other assets from getting mixed into the current training dataset.
- The default number of epochs for training is 10. However, it normally takes less 5 epochs to get strong detection results; to save time, you could change this number to 5 and fine-tune if the test results do not look promising.


# Best Practices

1. To get better detection results when testing from camera images, adjust the Intel RealSense camera settings in ``realsense-viewer``, saving them to `viewer_params.json`, and using the `--load_viewer_params` argument

# Troubleshooting

1. Problem: libpng warning: iCCP: known incorrect sRGB profile
	
	- Install `pngcrush` and `libpng-tools`:
	```  
	sudo apt install pngcrush
	sudo apt install libpng-tools
	```
	- Enter the directory containing all the images and run the following: 
	```
	pngcrush -n -q *.png
	```
	- For any images producing the warning `pngcrush: iCCP: Not recognizing known sRGB profile that has been edited`, run the following:
	```
	pngfix --strip=color --out=[output png] [input png]
	```