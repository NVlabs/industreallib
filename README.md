# IndustRealLib

This repo contains the real-world policy deployment code used in [Tang and Lin, et al., "IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to Reality," Robotics: Science and Systems (RSS), 2023](https://arxiv.org/abs/2305.17110). The goal of the repo is to allow researchers to be able to deploy a subset of our RL policies or deploy similar policies of their own. **The repo does not intend to be an all-in-one or long-term solution for sim-to-real deployment**, but simply shows one way it can be done.

---

## Table of Contents

1. [Setup Instructions](#setup-instructions)
2. [High-Level Code Structure](#high-level-code-structure)
3. [Running the Examples](#running-the-examples)
4. [Core Classes](#core-classes)
5. [Frequently Asked Questions](#frequently-asked-questions)
6. [Additional Information](#additional-information)

---

## Setup Instructions

### Hardware

<img src="media/hardware.png" alt="Franka robot, RealSense camera, and IndustRealKit assets" height="400"/>
<br/><br/>

#### Prerequisites:

The following hardware is required to use the repo. The models and versions used in the [IndustReal paper](https://arxiv.org/abs/2305.17110) are listed in parentheses:
- A working Franka robot (Panda)
- A working Intel RealSense D400 camera (D415) mounted to the robot
- Printed April Tags (a 3-inch and 6-inch 52h13 tag)
- A workstation or mini PC running an Ubuntu real-time kernel (for Ubuntu 20.04)
    - Typically, you would already be using such a workstation to directly communicate with the Franka robot
- A second workstation running an Ubuntu generic kernel (for Ubuntu 20.04) with an NVIDIA GPU (RTX 3080)
- Parts from [IndustRealKit](https://github.com/NVLabs/industrealkit)
    - The pegs, peg trays, and holes will be used in subsequent examples. **These parts are assumed to be located in the same region as our optical breadboard**, which is centered approximately 20 cm in front of the robot base, with its top surface approximately 4 cm above the robot base

### Software

<img src="media/software.png" alt="Communications between Franka and workstations" height="200"/>
<br/><br/>

#### Prerequisites:

- Set up [franka-interface](https://github.com/iamlab-cmu/franka-interface) on the workstation running the real-time kernel (refer to the [installation doc](https://github.com/iamlab-cmu/franka-interface/blob/master/docs/install.rst))
- Set up [frankapy](https://github.com/iamlab-cmu/frankapy) on the workstation running the generic kernel (refer to the [installation doc](https://github.com/iamlab-cmu/frankapy/blob/master/docs/install.rst))
- Make sure that the generic workstation can successfully control the Franka robot by running one of the [frankapy examples](https://github.com/iamlab-cmu/frankapy/tree/master/examples)
    - If any issues are encountered, please refer to frankapy's [Discord community](https://discord.gg/r6r7dttMwZ)
- Set up [Isaac Gym Preview Release 4](https://developer.nvidia.com/isaac-gym) on the generic workstation
- Clone and install [Isaac Gym Envs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) on the generic workstation in the same virtual environment or Conda environment used for frankapy

#### Installation:

- Clone this repo on the generic workstation
- Activate the virtual environment or Conda environment used for frankapy
- Install `wheel` (will prevent an `invalid command: bdist_wheel` error)<br>
    `pip install wheel`
- Install the repo in editable mode with<br>
    `pip install -e .`<br>
    Note: You can safely ignore incompatibility errors related to `wandb` and `ray`

#### Additional:

- Update [extrinsics_industreallib.json](src/industreallib/src/industreallib/perception/io/extrinsics_industreallib.json) with the camera's extrinsics matrix
    - If the extrinsics have not already been calibrated, the standalone calibration code used in the [IndustReal paper](https://arxiv.org/abs/2305.17110) is available in [calibrate_extrinsics.py](src/industreallib/perception/scripts/calibrate_extrinsics.py). However, please note that **policy performance can be sensitive to the quality of extrinsics calibration** due to potentially-large perception error; thus, it is recommended to also look into well-maintained third-party libraries like [easy-handeye](https://github.com/IFL-CAMP/easy_handeye)

---

## High-Level Code Structure

There are five fundamental building blocks within IndustRealLib:

- **Task classes** (e.g., the base class [IndustRealTaskBase](src/industreallib/tasks/classes/industreal_task_base.py), the child classes [IndustRealTaskPick](src/industreallib/tasks/classes/industreal_task_pick.py) and [IndustRealTaskPlace](src/industreallib/tasks/classes/industreal_task_place.py)), which define the methods needed for different task types (e.g., picking, placing)
- **Task instance configurations** (e.g., [pick_pegs.yaml](src/industreallib/tasks/instance_configs/pick_pegs.yaml), [pick_gears.yaml](src/industreallib/tasks/instance_configs/pick_gears.yaml)), which define customized instances of a particular task type (e.g., picking *pegs*, picking *gears*)
- **Sequence classes** (currently, only a base class [IndustRealSequenceBase](src/industreallib/sequences/classes/industreal_sequence_base.py)), which defines the methods needed for any sequence of task types (e.g., picking, placing, and inserting in sequence)
- **Sequence instance configurations** (e.g., [pick_place_insert_pegs.yaml](src/industreallib/sequences/instance_configs/pick_place_insert_pegs.yaml)), which define customized instances of a particular sequence of task instances (e.g., picking, placing, and inserting *pegs* in sequence)
- **Launch scripts** (i.e., [run_task.py](scripts/run_task.py), [run_sequence.py](scripts/run_sequence.py)), which launch a desired task instance or sequence instance, respectively
- **Utility scripts** (i.e., [run_simple_action.py](scripts/run_simple_action.py), [calibrate_extrinsics.py](src/industreallib/perception/scripts/calibrate_extrinsics.py)), which can execute useful procedures, such as running simple control actions or performing camera calibration

These building blocks should become intuitive after running examples and exploring the code.

Although less fundamental for understanding IndustRealLib, it is also important to note that at the beginning of most tasks and sequences, a common perception procedure is performed. This procedure consists of the following steps:

1. **Extrinsics parsing:** A camera extrinsics matrix is read from [extrinsics_industreallib.json](src/industreallib/src/industreallib/perception/io/extrinsics_industreallib.json)
    - The standalone calibration code used in the [IndustReal paper](https://arxiv.org/abs/2305.17110) is available in [calibrate_extrinsics.py](src/industreallib/perception/scripts/calibrate_extrinsics.py)
2. **Workspace mapping:** An image is captured, and the camera's image-space coordinates are mapped to real-world coordinates. An April Tag (e.g., a 3-inch 52h13 tag) must be in view
    - The mapping procedure is implemented in [map_workspace.py](src/industreallib/perception/scripts/map_workspace.py)
3. **Object detection:** Objects are detected in the image
    - The detection training code (based on [Mask R-CNN](https://arxiv.org/abs/1703.06870)) is available in the [perception training](src/industreallib/perception/scripts/training) directory, and the detection inference procedure is implemented in [detect_objects.py](src/industreallib/perception/scripts/detect_objects.py)
4. **Goal generation:** Goals for the robot's motion are generated

The perception procedure can be configured in [perception.yaml](src/industreallib/perception/configs/perception.yaml).

---

## Running the Examples

### Simple Control Actions

- Run a *get pose* action from [run_simple_action.py](scripts/run_simple_action.py) to test that IndustRealLib is installed properly:<br>
    `python run_simple_action.py --action get_pose`
- Run a *go upward* action:<br>
    `python run_simple_action.py -a go_upward`
- Explore the other actions in [run_simple_action.py](scripts/run_simple_action.py). Note that the actions themselves are implemented in [control_utils.py](src/industreallib/control/scripts/control_utils.py), mostly as thin wrappers around methods that are provided by frankapy

### Task Instances

The task instances require execution of perception models and/or RL policies. This repo contains a few detection models, as well as RL policies trained with [IndustRealSim](#related-repos). These models and policies were tested on either 2 or 3 robots with similar supporting hardware and software. Nevertheless, **performance or stability of the tasks cannot not be guaranteed when executing on any other system**, and getting them working on a new system may require significant time and effort. Please exercise caution when running the tasks as-is! For example, you may consider initially reducing the controller gains and/or actions scales by 1-2 orders of magnitude and gradually increasing them.

Also note that several examples will reference parts from [IndustRealKit](https://github.com/NVLabs/industrealkit).

#### 1. Reach:

By default, a set of position goals will be randomly generated, and the robot will move to each of them using an RL policy and the PLAI algorithm.

**Preparation:**

- Remove all objects from the work surface

**Execution:**

- Run a *go home* action:<br>
    `python run_simple_action.py -a go_home`
- Run the *Reach* task instance:<br>
    `python run_task.py --task_instance_config_name reach`
- Experiment with the options available in [reach.yaml](src/industreallib/tasks/instance_configs/reach.yaml). Be careful if increasing gains or action scales, especially when using PLAI or leaky PLAI

#### 2. Pick Pegs:


<table align="center">
    <tr>
        <th>Front View</th>
        <th>Side View</th>
    </tr>
    <tr>
        <td><img src="media/pick_front_view.gif" alt="front view of picking pegs" width="200"/></th>
        <td><img src="media/pick_side_view.gif"  alt="side view of picking pegs" width="200"/></th>
    </tr>
</table>


By default, round peg trays and rectangular peg trays from [IndustRealKit](https://github.com/NVLabs/industrealkit) will be detected. The robot will grasp a peg from its peg tray, move upward, drop the peg, return home, and repeat.

**Preparation:**

- Place a few peg trays on the work surface, and insert the corresponding pegs
- Be prepared to catch each peg as it is dropped

**Execution:**

- Run a *go home* action:<br>
    `python run_simple_action.py -a go_home`
- Run the *Pick Pegs* task instance:<br>
    `python run_task.py -t pick_pegs`
- Experiment with the options available in [pick_pegs.yaml](src/industreallib/tasks/instance_configs/pick_pegs.yaml). Again, be careful if increasing gains or action scales

#### 3. Pick Gears:

By default, small, medium, and large gears from [IndustRealKit](https://github.com/NVLabs/industrealkit) will be detected.

**Preparation:**

- Place a few gears on the work surface
- Be prepared to catch each gear as it is dropped

**Execution:**

- Run a procedure similar to that of *Pick Pegs*, but swapping out the task instance name as appropriate. The detection checkpoint and scene type in [perception.yaml](src/industreallib/perception/configs/perception.yaml) must also be changed (and reverted for subsequent peg examples)

#### 4. Place Pegs:

By default, round holes and rectangular holes from [IndustRealKit](https://github.com/NVLabs/industrealkit) will be detected. The *Place Pegs* task instance is typically executed as part of a sequence instance rather than alone, so it is assumed that the robot gripper is already grasping a relevant object. The robot will simply move to a location above each hole, return home, and repeat.

**Preparation:**

- Place a few *holes* (i.e., the 3D-printed trays from [IndustRealKit](https://github.com/NVLabs/industrealkit) into which pegs are intended to be inserted by the robot) on the work surface. Make sure there is nothing inserted in or obstructing the hole
- Place any peg into the robot gripper fingers. The peg should be placed high into the fingers, as the fingers will move very close to the top surface of the holes
    - As described in Section VI.B of the [IndustReal paper](https://arxiv.org/abs/2305.17110), a laser pointer could be used instead of a peg, allowing visualization of positional accuracy
- Command the gripper to close:<br>
    `python run_simple_action.py -a close_gripper`

**Execution:**

- Run a *go home* action:<br>
    `python run_simple_action.py -a go_home`
- Run the *Place Pegs* task instance:<br>
    `python run_task.py -t place_pegs`
- Experiment with the options available in [place_pegs.yaml](src/industreallib/tasks/instance_configs/place_pegs.yaml)

#### 5. Insert Pegs:


<table align="center">
    <tr>
        <th>8mm Rectangular Peg</th>
        <th>8mm Round Peg</th>
        <th>12mm Rectangular Peg</th>
        <th>12mm Round Peg</th>
        <th>16mm Rectangular Peg</th>
        <th>16mm Round Peg</th>
    </tr>
    <tr>
        <td><img src="media/insert_8mm_square.gif" alt="insert 8mm rectangular peg" height="200"/></th>
        <td><img src="media/insert_8mm_round.gif"  alt="insert 8mm round peg" height="200"/></th>
        <td><img src="media/insert_12mm_rect.gif"  alt="insert 12mm rectangular peg" height="200"/></th>
        <td><img src="media/insert_12mm_round.gif" alt="insert 12mm round peg" height="200"/></th>
        <td><img src="media/insert_16mm_rect.gif"  alt="insert 16mm rectangular peg" height="200"/></th>
        <td><img src="media/insert_16mm_round.gif" alt="insert 16mm round peg" height="200"/></th>
    </tr>
</table>


By default, the robot will enter guide mode to allow the user to guide the robot towards an inserted peg. The robot will then grasp the peg, move upward to remove it from the socket, perform a random pose perturbation, and execute a peg insertion policy. The robot will then release the peg, move upward, and return home.

**Preparation:**

- Place a single *hole* on the work surface, and insert the corresponding peg
    - Ideally, the *hole* should be bolted into a rigidly-mounted plate (e.g., the optical breadboard from [IndustRealKit](https://github.com/NVLabs/industrealkit)), and the plate should be located in approximately the same region as in the [IndustReal paper](https://arxiv.org/abs/2305.17110) (i.e., centered approximately 20 cm in front of the robot base, with its top surface approximately 4 cm above the robot base)

**Execution:**

- Run a *go home* action and an *open gripper* action:<br>
    `python run_simple_action.py -a go_home`<br>
    `python run_simple_action.py -a open_gripper`
- Run the *Insert Pegs* task instance:<br>
    `python run_task.py -t insert_pegs`
- Follow the instructions in the terminal to allow the robot to enter guide mode. Guide the robot such that the peg is centered between the gripper fingers and the peg is as deep into the fingers as possible. Close the fingers onto the part
    - Try to keep the elbow approximately upright and the gripper in an approximately top-down orientation, as the policy may not be robust to large deviations in joint configuration
- Follow the instructions in the terminal to exit guide mode. The robot will remove the peg from the socket, perform a pose perturbation, and execute the insertion policy
- Experiment with the options available in [insert_pegs.yaml](src/industreallib/tasks/instance_configs/insert_pegs.yaml)
    - Note: There was a bug that was recently discovered in the original code for leaky PLAI, where non-clamped positional errors accumulated actions twice. Thus, the action scales and positional error thresholds for leaky PLAI may need minor tuning to ensure full insertion

#### 6. Insert Gears:


<table align="center">
    <tr>
        <th>Small Gear</th>
        <th>Medium Gear</th>
        <th>Large Gear</th>
    </tr>
    <tr>
        <td><img src="media/insert_small_gear.gif" alt="insert small gear" width="200"/></th>
        <td><img src="media/insert_medium_gear.gif"  alt="insert medium gear" width="200"/></th>
        <td><img src="media/insert_large_gear.gif"  alt="insert large gear" width="200"/></th>
    </tr>
</table>


**Preparation:**

- Place the gear base on the work surface, and insert the medium gear onto the middle gearshaft
    - As in *Insert Pegs*, the gear base should ideally be bolted into a rigidly-mounted plate

**Execution:**

- Run a procedure similar to that of *Insert Pegs*, but swapping out the task instance name as appropriate. Remember that the detection checkpoint and scene type in [perception.yaml](src/industreallib/perception/configs/perception.yaml) may also need to be changed
- Experiment with the options available in [insert_gears.yaml](src/industreallib/tasks/instance_configs/insert_gears.yaml)

### Sequence Instances

Like task instances, sequence instances also require execution of perception models and/or RL policies; thus, performance or stability again cannot be guaranteed when executing on any other system. Furthermore, perception and control error may accumulate throughout long-horizon sequences. Please exercise caution.

#### 1. Pick-Place-Insert Pegs:

<img src="media/ppi_8mm_pegs.gif" alt="Pick-Place-Insert 8mm pegs" height="200"/>
<br/><br/>

By default, a round peg, a round *hole*, a rectangular peg, and a rectangular *hole* from [IndustRealKit](https://github.com/NVLabs/industrealkit) will be detected. The robot will execute the *Pick Pegs* task instance, the *Place Pegs* task instance, and the *Insert Pegs* task instance on the round peg-and-hole, followed by the same sequence on the rectangular peg-and-hole.

**Preparation:**

- Place one round peg tray, one same-sized round *hole*, one rectangular peg tray, and one same-sized rectangular *hole* on the work surface. Insert the corresponding pegs into the peg trays
    - As in *Insert Pegs*, the round *hole* and rectangular *hole* should ideally be bolted into a rigidly-mounted plate

**Execution:**

- Run a *go home* action and an *open gripper* action:<br>
    `python run_simple_action.py -a go_home`<br>
    `python run_simple_action.py -a open_gripper`
- Run the *Pick-Place-Insert Pegs* sequence instance:<br>
    `python run_sequence.py --sequence_instance_config_name pick_place_insert_pegs`
- Experiment with the options available in [pick_place_insert_pegs.yaml](src/industreallib/sequences/instance_configs/pick_place_insert_pegs.yaml)
    - Note: The sequence instance configuration files allow specification of the task instances that should compose the sequence, as well as the parts that each task instance should act on. When a sequence instance is executed, IndustRealLib will execute all task instances on the first set of parts, followed by all task instances on the second set of parts, and so on. In this example, (1) the task instances are specified as *Pick Pegs*, *Place Pegs*, and *Insert Pegs*, (2) the parts for *Pick Pegs* are specified as the round peg and rectangular peg, (3) the parts for *Place Pegs* are specified as the round *hole* and rectangular *hole*, and (4) the parts for *Insert Pegs* are also specified as the round *hole* and rectangular *hole*. Thus, the robot (1) picks up the round peg, places it above the round hole, and inserts it into the round hole, and (2) picks up the rectangular peg, places it above the rectangular hole, and inserts it into the rectangular hole

#### 2. Pick-Place-Insert Gears:

<img src="media/ppi_gear.gif" alt="Pick-Place-Insert gears" height="200"/>
<br/><br/>

By default, a small gear, a medium gear, a large gear, and a gear base from [IndustRealKit](https://github.com/NVLabs/industrealkit) will be detected. The robot will execute the *Pick Gears* task instance, the *Place Gears* task instance, and the *Insert Gears* task instance on the small gear, followed by the same sequence on the large gear and the medium gear.

**Preparation:**

- Place the gear base, the small gear, the medium gear, and the large gear on the work surface with a moderate amount of space between them. Orient the gear base such that the gearshaft for the small gear is closer to the positive x-axis in the robot base frame (i.e., farther away from the robot), whereas the gearshaft for the larger gear is closer to the negative x-axis (i.e., closer to the robot)
    - As in *Insert Gears*, the gear base should ideally be bolted into a rigidly-mounted plate

**Execution:**

- Run a procedure similar to that of *Pick-Place-Insert Pegs*, but swapping out the task instance name as appropriate. Remember that the detection checkpoint and scene type in [perception.yaml](src/industreallib/perception/configs/perception.yaml) may also need to be changed
- Experiment with the options available in [pick_place_insert_gears.yaml](src/industreallib/sequences/instance_configs/pick_place_insert_gears.yaml)
    - Note: This sequence instance configuration file specifies a [pick_place_insert_gears](src/industreallib/tasks/instance_configs/pick_place_insert_gears) subdirectory. In this subdirectory, task instance configuration files can be found for picking, placing, and inserting the small, medium, and large gears. The *Pick Gears* configuration files all reference the same RL policies, as is the case for the *Place Gears* and *Insert Gears* configuration files. However, the IndustRealLib gears detection model currently does not distinguish between the different shafts of the gear base. Thus, separate configuration files were created for each gear; each file contains a unique lateral offset value that specifies which shaft the corresponding gear should be assembled on

---

## Core Classes

Reading this section is not required, but may be useful for making modifications or extensions.

### Task classes

#### IndustRealTaskBase ([industreal_task_base.py](src/industreallib/tasks/classes/industreal_task_base.py))

IndustRealTaskBase is the base class for all task classes. IndustRealTaskBase contains methods that implement procedures needed by all task types, including the following:
- Getting goals (from random generation, perception, or guide mode)
- Going to goals (with a baseline method or with RL)
- Loading a policy that was trained in Isaac Gym
- Getting actions from the policy
- Converting actions to pose targets (e.g., using PLAI) and sending them to frankapy's task-space impedance controller 
- Doing a particular procedure before and/or after going to the goal (e.g., opening/closing the gripper, moving upward, returning home)

#### IndustRealTask\<Name\> (industreal_task_\<name\>.py)

There are four task classes: [IndustRealTaskReach](src/industreallib/tasks/classes/industreal_task_reach.py), [IndustRealTaskPlace](src/industreallib/tasks/classes/industreal_task_place.py), [IndustRealTaskPick](src/industreallib/tasks/classes/industreal_task_pick.py), and [IndustRealTaskInsert](src/industreallib/tasks/classes/industreal_task_insert.py). These task classes contain methods that implement procedures unique to each task type; for now, the only such procedure is getting observations from the robot (customized to each policy's observation space).

### Sequence classes

#### IndustRealSequenceBase ([industreal_sequence_base.py](src/industreallib/sequences/classes/industreal_sequence_base.py))

IndustRealSequenceBase is a common class for all sequences. (In more advanced applications, it may be used as a base class for sequence-specific classes; however, such an extension has not been needed for the [IndustReal paper](https://arxiv.org/abs/2305.17110).) This class contains methods that implement procedures needed by all sequences, including the following:
- Getting task instances that are needed for the given sequence instance
- Getting goals for each task instance (from perception)

---

## Frequently Asked Questions

- **Is the mass of the camera and camera mount provided to Franka's low-level torque controller?**
    Yes. To ensure that Franka's low-level torque controller adapts to the added mass of the camera and camera mount when performing gravity compensation, the mass of the end-effector was increased from a default value of 0.73 kg to a new value of 0.9 kg within Franka Desk (`Settings --> End Effector`)

- **Where should the April Tag be placed during each task?**
    It is recommended to place the April Tag in the same plane as the object bounding boxes. For example, in IndustReal, the detection model was trained to draw bounding boxes around the bases of the 3D-printed peg trays and holes. During each task, the April Tag was placed on the same plane as the bases of these parts, as can be seen in the [IndustReal videos](https://sites.google.com/nvidia.com/industreal).

- **The Franka is hitting virtual walls during extrinsics calibration or policy deployment. How can these walls be expanded?**
    The virtual walls can be expanded within [franka-interface](https://github.com/iamlab-cmu/franka-interface/blob/master/franka-interface/include/franka-interface/termination_handler/termination_handler.h#L76-L89). Please refer to frankapy's [Discord community](https://discord.gg/r6r7dttMwZ) for further discussion on this topic.

- **What is the *one-time offset* in the perception configuration file?**
    The *one-time offset* is described at the end of Appendix B4 in the [IndustReal paper](https://arxiv.org/abs/2305.17110). Empirically, the extrinsics calibration and workspace mapping procedure in IndustReal can produce a constant positional bias in the robot's goals, on the order of 3-7 millimeters along each axis. The one-time offset parameter allows this positional bias to be corrected before running an extensive set of experiments.

---

## Additional Information

### Citing IndustRealLib
If you use any of the IndustRealLib code in your work, please cite the following paper:
```
@inproceedings{
    tang2023industreal,
    author = {Bingjie Tang and Michael A Lin and Iretiayo Akinola and Ankur Handa and Gaurav S Sukhatme and Fabio Ramos and Dieter Fox and Yashraj Narang},
    title = {IndustReal: Transferring contact-rich assembly tasks from simulation to reality},
    booktitle = {Robotics: Science and Systems},
    year = {2023}
}
```

### Related repos
- Isaac Gym (simulate robots): [paper](https://arxiv.org/abs/2108.10470) | [website](https://developer.nvidia.com/isaac-gym) | [environments repo](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
- Factory (simulate contact-rich interactions in Isaac Gym): [paper](https://arxiv.org/abs/2205.03532) | [project website](https://sites.google.com/nvidia.com/factory) | [code](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/tree/main/isaacgymenvs/tasks/factory) | [shorter docs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/rl_examples.md#factory-fast-contact-for-robotic-assembly) | [longer docs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/factory.md)
- IndustRealKit (reproduce assets used in IndustReal): [paper](https://arxiv.org/abs/2305.17110) | [project website](https://sites.google.com/nvidia.com/industreal) | [assets](https://www.github.com/NVLabs/industrealkit)
- IndustRealSim (reproduce RL policy training algorithms used in IndustReal): [paper](https://arxiv.org/abs/2305.17110) | [project website](https://sites.google.com/nvidia.com/industreal) | [code](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/tree/main/isaacgymenvs/tasks/industreal) | [shorter docs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/rl_examples.md#industreal-transferring-contact-rich-simulation-tasks-from-simulation-to-reality) | [longer docs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/industreal.md)

### Support

Please create a GitHub issue.
 
Also note that due to time constraints, **we have no concrete plans to improve the public version of IndustRealLib beyond simple bugfixes**. However, if you significantly extend IndustRealLib or are inspired by the framework in your own work, please let us know, and we would be happy to link to your fork or repo!