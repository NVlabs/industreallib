# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Task utilities module.

This module defines utility functions for retrieving tasks.
"""

# Standard Library
import os
from pydoc import locate

# Third Party
import inflection
from omegaconf import OmegaConf


def get_task_instance_config(task_instance_config_name, task_instance_config_subdir):
    """Gets a task instance configuration from a YAML file."""
    if task_instance_config_subdir is not None:
        task_instance_config_path = (
            os.path.join(os.path.dirname(__file__), '..', 'instance_configs',
            task_instance_config_subdir, f"{task_instance_config_name}.yaml")
        )
    else:
        task_instance_config_path = (
            os.path.join(os.path.dirname(__file__), '..', 'instance_configs',
            f"{task_instance_config_name}.yaml")
        )

    if os.path.exists(task_instance_config_path):
        task_instance_config = OmegaConf.load(task_instance_config_path)
    else:
        raise ValueError(
            f"Task instance configuration file with path {task_instance_config_path} does not"
            " exist."
        )

    return task_instance_config


def get_task_instance(args, task_instance_config, in_sequence):
    """Gets an instance of a task with the specified configuration."""
    # Get name of corresponding class from YAML file
    task_class_name = task_instance_config["task"]["class"]

    # Get name of corresponding module (convert camel case to snake case and remove extraneous
    # underscore)
    task_module_name = inflection.underscore(task_class_name).replace("indust_real", "industreal")

    # Locate specified class within corresponding module
    task_class = locate(f"industreallib.tasks.classes.{task_module_name}.{task_class_name}")

    # Instantiate class
    task_instance = task_class(
        args=args, task_instance_config=task_instance_config, in_sequence=in_sequence
    )

    return task_instance
