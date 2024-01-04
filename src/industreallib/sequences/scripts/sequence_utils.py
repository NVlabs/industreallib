# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Sequence utilities module.

This module defines utility functions for retrieving sequences.
"""

# Standard Library
import os

# Third Party
from omegaconf import OmegaConf

# NVIDIA
from industreallib.sequences.classes.industreal_sequence_base import IndustRealSequenceBase


def get_sequence_instance_config(args):
    """Gets a sequence instance configuration from a YAML file."""
    sequence_instance_config_path = (
        os.path.join(os.path.dirname(__file__), '..', 'instance_configs',
        f"{args.sequence_instance_config_name}.yaml")
    )
    if os.path.exists(sequence_instance_config_path):
        sequence_instance_config = OmegaConf.load(sequence_instance_config_path)
    else:
        raise ValueError(
            f"Sequence instance configuration file with path {sequence_instance_config_path} does"
            " not exist."
        )

    return sequence_instance_config


def get_sequence_instance(args, sequence_instance_config):
    """Gets an instance of a sequence with the specified configuration."""
    # Instantiate class (will instantiate corresponding task instances)
    sequence_instance = IndustRealSequenceBase(
        args=args, sequence_instance_config=sequence_instance_config
    )

    return sequence_instance


def get_num_assemblies(sequence_instance_config):
    """Gets the number of assemblies to assemble."""
    # Get number of parts assigned to each task
    num_parts_all = []
    for task_instance_config_name in sequence_instance_config.sequence.task_instance_config_names:
        parts = sequence_instance_config.sequence[f"{task_instance_config_name}"].part_order
        num_parts = len(parts)
        num_parts_all.append(num_parts)

    # If all tasks have same number of parts
    if all(num_parts_curr == num_parts_all[0] for num_parts_curr in num_parts_all):
        num_assemblies = num_parts_all[0]
    else:
        raise ValueError(
            "Number of parts must be equal for all task instances in sequence instance."
        )

    return num_assemblies
