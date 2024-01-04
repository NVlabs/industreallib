# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""NVIDIA SRL industreallib package."""

# NVIDIA
from _industreallib import _get_version

# Set `__version__` attribute
__version__ = _get_version()

# Remove `_get_version` so it is not added as an attribute
del _get_version
