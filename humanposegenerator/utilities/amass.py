"""Provide various functions to interact with the AMASS dataset."""

import os
from typing import List

import numpy as np


def load_and_combine_poses(
    data_directories: List[str],
    joint_indices: List[int],
) -> np.ndarray:
    """
    Recursively search and load poses from `.npz` files in a given directory.

    Parameters
    ----------
    `data_directories: List[str]`
    The list of directories to search from.

    `joint_indices: List[int]`
    The indices of joints to load

    Returns
    -------
    A NumPy array with shape `(num_samples, num_joints, 3)`
    """
    combined_poses = []

    for directory in data_directories:
        for path, _, filenames in os.walk(directory):
            for filename in filenames:
                if not filename.endswith(".npz"):
                    continue

                data = np.load(os.path.join(path, filename))
                combined_poses.append(data["poses"][:, joint_indices])

    combined_poses = np.concatenate(combined_poses, axis=0)
    return combined_poses.reshape(-1, len(joint_indices) // 3, 3)
