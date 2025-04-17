"""Codes to load human poses from AMASS datasets."""

import os
from typing import Any, Dict, List

import numpy as np
import torch


def load_and_combine_amass_poses(
    data_directory: str, joint_indices: List[int]
) -> torch.Tensor:
    """
    Load and concatenate poses from `.npz` files in the AMASS dataset.

    Parameters
    ----------
    `parameters: Dict[str, Any]`
    Simulation parameters containing at least the following keys:
    `data_directory`, `joint_indices`, `device`, `data_type`, `total_sizee`

    Returns
    -------
    `torch.Tensor`
    A tensor with shape `(num_samples, num_joints, 3)`
    """
    combined_poses = []

    for path, _, filenames in os.walk(data_directory):
        for filename in filenames:
            if not filename.endswith(".npz"):
                continue

            data = np.load(os.path.join(path, filename))
            combined_poses.append(data["poses"][:, joint_indices])

    combined_poses = torch.tensor(
        np.concatenate(combined_poses, axis=0),
        device=parameters["device"],
        dtype=parameters["data_type"],
    )

    if parameters["total_size"] != -1:
        combined_poses = combined_poses[: parameters["total_size"]]

    return combined_poses.unflatten(
        dim=-1,
        sizes=(len(parameters["joint_indices"]) // 3, 3),
    )
