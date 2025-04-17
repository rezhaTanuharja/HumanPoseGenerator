import os
from typing import List

import numpy as np


def load_and_combine_amass_poses(
    data_directory: str,
    joint_indices: List[int],
) -> np.ndarray:
    combined_poses = []

    for path, _, filenames in os.walk(data_directory):
        for filename in filenames:
            if not filename.endswith(".npz"):
                continue

            data = np.load(os.path.join(path, filename))
            combined_poses.append(data["poses"][:, joint_indices])

    combined_poses = np.concatenate(combined_poses, axis=0)
    return combined_poses.reshape(-1, len(joint_indices) // 3, 3)
