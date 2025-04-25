from typing import Any, Callable, Dict

import torch

from humanposegenerator import models


def create_velocimeter(
    parameters: Dict[str, Any],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Produce a conditional velocity model.

    Parameters
    ----------
    `parameters: Dict[str, Any]`

    Returns
    -------
    A model to compute conditional velocity from `location` and `condition`
    """

    velocity_model = models.sequential.Assembly(parameters["velocimeter"])

    checkpoint = torch.load(parameters["velocity_checkpoint"], weights_only=True)

    state_dict = checkpoint["model_state_dict"]
    module_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    velocity_model.load_state_dict(module_state_dict)
    for param in velocity_model.parameters():
        param.requires_grad = False
    velocity_model.eval()
    velocity_model.to(parameters["device"])

    def velocimeter(location: torch.Tensor, condition: torch.Tensor):
        """
        Compute the conditional velocity at given location.

        Parameters
        ----------
        `location: torch.Tensor`
        A tensor with shape `(num_times, num_samples, num_joints, 3, 3)`

        `condition: torch.Tensor`
        A tensor with shape `(num_times, num_samples, 2 * num_frequencies)`

        Returns
        -------
        `torch.Tensor`
        A velocity tensor with shape `(num_times, num_samples, num_joints, 3)`
        """
        return velocity_model(
            location.flatten(-2),
            condition.unsqueeze(2).repeat(
                1,
                1,
                len(parameters["joint_indices"]) // 3,
                1,
            ),
        )

    return velocimeter
