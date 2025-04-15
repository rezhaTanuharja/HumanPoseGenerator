from typing import Any, Dict, Callable

import torch

from humanposegenerator.models.modulators import FiLM


def generate_speedometer(
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
    activation_layer = torch.nn.LeakyReLU(negative_slope=0.005)
    dropout_layer = torch.nn.Dropout(p=0.0)

    temporal_conditioner = torch.nn.Sequential(
        torch.nn.Linear(2 * parameters["num_sinusoids"], 32),
        activation_layer,
        torch.nn.Linear(32, 32),
        activation_layer,
        dropout_layer,
        torch.nn.Linear(32, 18),
        activation_layer,
    ).to(parameters["device"])

    spatial_head = torch.nn.Sequential(
        torch.nn.Linear(9, 80),
        activation_layer,
        dropout_layer,
        torch.nn.Linear(80, 80),
        activation_layer,
        torch.nn.Linear(80, 3),
    ).to(parameters["device"])

    velocity_model = FiLM(
        modulator=temporal_conditioner,
        main_block=spatial_head,
    ).to(
        parameters["device"],
    )

    checkpoint = torch.load(parameters["velocity_checkpoint"], weights_only=True)

    state_dict = checkpoint["model_state_dict"]
    module_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    velocity_model.load_state_dict(module_state_dict)
    for param in velocity_model.parameters():
        param.requires_grad = False
    velocity_model.eval()
    velocity_model.to(parameters["device"])

    def speedometer(location: torch.Tensor, condition: torch.Tensor):
        """
        Compute the conditional velocity at given location.

        Parameters
        ----------
        `location: torch.Tensor`
        A tensor with shape `(num_times, num_samples, num_joints, 3, 3)`

        `condition: torch.Tensor`
        A tensor with shape `(num_times, num_samples, 2 * num_sinusoids)`

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

    return speedometer
