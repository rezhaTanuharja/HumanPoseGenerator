from typing import Any, Callable, Dict, Tuple

import torch
from diffusionmodels.manifolds.rotationalgroups import SpecialOrthogonal3
from diffusionmodels.stochasticprocesses.multivariate.uniform import UniformSphere
from diffusionmodels.stochasticprocesses.univariate.inversetransforms import (
    InverseTransform,
)
from diffusionmodels.stochasticprocesses.univariate.inversetransforms.cumulativedistributions.heatequations import (
    PeriodicCumulativeEnergy,
)
from diffusionmodels.stochasticprocesses.univariate.inversetransforms.rootfinders.bisection import (
    Bisection,
)


def generate_diffuser(
    parameters: Dict[str, Any],
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate a diffuser based on training parameters.

    Parameters
    ----------
    `parameters: Dict[str, Any]`
    A dictionary containing training parameters

    Returns
    -------
    A function mapping `initial_condition` and `time` to `diffused_condition` and `increment`
    """
    axis_process = UniformSphere(
        dimension=3,
        device=parameters["device"],
        data_type=parameters["data_type"],
    )

    angle_distribution = PeriodicCumulativeEnergy(
        num_waves=parameters["num_waves"],
        mean_squared_displacement=parameters["mean_squared_displacement"],
        alpha=parameters["alpha"],
        device=parameters["device"],
        data_type=parameters["data_type"],
    )

    root_finder = Bisection(num_iterations=parameters["num_iterations"])

    angle_process = InverseTransform(
        distribution=angle_distribution,
        root_finder=root_finder,
        device=parameters["device"],
        data_type=parameters["data_type"],
    )

    manifold = SpecialOrthogonal3(
        device=parameters["device"],
        data_type=parameters["data_type"],
    )

    def diffuser(
        initial_condition: torch.Tensor,
        time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Diffuse an initial condition based on a predefined diffusion process.

        Parameters
        ----------
        `initial_condition: torch.Tensor`
        A tensor with shape `(num_times, num_samples, num_joints, 3, 3)`

        `time: torch.Tensor`
        A tensor with shape `(num_times,)`

        Returns
        -------
        `Tuple[torch.Tensor, torch.Tensor]`
        The diffused condition and the difference between the diffused and initial conditions.
        Each is a tensor with shape `(num_times, num_samples, num_joints, 3, 3)`
        """
        rotation_matrices = manifold.exp(
            torch.eye(3, device=parameters["device"], dtype=parameters["data_type"]),
            initial_condition,
        )

        axis = axis_process.at(time).sample(21 * parameters["batch_size"])
        angle = angle_process.at(time).sample(21 * parameters["batch_size"])
        angle.requires_grad_()

        y = torch.log(angle_process.density(angle))
        y.sum().backward()

        lamb = 2.0

        theta = torch.einsum(
            "i..., i... -> i...",
            torch.exp(lamb * (time - 1.6)),
            2.0
            * torch.acos(
                torch.clip(1.0 - (angle / torch.pi) ** 0.75, min=0.0, max=1.0),
            ),
        )

        axis_angle = torch.einsum("...i, ... -> ...i", axis, theta)

        noisy_poses = manifold.exp(
            rotation_matrices.unsqueeze(0),
            axis_angle.unflatten(1, (4, 21)),
        )

        relative_poses = torch.einsum(
            "...ji, ...jk -> ...ik",
            rotation_matrices,
            noisy_poses,
        )

        return noisy_poses, relative_poses

    return diffuser
