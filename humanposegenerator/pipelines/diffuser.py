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


def create_empirical_diffuser(
    parameters: Dict[str, Any],
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate an empirical diffuser based on training parameters.

    Note
    ----
    An empirical diffuser transforms clean poses into noisy poses.
    This diffuser cannot evaluate the analytical velocity vector field.
    It can only compute the vectors connecting clean and noisy poses (increment).

    Parameters
    ----------
    `parameters: Dict[str, Any]`
    A dictionary containing training parameters

    Returns
    -------
    A function mapping `(initial_condition, time)` to `(diffused_condition, increment)`
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

    def empirical_diffuser(
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

        axis = axis_process.at(time).sample(
            len(parameters["joint_indices"]) // 3 * parameters["batch_size"],
        )
        angle = angle_process.at(time).sample(
            len(parameters["joint_indices"]) // 3 * parameters["batch_size"],
        )

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
            axis_angle.unflatten(
                1,
                (parameters["batch_size"], len(parameters["joint_indices"]) // 3),
            ),
        )

        relative_poses = torch.einsum(
            "...ji, ...jk -> ...ik",
            rotation_matrices,
            noisy_poses,
        )

        return noisy_poses, relative_poses

    return empirical_diffuser


def create_analytical_diffuser(
    parameters: Dict[str, Any],
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate an analytical diffuser based on training parameters.

    Note
    ----
    An analytical diffuser transforms clean poses into noisy poses.
    This diffuser can evaluate the analytical velocity vector field at noisy poses.

    Parameters
    ----------
    `parameters: Dict[str, Any]`
    A dictionary containing training parameters

    Returns
    -------
    A function mapping `(initial_condition, time)` to `(diffused_condition, velocity)`
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

    def analytical_diffuser(
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
        The shapes of the tensors are:
        `(num_times, num_samples, num_joints, 3, 3)` and
        `(num_times, num_samples, num_joints, 3)` respectively.
        """
        axis = axis_process.at(time).sample(
            parameters["batch_size"],
        )

        angle = angle_process.at(time).sample(
            parameters["batch_size"],
        )
        angle.requires_grad_()

        log_density = torch.log(angle_process.density(angle))
        log_density.sum().backward()

        angle_velocity = torch.einsum("i..., i... -> i...", 6.0 * time**3, angle.grad)

        lamb = 2.0

        theta = torch.einsum(
            "i..., i... -> i...",
            torch.exp(lamb * (time - 1.6)),
            2.0
            * torch.acos(
                torch.clip(1.0 - (angle / torch.pi) ** 0.75, min=0.0, max=1.0),
            ),
        )

        angle = angle / torch.pi

        angle_velocity = -lamb * theta + 3.0 / (2.0 * torch.pi) * torch.einsum(
            "i..., i... -> i...",
            torch.exp(lamb * (time - 1.6)),
            angle_velocity
            / torch.clip(torch.sqrt(2.0 * angle**1.25 - angle**2), min=1e-12),
        )

        axis_angle = torch.einsum("...i, ... -> ...i", axis, theta)
        velocity = torch.einsum("...i, ... -> ...i", axis, angle_velocity)

        noisy_poses = manifold.exp(
            initial_condition.unsqueeze(0),
            axis_angle,
        )

        return noisy_poses, velocity

    return analytical_diffuser
