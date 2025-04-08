import torch
from typing import Any, Dict


from diffusionmodels.manifolds.rotationalgroups import SpecialOrthogonal3
from diffusionmodels.stochasticprocesses.multivariate.uniform import UniformSphere
from diffusionmodels.stochasticprocesses.univariate.inversetransforms.cumulativedistributions.heatequations import (
    PeriodicCumulativeEnergy,
)
from diffusionmodels.stochasticprocesses.univariate.inversetransforms.rootfinders.bisection import (
    Bisection,
)
from diffusionmodels.stochasticprocesses.univariate.inversetransforms import (
    InverseTransform,
)
from torch.nn import init


def generate_diffuser(parameters: Dict[str, Any]):
    _axis_process = UniformSphere(
        dimension=3, device=parameters["device"], data_type=parameters["data_type"]
    )

    _angle_distribution = PeriodicCumulativeEnergy(
        num_waves=parameters["num_waves"],
        mean_squared_displacement=parameters["mean_squared_displacement"],
        alpha=parameters["alpha"],
        device=parameters["device"],
        data_type=parameters["data_type"],
    )

    _root_finder = Bisection(num_iterations=parameters["num_iterations"])

    _angle_process = InverseTransform(
        distribution=_angle_distribution,
        root_finder=_root_finder,
        device=parameters["device"],
        data_type=parameters["data_type"],
    )

    _manifold = SpecialOrthogonal3(
        device=parameters["device"], data_type=parameters["data_type"]
    )

    def diffuse(initial_condition: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        axis = _axis_process.at(time).sample(52 * parameters["batch_size"])
        angle = _angle_process.at(time).sample(52 * parameters["batch_size"])
        angle.requires_grad_()

        y = torch.log(_angle_process.density(angle))
        y.sum().backward()

        lamb = 2.0

        theta = torch.einsum(
            "i..., i... -> i...",
            torch.exp(lamb * (time - 1.6)),
            2.0
            * torch.acos(
                torch.clip(1.0 - (angle / torch.pi) ** 0.75, min=0.0, max=1.0)
            ),
        )

        axis_angle = torch.einsum("...i, ... -> ...i", axis, theta)

        return _manifold.exp(initial_condition.unsqueeze(0), axis_angle.unflatten(1, (2, 52)))
        # return _manifold.exp(initial_condition.unsqueeze(0), axis_angle.unsqueeze(2))

    return diffuse
