from typing import Callable

import torch
from diffusionmodels import manifolds

from .interfaces import Solver


class ExplicitEuler(Solver):
    def __init__(
        self,
        velocity_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        time_encoder: Callable[[torch.Tensor], torch.Tensor],
        manifold: manifolds.interfaces.Manifold,
        device: torch.device,
        data_type: torch.dtype,
    ) -> None:
        self._velocity_model = velocity_model
        self._time_encoder = time_encoder
        self._manifold = manifold
        self._device = device
        self._data_type = data_type

    def solve(
        self,
        initial_condition: torch.Tensor,
        initial_time: float,
        num_time_steps: int,
    ) -> torch.Tensor:
        num_samples = initial_condition.shape[0]
        time_increment = initial_time / num_time_steps

        time = torch.tensor(
            [
                initial_time,
            ],
            device=self._device,
            dtype=self._data_type,
        )

        for _ in range(num_time_steps):
            encoded_time = self._time_encoder(time).repeat(num_samples, 1, 1)

            initial_condition = self._manifold.exp(
                initial_condition,
                time_increment
                * self._velocity_model(
                    initial_condition.flatten(-3),
                    encoded_time.flatten(-2),
                ).view(num_samples, 21, 3),
            )

            time = time - time_increment

        return initial_condition
