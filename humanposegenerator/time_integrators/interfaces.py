from abc import ABC, abstractmethod
from typing import Callable

from diffusionmodels import manifolds

import torch


class Solver(ABC):
    def __init__(
        self,
        velocity_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        time_encoder: Callable[[float], torch.Tensor],
        manifold: manifolds.interfaces.Manifold,
        device: torch.device,
        data_type: torch.dtype,
    ) -> None:
        """
        Instantiate a numerical solver for reverse diffusion.

        Parameters
        ----------
        `velocity_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`
        A model that takes `position, condition` as input and output `velocity`.

        `time_encoder: Callable[[float], torch.Tensor]`
        An encoder that convert a float to a tensor for neural network input.

        `manifold: Manifold`
        The manifold on which data lives.

        `device: torch.device`
        The hardware where all tensor attributes live.

        `data_type: torch.dtype`
        The type of floating point.
        """
        self._velocity_model = velocity_model
        self._time_encoder = time_encoder
        self._manifold = manifold
        self._device = device
        self._data_type = data_type

    @abstractmethod
    def solve(
        self,
        initial_condition: torch.Tensor,
        initial_time: float,
        num_time_steps: int,
    ) -> torch.Tensor:
        """
        Solve the reverse diffusion process.

        Parameters
        ----------
        `initial_condition: torch.Tensor`
        The position of points at `time = initial_time`

        `initial_time: float`
        The time from which to move the points backward in time.

        `num_time_steps: int`
        The number of time steps for numerical solver.

        Returns
        -------
        `torch.Tensor`
        The position at `time = 0`
        """
        raise NotImplementedError
