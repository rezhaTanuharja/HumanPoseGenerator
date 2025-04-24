from typing import Optional, Tuple, cast

import torch

from humanposegenerator import utilities


class MultiLayerPerceptron(torch.nn.Module):
    """Custom class of Multi-Layer Perceptron for rapid prototyping."""

    def __init__(
        self,
        signal_shapes: Tuple[int, ...],
        dropout_at: Tuple[int, ...],
        activation_layer: torch.nn.Module,
        *,
        drop_last_activation: bool = False,
    ) -> None:
        """
        Create an MLP using only `torch.nn.Linear`, `torch.nn.Dropout`, and `activation_layer`.

        Parameters
        ----------
        `signal_shapes: Tuple[int, ...]`
        The shapes of the signal as it pass through the block.

        `dropout_at: Tuple[int, ...]`
        The indices of `torch.nn.Linear` - `activation_layer` pair to put a dropout after.

        `activation_layer: torch.nn.Module`
        The activation layer to use in the MLP.

        `drop_last_activation: bool = False`
        Flag if the MLP should omit the last activation layer.
        """
        super().__init__()

        self._layers = torch.nn.ModuleList()

        for i in range(len(signal_shapes) - 1):
            self._layers.append(torch.nn.Linear(signal_shapes[i], signal_shapes[i + 1]))

            if i in dropout_at:
                self._layers.append(torch.nn.Dropout())

            self._layers.append(activation_layer)

        if drop_last_activation:
            self._layers = self._layers[:-1]

    def forward(
        self,
        signal: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Propagate input forward through the MLP.

        Parameters
        ----------
        `signal: torch.Tensor`
        A tensor with shape `(..., num_neurons_in_first_layer)`.

        `condition: torch.Tensor` (unused)

        Returns
        -------
        `torch.Tensor`
        A tensor with shape `(..., num_neurons_in_last_layer)`A.
        """
        utilities.typing.unused_variables(condition)

        for layer in cast(torch.nn.ModuleList, self._layers):
            signal = layer(signal)

        return signal
