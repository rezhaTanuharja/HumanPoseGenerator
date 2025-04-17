from typing import Tuple

import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        layer_sizes: Tuple[int, ...],
        dropout_at: Tuple[int, ...],
        activation_layer: torch.nn.Module,
    ) -> None:
        super().__init__()
        self._layers = torch.nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self._layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self._layers.append(activation_layer)
            if i in dropout_at:
                self._layers.append(torch.nn.Dropout())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)

        return x


class FiLM(torch.nn.Module):
    """
    Minimal implementation of the Feature-wise Linear Modulation.
    Consists of a single modulator which modulates the input of a main block.

    Example Usage
    -------------
    ```python
    modulator = torch.nn.Linear(20, 128)
    main_block = torch.nn.Linear(64, 16)
    model = FiLM(modulator, main_block)

    signal = torch.rand(size=(64,))
    condition = torch.rand(size=(20,))

    prediction = model(signal, condition)
    ```
    """

    def __init__(
        self,
        modulator: torch.nn.Sequential,
        main_block: torch.nn.Sequential,
    ) -> None:
        """
        Instantiate a FiLM module by specifying a modulator and a main block.

        Parameters
        ----------
        `modulator: torch.nn.Module`
        A module that process `condition` and output multiplicative and additive modulations.

        `main_block: torch.nn.Module`
        A module that process modulated input and output prediction.
        """
        super(FiLM, self).__init__()
        self._modulator = modulator
        self._head = main_block

    def forward(self, signal: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Generate a prediction from a conditioned signal.

        Parameters
        ----------
        `signal: torch.Tensor`
        A tensor with shape `(..., main_block_input_size)`

        `condition: torch.Tensor`
        A tensor with shape `(..., modulator_input_size)`

        Returns
        -------
        `torch.Tensor`
        A prediction based on `signal` conditioned to `condition` with shape `(... main_block_output_size)`.
        """
        gamma, beta = self._modulator(condition).chunk(2, dim=-1)
        return self._head(gamma * signal + beta)
