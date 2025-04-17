from typing import Any, Dict, List, cast

import torch

from humanposegenerator import models


class Assembly(torch.nn.Module):
    """A model assembled by chaining several submodels."""

    def __init__(self, components: List[Dict[str, Any]]):
        """
        Build the model from a config.

        Parameters
        ----------
        `components: List[Dict[str, Any]]`
        A list of components. Each component is a dictionary with subcomponent names as keys and their constructor arguments (as a dictionary) as values.

        Example Usage
        -------------
        Building a model consisting of one FiLM block and one MLP block:

        ```python

            config = [
                {
                    "modulator": {
                        "layer_sizes": (16, 60, 60, 16),
                        "dropout_at": (1,),
                        "activation_layer": activation_layer,
                    },
                    "main_block": {
                        "layer_sizes": (8, 32, 32, 10),
                        "dropout_at": (1,),
                        "activation_layer": activation_layer,
                    },
                },
                {
                    "mlp": {
                        "layer_sizes": (10, 20, 4),
                        "dropout_at": (1,),
                        "activation_layer": activation_layer,
                    },
                },
            ]

            model = Assembly(config)
        ```
        """
        super().__init__()

        self._layers = torch.nn.ModuleList()

        for component in components:
            if "mlp" in component:
                self._layers.append(
                    models.submodels.mlp.MultiLayerPerceptron(**component["mlp"]),
                )

            if "modulator" in component:
                modulator = models.submodels.mlp.MultiLayerPerceptron(
                    **component["modulator"],
                )

                self._layers.append(models.submodels.film.FiLM(modulator=modulator))

    def forward(self, signal: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward propagation through the model.

        Parameters
        ----------
        `signal: torch.Tensor`
        The input signal from which a prediction is made.

        `condition: torch.Tensor`
        The condition applied to the input signal.

        Returns
        -------
        A prediction made based on `signal` conditioned on `condition`.
        """
        for layer in cast(torch.nn.ModuleList, self._layers):
            signal = layer(signal, condition)

        return signal
