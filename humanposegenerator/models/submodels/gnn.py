from typing import Optional, Tuple, cast

import torch
import torch_geometric.nn


from humanposegenerator import utilities


class GraphNeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        signal_shapes: Tuple[int, ...],
        activation_layer: torch.nn.Module,
        edge_index: torch.Tensor,
        *,
        drop_last_activation: bool = False,
    ):
        super().__init__()

        self._edge_index = edge_index
        self._layers = torch.nn.ModuleList()

        for i in range(len(signal_shapes) - 1):
            self._layers.append(
                torch_geometric.nn.GCNConv(signal_shapes[i], signal_shapes[i + 1]),
            )
            self._layers.append(activation_layer)

        if drop_last_activation:
            self._layers = self._layers[:-1]

    def forward(
        self,
        signal: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._edge_index = self._edge_index.to(signal.device)
        utilities.typing.unused_variables(condition)

        signal = signal.unflatten(dim=-1, sizes=(signal.shape[-1] // 9, 9))

        for layer in cast(torch.nn.ModuleList, self._layers):
            if isinstance(layer, torch_geometric.nn.GCNConv):
                signal = layer(signal, self._edge_index)
                continue

            signal = layer(signal)

        return signal.flatten(-2)
