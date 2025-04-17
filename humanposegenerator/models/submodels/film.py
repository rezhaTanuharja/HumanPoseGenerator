import torch


class FiLM(torch.nn.Module):
    """
    Minimal implementation of the Feature-wise Linear Modulation layer.

    Example Usage:

    ```python
        modulator = torch.nn.Linear(20, 128)
        main_block = torch.nn.Linear(64, 16)
        model = FiLM(modulator, main_block)

        signal = torch.rand(size=(64,))
        condition = torch.rand(size=(20,))

        prediction = model(signal, condition)
    ```

    In the example, `128 = 2 * 64` and the prediction will have shape `(..., 16)`.
    """

    def __init__(
        self,
        modulator: torch.nn.Module,
    ) -> None:
        """
        Instantiate a FiLM layer by specifying a modulator and a main block.

        Parameters
        ----------
        `modulator: torch.nn.Module`
        A module that process `condition` and output multiplicative and additive modulations.

        `main_block: torch.nn.Module`
        A module that process modulated input and output prediction.
        """
        super(FiLM, self).__init__()
        self._modulator = modulator

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
        A prediction based on `signal` conditioned to `condition`.
        """
        gamma, beta = self._modulator(condition).chunk(2, dim=-1)
        return gamma * signal + beta
