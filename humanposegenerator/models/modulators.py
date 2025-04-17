"""
Provides models that modulates inputs using preprocessed conditions.

Classes
-------
`FiLM`  : a minimal implementation of Feature-wise Linear Modulation.
"""

from typing import Any, Dict, List, Tuple

import torch


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


class NewFiLM(torch.nn.Module):
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
        pre_modulator: torch.nn.Sequential,
        pos_modulator: torch.nn.Sequential,
        pre_block: torch.nn.Sequential,
        pos_block: torch.nn.Sequential,
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
        super(NewFiLM, self).__init__()
        self._pre_modulator = pre_modulator
        self._pos_modulator = pos_modulator
        # self._head = main_block
        self._pre_block = pre_block
        self._pos_block = pos_block

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
        gamma, beta = self._pre_modulator(condition).chunk(2, dim=-1)

        output = self._pre_block(gamma * signal + beta)

        gamma, beta = self._pos_modulator(condition).chunk(2, dim=-1)
        # return self._head(gamma * signal + beta)
        return self._pos_block(gamma * output + beta)


def generate_film_model(
    parameters: Dict[str, Any],
) -> Tuple[torch.nn.Module, List[Tuple[torch.nn.Sequential, str]]]:
    """
    Generate a `FiLM` model with a predefined structure.

    Parameters
    ----------
    A dictionary containing at least the following items:
    - `"device"`: the hardware where tensors live,
    - `"joint_indices"`: the indices list of joints to be predicted
    - `"num_frequencies"`: the number of sinusoids used to encode time,
    - `"hidden_size_modulator"`: the number of neurons in `modulator`'s hidden layers,
    - `"hidden_size_main_block"`: the number of neurons in `main_block`'s hidden layers,
    - `"negative_slope"`: the gradient of `LeakyReLU` in the second quadrant,

    Returns
    -------
    A pair of model and its prune-able weights
    """
    modulator = torch.nn.Sequential(
        torch.nn.Linear(
            2 * parameters["num_frequencies"],
            parameters["hidden_size_modulator"],
        ),
        torch.nn.LeakyReLU(negative_slope=parameters["negative_slope"]),
        torch.nn.Linear(
            parameters["hidden_size_modulator"],
            parameters["hidden_size_modulator"],
        ),
        torch.nn.LeakyReLU(negative_slope=0.005),
        torch.nn.Dropout(),
        torch.nn.Linear(
            parameters["hidden_size_modulator"],
            # len(parameters["joint_indices"]) * 6,
            2 * parameters["hidden_size_main_block"],
        ),
        torch.nn.LeakyReLU(negative_slope=parameters["negative_slope"]),
    ).to(parameters["device"])

    main_block = torch.nn.Sequential(
        torch.nn.Linear(
            len(parameters["joint_indices"]) * 3,
            parameters["hidden_size_main_block"],
        ),
        torch.nn.LeakyReLU(negative_slope=parameters["negative_slope"]),
        torch.nn.Dropout(),
        torch.nn.Linear(
            parameters["hidden_size_main_block"],
            parameters["hidden_size_main_block"],
        ),
        torch.nn.LeakyReLU(negative_slope=parameters["negative_slope"]),
        torch.nn.Linear(
            parameters["hidden_size_main_block"],
            len(parameters["joint_indices"]),
        ),
    ).to(parameters["device"])

    prunable_weights = [
        (modulator[2], "weight"),
        (modulator[5], "weight"),
        (main_block[0], "weight"),
        (main_block[3], "weight"),
        (main_block[5], "weight"),
        # (pre_block[0], "weight"),
        # (pre_block[3], "weight"),
        # (pos_block[0], "weight"),
    ]

    # model = main_block
    model = FiLM(modulator, main_block)
    model.train()

    return model, prunable_weights


def generate_new_film_model(
    parameters: Dict[str, Any],
) -> Tuple[torch.nn.Module, List[Tuple[torch.nn.Sequential, str]]]:
    """
    Generate a `FiLM` model with a predefined structure.

    Parameters
    ----------
    A dictionary containing at least the following items:
    - `"device"`: the hardware where tensors live,
    - `"joint_indices"`: the indices list of joints to be predicted
    - `"num_frequencies"`: the number of sinusoids used to encode time,
    - `"hidden_size_modulator"`: the number of neurons in `modulator`'s hidden layers,
    - `"hidden_size_main_block"`: the number of neurons in `main_block`'s hidden layers,
    - `"negative_slope"`: the gradient of `LeakyReLU` in the second quadrant,

    Returns
    -------
    A pair of model and its prune-able weights
    """
    pre_modulator = torch.nn.Sequential(
        torch.nn.Linear(
            2 * parameters["num_frequencies"],
            parameters["hidden_size_modulator"],
        ),
        torch.nn.LeakyReLU(negative_slope=parameters["negative_slope"]),
        torch.nn.Linear(
            parameters["hidden_size_modulator"],
            parameters["hidden_size_modulator"],
        ),
        torch.nn.LeakyReLU(negative_slope=0.005),
        torch.nn.Dropout(),
        torch.nn.Linear(
            parameters["hidden_size_modulator"],
            len(parameters["joint_indices"]) * 6,
        ),
        torch.nn.LeakyReLU(negative_slope=parameters["negative_slope"]),
    ).to(parameters["device"])

    pos_modulator = torch.nn.Sequential(
        torch.nn.Linear(
            2 * parameters["num_frequencies"],
            parameters["hidden_size_modulator"],
        ),
        torch.nn.LeakyReLU(negative_slope=parameters["negative_slope"]),
        torch.nn.Linear(
            parameters["hidden_size_modulator"],
            parameters["hidden_size_modulator"],
        ),
        torch.nn.LeakyReLU(negative_slope=0.005),
        torch.nn.Dropout(),
        torch.nn.Linear(
            parameters["hidden_size_modulator"],
            # len(parameters["joint_indices"]) * 6,
            2 * parameters["hidden_size_main_block"],
        ),
        torch.nn.LeakyReLU(negative_slope=parameters["negative_slope"]),
    ).to(parameters["device"])

    pre_block = torch.nn.Sequential(
        torch.nn.Linear(
            len(parameters["joint_indices"]) * 3,
            parameters["hidden_size_main_block"],
        ),
        torch.nn.LeakyReLU(negative_slope=parameters["negative_slope"]),
        torch.nn.Dropout(),
        torch.nn.Linear(
            parameters["hidden_size_main_block"],
            parameters["hidden_size_main_block"],
        ),
    ).to(parameters["device"])

    pos_block = torch.nn.Sequential(
        torch.nn.LeakyReLU(negative_slope=parameters["negative_slope"]),
        torch.nn.Linear(
            parameters["hidden_size_main_block"],
            len(parameters["joint_indices"]),
        ),
    ).to(parameters["device"])

    prunable_weights = [
        (pre_modulator[0], "weight"),
        (pre_modulator[2], "weight"),
        (pre_modulator[5], "weight"),
        (pos_modulator[0], "weight"),
        (pos_modulator[2], "weight"),
        (pos_modulator[5], "weight"),
        (pre_block[0], "weight"),
        (pre_block[3], "weight"),
        (pos_block[1], "weight"),
    ]

    # model = main_block
    # model = torch.nn.Sequential(
    #     pre_block,
    #     pos_block,
    # )
    model = NewFiLM(pre_modulator, pos_modulator, pre_block, pos_block)
    model.train()

    return model, prunable_weights
