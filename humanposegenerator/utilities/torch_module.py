from typing import Any, Dict, List, Tuple

import torch


def set_dropout_rate(model: torch.nn.Module, rate: float) -> None:
    """
    Set a dropout probability for all `Dropout` layer in the model, if any.

    Parameters
    ----------
    `model: torch.nn.Module`
    The model in which the dropout layers' probability is going to be set.

    `rate: float`
    The probability of all dropout layer(s) will be set to this value.
    """
    for child in model.modules():
        if isinstance(child, torch.nn.Dropout):
            child.p = rate


def get_parameters(
    model: torch.nn.Module,
    parameter_name: str,
) -> List[Tuple[str, torch.nn.Module, str]]:
    """
    Collect a certain type of parameter from the model.

    Parameters
    ----------
    `model: torch.nn.Module`
    The model from which the parameters are extracted.

    `parameter: str`
    The name of parameter to extract.

    Returns
    -------
    A list of tuples `(name, module, parameter)`.
    """
    parameters_to_prune = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        if getattr(module, parameter_name, None) is not None:
            parameters_to_prune.append((name, module, parameter_name))

    return parameters_to_prune


def get_state_dict(
    model: torch.nn.Module,
) -> Dict[str, Any]:
    state_dict = model.state_dict()

    for name, module in model.named_modules():
        if hasattr(module, "weight_orig"):
            state_dict[f"{name}.weight_orig"] = module.weight_orig
            state_dict[f"{name}.weight_mask"] = module.weight_mask

    return state_dict
