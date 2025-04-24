import torch


def latin_hypercube_sampling(
    lower_bound: float,
    upper_bound: float,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Perform one-dimensional sampling using LHS.
    This is very basic, only generates one sample per interval.

    Parameters
    ----------
    `num_samples: int`
    The number of samples to generate, equals to the number of intervals

    `device: torch.device`
    The hardware where tensors live

    Returns
    -------
    `torch.Tensor`
    Samples in the shape of `(num_samples,)`, each element of which lies in `[0, 1]`
    """

    intervals = torch.linspace(
        lower_bound,
        upper_bound,
        num_samples + 1,
        device=device,
    )[:-1]

    jitter = (
        (upper_bound - lower_bound)
        * torch.rand(num_samples, device=device)
        / num_samples
    )

    return intervals + jitter
