import torch


def latin_hypercube_sampling(num_samples: int, device: torch.device) -> torch.Tensor:
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

    intervals = torch.linspace(0, 1, num_samples + 1, device=device)[:-1]
    jitter = torch.rand(num_samples, device=device) / num_samples

    samples = intervals + jitter

    return samples
