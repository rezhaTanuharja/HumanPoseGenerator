"""Various utility methods to perform sampling."""

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


def sample_uniform_so3(num_samples: int) -> torch.Tensor:
    """
    Generate uniform samples on SO(3) based on Shoemake's method.

    Parameters
    ----------
    `num_samples: int`
    The number of uniformly distributed samples to generate.

    Returns
    -------
    `torch.Tensor`
    A tensor with shape `(num_samples, 3, 3)`
    """
    u1, u2, u3 = torch.rand(3, num_samples)

    q0 = torch.sqrt(1.0 - u1) * torch.sin(2 * torch.pi * u2)
    q1 = torch.sqrt(1.0 - u1) * torch.cos(2 * torch.pi * u2)
    q2 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q3 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    return torch.stack(
        [
            1 - 2 * (q2**2 + q3**2),
            2 * (q1 * q2 - q0 * q3),
            2 * (q1 * q3 + q0 * q2),
            2 * (q1 * q2 + q0 * q3),
            1 - 2 * (q1**2 + q3**2),
            2 * (q2 * q3 - q0 * q1),
            2 * (q1 * q3 - q0 * q2),
            2 * (q2 * q3 + q0 * q1),
            1 - 2 * (q1**2 + q2**2),
        ],
        dim=-1,
    ).reshape(num_samples, 3, 3)
