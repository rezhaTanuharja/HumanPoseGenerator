import torch


def time_encoder(time: torch.Tensor, period: float, num_waves: int):
    waves = torch.arange(1, num_waves + 1, device=torch.device("cuda")).unsqueeze(0)

    encoding = torch.cat(
        [
            torch.sin(2.0 * torch.pi * time.unsqueeze(-1) / period * waves),
            torch.cos(2.0 * torch.pi * time.unsqueeze(-1) / period * waves),
        ],
        dim=-1,
    )

    return encoding
