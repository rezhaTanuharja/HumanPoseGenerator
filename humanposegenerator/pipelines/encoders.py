from typing import Any, Dict

import torch


def generate_encoder(parameters: Dict[str, Any]):
    def time_encoder(time: torch.Tensor):
        waves = torch.arange(
            1,
            parameters["num_sinusoids"] + 1,
            device=parameters["device"],
        ).unsqueeze(0)

        encoded_time = torch.cat(
            [
                torch.sin(
                    2.0 * torch.pi * time.unsqueeze(-1) / parameters["period"] * waves,
                ),
                torch.cos(
                    2.0 * torch.pi * time.unsqueeze(-1) / parameters["period"] * waves,
                ),
            ],
            dim=-1,
        )

        return encoded_time.unsqueeze(1).repeat(1, parameters["batch_size"], 1)

    return time_encoder
