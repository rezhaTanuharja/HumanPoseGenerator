import torch


class FiLM(torch.nn.Module):
    def __init__(self, modulator: torch.nn.Module, head: torch.nn.Module) -> None:
        super(FiLM, self).__init__()
        self._modulator = modulator
        self._head = head

    def forward(self, x, cond):
        gamma, beta = self._modulator(cond).chunk(2, dim=-1)
        return self._head(gamma * x + beta)
