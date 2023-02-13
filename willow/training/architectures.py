import torch, torch.nn as nn

class WaveNet(nn.Module):
    """Neural network architecture following Espinosa et al. (2022)."""

    def __init__(self,
        n_in: int,
        n_out: int,
        branch_dims: list[int]=[64, 32]
    ) -> None:
        """
        Initialize a WaveNet model.

        Parameters
        ----------
        n_in : Number of input features.
        n_out : Number of output features.
        branch_dims : List of dimensions of the layers to include in each of the
            level-specific branches.

        """

        super().__init__()

        shared = [nn.BatchNorm1d(n_in), nn.Linear(n_in, 256), nn.ReLU()]
        for _ in range(4):
            shared.append(nn.Linear(256, 256))
            shared.append(nn.ReLU())

        shared.append(nn.Linear(256, branch_dims[0]))
        shared.append(nn.ReLU())

        branches = []
        for _ in range(n_out):
            args: list[nn.Module] = []
            for a, b in zip(branch_dims[:-1], branch_dims[1:]):
                args.append(nn.Linear(a, b))
                args.append(nn.ReLU())

            args.append(nn.Linear(branch_dims[-1], 1))
            branches.append(nn.Sequential(*args))

        self.shared = nn.Sequential(*shared)
        self.branches = nn.ModuleList(branches)

        self.shared.apply(_xavier_init)
        for branch in self.branches:
            branch.apply(_xavier_init)

        self.double()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the network to a `Tensor` of input features.

        Parameters
        ----------
        X : `Tensor` of input features.

        Returns
        -------
        output : `Tensor` of predicted outputs.

        """

        Z, levels = self.shared(X), []
        for branch in self.branches:
            levels.append(branch(Z).squeeze())

        return torch.vstack(levels).T

def _xavier_init(layer: nn.Module) -> None:
    """
    Apply Xavier initialization to a layer if it is an `nn.Linear`.

    Parameters
    ----------
    layer : Linear to potentially initialize.

    """

    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
