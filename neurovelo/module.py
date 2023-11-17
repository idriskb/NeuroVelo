import torch
import torch.nn as nn


class LatentODE(nn.Module):
    """
    A class modelling the latent splicing dynamics.

    Parameters
    ----------
    n_latent
        Dimension of latent space.
        (Default: 20)
    n_hidden
        The dimensionality of the hidden layer for the ODE function.
        (Default: 128)
    """

    def __init__(
        self,
        n_latent: int = 20,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """
        Compute the gradient at a given time t and a given state x.

        Parameters
        ----------
        t
            A given time point.
        x
            A given spliced latent state.

        Returns
        ----------
        :class:`torch.Tensor`
            A tensor
        """
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out


class Encoder(nn.Module):
    """
    Linear encoder class for dimensionality reduction and features selection.

    Parameters
    ----------
    n_int
        Number of genes.
    n_latent
        Dimension of latent space.
        (Default: 20)
    n_hidden
        The dimensionality of the hidden layer for the AutoEncoder.
        (Default: 128)
    batch_norm
        Whether to include `BatchNorm` layer or not.
        (Default: 'False')
    """

    def __init__(
        self,
        n_int: int,
        n_latent: int = 20,
        n_hidden: int = 128,
        batch_norm: bool = False,
    ):
        super().__init__()
        
        self.n_latent = n_latent
        self.fc = nn.Sequential()
        self.fc.add_module('L1', nn.Linear(n_int, n_hidden))
        if batch_norm:
            self.fc.add_module('N1', nn.BatchNorm1d(n_hidden))
        self.fc2 = nn.Linear(n_hidden, n_latent)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, x:torch.Tensor) -> tuple:
        x = self.fc(x)
        out = self.fc2(x)
        t = self.fc3(x).sigmoid()
        return t, out


class Decoder(nn.Module):
    """
    Linear decoder class to reconstruct the original counts based on the latent representation.

    Parameters
    ----------
    n_latent
        Dimension of latent space.
        (Default: 20)
    n_int
        Number of genes.
    n_hidden
        The dimensionality of the hidden layer for the AutoEncoder.
        (Default: 128)
    batch_norm
        Whether to include `BatchNorm` layer or not.
        (Default: 'False')
    """

    def __init__(
        self,
        n_int: int,
        n_latent: int = 20,
        n_hidden: int = 128,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.fc = nn.Sequential()
        self.fc.add_module('L1', nn.Linear(n_latent, n_hidden))
        if batch_norm:
            self.fc.add_module('N1', nn.BatchNorm1d(n_hidden))
        self.fc2 = nn.Linear(n_hidden, n_int)

    def forward(self, z: torch.Tensor):
        out = self.fc(z)
        recon_x = self.fc2(out)
        return recon_x
