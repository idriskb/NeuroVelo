import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from typing import Optional
from typing_extensions import Literal

from .module import LatentODE, Encoder, Decoder


class TNODE(nn.Module):
    """
    Class to automatically infer treatment wise cellular dynamics using autoencoder and neural ODE.

    Parameters
    ----------
    n_int
        Number of genes.
    n_latent
        Dimension of latent space.
        (Default: 20)
    n_sample
        Number of samples or treatment as independant ODEs
        (Default: 1)
    n_ode_hidden
        The dimensionality of the hidden layer for the ODE function.
        (Default: 128)
    n_vae_hidden
        The dimensionality of the hidden layer for the AutoEncoder.
        (Default: 128)
    batch_norm
        Whether to include `BatchNorm` layer.
        (Default: 'False')
    ode_method
        ODE Solver method.
        (Default: 'euler')
    same_ode
        Whether to initialize all ODEs with same initialization
        (Default: 'False')
    """

    def __init__(
        self,
        n_int: int,
        n_latent: int = 20,
        n_sample: int = 1,
        n_ode_hidden: int = 128,
        n_vae_hidden: int = 128,
        batch_norm: bool = False,
        ode_method: str = 'dopri5',
        same_ode: bool = False,
        pre_ptime: bool = False,
        reconstruct_xt: bool = False,
    ):
        super().__init__()
        self.n_int = n_int
        self.n_latent = n_latent
        self.n_ode_hidden = n_ode_hidden
        self.n_vae_hidden = n_vae_hidden
        self.batch_norm = batch_norm
        self.ode_method = ode_method
        self.n_sample = n_sample
        self.same_ode = same_ode
        self.pre_ptime = pre_ptime
        self.reconstruct_xt = reconstruct_xt

        self.lode_func = nn.ModuleDict({f'node_{i}': LatentODE(n_latent, n_ode_hidden) for i in range(self.n_sample)})  
        self.encoder = Encoder(n_int, n_latent, n_vae_hidden, batch_norm)
        self.decoder = Decoder(n_int, n_latent, n_vae_hidden, batch_norm)
        self.beta, self.lam = torch.nn.Parameter(torch.Tensor(self.n_latent), requires_grad=True), torch.nn.Parameter(torch.Tensor(self.n_latent), requires_grad=True)
        nn.init.uniform_(self.beta)
        nn.init.uniform_(self.lam)
    def forward(self, s: torch.Tensor, u: torch.Tensor, Pt: torch.Tensor, g: torch.Tensor) -> tuple:
        """
        Given the transcriptomes and the treatments of cells, this function predicts the time, latent space and dynamics of the cells.

        Parameters
        ----------
        s
            Spliced reads.
        u
            Unspliced reads.
        Pt
            Pseudotime tensor (Only if you want your velocity to follow specific pseudotime)
        g
            Sample/Treatment index.

        Returns
        ----------
        5-tuple of :class:`torch.Tensor`
            Tensors for loss, including:
            1) total loss,
            2) reconstruction loss from encoder-derived latent space,
            3) reconstruction loss from ODE-solver latent space,
        """
        if self.pre_ptime:
            Ts, z_s = self.encoder(s)
            _, z_u = self.encoder(u)
            Ts = Ts.ravel()
            index = torch.argsort(Ts)
            Ts = Ts[index]
            s = s[index]
            z_s = z_s[index]
            u = u[index]
            z_u = z_u[index]
            g = g[index]
            Pt = Pt[index]

            index2 = (Ts[:-1] != Ts[1:])
            index2 = torch.cat((index2, torch.tensor([True]).to(index2.device))) ## index2 is used to get unique time points as odeint requires strictly increasing/decreasing time points
            Ts = Ts[index2]
            s = s[index2]
            z_s = z_s[index2]
            z_u = z_u[index2]
            u = u[index2]
            g = g[index2]
            Pt = Pt[index2]
            pt_loss = F.mse_loss(Pt, Ts, reduction='mean')
        else:
            Ts, z_s = self.encoder(s)
            _, z_u = self.encoder(u)
            Ts = Ts.ravel()
            index = torch.argsort(Ts)
            Ts = Ts[index]
            s = s[index]
            z_s = z_s[index]
            u = u[index]
            z_u = z_u[index]
            g = g[index]
            index2 = (Ts[:-1] != Ts[1:])
            index2 = torch.cat((index2, torch.tensor([True]).to(index2.device))) ## index2 is used to get unique time points as odeint requires strictly increasing/decreasing time points
            Ts = Ts[index2]
            s = s[index2]
            z_s = z_s[index2]

            z_u = z_u[index2]
            u = u[index2]
            g = g[index2]
            pt_loss = 0
        pred_z = torch.empty(s.size(0), self.n_latent).to(s.device)
        z_div = torch.Tensor([0]).to(s.device)
        for i in range(self.n_sample):
            mask = g == i
            
            if mask.sum() == 0:
                continue
                
            zsm = z_s[mask]
            zum = z_u[mask]
            Tm = Ts[mask]
            z0 = zsm[0]
            

            pred_z[mask] = odeint(self.lode_func[f'node_{i}'], z0, Tm, method = self.ode_method).view(-1, self.n_latent)
            z_div += F.mse_loss(self.lode_func[f'node_{i}'](Tm, pred_z[mask]), (torch.exp(self.beta)*zum-torch.exp(self.lam)*zsm), reduction='mean')
        pred_x_s = self.decoder(z_s)
        pred_x_u = self.decoder(z_u)
        if self.reconstruct_xt:
            pred_x_t = self.decoder(pred_z)
            recon_loss_xt = F.mse_loss(s, pred_x_t, reduction='mean')
        else:
            recon_loss_xt = F.mse_loss(z_s, pred_z, reduction='mean')
        recon_loss_ec = F.mse_loss(s, pred_x_s, reduction='mean')
        recon_loss_ec_u = F.mse_loss(u, pred_x_u, reduction='mean')

        loss =recon_loss_ec + recon_loss_ec_u + z_div + recon_loss_xt + pt_loss
        return loss, recon_loss_ec, recon_loss_ec_u, z_div
    
