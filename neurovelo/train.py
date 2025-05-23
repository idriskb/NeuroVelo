import torch
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from typing import Optional, Union
from typing_extensions import Literal
import numpy as np
from anndata import AnnData
from scipy import sparse
from scipy.sparse import spmatrix
from scipy.sparse import issparse
from tqdm import tqdm
import os
from collections import defaultdict
from sklearn.decomposition import PCA
from .model import TNODE
from .data import split_data, MakeDataset



class Trainer:
    """
    Class for implementing the scTour training process.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    percent
        The percentage of cells used for model training. Default to 0.2 when the cell number > 10,000 and to 0.9 otherwise.
    n_latent
        The dimensionality of the latent space.
        (Default: 5)
    n_ode_hidden
        The dimensionality of the hidden layer for the latent ODE function.
        (Default: 25)
    n_vae_hidden
        The dimensionality of the hidden layer for the VAE.
        (Default: 128)
    batch_norm
        Whether to include a `BatchNorm` layer.
        (Default: `False`)
    odesample_obs
        key in anndata observation for ODE sample (This is different from celltype, use it only when you want to modelize different dynamics)
        (Default:None)
    ode_method
        The solver for integration.
        (Default: `'euler'`)
    nepoch
        Number of epochs.
    batch_size
        The batch size during training.
        (Default: 1024)
    lr
        The learning rate.
        (Default: 1e-3)
    wt_decay
        The weight decay for Adam optimizer.
        (Default: 1e-6)
    eps
        The eps for Adam optimizer.
        (Default: 0.01)
    random_state
        The seed for generating random numbers.
        (Default: 0)
    use_gpu
        Whether to use GPU when available
        (Default: True)
    layer
        RNA reads to use, either spliced or moments
        (Default: spliced)
    use_pca
        Whether to initialize the linear encoder with PCA
        (Default: False)
    reconstruct_xt
        Whether to recontsuct x(t) instead of z(t)
        (Default: False)
    """

    def __init__(
        self,
        adata: AnnData,
        percent: Optional[float] = None,
        odesample_obs: Optional[str] = None,
        n_latent: int = 5,
        n_ode_hidden: int = 25,
        n_vae_hidden: int = 128,
        batch_norm: bool = False,
        ode_method: str = 'dopri5',
        nepoch: Optional[int] = None,
        batch_size: int = 1024,
        lr: float = 1e-3,
        wt_decay: float = 1e-6,
        eps: float = 0.01,
        random_state: int = 0,
        use_gpu: bool = False,
        n_sample: int = 1,
        layer = 'spliced',
        use_pca: bool = False,
        pre_ptime: bool = False,
        reconstruct_xt: bool = False,
    ):
        self.adata = adata
        self.percent = percent
        self.precent = percent
        self.n_cells = adata.n_obs
        self.batch_size = batch_size
        self.nepoch = nepoch
        self.lr = lr
        self.wt_decay = wt_decay
        self.eps = eps
        self.n_latent = n_latent
        self.n_vae_hidden = n_vae_hidden 
        self.odesample_obs = odesample_obs
        self.pre_ptime = pre_ptime
        self.reconstruct_xt = reconstruct_xt
        
        if layer == 'spliced':
            self.layer = ['spliced','unspliced']
            print('Using spliced and unspliced reads')
        else:
            self.layer = ['Ms', 'Mu']
            print('Using spliced and unspliced moments')

        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        #Convert sparse input to matrix
        if issparse(adata.layers[self.layer[0]]):
            self.adata.layers[self.layer[0]], self.adata.layers[self.layer[1]] = adata.layers[self.layer[0]].toarray(), adata.layers[self.layer[1]].toarray()

        
        if odesample_obs is None:
            self.adata.obs[odesample_obs] = 0
        assert self.adata.obs[odesample_obs].unique().size == n_sample
        self.n_int = adata.n_vars
        self.model_kwargs = dict(
            n_int = self.n_int,
            n_latent = self.n_latent,
            n_sample = n_sample,
            n_ode_hidden = n_ode_hidden,
            n_vae_hidden = n_vae_hidden,
            batch_norm = batch_norm,
            ode_method = ode_method,
            pre_ptime = pre_ptime,
            reconstruct_xt = reconstruct_xt
        )
        self.model = TNODE(**self.model_kwargs)
        self.log = defaultdict(list)
        
        if pre_ptime:
            ptime = adata.obs['latent_time']
        else:
            adata.obs['latent_time'] = np.ones(adata.n_obs)
        
        if use_pca:
            w1, w2 = self.get_init_pca()
            self.model.encoder.fc.L1.weight.data.copy_(w1)
            self.model.encoder.fc2.weight.data.copy_(w2)
            self.adata.layers[self.layer[0]] = self.adata.layers[self.layer[0]] - self.adata.layers[self.layer[0]].mean(0)
            self.adata.layers[self.layer[1]] = self.adata.layers[self.layer[1]] - self.adata.layers[self.layer[1]].mean(0)

        
        gpu = torch.cuda.is_available() and use_gpu
        if gpu:
            torch.cuda.manual_seed(random_state)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        
    def get_init_pca(self):
        pca1 = PCA(self.n_vae_hidden)
        pca2 = PCA(self.n_latent)
        reduced_x = pca1.fit_transform(self.adata.layers[self.layer[0]])
        pca2.fit(reduced_x)
        return torch.Tensor(pca1.components_), torch.Tensor(pca2.components_)
    
    def get_data_loaders(self) -> None:
        """
        Generate Data Loaders for training and validation datasets.
        """

        train_data, val_data = split_data(self.adata, self.percent,self.odesample_obs, self.layer)
        self.train_dataset = MakeDataset(train_data)
        self.val_dataset = MakeDataset(val_data)

        self.train_dl = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.val_dl = DataLoader(self.val_dataset, batch_size = self.batch_size)


    def train(self):
        self.get_data_loaders()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr = self.lr, weight_decay = self.wt_decay, eps = self.eps)
        #self.optimizer = torch.optim.SGD(params, lr=self.lr, weight_decay=self.wt_decay, momentum=0.9)
        with tqdm(total=self.nepoch, unit='epoch') as t:
            for tepoch in range(t.total):
                train_loss = self.on_epoch_train(self.train_dl)
                val_loss = self.on_epoch_val(self.val_dl)
                self.log['train_loss'].append(train_loss.item())
                self.log['validation_loss'].append(val_loss.item())
                t.set_description(f"Epoch {tepoch + 1}")
                t.set_postfix({'train_loss': train_loss.item(), 'val_loss':val_loss.item()}, refresh=False)
                t.update()


    def on_epoch_train(self, DL) -> float:
        """
        Go through the model and update the model parameters.

        Parameters
        ----------
        DL
            DataLoader for training dataset.

        Returns
        ----------
        float
            Training loss for the current epoch.
        """
        self.model.train()
        total_loss = .0
        ss = 0
        for X, Y, T, S in DL:
            self.optimizer.zero_grad()
            X = X.to(self.device)
            Y = Y.to(self.device)
            T = T.to(self.device)
            S = S.to(self.device)
            loss, _, _, _ = self.model(X,Y,T,S)
            loss.backward()
            self.optimizer.step()

            total_loss += loss * X.size(0)
            ss += X.size(0)

        train_loss = total_loss/ss
        return train_loss


    @torch.no_grad()
    def on_epoch_val(self, DL) -> float:
        """
        Validate using validation dataset.

        Parameters
        ----------
        DL
            DataLoader for validation dataset.

        Returns
        ----------
        float
            Validation loss for the current epoch.
        """

        self.model.eval()
        total_loss = .0
        ss = 0
        for X, Y, T, S in DL:
            X = X.to(self.device)
            Y = Y.to(self.device)
            T = T.to(self.device)
            S = S.to(self.device)
            loss, _,_,_ = self.model(X, Y, T, S)
            total_loss += loss * X.size(0)
            ss += X.size(0)

        val_loss = total_loss/ss
        return val_loss

    def save_model(
        self,
        save_dir: str,
        save_prefix: str,
    ) -> None:
        """
        Save the model.

        Parameters
        ----------
        save_dir
            The directory where the model is saved.
        save_prefix
            The prefix for model name.
        """

        save_path = os.path.abspath(os.path.join(save_dir, f'{save_prefix}.pth'))
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'var_names': self.adata.var_names,
                'nepoch': self.nepoch,
                'random_state': self.random_state,
                'percent': self.percent,
                'model_kwargs': self.model_kwargs,
                'log': self.log,
            },
            save_path
        )
