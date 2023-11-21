import numpy as np
from scipy.sparse import issparse
from .model import TNODE
from torch.autograd.functional import jacobian
import torch
from sklearn.metrics.pairwise import cosine_similarity
import copy
import anndata as ad
import pandas as pd

#Class to perform gene ranking. Output can be used directly in gseapy for pathways
class ModelAnalyzer:
    def __init__(self, data, n_vectors, models_path):
        self.data = data
        self.n_vectors = n_vectors
        self.models_path = models_path

    def post_training_enc(self, layer='spliced'):
        with torch.no_grad():
            if issparse(self.data.layers[layer]):
                self.data.obs['ptime'], self.data.obsm['X_z'] = self.model.encoder(
                    torch.Tensor(self.data.layers[layer].toarray()))
            else:
                self.data.obs['ptime'], self.data.obsm['X_z'] = self.model.encoder(
                    torch.Tensor(self.data.layers[layer]))
            self.data.obsm['X_z'] = self.data.obsm['X_z'].numpy()
            return self.data

    def load_model(self, model_path):
        model_dict = torch.load(model_path)
        model_kwargs = model_dict['model_kwargs']
        self.model = TNODE(**model_kwargs)
        self.model.load_state_dict(model_dict['model_state_dict'])
        return self.model

    def compute_jacobian_eigenvalues(self):
        eigen_values = {}
        gene_dic = {}
        gene_dic_eig = {}
        self.data = self.post_training_enc()
        for j in range(self.model.n_sample):
            mask = self.data.obs['sample'] == j
            masked = self.data[mask]
            cop = np.mean(masked.obsm['X_z'], axis=0)

            self.model.eval()
            jac0 = jacobian(self.model.lode_func[f'node_{j}'], (torch.Tensor([1]), torch.Tensor(cop)))[1]

            val, vec = np.linalg.eig(jac0)
            _, sorted_indices = np.unique(-np.absolute(val), return_index=True)

            sorted_indices = sorted_indices[:self.n_vectors]

            eigen_values[masked.obs['treatment'][0]] = val[sorted_indices]
            decoded_v = self.model.decoder(torch.Tensor(vec[:, sorted_indices].T)).detach().numpy()
            indic = np.argsort(-np.abs(decoded_v))

            gene_dic_eig[masked.obs['treatment'][0]] = decoded_v
            gene_dic[masked.obs['treatment'][0]] = self.data.var.index.to_numpy()[indic]

        return eigen_values, gene_dic, gene_dic_eig

    def models_output(self):
        self.results = {}

        for o, model_path in enumerate(self.models_path):
            model_key = f'model {o}'
            self.model = self.load_model(model_path)

            eigen_values, genes, eigen_vectors = self.compute_jacobian_eigenvalues()

            self.results[model_key] = {
                'eigen_values': eigen_values,
                'genes': genes,
                'eigen_vectors': eigen_vectors
            }

        return self.results

    def calculate_eigenvector_similarity(self, treatment):
        sim = None
        for n, model_a in enumerate(self.results.values()):
            a = model_a['eigen_vectors'][treatment]
            for m, model_b in enumerate(self.results.values()):
                if n != m:
                    b = model_b['eigen_vectors'][treatment]
                    cos_matrix = cosine_similarity(a, b)

                    eigen_sim = np.array(
                        [np.ones(self.n_vectors, dtype=int) * n, np.ones(self.n_vectors, dtype=int) * m,
                         np.arange(self.n_vectors, dtype=int), np.argmax(np.abs(cosine_similarity(a, b)), axis=1)]).T
                    if sim is None:
                        sim = eigen_sim
                    else:
                        sim = np.append(sim, eigen_sim, axis=0)
        self.similarity_matrix = sim
        return sim


    def gene_order(self, gene_df):
        self.new_df = gene_df.copy()
        for column in self.new_df.columns:
            self.new_df.loc[:, column] = self.new_df.index[self.new_df.loc[:, column].argsort()]
        self.new_df.index = np.arange(self.new_df.shape[0])+1
        return self.new_df

    def gene_ranking(self):
        all_gene_rank, all_gene_mean = {}, {} #Dictionary to save all information about genes ranking accross treatment and eigenvector
        gene_eigen_treatment_mean = {} #The output dictionary. It contains DataFrame of each treatment with genes average rank for each eigenvector.
        gene_eigen_treatment_order = {}
        for t in self.data.obs['treatment'].unique():
            sim = self.calculate_eigenvector_similarity(t)
            treatment = t
            model = 0
            eigens = np.arange(self.n_vectors)
            all_gene_rank[treatment] = {}
            all_gene_mean[treatment] = {}
            for eigen in eigens:
                mask = (self.similarity_matrix[:, 0].astype(int) == model) & (
                            self.similarity_matrix[:, 2].astype(int) == eigen)
                new_ev = self.similarity_matrix[mask]

                l = self.results[f'model {model}']['genes'][treatment][eigen]
                for i in range(new_ev.shape[0]):
                    l = np.vstack((l, self.results[f'model {new_ev[i, 1]}']['genes'][treatment][new_ev[i, 3]]))

                genes_rank = {}
                mean_gene = {}
                for k in self.data.var.index:
                    genes_rank[k] = np.where(l == k)[1] + 1
                    mean_gene[k] = np.mean(genes_rank[k])

                all_gene_rank[treatment][eigen] = genes_rank
                all_gene_mean[treatment][eigen] = mean_gene
            
            gene_eigen_treatment_mean[treatment] = pd.DataFrame(all_gene_mean[treatment])
            gene_eigen_treatment_order[treatment] = self.gene_order(gene_eigen_treatment_mean[treatment])
            
        self.all_gene_rank = all_gene_rank
        return gene_eigen_treatment_order,gene_eigen_treatment_mean



def latent_data(adata, model_pth,layer='spliced'):
    model_dic = torch.load(model_pth)
    kwargs = model_dic['model_kwargs']
    trained_model = TNODE(**kwargs)
    trained_model.load_state_dict(model_dic['model_state_dict'])
    trained_model.eval()

    if issparse(adata.layers[layer]):
        encoded = trained_model.encoder(torch.Tensor(adata.layers[layer].toarray()))
    else:
        encoded = trained_model.encoder(torch.Tensor(adata.layers[layer]))

    ptime , z_s = encoded[0].detach().numpy(), encoded[1].detach().numpy()
    vf = np.empty(encoded[1].shape)
    for j in adata.obs['sample'].unique():
        mask = adata.obs['sample'] == j
        vf[mask] = trained_model.lode_func[f'node_{j}'](encoded[0][mask].values, encoded[1][mask]).detach().numpy()
    
    adata.obs['ptime'] = ptime
    latent_adata = ad.AnnData(z_s, 
                          obsm={'X_z': z_s},
                         layers = {'spliced': z_s,
                                   'spliced_velocity': vf},
                          obs=adata.obs)
    del adata.obs['ptime']
    return latent_adata
