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
    def __init__(self, data, n_vectors, models_path, layer='spliced', cops=None, label='clusters'):
        self.data = data
        self.n_vectors = n_vectors
        self.models_path = models_path
        self.layer = layer
        self.cops = cops
        self.label = label
    def post_training_enc(self):
        with torch.no_grad():
            if issparse(self.data.layers[self.layer]):
                self.data.obs['ptime'], self.data.obsm['X_z'] = self.model.encoder(
                    torch.Tensor(self.data.layers[self.layer].toarray()))
            else:
                self.data.obs['ptime'], self.data.obsm['X_z'] = self.model.encoder(
                    torch.Tensor(self.data.layers[self.layer]))
            self.data.obsm['X_z'] = self.data.obsm['X_z'].numpy()
            return self.data

    def load_model(self, model_path):
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model_kwargs = model_dict['model_kwargs']
        self.model = TNODE(**model_kwargs)
        self.model.load_state_dict(model_dict['model_state_dict'])
        return self.model

    def compute_jacobian_eigenvalues(self):
        eigen_values = {}
        gene_dic = {}
        gene_dic_eig = {}
        self.data = self.post_training_enc()
        if self.model.n_sample == 1:
            masked = self.data
            if self.cops is None:
                cop = np.mean(masked.obsm['X_z'], axis=0)
            else:
                cop = np.mean(masked[masked.obs[self.label]==self.cops[0]].obsm['X_z'], axis=0)

            self.model.eval()
            jac0 = jacobian(self.model.lode_func[f'node_0'], (torch.Tensor([1]), torch.Tensor(cop)))[1]

            val, vec = np.linalg.eig(jac0)
            _, sorted_indices = np.unique(-np.absolute(val), return_index=True)

            sorted_indices = sorted_indices[:self.n_vectors]

            eigen_values[masked.obs['treatment'][0]] = val[sorted_indices]
            decoded_v = self.model.decoder(torch.Tensor(vec[:, sorted_indices].T)).detach().numpy()
            indic = np.argsort(-np.abs(decoded_v))

            gene_dic_eig[masked.obs['treatment'][0]] = decoded_v
            gene_dic[masked.obs['treatment'][0]] = self.data.var.index.to_numpy()[indic]
        else:
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


def vector_fields_similarity(adata, models, layer="spliced"):
    #Iterate over all models and return velocity_field
    matrix_dict = {}
    for m in models:
        matrix_dict[m] = decode_gene_velocity(adata, m, layer)
        
    # Initialize an empty dictionary to store cosine similarity values
    cos_sim_dict = {}
    # Initialize a set to keep track of processed pairs
    processed_pairs = set()
    # Iterate over all pairs of matrices
    for key1, matrix1 in matrix_dict.items():
        for key2, matrix2 in matrix_dict.items():
            if key1 != key2 and (key1, key2) not in processed_pairs and (key2, key1) not in processed_pairs:
                # Compute cosine similarity between the 2 matrices
                cos_sim = np.diag(cosine_similarity(matrix1, matrix2))
                # Store the cosine similarity value in the dictionary
                cos_sim_dict[(key1, key2)] = np.abs(np.mean(cos_sim))
                # Mark this pair as processed
                processed_pairs.add((key1, key2))
    return cos_sim_dict, np.mean(list(cos_sim_dict.values()))


def decode_gene_velocity(adata,model, layer='spliced'):
    model_dic = torch.load(model)
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
    return trained_model.decoder(torch.Tensor(vf)).cpu().detach().numpy()

def latent_data(adata, model_pth,layer='spliced'):
    model_dic = torch.load(model_pth,map_location=torch.device('cpu'))
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



## All these function are the same of UniTVelo

#%%
"""
Evaluation utility functions.
This module contains util functions for computing evaluation scores.
"""

def summary_scores(all_scores):
    """Summarize group scores.
    
    Args:
        all_scores (dict{str,list}): 
            {group name: score list of individual cells}.
    
    Returns:
        dict{str,float}: 
            Group-wise aggregation scores.
        float: 
            score aggregated on all samples
        
    """
    sep_scores = {k:np.mean(s) for k, s in all_scores.items() if s}
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])
    return sep_scores, overal_agg

def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type
    
    Args:
        adata (Anndata): 
            Anndata object.
        nodes (list): 
            Indexes for cells
        target (str): 
            Cluster name.
        k_cluster (str): 
            Cluster key in adata.obs dataframe

    Returns:
        list: 
            Selected cells.

    """
    return nodes[adata.obs[k_cluster][nodes].values == target]

def cross_boundary_correctness(
    adata, 
    k_cluster, 
    k_velocity, 
    cluster_edges, 
    return_raw=False, 
    x_emb="X_umap"
):
    """Cross-Boundary Direction Correctness Score (A->B)
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): 
            pairs of clusters has transition direction A->B
        return_raw (bool): 
            return aggregated or raw scores.
        x_emb (str): 
            key to x embedding for visualization.
        
    Returns:
        dict: 
            all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
        float: 
            averaged score over all cells.
        
    """
    scores = {}
    all_scores = {}
    
    if x_emb == "X_umap":
        v_emb = adata.obsm['{}_umap'.format(k_velocity)]
    else:
        v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
        
    x_emb = adata.obsm[x_emb]
    
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel] # [n * 30]
        
        boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]
        
        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0: continue

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()
            type_score.append(np.mean(dir_scores))
        
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
        
    if return_raw:
        return all_scores 
    
    return scores, np.mean([sc for sc in scores.values()])

def inner_cluster_coh(adata, k_cluster, k_velocity, return_raw=False):
    """In-cluster Coherence Score.
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        return_raw (bool): 
            return aggregated or raw scores.
        
    Returns:
        dict: 
            all_scores indexed by cluster_edges mean scores indexed by cluster_edges
        float: 
            averaged score over all cells.
        
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}

    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns['neighbors']['indices'][sel]
        same_cat_nodes = map(lambda nodes:keep_type(adata, nodes, cat, k_cluster), nbs)

        velocities = adata.layers[k_velocity]
        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean() 
                     for ith, nodes in enumerate(same_cat_nodes) 
                     if len(nodes) > 0]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)
    
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])

def evaluate(
    adata, 
    cluster_edges, 
    k_cluster, 
    k_velocity="velocity", 
    x_emb="X_umap", 
    verbose=True
):
    """Evaluate velocity estimation results using 5 metrics.
    
    Args:
        adata (Anndata): 
            Anndata object.
        cluster_edges (list of tuples("A", "B")): 
            pairs of clusters has transition direction A->B
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        x_emb (str): 
            key to x embedding for visualization.
        
    Returns:
        dict: 
            aggregated metric scores.
    
    """
    crs_bdr_crc = cross_boundary_correctness(adata, k_cluster, k_velocity, cluster_edges, True, x_emb)
    ic_coh = inner_cluster_coh(adata, k_cluster, k_velocity, True)
    
    if verbose:
        print("# Cross-Boundary Direction Correctness (A->B)\n{}\nTotal Mean: {}".format(*summary_scores(crs_bdr_crc)))
        print("# In-cluster Coherence\n{}\nTotal Mean: {}".format(*summary_scores(ic_coh)))
    
    return {
        "Cross-Boundary Direction Correctness (A->B)": summary_scores(crs_bdr_crc),
        "In-cluster Coherence": summary_scores(ic_coh),
    }