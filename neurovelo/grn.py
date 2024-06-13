import numpy as np
import torch
from scipy.sparse import issparse
from torch.autograd.functional import jacobian
import networkx as nx
import matplotlib.pyplot as plt
from .model import TNODE
from .utils import latent_data
import pandas as pd
from matplotlib.lines import Line2D

class GraphMaker:
    def __init__(self, adata, model_path, layer='spliced'):
        self.adata = adata
        self.model_path = model_path
        self.layer = layer

    def assign_samples(self, adata, n_sam=1):
        # Assuming this function assigns samples to adata
        adata.obs['sample'] = 0
        return adata

    def graph_maker(self, g, n_frames=30, celltype_to_linearize=None, take_windows=False, ci=0, take_smallest_ptime=False, reverse=False):
        s = 0
        gi = [np.where(self.adata.var.index == gene)[0][0] for gene in g if gene in self.adata.var.index]
        self.adata = self.assign_samples(self.adata, n_sam=1)

        model_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        model_kwargs = model_dict['model_kwargs']
        model = TNODE(**model_kwargs)
        model.load_state_dict(model_dict['model_state_dict'])

        latent_adata = latent_data(self.adata, self.model_path, layer=self.layer)
        if reverse:
            latent_adata.obs['ptime'] = 1 - latent_adata.obs['ptime']
        window_size = int(latent_adata.n_obs / n_frames)
        cell_of_ptime, cop = self._get_cop(latent_adata, celltype_to_linearize, take_windows, ci, window_size, take_smallest_ptime)

        model.eval()
        jac0 = jacobian(model.lode_func[f'node_{s}'], (torch.Tensor([1]), torch.Tensor(cop)))[1]
        combined_weights, _ = self.get_wab(model)
        combined_weights_enc, _ = self.get_wab_encoder(model)
        A = combined_weights @ (jac0.numpy()) @ combined_weights_enc
        cell_dict = latent_adata[cell_of_ptime].obs['clusters'].value_counts().to_dict()
        sorted_cell_value = {k: v for k, v in sorted(cell_dict.items(), key=lambda item: item[1], reverse=True)}
        return A, gi, g, np.mean(latent_adata[cell_of_ptime].obs['ptime']), sorted_cell_value, cell_of_ptime

    def _get_cop(self, latent_adata, celltype_to_linearize, take_windows, ci, window_size, take_smallest_ptime):
        if take_windows and celltype_to_linearize is None:
            cell_of_ptime = latent_adata.obs['ptime'].sort_values().index[ci:ci+window_size]
        elif take_windows and celltype_to_linearize is not None:
            cell_of_ptime = latent_adata[latent_adata.obs['clusters'].isin(celltype_to_linearize)].obs['ptime'].sort_values().index[ci:ci+window_size]
        elif celltype_to_linearize is not None:
            cell_of_ptime = latent_adata[latent_adata.obs['clusters'].isin(celltype_to_linearize)].index
        elif take_smallest_ptime and celltype_to_linearize is not None:
            cell_of_ptime = latent_adata[latent_adata.obs['clusters'].isin(celltype_to_linearize)].obs['ptime'].sort_values().index[:50]
        else:
            cell_of_ptime = latent_adata.obs['ptime'].index

        return cell_of_ptime,np.mean(latent_adata[cell_of_ptime].obsm['X_z'], axis=0)

    def create_graph(self, A, gi, g, threshold_a, threshold_b=None):
        heatmap_data = self.give_small(A, gi).numpy()
        if threshold_b is None:
            significant_connections = np.where(abs(heatmap_data) > threshold_a)
        else:
            significant_connections = np.where((abs(heatmap_data) > threshold_a) & (abs(heatmap_data) < threshold_b))

        G = nx.MultiDiGraph()
        num_nodes = heatmap_data.shape[0]
        for i in range(num_nodes):
            G.add_node(g[i])

        for i, j in zip(significant_connections[0], significant_connections[1]):
            if i != j:
                weight = heatmap_data[i, j]
                G.add_edge(g[i], g[j], weight=weight, direction='forward')

        return G

    def give_small(self, matrix, gi):
        return matrix[gi][:, gi].squeeze()

    def get_wab(self, model):
        weights = []
        for name, param in model.decoder.named_parameters():
            if 'weight' in name or 'bias' in name:
                weights.append(param.data)

        combined_weights = torch.matmul(weights[2], weights[0])
        combined_bias = torch.matmul(weights[2], weights[1]) + weights[3]
        return combined_weights, combined_bias

    def get_wab_encoder(self, model):
        weights = []
        for name, param in model.encoder.named_parameters():
            if 'weight' in name and 'fc3' not in name:
                weights.append(param.data)
            elif 'bias' in name and 'fc3' not in name:
                weights.append(param.data)

        combined_weights = torch.mm(weights[2], weights[0])
        combined_bias = torch.mv(weights[2], weights[1]) + weights[3]
        return combined_weights, combined_bias

    def plot_custom_graph(self, G, node_color='#EAE0D5', positive_color='#A3320B', negative_color='#61988E', f_color='black', edge_width=1,ax_network=None,network_node_file=None):
        if ax_network is None:
            fig, ax_network = plt.subplots(figsize=(12, 8))
        
        pos = nx.spring_layout(G)  # Initial position layout
        if network_node_file is not None:
            network_node = pd.read_csv(network_node_file, delimiter='\t', index_col=0)
            # Filter out nodes not present in the network_node file
            nodes_to_remove = [node for node in pos.keys() if node not in network_node.index]
            G.remove_nodes_from(nodes_to_remove)
            pos = {node: pos[node] for node in pos if node in network_node.index}
            print("The following genes are either not protein coding genes or not found in string-db: ", nodes_to_remove)
            for i in pos.keys():
                pos[i][0] = network_node.loc[i, 'x_position']
                pos[i][1] = 1 - network_node.loc[i, 'y_position']
                
        nx.draw_networkx_nodes(G, pos, ax=ax_network, node_color=node_color, node_size=1500)
        nx.draw_networkx_labels(G, pos, ax=ax_network, font_family='DejaVu Sans', font_size=10, font_color=f_color)

        for u, v, data in G.edges(data=True):
            color = positive_color if data['weight'] > 0 else negative_color
            width = abs(data['weight']) * edge_width
            nx.draw_networkx_edges(G, pos, ax=ax_network, edgelist=[(u, v)], width=width, edge_color=color, arrows=True, arrowsize=10, connectionstyle=f"arc3,rad={0.10}", node_size=1500)
            
        # Create custom legend
        legend_elements = [
            Line2D([0], [0], color=positive_color, lw=4, label='Positive'),
            Line2D([0], [0], color=negative_color, lw=4, label='Negative')
        ]
        ax_network.legend(handles=legend_elements, loc='best')
        
        plt.title('Gene Interaction Network')
        plt.show()
