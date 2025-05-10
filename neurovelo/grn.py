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
import scvelo as scv
from matplotlib.animation import FuncAnimation, PillowWriter

class GraphMaker:
    def __init__(self, adata, model_path, layer='spliced'):
        self.adata = adata
        self.model_path = model_path
        self.layer = layer

    def assign_samples(self, adata, n_sam=1):
        # Assuming this function assigns samples to adata
        adata.obs['sample'] = 0
        return adata

    def graph_maker(self, g, n_frames=30, celltype_to_linearize=None, take_windows=False, ci=0, reverse=False):
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
        cell_of_ptime, cop = self._get_cop(latent_adata, celltype_to_linearize, take_windows, ci, window_size)

        model.eval()
        jac0 = jacobian(model.lode_func[f'node_{s}'], (torch.Tensor([1]), torch.Tensor(cop)))[1]
        combined_weights, _ = self.get_wab(model)
        combined_weights_enc, _ = self.get_wab_encoder(model)
        A = combined_weights @ (jac0.numpy()) @ combined_weights_enc
        A = A.T #so the graph is inherently from i to j -opposite to what written in the paper
        cell_dict = latent_adata[cell_of_ptime].obs['clusters'].value_counts().to_dict()
        sorted_cell_value = {k: v for k, v in sorted(cell_dict.items(), key=lambda item: item[1], reverse=True)}
        return A, gi, g, np.mean(latent_adata[cell_of_ptime].obs['ptime']), sorted_cell_value, cell_of_ptime

    def _get_cop(self, latent_adata, celltype_to_linearize, take_windows, ci, window_size):
        if take_windows and celltype_to_linearize is None:
            cell_of_ptime = latent_adata.obs['ptime'].sort_values().index[ci:ci+window_size]
        elif take_windows and celltype_to_linearize is not None:
            cell_of_ptime = latent_adata[latent_adata.obs['clusters'].isin(celltype_to_linearize)].obs['ptime'].sort_values().index[ci:ci+window_size]
        elif celltype_to_linearize is not None:
            cell_of_ptime = latent_adata[latent_adata.obs['clusters'].isin(celltype_to_linearize)].index
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



class GraphMakerAnimation:
    def __init__(self, adata, model_path,cell_label, layer='spliced', reverse=False, show_scatter=False):
        self.adata = adata
        self.model_path = model_path
        self.layer = layer
        self.adata_subset = None 
        self.fig = None
        self.ax_network = None
        self.ax_scatter = None
        self.animation = None
        self.celltype_to_linearize = None
        self.reverse = reverse
        self.fix_pos = None
        self.cell_label = cell_label
        self.show_scatter = show_scatter
    def assign_samples(self, adata, n_sam=1):
        adata.obs['sample'] = 0
        return adata

    def graph_maker(self, g, celltype_to_linearize=None, take_windows=False, ci=0, reverse=False):
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

        if self.celltype_to_linearize is not None:
            latent_adata = latent_adata[latent_adata.obs[self.cell_label].isin(self.celltype_to_linearize)]

        self.sorted_cell_index = latent_adata.obs['ptime'].sort_values().index.tolist()

        self.window_size = int(len(self.sorted_cell_index) / self.n_frames)
        cell_of_ptime, cop = self._get_cop(latent_adata, celltype_to_linearize, take_windows, ci, self.window_size)

        model.eval()
        jac0 = jacobian(model.lode_func[f'node_{s}'], (torch.Tensor([1]), torch.Tensor(cop)))[1]
        combined_weights, _ = self.get_wab(model)
        combined_weights_enc, _ = self.get_wab_encoder(model)
        A = combined_weights @ (jac0.numpy()) @ combined_weights_enc
        A = A.T #so the graph is inherently from i to j -opposite to what written in the paper
        cell_dict = latent_adata[cell_of_ptime].obs[self.cell_label].value_counts().to_dict()
        sorted_cell_value = {k: v for k, v in sorted(cell_dict.items(), key=lambda item: item[1], reverse=True)}
        return A, gi, g, np.mean(latent_adata[cell_of_ptime].obs['ptime']), sorted_cell_value, cell_of_ptime

    def _get_cop(self, latent_adata, celltype_to_linearize, take_windows, ci, window_size):
        if take_windows and celltype_to_linearize is None:
            cell_of_ptime = latent_adata.obs['ptime'].sort_values().index[ci:ci+window_size]
        elif take_windows and celltype_to_linearize is not None:
            cell_of_ptime = latent_adata[latent_adata.obs[self.cell_label].isin(celltype_to_linearize)].obs['ptime'].sort_values().index[ci:ci+window_size]
        elif celltype_to_linearize is not None:
            cell_of_ptime = latent_adata[latent_adata.obs[self.cell_label].isin(celltype_to_linearize)].index
        else:
            cell_of_ptime = latent_adata.obs['ptime'].index
        return cell_of_ptime, np.mean(latent_adata[cell_of_ptime].obsm['X_z'], axis=0)

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

    def plot_custom_graph(self, G, node_color='#EAE0D5', positive_color='#A3320B', negative_color='#61988E', 
                         f_color='black', edge_width=1, ax_network=None, network_node_file=None):
        if ax_network is None:
            fig, ax_network = plt.subplots(figsize=(12, 8))
        
        pos = nx.spring_layout(G, seed=42)
        if network_node_file is not None:
            network_node = pd.read_csv(network_node_file, delimiter='\t', index_col=0)
            nodes_to_remove = [node for node in pos.keys() if node not in network_node.index]
            G.remove_nodes_from(nodes_to_remove)
            pos = {node: pos[node] for node in pos if node in network_node.index}
            print("Removed genes:", nodes_to_remove)
            for i in pos.keys():
                pos[i][0] = network_node.loc[i, 'x_position']
                pos[i][1] = 1 - network_node.loc[i, 'y_position']
                
        nx.draw_networkx_nodes(G, pos, ax=ax_network, node_color=node_color, node_size=1500)
        nx.draw_networkx_labels(G, pos, ax=ax_network, font_family='DejaVu Sans', font_size=10, font_color=f_color)

        for u, v, data in G.edges(data=True):
            color = positive_color if data['weight'] > 0 else negative_color
            width = abs(data['weight']) * edge_width
            nx.draw_networkx_edges(G, pos, ax=ax_network, edgelist=[(u, v)], width=width, edge_color=color, 
                                   arrows=True, arrowsize=10, connectionstyle=f"arc3,rad={0.10}", node_size=1500)
            
        legend_elements = [
            Line2D([0], [0], color=positive_color, lw=4, label='Positive'),
            Line2D([0], [0], color=negative_color, lw=4, label='Negative')
        ]
        ax_network.legend(handles=legend_elements, loc='best')
        plt.title('Gene Interaction Network')
        plt.show()

    def create_animation(self, g, path_to_save ,adata_subset=None, n_frames=20, threshold_a=0.01, threshold_b=None, 
                        network_node_file=None, interval=100, celltype_to_linearize = None):
        if adata_subset is None:
            self.adata_subset = self.adata.copy()
        else:
            self.adata_subset = adata_subset.copy()
        self.celltype_to_linearize = celltype_to_linearize
        self.g = g
        self.n_frames = n_frames
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.network_node_file = network_node_file


        self.graph_maker(g, self.celltype_to_linearize, take_windows=True)

        self.fig, (self.ax_network, self.ax_scatter) = plt.subplots(1, 2, figsize=(25, 10))

        frames = range(0, len(self.sorted_cell_index), self.window_size)
        
        valid_frames = (frame for frame in frames if self._is_valid_frame(frame))
        self.animation = FuncAnimation(self.fig, self._update, frames=valid_frames, 
                                      interval=interval, blit=False)
        self.animation.save(path_to_save, writer=PillowWriter(fps=2))
        plt.show()

    def _is_valid_frame(self, start_idx):
        window = self.sorted_cell_index[start_idx : start_idx + self.window_size]
        return len(window) > 0
    
    def _is_all_na(self, cell_n):
        self.adata_subset.obs['fixed_clusters'] = self.adata_subset.obs[self.cell_label]
        self.adata_subset.obs['changed_clusters'] = pd.Categorical(
            [pd.NA] * len(self.adata_subset),
            categories=self.adata_subset.obs['fixed_clusters'].cat.categories
        )

        cell_index = self.adata_subset.obs_names[cell_n:cell_n + self.n_frames]
        for cell in cell_index:
            if cell in self.adata_subset.obs_names:
                idx = self.adata_subset.obs_names.get_loc(cell)
                cluster = self.adata_subset.obs['fixed_clusters'][idx]
                self.adata_subset.obs.at[cell, 'changed_clusters'] = cluster

        return self.adata_subset.obs['changed_clusters'].isna().all()

    
    def _update(self, cell_n):
        self.ax_network.clear()
        self.ax_network.axis('off')
        self.ax_scatter.clear()
        # Get network data
        A, gi, _, aptime, _, cell_index = self.graph_maker(
            self.g, take_windows=True, ci=cell_n, celltype_to_linearize=self.celltype_to_linearize, reverse=self.reverse
        )
        print(cell_index.shape)
        G = self.create_graph(A, gi, self.g, self.threshold_a, self.threshold_b)
        # Draw network
        if self.network_node_file:
            network_node = pd.read_csv(self.network_node_file, sep='\t', index_col=0)
            nodes_to_remove = [n for n in G.nodes if n not in network_node.index]
            G.remove_nodes_from(nodes_to_remove)
            self.fix_pos = {n: [network_node.loc[n, 'x_position'], 1 - network_node.loc[n, 'y_position']] 
                   for n in G.nodes}
        elif self.fix_pos is None:
            self.fix_pos = nx.spring_layout(G)
        
        pos = self.fix_pos

        nx.draw_networkx_nodes(G, pos, ax=self.ax_network, node_color='#EAE0D5', node_size=1500)
        nx.draw_networkx_labels(G, pos, ax=self.ax_network, font_size=10, font_color='#0A0908')

        for u, v, data in G.edges(data=True):
            color = '#A3320B' if data['weight'] > 0 else '#61988E'
            width = abs(data['weight']) * 5
            nx.draw_networkx_edges(G, pos, ax=self.ax_network, edgelist=[(u, v)], width=width, edge_color=color, 
                                   arrows=True, arrowsize=10, connectionstyle=f"arc3,rad={0.10}", node_size=1500)

        self.ax_network.set_title(f"Time: {aptime:.4f}")

        if self.show_scatter:
            # Update scatter plot
            self.adata_subset.obs['changed_clusters'] = pd.Categorical(
                [pd.NA] * len(self.adata_subset),
                categories=self.adata_subset.obs[self.cell_label].cat.categories
            )
            for cell in cell_index:
                if cell in self.adata_subset.obs_names:
                    self.adata_subset.obs.at[cell, 'changed_clusters'] = self.adata_subset.obs.at[cell, self.cell_label]
            
            has_umap, has_tsne = 'X_umap' in self.adata.obsm, 'X_tsne' in self.adata.obsm

            if has_umap:
                scv.pl.umap(self.adata_subset, color='changed_clusters', ax=self.ax_scatter, 
                            title='Network Cells', show=False, legend_loc='right margin')
            if has_tsne:
                scv.pl.tsne(self.adata_subset, ax=self.ax_scatter, 
                            title='Network Cells', show=False, legend_loc='right margin')
            else:
                print("No UMAP or TSNE embedding. Consider computing it first")

        return self.ax_network, self.ax_scatter
