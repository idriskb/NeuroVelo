# NeuroVelo: interpretable learning of cellular dynamics
NeuroVelo: physics-based interpretable learning of cellular dynamics. It is implemented on Python3 and PyTorch, the model estimate velocity field and genes that drives the splicing dynamics.

![Model](https://github.com/idriskb/NeuroVelo/blob/main/figures/model_final.png?raw=true)

The main contributions of NeuroVelo are,

- Using linear projection and embedding for spliced and unspliced RNA to keep interpretability.
- Introducing sample specific velocity estimation, and sample specific interpretation of cellular dynamics.
- Presenting a loss function based on splicing dynamics.

## Installation

```python3
pip install neurovelo
```

or

```python3
pip install git+https://github.com/idriskb/NeuroVelo
```

To avoid potential conflict, it is advised to create a seperate virtual envrionment to run the method
## Getting started

1. Import package

```python3
from neurovelo.train import Trainer
from neurovelo.utils import ModelAnalyzer, latent_data
```

1. Training

```python3
model = Trainer(adata, sample_obs='sample')
mode.train()
model.save_model('/to/folder/','trained.pth')
```

1. Visualization

```python3
latent_adata = latent_data(adata, '/to/folder/trained.pth')
scv.pp.neighbors(latent_adata, use_rep='X_z', n_neighbors=20)
sc.tl.umap(latent_adata,  min_dist=0.1)
scv.tl.velocity_graph(latent_adata, vkey='spliced_velocity', xkey='spliced')
scv.tl.velocity_embedding(latent_adata, vkey='spliced_velocity', basis='umap')
scv.pl.velocity_embedding_stream(latent_adata,basis='umap',vkey='spliced_velocity', color='sample')
```

1. Analysis

```python3
analyzer = ModelAnalyzer(adata, n_vectors=10, '/path/to/trained_models/')
results = analyzer.models_output()
gene_ranking_order, gene_ranking_mean = analyzer.gene_ranking() #gene_ranking_mean can be directly used with prerank gene set enrichment analysis
```
   
