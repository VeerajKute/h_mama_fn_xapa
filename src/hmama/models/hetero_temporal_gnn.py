import torch
import torch.nn as nn
import math
from torch_geometric.nn import HGTConv, to_hetero
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import HeteroData

class Time2Vec(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.w = nn.Parameter(torch.randn(out_dim))
        self.b = nn.Parameter(torch.randn(out_dim))

    def forward(self, t):
        # t: (E,1) or (N,1)
        # simple sinusoidal mapping
        # output shape: (..., out_dim)
        x = t.unsqueeze(-1)  # (...,1)
        return torch.sin(x * self.w + self.b)

class HeteroTemporalGNN(nn.Module):
    def __init__(self, metadata, in_dims: dict, hidden_dim=128, out_dim=64, n_layers=2, heads=2):
        super().__init__()
        # metadata: metadata() from HeteroData
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        # Linear projections per node type
        self.input_proj = nn.ModuleDict()
        for ntype, dim in in_dims.items():
            self.input_proj[ntype] = nn.Linear(dim, hidden_dim)
        # time2vec for edge time encoding
        self.t2v = Time2Vec(out_dim=16)
        # stack of HGTConv layers
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            conv = HGTConv(in_channels=hidden_dim, out_channels=hidden_dim, metadata=metadata, heads=heads)
            self.convs.append(conv)
        self.post_pool = global_mean_pool
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: HeteroData):
        # data.x_dict expected: node type -> x
        x_dict = {}
        for ntype, x in data.x_dict.items():
            x_dict[ntype] = torch.relu(self.input_proj[ntype](x))
        # optionally inject edge time attr as relation-specific features (PyG HGTConv doesn't accept edge_attr directly)
        # We'll rely on node features and relation types; for more advanced temporal GNNs, replace HGTConv with TGAT/HGAT
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: torch.relu(v) for k,v in x_dict.items()}
        # pool post nodes to obtain per-post embeddings
        if 'post' in x_dict:
            post_emb = self.out_proj(x_dict['post'])
        else:
            # fallback: aggregate any node
            any_node = next(iter(x_dict.values()))
            post_emb = self.out_proj(any_node)
        return post_emb
