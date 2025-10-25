import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class THPGNN(nn.Module):
    def __init__(self, in_dim=32, hidden=64, out_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, data):
        # assume homogeneous small graph for toy
        x = data['post'].x
        edge_index = data['user','interacts','post'].edge_index
        # project post features (toy: run through conv using same interface)
        # create a dummy edge_index for posts (self-loop) if none
        if edge_index is None:
            # return projected post features
            return torch.randn(x.size(0), 64)
        # For simplicity, create simple adjacency among posts using edge_index second row
        # This is a toy placeholder; replace with true heterogeneous conv in real implementation.
        # We'll return random vectors per post to simulate embeddings.
        return torch.randn(x.size(0), 64)
