import numpy as np
import torch
import torch.nn as nn

import dgl
from dgl.nn.pytorch.conv import GATv2Conv

from gnns.graphs.GNN import GNN

class GATv2ConvEncoder(GNN):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim)

        from utils.env import edge_types

        hidden_dim = kwargs.get('hidden_dim', 32)
        n_heads = kwargs.get('n_heads', 2)

        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.conv = GATv2Conv(hidden_dim, hidden_dim, n_heads)
        self.g_embed = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, g):
        g = np.array(g).reshape((1, -1)).tolist()[0]
        g = dgl.batch(g)
        h = self.linear_in(g.ndata["feat"].float().squeeze(dim=1))
        h = self.conv(g, h)
        n, m, k = h.shape
        h = h.view(n, m*k)
        g.ndata['h'] = h
        g.ndata["is_root"] = g.ndata["is_root"].float()
        hg = dgl.sum_nodes(g, 'h', weight='is_root')
        return self.g_embed(hg)
