import torch.nn as nn
from layers import StructuralAttentionLayer, TemporalAttentionLayer


class DySAT(nn.Module):
    """build_net port. One-hot input == a row lookup, so nn.Embedding."""
    def __init__(self, n_nodes, T, 
                 num_features=128, 
                 structural_layer_config=(128,), structural_head_config=(16,),
                 temporal_layer_config=128, temporal_head_config=16,
                 spatial_drop=0.1, temporal_drop=0.5):
        super().__init__()
        self.node_table = nn.Embedding(n_nodes, num_features)
        nn.init.xavier_uniform_(self.node_table.weight)

        layers, ind = [], num_features
        for out_dim, heads in zip(structural_layer_config, structural_head_config):
            layers.append(StructuralAttentionLayer(ind, out_dim, heads,
                                                   spatial_drop, spatial_drop))
            ind = out_dim
        self.struct = nn.ModuleList(layers)
        self.temporal = TemporalAttentionLayer(ind, temporal_head_config, T, temporal_drop)

    def structural_one(self, src, dst, n):
        """ src, dst : edge list (source & destination) | n = number of nodes """
        h = self.node_table.weight
        for layer in self.struct:
            h = layer(h, src, dst, n)
        return h
