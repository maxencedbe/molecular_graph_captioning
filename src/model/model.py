import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool

node_feat_dim = 177
edge_feat_dim = 30
hidden_dim = 128
projection_dim = 256

class GEncoder(nn.Module):
    def __init__(self, num_layers=3, in_dim=node_feat_dim, hidden_dim=hidden_dim):
        super(GEncoder, self).__init__()

        self.layers = nn.ModuleList()
        current_dim = in_dim

        self.layers.append(MessagePassing(current_dim, hidden_dim))
        current_dim = hidden_dim

        for _ in range(num_layers - 1):
            self.layers.append(MessagePassing(current_dim, hidden_dim))
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)

        h_graph = global_add_pool(x, batch)

        z_graph = self.projection_head(h_graph)

        return z_graph


class MessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MessagePassing, self).__init__(aggr='add', flow='source_to_target')

        mlp_in_dim = in_channels + edge_feat_dim

        self.mlp_message = nn.Sequential(
            nn.Linear(mlp_in_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.mlp_update = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        message_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.mlp_message(message_input)
    
    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_update(update_input)