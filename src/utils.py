import torch
import torch.nn.functional as F


def contrastive_loss(z_graph, z_text, temp=0.07):
    z_graph = F.normalize(z_graph, p=2, dim=1)
    z_text = F.normalize(z_text, p=2, dim=1)

    similarity_matrix = torch.matmul(z_graph, z_text.T) / temp

    batch_size = z_graph.size(0)
    target = torch.arange(batch_size, device=z_graph.device)

    loss_graph_to_text = F.cross_entropy(similarity_matrix, target)
    loss_text_to_graph = F.cross_entropy(similarity_matrix.T, target)

    total_loss = (loss_graph_to_text + loss_text_to_graph) / 2

    return total_loss