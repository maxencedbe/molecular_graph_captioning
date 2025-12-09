import torch
from torch_geometric.data import InMemoryDataset


class PubChemDataset(InMemoryDataset):
    def __init__(self, path):
        super(PubChemDataset, self).__init__()
        self.data, self.slices = torch.load(path)
    
    def __getitem__(self, idx):
        return self.get(idx)

if __name__ == '__main__':
    dataset = PubChemDataset('./pretrain.pt')
    print(dataset[0])
    