import torch
from torch.utils.data import Dataset

class MultiLabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y_multi = torch.LongTensor(y['Attack Type'].values)
        self.y_binary = torch.FloatTensor(y['status'].values).unsqueeze(1)

    def __len__(self):
        return self.X.shape[0] # return the sample size

    def __getitem__(self, idx):
        return self.X[idx], self.y_multi[idx], self.y_binary[idx]