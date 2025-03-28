import torch
import torch.nn as nn 

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, neurons_per_layers, dropout_rate=0.2):
        super(MultiLayerPerceptron, self).__init__()

        layers = []

        # hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, neurons_per_layers))
            layers.append(nn.BatchNorm1d(neurons_per_layers))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = neurons_per_layers

        self.network = nn.Sequential(*layers) # create the network

        self.multi_output = nn.Linear(neurons_per_layers, output_dim)   # multiclass output
        self.binary_output = nn.Sequential(   # binary output
            nn.Linear(neurons_per_layers, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.network(x)
        multi_class_out = self.multi_output(features)
        binary_out = self.binary_output(features)
        return multi_class_out, binary_out
