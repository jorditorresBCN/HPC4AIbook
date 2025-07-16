from typing import List

import torch.nn as nn


class MyCustomModel(nn.Module):
    def __init__(
        self,
        n_classes: int = 200,
        resolution: int = 64,
        intermidiate_dimensions: List[int] = [8192, 4096, 2048, 1024, 512, 256, 128, 64],
    ):
        super().__init__()
        self.n_classes = n_classes
        self.resolution = resolution
        self.intermidiate_dimensions = intermidiate_dimensions

        # Build model
        layer_dims = [resolution * resolution] + intermidiate_dimensions + [n_classes]
        layers = []
        for input_dim, output_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_features=input_dim, out_features=output_dim))
            if output_dim != n_classes:
                layers.append(nn.ReLU())

        self.model = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
