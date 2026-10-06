"""InceptSADNet ablation without electrode-spatial convolution."""

import torch
from models.InceptSADNet import Config as BaseConfig
from models.InceptSADNet import ConvEmbedding, Model as BaseModel
from torch import nn


class Config(BaseConfig):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.model_name = "InceptSADNet_no_Spatial"


class ConvEmbeddingWithoutSpatialConv(ConvEmbedding):
    def __init__(self, embed_size):
        super().__init__(embed_size)
        total_filters = self.F1 * len(self.kernel_lengths)
        spatial_features = total_filters * self.D
        self.spatial_conv = nn.Identity()
        self.spatial_channel_projection = nn.Conv2d(
            total_filters,
            spatial_features,
            kernel_size=(1, 1),
            groups=total_filters,
            bias=False,
        )
        self.spatial_norm = nn.BatchNorm2d(spatial_features)
        self.spatial_pool = nn.AvgPool2d((1, 4))
        self.spatial_dropout = nn.Dropout(0.5)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        branches_out = [branch(x) for branch in self.branches]
        x = self.spatial_channel_projection(torch.cat(branches_out, dim=1))
        x = self.spatial_norm(x)
        x = nn.functional.elu(x)
        x = x.mean(dim=2, keepdim=True)
        x = self.spatial_pool(x)
        x = self.spatial_dropout(x)
        x = self.se_block(x)
        return self.projection(x)


class Model(BaseModel):
    def __init__(self, config, embed_size=40, depth=3, n_classes=3, **kwargs):
        super().__init__(
            config,
            embed_size=embed_size,
            depth=depth,
            n_classes=n_classes,
            **kwargs,
        )
        self.conv = ConvEmbeddingWithoutSpatialConv(embed_size)
