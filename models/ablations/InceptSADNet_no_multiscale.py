"""InceptSADNet ablation using one temporal convolution branch."""

from models.InceptSADNet import Config as BaseConfig
from models.InceptSADNet import ConvEmbedding, Model as BaseModel
from torch import nn


class Config(BaseConfig):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.model_name = "InceptSADNet_no_MultiScale"


class ConvEmbeddingSingleScale(ConvEmbedding):
    def __init__(self, embed_size):
        super().__init__(embed_size)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.F1 * 3, (1, 31), padding=(0, 15), bias=False),
                    nn.BatchNorm2d(self.F1 * 3),
                    nn.ELU(),
                )
            ]
        )


class Model(BaseModel):
    def __init__(self, config, embed_size=40, depth=3, n_classes=3, **kwargs):
        super().__init__(
            config,
            embed_size=embed_size,
            depth=depth,
            n_classes=n_classes,
            **kwargs,
        )
        self.conv = ConvEmbeddingSingleScale(embed_size)
