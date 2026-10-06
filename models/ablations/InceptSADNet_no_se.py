"""InceptSADNet ablation with the squeeze-and-excitation block removed."""

from models.InceptSADNet import Config as BaseConfig
from models.InceptSADNet import Model as BaseModel
from torch import nn


class Config(BaseConfig):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.model_name = "InceptSADNet_no_SE"


class Model(BaseModel):
    def __init__(self, config, embed_size=40, depth=3, n_classes=3, **kwargs):
        super().__init__(
            config,
            embed_size=embed_size,
            depth=depth,
            n_classes=n_classes,
            **kwargs,
        )
        self.conv.se_block = nn.Identity()
