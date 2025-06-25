import wespeaker
from feature_processors.BaseProcessor import BaseProcessor
import torch.nn as nn


class ResNet293(nn.Module, BaseProcessor):
    def __init__(self, input_dim=1024, output_dim=1024, pooling_func='TSTP', two_emb_layer=False):
        super(ResNet293, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = wespeaker.models.ResNet293(
            feat_dim=input_dim,
            embed_dim=output_dim,
            pooling_func=pooling_func,
            two_emb_layer=two_emb_layer
        )

    def forward(self, x):
        return self.model(x)