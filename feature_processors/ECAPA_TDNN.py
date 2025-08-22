from wespeaker.models.ecapa_tdnn import ECAPA_TDNN_c512 as build_ECAPA_TDNN
from feature_processors.BaseProcessor import BaseProcessor
import torch.nn as nn


class ECAPA_TDNN(nn.Module, BaseProcessor):
    def __init__(self, input_dim=128, output_dim=512, pooling_func='TSTP', two_emb_layer=False):
        super(ECAPA_TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = build_ECAPA_TDNN(
            feat_dim=input_dim,
            embed_dim=output_dim
        )

    def forward(self, x):
        _, output = self.model(x)
        return output