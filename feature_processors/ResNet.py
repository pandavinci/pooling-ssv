from wespeaker.models.resnet import ResNet293 as build_ResNet293
from wespeaker.models.resnet import ResNet221 as build_ResNet221
from wespeaker.models.resnet import ResNet152 as build_ResNet152
from wespeaker.models.resnet import ResNet101 as build_ResNet101
from wespeaker.models.resnet import ResNet50 as build_ResNet50
from wespeaker.models.resnet import ResNet34 as build_ResNet34
from wespeaker.models.resnet import ResNet18 as build_ResNet18
from feature_processors.BaseProcessor import BaseProcessor
import torch.nn as nn


class ResNet293(nn.Module, BaseProcessor):
    def __init__(self, input_dim=128, output_dim=128, pooling_func='TSTP', two_emb_layer=False):
        super(ResNet293, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = build_ResNet293(
            feat_dim=input_dim,
            embed_dim=output_dim,
            pooling_func=pooling_func,
            two_emb_layer=two_emb_layer
        )

    def forward(self, x):
        return self.model(x)
    
class ResNet221(nn.Module, BaseProcessor):
    def __init__(self, input_dim=128, output_dim=128, pooling_func='TSTP', two_emb_layer=False):
        super(ResNet221, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = build_ResNet221(
            feat_dim=input_dim,
            embed_dim=output_dim,
            pooling_func=pooling_func,
            two_emb_layer=two_emb_layer
        )

    def forward(self, x):
        return self.model(x)
    
class ResNet152(nn.Module, BaseProcessor):
    def __init__(self, input_dim=128, output_dim=128, pooling_func='TSTP', two_emb_layer=False):
        super(ResNet152, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = build_ResNet152(
            feat_dim=input_dim,
            embed_dim=output_dim,
            pooling_func=pooling_func,
            two_emb_layer=two_emb_layer
        )

    def forward(self, x):
        return self.model(x)

class ResNet101(nn.Module, BaseProcessor):
    def __init__(self, input_dim=128, output_dim=128, pooling_func='TSTP', two_emb_layer=False):
        super(ResNet101, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = build_ResNet101(
            feat_dim=input_dim,
            embed_dim=output_dim,
            pooling_func=pooling_func,
            two_emb_layer=two_emb_layer
        )

    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module, BaseProcessor):
    def __init__(self, input_dim=128, output_dim=128, pooling_func='TSTP', two_emb_layer=False):
        super(ResNet50, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = build_ResNet50(
            feat_dim=input_dim,
            embed_dim=output_dim,
            pooling_func=pooling_func,
            two_emb_layer=two_emb_layer
        )

    def forward(self, x):
        return self.model(x)

class ResNet34(nn.Module, BaseProcessor):
    def __init__(self, input_dim=128, output_dim=128, pooling_func='TSTP', two_emb_layer=False):
        super(ResNet34, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = build_ResNet34(
            feat_dim=input_dim,
            embed_dim=output_dim,
            pooling_func=pooling_func,
            two_emb_layer=two_emb_layer
        )

    def forward(self, x):
        return self.model(x)

class ResNet18(nn.Module, BaseProcessor):
    def __init__(self, input_dim=128, output_dim=128, pooling_func='TSTP', two_emb_layer=False):
        super(ResNet18, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = build_ResNet18(
            feat_dim=input_dim,
            embed_dim=output_dim,
            pooling_func=pooling_func,
            two_emb_layer=two_emb_layer
        )

    def forward(self, x):
        return self.model(x)
    