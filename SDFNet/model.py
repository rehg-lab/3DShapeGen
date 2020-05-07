import torch
import torch.nn as nn
from torchvision import models
import config

class SDFNet(nn.Module):
    ''' SDFNet 3D regressor class

    Args:
        input_point_dim: dimension of input points, default to 3
        latent_dim: dimension of conditioned code, default to 256
        size_hidden: dimension of points block hidden size, default to 256
        pretrained: whether the encoder is ImageNet pretrained, 
            default to False

    '''
    def __init__(self, input_point_dim=3, latent_dim=256, size_hidden=256, pretrained=False):
        super().__init__()

        self.encoder = Encoder(latent_dim, pretrained=pretrained)
        self.decoder = Decoder(input_point_dim, latent_dim, size_hidden)

    def forward(self, points, inputs):
        assert points.size(0) == inputs.size(0)
        batch_size = points.size(0)
        latent_feats = self.encoder(inputs)
        score = self.decoder(points, latent_feats)
        return score

class Encoder(nn.Module):
    def __init__(self, latent_dim, pretrained):
        super().__init__()
        self.features = models.resnet18(\
            pretrained=pretrained)
        # Reinitialize the first conv layer for D+N inputs
        if config.path['input_image_path'] is None:
            self.features.conv1 = nn.Conv2d(4,\
                                            64,\
                                            kernel_size=7,\
                                            stride=2,\
                                            padding=3,\
                                            bias=False)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        feat = self.features(x)
        latent_feat = self.fc(feat)
        return latent_feat 

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, size_hidden):
        super().__init__()
        self.fc_p = nn.Conv1d(input_dim, size_hidden, 1)
        
        self.block0 = CResnetBlockConv(latent_dim, size_hidden)
        self.block1 = CResnetBlockConv(latent_dim, size_hidden)
        self.block2 = CResnetBlockConv(latent_dim, size_hidden)
        self.block3 = CResnetBlockConv(latent_dim, size_hidden)
        self.block4 = CResnetBlockConv(latent_dim, size_hidden)

        self.bn = CBatchNorm(latent_dim, size_hidden)

        self.fc_out = nn.Conv1d(size_hidden, 1, 1)

        self.actvn = nn.ReLU()

    def forward(self, p, c):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out

class CBatchNorm(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.feature_dim = feature_dim
        self.conv_gamma = nn.Conv1d(self.latent_dim, self.feature_dim, 1)
        self.conv_beta = nn.Conv1d(self.latent_dim, self.feature_dim, 1)
        self.bn = nn.BatchNorm1d(self.feature_dim, affine=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        latent = c
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.latent_dim)

        # c is assumed to be of size batch_size x latent_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out

class CResnetBlockConv(nn.Module):
    def __init__(self, latent_dim, size_in, size_hidden=None, size_out=None):
        super().__init__()
        if size_hidden is None:
            size_hidden = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_hidden = size_hidden
        self.size_out = size_out

        self.bn_0 = CBatchNorm(\
            latent_dim, self.size_in)
        self.bn_1 = CBatchNorm(\
            latent_dim, self.size_hidden)

        self.fc_0 = nn.Conv1d(self.size_in, self.size_hidden, 1)
        self.fc_1 = nn.Conv1d(self.size_hidden, self.size_out, 1)
        self.actvn = nn.ReLU()

        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        return x + dx
 
