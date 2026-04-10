"""Architecture A: Multi-Layer Perceptron (MLP) GAN (Baseline)."""

import torch
import torch.nn as nn
import numpy as np

class MLPGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(6, 3, 256, 256)):
        super(MLPGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.output_dim = int(np.prod(img_shape))

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.output_dim),
            nn.Tanh() # Outputs values in [-1, 1]
        )

    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        img_flat = self.model(z)
        # Reshape to (batch_size, 6, C, H, W)
        img = img_flat.view(img_flat.size(0), *self.img_shape)
        return img


class MLPDiscriminator(nn.Module):
    def __init__(self, img_shape=(6, 3, 256, 256)):
        super(MLPDiscriminator, self).__init__()
        self.img_shape = img_shape
        self.input_dim = int(np.prod(img_shape))

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
            # No Sigmoid here; we will use BCEWithLogitsLoss during training for better stability
        )

    def forward(self, img):
        # img shape: (batch_size, 6, C, H, W)
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
