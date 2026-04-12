"""Architecture B: Multi-View Convolutional GAN (DCGAN Variant)."""
import torch
import torch.nn as nn
import numpy as np

class DCGANGenerator(nn.Module):
    """
    Generator for the DCGAN variant.
    
    This model takes a random latent vector (noise) and uses strided
    convolutional layers, batchNorm layers, and leakyReLU activations.
    The fractionally-strided convolution layers upsample the features
    into a completed image output. 

    T architecture takes into account spatial relationships, making it
    more capable of generating realistic images with coherent structure.
    
    Each 'block' doubles the spatial dimensions (H and W) while halving the
    number of feature maps, until we reach the target output size.
    """
    def __init__(self, latent_dim=100, img_shape=(6, 3, 64, 64), base_filters=64):
        """
        Args:
            latent_dim (int): Size of the input random noise vector 'z'.
            img_shape (tuple): The expected shape of the output tensor
                               (Segments, Channels, Height, Width).
            base_filters (int): Base number of convolutional filters. Controls
                                model capacity — higher means more parameters.
        """
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        segments, channels, height, width = img_shape
        
        # We generate all 6 segments at once by treating them as channels
        # e.g. 6 segments × 3 channels = 18 output channels
        self.out_channels = segments * channels
        self.segments = segments
        self.channels = channels
        
        bf = base_filters  # Shorthand
        
        def block(in_channels, out_channels, normalize=True):
            """A single upsampling transposed convolution block."""
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 
                                          kernel_size=4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            # Input: (batch, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, bf * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bf * 16),
            nn.ReLU(inplace=True),
            # Shape: (batch, bf*16, 4, 4)

            *block(bf * 16, bf * 8),
            # Shape: (batch, bf*4, 8, 8)
            
            *block(bf * 8, bf * 4),
            # Shape: (batch, bf*4, 8, 8)
            
            *block(bf * 4, bf * 2),
            # Shape: (batch, bf*2, 16, 16)
            
            *block(bf * 2, bf),
            # Shape: (batch, bf, 32, 32)
            
            # Final layer: upsample to target size, no BatchNorm, Tanh output
            nn.ConvTranspose2d(bf, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Shape: (batch, segments*channels, 64, 64)
        )

    def forward(self, z):
        """
        Forward pass of the Generator.
        
        Args:
            z (Tensor): Random noise vector of shape (batch_size, latent_dim).
            
        Returns:
            Tensor: Generated fake images of shape (batch_size, 6, C, H, W).
        """
        # Reshape z from (batch, latent_dim) to (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        
        # Generate flat output: (batch, segments*channels, H, W)
        out = self.model(z)

        # Resize the output to 224x224 (224 isn't a power of 2 so we can't reach it directly)
        target_h, target_w = self.img_shape[2], self.img_shape[3]
        if out.shape[-2] != target_h or out.shape[-1] != target_w:
            out = torch.nn.functional.interpolate(
                out, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Reshape to (batch, segments, channels, H, W)
        batch_size = out.size(0)
        out = out.view(batch_size, self.segments, self.channels, out.size(2), out.size(3))
        return out


class DCGANDiscriminator(nn.Module):
    """
    Discriminator for the DCGAN.
    
    This model uses strided convolutions to progressively downsample the input
    image, extracting features at each layer.
    
    Uses LeakyReLU (not ReLU) as is standard for GAN discriminators, to allow
    small gradients for negative values and prevent dead neurons.
    """
    def __init__(self, img_shape=(6, 3, 64, 64), base_filters=64):
        """
        Args:
            img_shape (tuple): The expected shape of the input tensor
                               (Segments, Channels, Height, Width).
            base_filters (int): Base number of convolutional filters. Should
                                match the Generator's base_filters.
        """
        super(DCGANDiscriminator, self).__init__()
        self.img_shape = img_shape
        segments, channels, height, width = img_shape
        
        # Input channels: all 6 segments treated as channels
        self.in_channels = segments * channels
        
        bf = base_filters  # Shorthand

        def block(in_channels, out_channels, normalize=True):
            """A single downsampling strided convolution block."""
            layers = [nn.Conv2d(in_channels, out_channels,
                                kernel_size=4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Input: (batch, segments*channels, w, h) - supports variable input size, assumes 224
            *block(self.in_channels, bf, normalize=False),  # H to H/2
            *block(bf, bf * 2),                             # H/2 to H/4
            *block(bf * 2, bf * 4),                         # H/4 to H/8
            *block(bf * 4, bf * 8),                         # H/8 to H/16
            *block(bf * 8, bf * 16),                        # H/16 to H/32
            nn.AdaptiveAvgPool2d(4),                        # H/32 to 4x4
            nn.Conv2d(bf * 16, 1, kernel_size=4, stride=1, padding=0, bias=False)  # 4x4 → 1
        )

    def forward(self, x):
        """
        Forward pass of the Discriminator.
        
        Args:
            x (Tensor): Real or fake images of shape (batch_size, 6, C, H, W).
            
        Returns:
            Tensor: A single logit per image predicting real vs. fake.
        """
        batch_size = x.size(0)
        
        # Reshape from (batch, segments, channels, H, W) 
        # to (batch, segments*channels, H, W)
        x = x.view(batch_size, self.in_channels, x.size(3), x.size(4))
        
        validity = self.model(x)
        
        # Flatten from (batch, 1, 1, 1) to (batch, 1)
        return validity.view(batch_size, -1)