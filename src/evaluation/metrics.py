"""Evaluation metrics: Adversarial Success Rate (Fooling Rate), FID."""
import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights


def calculate_fooling_rate(fake_notes, target_model, batch_size=64):
    """
    Calculates the rate at which fake notes fool the target model.

    Args:
        fake_notes (Tensor): (N, 6, C, H, W) tensor of generated notes on CPU.
        target_model (nn.Module): Classifier trained to distinguish real/fake notes.
        batch_size (int): Number of notes to evaluate per forward pass.

    Returns:
        float: Fraction of generated notes that fooled the target model.
    """
    device = next(target_model.parameters()).device
    target_model.eval()

    fooled = 0
    with torch.no_grad():
        for start in range(0, len(fake_notes), batch_size):
            batch = fake_notes[start : start + batch_size].to(device)  # (B, 6, C, H, W)
            B, S, C, H, W = batch.shape

            if (H, W) != (224, 224):
                # This is for the MLP model - since it outputs 64x64 images (224x224 results in too many parameters)
                batch = batch.view(B * S, C, H, W)
                batch = torch.nn.functional.interpolate(batch, size=(224, 224), mode='bilinear', align_corners=False)
                batch = batch.view(B, S, C, *(224,224))

            preds = target_model(batch).bool()
            fooled += preds.sum().item()

    return fooled / len(fake_notes)



class InceptionFeatureExtractor(nn.Module):
    """
    A class that implements the inceptionv3 CNN, a standard for evaluating
    generative image models. 

    Features are a 2048-d vector.
    """
    def __init__(self, device):
        super().__init__()
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        self.model.fc = nn.Identity()
        self.model.eval()
        self.model.to(device)
        self.device = device

    def forward(self, x):
        with torch.no_grad(): # We never want to train this model
            # Resize to 299x299
            x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

            return self.model(x)


def get_activations(imgs, model, batch_size=64):
    """
    Extract Inception feature vectors from a flat image tensor.

    Args:
        imgs (Tensor): (N, C, H, W) float tensor, values in [-1, 1].
        model (InceptionFeatureExtractor): Feature extractor.
        batch_size (int): Batch size for inference. Lower if running out of memory.

    Returns:
        np.ndarray: (N, 2048) feature matrix.
    """
    activations = []
    for start in range(0, len(imgs), batch_size):
        batch = imgs[start : start + batch_size].to(model.device)
        batch = (batch + 1) / 2  # [-1, 1] -> [0, 1]
        feats = model(batch)
        activations.append(feats.cpu().numpy())
        del batch, feats
        torch.cuda.empty_cache()

    return np.concatenate(activations, axis=0)


def compute_stats(activations):
    """
    Takes in the features extracted from the real or fake datasets and treats them
    as if drawn from a multidimensional Gaussian distribution. Returns the mean
    and covariance of the feature vectors.

    Args:
        activations: A list of feature vectors extracted from the dataset. Each entry is a vector corresponding to an image.
                
    Returns:
        mu (Float): the mean of the feature vectors
        sigma (Float): the covariance of the feature vectors  
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_fid(real_imgs, fake_imgs, batch_size=64):
    """
    Calculates FID between two sets of images.

    Args:
        real_imgs (Tensor): (N, C, H, W) flat tensor of real segments.
        fake_imgs (Tensor): (N, C, H, W) flat tensor of generated segments.
        batch_size (int): Batch size for Inception inference.

    Returns:
        float: FID score. Lower is better.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_model = InceptionFeatureExtractor(device)

    real_acts = get_activations(real_imgs, feature_model, batch_size)
    fake_acts = get_activations(fake_imgs, feature_model, batch_size)

    mu_r, sigma_r = compute_stats(real_acts)
    mu_f, sigma_f = compute_stats(fake_acts)
    
    diff = mu_r - mu_f

    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)