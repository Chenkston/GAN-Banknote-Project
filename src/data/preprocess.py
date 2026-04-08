"""Preprocessing utilities for the dataset (e.g., normalization to [-1, 1])."""

import torch

def normalize_segments(segments):
    """
    Normalizes 6-segment image tensors from [0, 255] or [0, 1] to [-1, 1].
    Expected input shape: (6, C, H, W) or (6, H, W, C).
    
    This is required for GANs where the Generator output layer uses a Tanh activation.
    """
    if segments.max() > 1.0:
        # Assume data is in the range [0, 255]
        segments = segments / 255.0
        
    # Now data is in [0, 1]. Transform to [-1, 1]
    # Formula: (x * 2) - 1
    segments = (segments * 2.0) - 1.0
    return segments

def denormalize_segments(segments):
    """
    Denormalizes 6-segment image tensors from [-1, 1] back to [0, 1].
    This is useful for plotting images with matplotlib or saving them.
    """
    segments = (segments + 1.0) / 2.0
    return segments.clamp(0, 1)
