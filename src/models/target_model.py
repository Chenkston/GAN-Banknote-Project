"""The target model (Victim): Chowdhury et al. multi-view CNN.

Decision Making Process:
------------------------
1. Why DenseNet121?
   According to the original paper's performance table, DenseNet121 achieved 
   the highest metrics across the board (Accuracy: 0.989, Precision: 0.99, 
   Recall: 0.98, F1: 0.99) compared to ResNet50, VGG16, etc. Therefore, we 
   select this architecture as the primary "victim" for our White-Box attack 
   to prove our GANs can defeat their best defense.

2. Why re-train in PyTorch instead of loading their provided weights?
   The authors provided a Keras/TensorFlow `.h5` weights file. Our adversarial 
   attack relies on PyTorch's `autograd` engine to mathematically calculate 
   gradients backward through the Target Model and into our DCGAN Generator 
   (the custom adversarial loss). Converting a complex, 6-branch concatenating 
   Keras model to PyTorch via ONNX frequently results in broken computational 
   graphs or silent gradient failures. 
   
   To guarantee stable adversarial training, we implement their exact multi-view 
   architecture natively in PyTorch and will re-train it on the dataset to act 
   as a perfect, differentiable proxy model.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class TargetCNN(nn.Module):
    """
    Multi-view DenseNet121 Target Model.
    
    Architecture based on Chowdhury et al.:
    - Takes a tensor of 6 banknote segments.
    - Passes each segment through a shared DenseNet121 feature extractor (pre-trained on ImageNet).
    - Flattens and concatenates the 6 feature vectors.
    - Passes the concatenated vector through final Dense layers for binary classification.
    """
    def __init__(self):
        super(TargetCNN, self).__init__()
        
        # Load the base DenseNet121, pre-trained on ImageNet
        base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # We only want the feature extractor part, not the final classification layer
        self.feature_extractor = base_model.features
        
        # Global Average Pooling to reduce spatial dimensions (H, W) to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # DenseNet121 features output 1024 channels. 
        # Since we have 6 segments, the concatenated vector will be 6 * 1024 = 6144
        self.concat_dim = 1024 * 6
        
        # Final classification head (as inferred from standard Keras Concatenate -> Dense)
        self.classifier = nn.Sequential(
            nn.Linear(self.concat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1) # Single output logit (Genuine vs Counterfeit)
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, 6, C, H, W).
            
        Returns:
            Tensor: Single logit output of shape (batch_size, 1).
        """
        batch_size = x.size(0)
        num_segments = x.size(1) # Should be 6
        
        segment_features = []
        
        # Process each of the 6 segments through the shared feature extractor
        for i in range(num_segments):
            # Extract the i-th segment across the batch: shape (batch_size, C, H, W)
            segment = x[:, i, :, :, :]
            
            # Extract features
            features = self.feature_extractor(segment)
            
            # Pool and flatten
            pooled = self.pool(features)
            flattened = torch.flatten(pooled, 1) # shape (batch_size, 1024)
            
            segment_features.append(flattened)
            
        # Concatenate the 6 feature vectors along the feature dimension
        # Resulting shape: (batch_size, 6144)
        concatenated = torch.cat(segment_features, dim=1)
        
        # Final classification
        out = self.classifier(concatenated)
        
        return out
