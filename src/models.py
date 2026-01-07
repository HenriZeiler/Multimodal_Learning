"""
Model architectures for multimodal learning.

This module contains various neural network architectures for multimodal fusion,
including early fusion, late fusion, and hybrid approaches.
"""

import torch
import torch.nn as nn


class MultimodalModel(nn.Module):
    """Base class for multimodal learning models."""
    
    def __init__(self):
        super(MultimodalModel, self).__init__()
    
    def forward(self, x):
        """Forward pass through the model."""
        raise NotImplementedError("Subclasses must implement forward method")


class EarlyFusionModel(MultimodalModel):
    """Early fusion model that combines modalities at the input level."""
    
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EarlyFusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward pass through early fusion model."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LateFusionModel(MultimodalModel):
    """Late fusion model that combines modalities at the decision level."""
    
    def __init__(self, modality_dims, hidden_dim, num_classes):
        super(LateFusionModel, self).__init__()
        self.modality_encoders = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in modality_dims
        ])
        self.fusion = nn.Linear(hidden_dim * len(modality_dims), num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, modalities):
        """Forward pass through late fusion model."""
        encoded = [self.relu(encoder(mod)) for encoder, mod in 
                   zip(self.modality_encoders, modalities)]
        fused = torch.cat(encoded, dim=1)
        output = self.fusion(fused)
        return output
