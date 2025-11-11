import os

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

from music_recommender.src.dataloaders import get_dataloaders
from music_recommender.src.image_utils import transforms

class ConvNextTinyEncoder(nn.Module):
    def __init__(self, pretrained: bool | str = True):
        super(ConvNextTinyEncoder, self).__init__()
        self.convnext_tiny = convnext_tiny(pretrained=(pretrained if pretrained is True else None))
        self.convnext_tiny.classifier = nn.Identity()
        if pretrained is not False and os.path.exists(pretrained):
            self.convnext_tiny.load_state_dict(torch.load(pretrained))
            print(f"Loaded model weights from {pretrained}")
        # If pretrained is False or path not found, model will remain randomly initialized.

    def forward(self, images):
        # images: tensor of images with shape (N, V, C, H, W) or (N, C, H, W)
        if images.ndimension() == 4:  # If input is (N, C, H, W), add a view dimension
            images = images.unsqueeze(1)  # shape becomes (N, 1, C, H, W)
        embeddings = []
        # Iterate over each "view" (e.g., multiple parts per sample)
        for view_idx in range(images.shape[1]):
            # Extract the batch of images for this view and get embeddings
            embedding = self.convnext_tiny(images[:, view_idx])  # output shape: (N, feature_dim, H_feat, W_feat)
            embeddings.append(embedding)
        # Concatenate embeddings from all views along the feature dimension
        concatenated = torch.cat(embeddings, dim=1)
        # If ConvNeXt outputs spatial feature maps, squeeze them to 1D
        return concatenated.squeeze(dim=(2, 3))

    def save(self, path: str):
        model_path = os.path.join(path, 'model_weights.pth')
        torch.save(self.convnext_tiny.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")
