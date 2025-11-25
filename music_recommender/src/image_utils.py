import torch
import torchvision.transforms as T
from torchvision.transforms import v2

transforms = v2.Compose([
    T.ToTensor(),
    T.Resize(size=(224,224)),
    v2.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
