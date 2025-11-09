import os
import warnings

import torch
from torch.utils.data import DataLoader

from music_recommender.src.triplet_loss import TripletLoss
from music_recommender.src.trtiplet_dataset import TripletRecommendationDataset

warnings.filterwarnings('ignore')

from music_recommender.src.evaluate import evaluate
from music_recommender.src.image_utils import transforms
from music_recommender.src.model import ConvNextTinyEncoder
from music_recommender.src.utils import get_config


def main(config):
    # ["Chorus", "Verse"]
    val_loader = DataLoader(
        TripletRecommendationDataset(
            annotations_file=config["val_annotations_file"],
            music_dir=config["music_dir"],
            music_parts=config["music_parts"],
            transforms=transforms,
            temp_dir=config["temp_dir"],
        ),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    model = ConvNextTinyEncoder(
        pretrained=os.path.join(config["models_path"], "model_weights.pth"))
    criterion = TripletLoss(margin=config["triplet_loss_margin"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate(model=model, data={"val_loader": val_loader},
                       criterion=criterion, device=device)

    print(metrics)


if __name__ == "__main__":
    config = get_config()
    main(config)
