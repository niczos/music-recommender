import json
import os
import warnings

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from music_recommender.src.image_utils import transforms
from music_recommender.src.model import ConvNextTinyEncoder
from music_recommender.src.triplet_loss import TripletLoss, train_model
from music_recommender.src.triplet_dataset import TripletRecommendationDataset
from music_recommender.src.utils import get_config, generate_experiment_name

warnings.filterwarnings('ignore')

def plot_loss_history(training_loss_history, validation_loss_history, filepath='loss_history.png'):
    """
    Plot the training and validation loss history.
    """
    epochs = range(1, len(training_loss_history) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_loss_history, label='Training Loss')
    plt.plot(epochs, validation_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.show()

def main(config, resume_epoch=0, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = generate_experiment_name(prefix="triplet_ssrl_learning")
    if checkpoint_path is None:
        results_dir = os.path.join(config["output_dir"], experiment_name)
        os.mkdir(results_dir)
        config_filename = "config.json"
    else:
        results_dir = checkpoint_path
        config_filename = "new_config.json"
    config_path = os.path.join(results_dir, config_filename)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config file copied to: {config_path}")
    
    # Initialize model, dataset, and data loaders
    model = ConvNextTinyEncoder(pretrained=False)
    train_loader = DataLoader(
        TripletRecommendationDataset(
            annotations_file=config["annotations_file"],
            music_dir=config["music_dir"],
            music_parts=config["music_parts"],
            transforms=transforms,
            temp_dir=config["temp_dir"]
        ),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TripletRecommendationDataset(
            annotations_file=config["val_annotations_file"],
            music_dir=config["music_dir"],
            music_parts=config["music_parts"],
            transforms=transforms,
            temp_dir=config["temp_dir"]
        ),
        batch_size=config["batch_size"],
        shuffle=False,
    )
    criterion = TripletLoss(margin=config["triplet_loss_margin"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # Train the model
    training_loss_history, validation_loss_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config["num_epochs"],
        checkpoint_path=results_dir,
        resume_epoch=resume_epoch
    )
    
    # Save the trained model and plot loss history
    model.save(path=results_dir)
    plot_loss_history(training_loss_history, validation_loss_history,
                      filepath=os.path.join(results_dir, "loss_history.png"))

if __name__ == "__main__":
    config = get_config()
    main(config=config)
