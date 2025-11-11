import json
import os
import warnings

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from music_recommender.src.image_utils import transforms
from music_recommender.src.model import ConvNextTinyEncoder
from music_recommender.src.triplet_loss import TripletLoss, train_model
from music_recommender.src.triplet_dataset import TripletRecommendationDataset
from music_recommender.src.utils import get_config, generate_experiment_name

warnings.filterwarnings('ignore')

# ===== NEW: bezpieczny wrapper i collate =====
class SafeTripletDataset(torch.utils.data.Dataset):
   
    def __init__(self, base_ds):
        self.base = base_ds
        self.skipped_missing = 0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        try:
            return self.base[idx]
        except FileNotFoundError:
            # brak pliku spektrogramu -> pomijamy
            self.skipped_missing += 1
            return None

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    # Jeżeli cały batch był do wyrzucenia, spróbujemy oddać pusty batch
    # o poprawnym typie — default_collate na pustej liście wywali błąd,
    # więc w tym rzadkim wypadku zwracamy None i trening pominie taki batch,
    # jeśli train_model to obsługuje; jeżeli nie, lepiej utrzymywać batch_size
    # i dane tak, aby do tej sytuacji nie dochodziło.
    if not batch:
        # Minimalny bezpieczny fallback: zwróć None.
        # Jeśli Twój train_model nie toleruje None, ustaw batch_size mniejsze
        # lub zadbaj, by w batchu zawsze coś było.
        return None
    return default_collate(batch)
# ============================================

def plot_loss_history(training_loss_history, validation_loss_history, filepath='loss_history.png'):
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

    base_train = TripletRecommendationDataset(
        annotations_file=config["annotations_file"],
        music_dir=config["music_dir"],
        music_parts=config["music_parts"],
        transforms=transforms,
        temp_dir=config["temp_dir"]
    )
    train_ds = SafeTripletDataset(base_train)

    base_val = TripletRecommendationDataset(
        annotations_file=config["val_annotations_file"],
        music_dir=config["music_dir"],
        music_parts=config["music_parts"],
        transforms=transforms,
        temp_dir=config["temp_dir"]
    )
    val_ds = SafeTripletDataset(base_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,            # ważne dla poprawnego liczenia skipped
        collate_fn=safe_collate,  # usuwa None
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate,
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

    # ===== NEW: raport pominięć =====
    skipped_total = train_ds.skipped_missing + val_ds.skipped_missing
    print(f"[INFO] Pominietych próbek (brak spektrogramu): "
          f"train={train_ds.skipped_missing}, val={val_ds.skipped_missing}, razem={skipped_total}")

if __name__ == "__main__":
    config = get_config()
    main(config=config)
