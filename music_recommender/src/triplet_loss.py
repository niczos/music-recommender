import csv
import os
import time
import math

import torch
import torch.nn as nn


# === Triplet Loss ===
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        distance_positive = torch.nn.functional.pairwise_distance(anchor, positive)
        distance_negative = torch.nn.functional.pairwise_distance(anchor, negative)
        target = torch.ones_like(distance_positive)
        loss = self.loss_fn(distance_positive, distance_negative, target)
        return loss


# === Walidacja: pomijanie pustych batchy ===
def validation_step(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    total = 0
    skipped_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                skipped_batches += 1
                continue
            try:
                anchor, positive, negative = batch
            except Exception:
                # np. pusty/niepoprawny batch po filtrze
                skipped_batches += 1
                continue

            anchor, positive, negative = (
                anchor.to(device), positive.to(device), negative.to(device)
            )
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            loss = criterion(emb_a, emb_p, emb_n)
            val_loss += loss.item()
            total += 1

    avg = (val_loss / total) if total > 0 else float("nan")
    if skipped_batches:
        print(f"[VAL] Pomięte batchy (puste/niepoprawne): {skipped_batches}")
    return avg


# === Pojedynczy krok treningu ===
def train_step(model, data, criterion, optimizer, device):
    anchor, positive, negative = data
    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

    # Forward
    anchor_output = model(anchor)
    positive_output = model(positive)
    negative_output = model(negative)

    # Loss
    loss = criterion(anchor_output, positive_output, negative_output)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# === Główna pętla treningowa (z pomijaniem pustych batchy) ===
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    checkpoint_path,
    resume_epoch=0,
    log_interval=10,
    patience=15,
):
    """
    Trains a model with detailed logging, timing information, saves checkpoints, and logs to a CSV.
    Supports resuming training from a specific epoch and early stopping.

    Args:
        model: The model to train.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data (optional).
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use (e.g., 'cuda' or 'cpu').
        num_epochs: Total number of training epochs.
        checkpoint_path: Path to save model checkpoints.
        resume_epoch: Epoch number to resume training from (0 for starting from scratch).
        log_interval: Frequency (in batches) to log training progress.
        patience: Number of epochs to wait for improvement before early stopping.

    Returns:
        (training_loss_history, validation_loss_history)
    """
    model.to(device)

    training_loss_history = []
    validation_loss_history = [] if val_loader is not None else None

    assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist."
    start_epoch = resume_epoch

    # Wznowienie
    if resume_epoch > 0:
        checkpoint_filename = os.path.join(checkpoint_path, f"model_epoch_{resume_epoch}.pth")
        if os.path.isfile(checkpoint_filename):
            print(f"Resuming training from checkpoint: {checkpoint_filename}")
            checkpoint = torch.load(checkpoint_filename, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            training_loss_history = checkpoint.get("training_loss_history", [])
            if val_loader is not None:
                validation_loss_history = checkpoint.get("validation_loss_history", [])
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found for epoch {resume_epoch}. Starting from scratch.")
            start_epoch = 0

    print(f"Starting training from epoch {start_epoch + 1} to {num_epochs} on {device}.")

    # CSV log
    csv_log_file = os.path.join(checkpoint_path, "training_log.csv")
    file_exists = os.path.isfile(csv_log_file)
    with open(csv_log_file, mode="a", newline="") as csvfile:
        fieldnames = ["epoch", "batch", "batch_loss", "batch_time", "avg_epoch_loss", "validation_loss", "epoch_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            processed_batches = 0  # licznik faktycznie policzonych batchy (po pominięciach)
            skipped_batches = 0

            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            model.train()

            for batch_idx, data in enumerate(train_loader):
                batch_start_time = time.time()

                # Pusty batch po filtrze / collate
                if data is None:
                    skipped_batches += 1
                    continue

                try:
                    loss = train_step(model, data, criterion, optimizer, device)
                except Exception:
                    # np. nieoczekiwany problem z konkretną próbką — pomijamy batch
                    skipped_batches += 1
                    continue

                epoch_loss += loss
                processed_batches += 1

                batch_time = time.time() - batch_start_time
                if (batch_idx + 1) % log_interval == 0:
                    print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss:.6f}, Time: {batch_time:.2f}s", end="\r")

                # logujemy tylko realne batch'e
                writer.writerow({
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "batch_loss": loss,
                    "batch_time": batch_time,
                    "avg_epoch_loss": None,
                    "validation_loss": None,
                    "epoch_time": None,
                })

            # podsumowanie epoki
            print("")  # nowa linia po progressie
            if skipped_batches:
                print(f"[TRAIN] Pomięte batchy (puste/niepoprawne): {skipped_batches}")

            if processed_batches > 0:
                avg_epoch_loss = epoch_loss / processed_batches
            else:
                avg_epoch_loss = float("nan")
                print("[WARN] Brak przetworzonych batchy w tej epoce (wszystko pominięte).")

            training_loss_history.append(avg_epoch_loss)

            # walidacja
            if val_loader is not None:
                model.eval()
                validation_loss = validation_step(model, val_loader, criterion, device)
                # zabezpieczenie: NaN/None -> traktuj jako „brak poprawy”
                if validation_loss is None or not isinstance(validation_loss, float) or not math.isfinite(validation_loss):
                    validation_loss = float("inf")
                validation_loss_history.append(validation_loss)

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                scheduler.step(validation_loss)

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch [{epoch + 1}/{num_epochs}] Average Training Loss: {avg_epoch_loss:.6f}, Epoch Time: {epoch_time:.2f}s")
            if val_loader is not None and math.isfinite(validation_loss):
                print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {validation_loss:.6f}")
            elif val_loader is not None:
                print("Epoch Validation Loss: NaN/Inf (pominięto w kryterium poprawy)")

            # log końca epoki
            writer.writerow({
                "epoch": epoch + 1,
                "batch": None,
                "batch_loss": None,
                "batch_time": None,
                "avg_epoch_loss": avg_epoch_loss,
                "validation_loss": (validation_loss if (val_loader and math.isfinite(validation_loss)) else None),
                "epoch_time": epoch_time,
            })

            # checkpoint
            checkpoint_filename = os.path.join(checkpoint_path, f"model_epoch_{epoch + 1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_epoch_loss,
                "training_loss_history": training_loss_history,
                "validation_loss_history": validation_loss_history,
            }, checkpoint_filename)
            print(f"Saved checkpoint: {checkpoint_filename}")

            # early stopping
            if val_loader is not None and epochs_no_improve == patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    print("\nTraining complete.")
    return training_loss_history, validation_loss_history
