import csv
import os
import time

import torch
import torch.nn as nn


# Define the Triplet Loss Function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        distance_positive = torch.nn.functional.pairwise_distance(anchor,
                                                                  positive)
        distance_negative = torch.nn.functional.pairwise_distance(anchor,
                                                                  negative)
        target = torch.ones_like(distance_positive)
        loss = self.loss_fn(distance_positive, distance_negative, target)
        return loss


def validation_step(model, val_loader, criterion, device):
    """
    Performs a validation step on the model using the provided data loader.

    Args:
        model (torch.nn.Module): The model to be validated.
        val_loader (torch.utils.data.DataLoader): The data loader for validation data.
        criterion (torch.nn.Module): The loss function to be used.
        device (torch.device): The device to use for validation (CPU or GPU).

    Returns:
        float: The average validation loss for the current epoch.
    """

    model.eval()  # Set model to evaluation mode
    model.to(device)
    epoch_loss = 0.0

    # Iterate over batches in the validation loader
    for anchor, positive, negative in val_loader:
        anchor, positive, negative = anchor.to(device), positive.to(
            device), negative.to(device)

        # Forward pass
        with torch.no_grad():
            # Forward pass
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

        # Compute loss
        loss = criterion(anchor_output, positive_output, negative_output)

        epoch_loss += loss.item()  # Accumulate loss for the epoch

    # Calculate average loss for the validation epoch
    avg_loss = epoch_loss / len(val_loader)

    return avg_loss


# Function to Perform a Single Training Step
def train_step(model, data, criterion, optimizer, device):
    anchor, positive, negative = data
    anchor, positive, negative = anchor.to(device), positive.to(
        device), negative.to(device)

    # Forward pass
    anchor_output = model(anchor)
    positive_output = model(positive)
    negative_output = model(negative)

    # Compute loss
    loss = criterion(anchor_output, positive_output, negative_output)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs, checkpoint_path, resume_epoch=0, log_interval=10, patience=5):
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
        Tuple of training loss history and validation loss history (if validation is used).
    """
    model.to(device)

    training_loss_history = []
    validation_loss_history = [] if val_loader is not None else None

    # checkpoint directory
    assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist."

    start_epoch = resume_epoch

    if resume_epoch > 0:
        checkpoint_filename = os.path.join(checkpoint_path, f'model_epoch_{resume_epoch}.pth')
        if os.path.isfile(checkpoint_filename):
            print(f"Resuming training from checkpoint: {checkpoint_filename}")
            checkpoint = torch.load(checkpoint_filename)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            training_loss_history = checkpoint.get('training_loss_history', [])
            validation_loss_history = checkpoint.get('validation_loss_history', []) if val_loader is not None else None
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found for epoch {resume_epoch}. Starting from scratch.")
            start_epoch = 0

    print(f"Starting training from epoch {start_epoch + 1} to {num_epochs} on {device}.")

    # Initialize CSV logging
    csv_log_file = os.path.join(checkpoint_path, "training_log.csv")
    file_exists = os.path.isfile(csv_log_file)
    with open(csv_log_file, mode='a', newline='') as csvfile:
        fieldnames = ['epoch', 'batch', 'batch_loss', 'batch_time', 'avg_epoch_loss', 'validation_loss', 'epoch_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = len(train_loader)

            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

            model.train()

            for batch_idx, data in enumerate(train_loader):
                batch_start_time = time.time()
                loss = train_step(model, data, criterion, optimizer, device)
                epoch_loss += loss

                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time

                if (batch_idx + 1) % log_interval == 0 or batch_idx == num_batches - 1:
                    print(
                        f"  Batch [{batch_idx + 1}/{num_batches}] Loss: {loss:.6f}, Time: {batch_time:.2f}s",
                        end='\r')

                writer.writerow({
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'batch_loss': loss,
                    'batch_time': batch_time,
                    'avg_epoch_loss': None,
                    'validation_loss': None,
                    'epoch_time': None
                })

            print("")

            avg_epoch_loss = epoch_loss / num_batches
            training_loss_history.append(avg_epoch_loss)

            if val_loader is not None:
                model.eval()
                validation_loss = validation_step(model, val_loader, criterion, device)
                validation_loss_history.append(validation_loss)

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            print(
                f'Epoch [{epoch + 1}/{num_epochs}] Average Training Loss: {avg_epoch_loss:.6f}, Epoch Time: {epoch_time:.2f}s')
            if val_loader is not None:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {validation_loss:.6f}')
            else:
                print("No validation performed.")

            writer.writerow({
                'epoch': epoch + 1,
                'batch': None,
                'batch_loss': None,
                'batch_time': None,
                'avg_epoch_loss': avg_epoch_loss,
                'validation_loss': validation_loss if val_loader else None,
                'epoch_time': epoch_time
            })

            checkpoint_filename = os.path.join(checkpoint_path, f'model_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'training_loss_history': training_loss_history,
                'validation_loss_history': validation_loss_history,
            }, checkpoint_filename)
            print(f"Saved checkpoint: {checkpoint_filename}")

            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    print("\nTraining complete.")

    return training_loss_history, validation_loss_history
