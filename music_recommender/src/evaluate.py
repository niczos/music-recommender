import torch.utils


# TODO come back after training
from music_recommender.src.triplet_loss import validation_step


def evaluate_on_data(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion, device):
    return validation_step(model, dataloader, criterion, device)



def evaluate(model: torch.nn.Module, data: dict[str, torch.utils.data.DataLoader], criterion, device):
    results_dict = {}
    for dataset_name, dataloader in data.items():
        results_dict[dataset_name] = evaluate_on_data(model, dataloader=dataloader, criterion=criterion, device=device)

    return results_dict
