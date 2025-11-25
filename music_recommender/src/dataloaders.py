from typing import Callable

from torch.utils.data import DataLoader

from music_recommender.src.audio_dataset import AudioDataset


def get_dataloaders(annotations_file: str, music_dir: str, temp_dir: str, music_parts: list[str],
                    transforms: Callable, batch_size: int):
    ds = RecommendationDataset(annotations_file=annotations_file,
                               music_dir=music_dir,
                               music_parts=music_parts,
                               transforms=transforms,
                               temp_dir=temp_dir)


    train_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return train_dataloader
