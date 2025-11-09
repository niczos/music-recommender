import os
from typing import Callable
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
from music_recommender.src.consts import IMAGE_SIZE
import pandas as pd
import torch
from torch.utils.data import Dataset

from music_recommender.src.audio_processing import AudioSample
from music_recommender.src.image_utils import transforms


class RecommendationDataset(Dataset):
    def __init__(self, annotations_file: str, music_dir: str, temp_dir: str, music_parts: list[str],
                 transforms: Callable):
        self.annotations_path = annotations_file
        self.music_parts = music_parts
        self.img_labels = self.read_annotations()
        self.music_dir = music_dir
        self.transform = transforms
        self.temp_dir = temp_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        column_names = self.img_labels.columns
        row = {column_name: value for column_name, value in zip(column_names, self.img_labels.iloc[idx].values)}
        images = []
        audio_path = os.path.join(self.music_dir, f"{row[('filename', '')]}.wav".replace(":", "：").replace("\"", "＂").replace("/", "⧸"))
        audio = AudioSample(filepath=audio_path, temp_dir=self.temp_dir)
        # TODO paralellize
        for part in self.music_parts:
            spectrogram = audio.to_spectrogram(beginning=row[(part, "beginning_time")], end=row[(part, "end_time")])
            transformed_image = self.transform(spectrogram)
            images.append(transformed_image)
        return torch.stack(images)

    def get_title(self, idx: int):
        column_names = self.img_labels.columns
        row = {column_name: value for column_name, value in zip(column_names, self.img_labels.iloc[idx].values)}
        return row[('filename', '')]

    def read_annotations(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotations_path)
        df = df[df["type"].isin(self.music_parts)]
        df = df.drop_duplicates(subset=['salami_id', "type"], keep="first")
        df = df.pivot(index=['salami_id', 'filename'], columns='type', values=['beginning_time', 'end_time'])
        df = df.dropna()
        df.columns = df.columns.reorder_levels(order=[1, 0])
        df = df.reset_index()
        return df.dropna()

    def get_sample_by_title(self, title: str):
        sample_idx: int = self.img_labels[self.img_labels[('filename', '')] == title].index[0]
        return self.__getitem__(sample_idx), sample_idx


if __name__ == '__main__':
    output_folder = r"C:\Users\skrzy\Music\sample_music"
    annotations_file = os.path.join(output_folder, 'metadata.csv')

    ds = RecommendationDataset(annotations_file=annotations_file,
                               music_dir=output_folder,
                               music_parts=["Chorus", "Verse"],
                               transforms=transforms,
                               temp_dir=output_folder)
    for el in ds:
        for image in el:
            assert image.shape == (3, *IMAGE_SIZE), image.shape
