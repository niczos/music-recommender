import os
from typing import Callable
import warnings
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from music_recommender.src.image_utils import transforms

IMAGE_SIZE = (224, 224)

class RecommendationDataset(Dataset):
    def __init__(self, annotations_file: str, music_dir: str, temp_dir: str, music_parts: list[str],
                 transforms: Callable):
        self.annotations_path = annotations_file
        self.music_parts = music_parts
        self.img_labels = self.read_annotations()
        self.music_dir = music_dir
        self.transform = transforms
        self.temp_dir = temp_dir

        # Lista (filename, part_type, segment_index)
        self.samples = self._build_samples_list()

    def _build_samples_list(self):
        samples = []
        for _, row in self.img_labels.iterrows():
            filename = row[('filename', '')]
            for part in self.music_parts:
                index = 0
                while True:
                    file_name = f"{filename}_{part.lower()}_{index}.wav.png"
                    file_path = os.path.join(self.temp_dir, file_name)
                    if os.path.isfile(file_path):
                        samples.append((filename, part, index))
                        index += 1
                    else:
                        break
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        filename, part, index = self.samples[idx]
        file_name = f"{filename}_{part.lower()}_{index}.wav.png"
        file_path = os.path.join(self.temp_dir, file_name)

        img = Image.open(file_path).convert("RGB")
        img = self.transform(img)

        return img, filename

    def get_title(self, idx: int):
        _, filename, _ = self.samples[idx]
        return filename

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
        for idx, (filename, _, _) in enumerate(self.samples):
            if filename == title:
                return self.__getitem__(idx), idx
        raise ValueError(f"Nie znaleziono utworu: {title}")


if __name__ == '__main__':
    output_folder = r"C:\Users\skrzy\Music\sample_music"
    annotations_file = os.path.join(output_folder, 'metadata.csv')

    ds = RecommendationDataset(annotations_file=annotations_file,
                               music_dir=output_folder,
                               music_parts=["Chorus", "Verse"],
                               transforms=transforms,
                               temp_dir=output_folder)
    for img, title in ds:
        assert img.shape == (3, *IMAGE_SIZE), img.shape
