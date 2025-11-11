import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image

from music_recommender.src.audio_dataset import RecommendationDataset
from music_recommender.src.image_utils import transforms as default_transforms
from music_recommender.src.utils import get_config

class TripletRecommendationDataset(RecommendationDataset):
    def __init__(self, annotations_file: str, music_dir: str, temp_dir: str, music_parts: list[str], transforms=None):
        # Use default image transformations if none provided
        if transforms is None:
            transforms = default_transforms
        # Initialize the base dataset (loads annotations and sets up DataFrame of segments)
        super().__init__(annotations_file=annotations_file, music_dir=music_dir, temp_dir=temp_dir, 
                         music_parts=music_parts, transforms=transforms)
        # Path to the directory containing precomputed spectrogram images
        self.spectrogram_dir = os.path.join(music_dir, "spectrograms")
    
    def __getitem__(self, idx: int):
        # Retrieve metadata for the track at the given index
        column_names = self.img_labels.columns
        row_values = self.img_labels.iloc[idx].values
        row = {col_name: val for col_name, val in zip(column_names, row_values)}
        
        # Randomly select a segment type (e.g., "Chorus" or "Verse") present in this track for anchor & positive
        type_columns = [name[0] for name in column_names if any(name[0].startswith(part) for part in self.music_parts)]
        available_segments = [name for name in type_columns if not pd.isna(row[(name, "beginning_time")])]
        segment_types = list({name.split("_")[0] for name in available_segments})
        selected_type = random.choice(segment_types)
        
        # Choose two different segments of the selected type for the anchor and positive examples
        segment_choices = [name for name in available_segments if name.startswith(selected_type)]
        if len(segment_choices) >= 2:
            chosen_segments = random.sample(segment_choices, 2)  # two distinct segments
        else:
            chosen_segments = random.choices(segment_choices, k=2)  # fallback (in case of only one segment, not expected)
        
        images = []
        # Base track name for constructing file paths
        base_name = str(row[("filename", "")])
        sanitized_name = base_name.replace(":", "ďĽš").replace("\"", "ďĽ‚").replace("/", "â§¸")
        
        # Load and transform the anchor and positive spectrogram images
        for segment in chosen_segments:
            image_file = f"{sanitized_name}_{segment}.wav.png"
            image_path = os.path.join(self.spectrogram_dir, image_file)
            img = Image.open(image_path).convert("RGB")
            images.append(self.transform(img))
        
        # Select a different track index for the negative example
        other_idx = random.randrange(len(self))
        while other_idx == idx:
            other_idx = random.randrange(len(self))
        other_values = self.img_labels.iloc[other_idx].values
        other_row = {col_name: val for col_name, val in zip(column_names, other_values)}
        
        # Choose a random segment from the other track as the negative example
        base_name_neg = str(other_row[("filename", "")])
        sanitized_name_neg = base_name_neg.replace(":", "ďĽš").replace("\"", "ďĽ‚").replace("/", "â§¸")
        other_segments = [name for name in type_columns if not pd.isna(other_row[(name, "beginning_time")])]
        neg_segment = random.choice(other_segments)
        neg_image_file = f"{sanitized_name_neg}_{neg_segment}.wav.png"
        neg_image_path = os.path.join(self.spectrogram_dir, neg_image_file)
        img_neg = Image.open(neg_image_path).convert("RGB")
        images.append(self.transform(img_neg))
        
        # Return [anchor_image_tensor, positive_image_tensor, negative_image_tensor]
        return images

    def read_annotations(self) -> pd.DataFrame:
        # Load the annotations CSV and filter to the desired segment types
        df = pd.read_csv(self.annotations_path)
        df = df[df["type"].isin(self.music_parts)]
        df = df.dropna(subset=["filename"])
        # Keep only tracks that have more than one segment of a desired type (needed for triplet formation)
        counts = df.groupby(["salami_id", "type"]).size().reset_index(name="count")
        valid_combos = counts[counts["count"] > 1][["salami_id", "type"]]
        df = valid_combos.merge(df, on=["salami_id", "type"], how="left")
        # Label each segment with an index per track/type (e.g., Chorus_0, Chorus_1, etc.)
        df["segment_index"] = df.groupby(["salami_id", "type"]).cumcount()
        df["counted_type"] = df["type"] + "_" + df["segment_index"].astype(str)
        # Pivot to wide format so each track is one row with segment timing columns
        df = df.pivot(index=["salami_id", "filename"], columns="counted_type", values=["beginning_time", "end_time"])
        df.columns = df.columns.reorder_levels(order=[1, 0])  # set first level as segment name, second as time attribute
        df = df.reset_index()
        return df

# If run as a script, allow quick testing of dataset functionality
if __name__ == "__main__":
    config = get_config()
    dataset = TripletRecommendationDataset(
        annotations_file=config["annotations_file"],
        music_dir=config["music_dir"],
        temp_dir=config["temp_dir"],
        music_parts=config["music_parts"]
    )
    print(f"Total tracks in dataset: {len(dataset)}")
    anchor_img, positive_img, negative_img = dataset[0]
    print(f"Anchor image tensor shape: {anchor_img.shape}")
