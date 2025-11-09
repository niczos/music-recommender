import os

import pandas as pd
from tqdm import tqdm

from music_recommender.src.audio_processing import AudioSample

if __name__ == "__main__":
    folder_path = r"/home/nika/music-recommender-system/sample_music/"
    temp_dir = r"C:\Users\skrzy\Music\sample_music"
    folder_path = r"C:\Users\skrzy\Music\sample_music"
    output_dir = r"C:\Users\skrzy\Music\cut_dataset"

    annotations_path = r'C:\Users\skrzy\Music\sample_music\metadata.csv'
    music_parts = ['Chorus', 'Verse']

    df = pd.read_csv(annotations_path)
    df = df[df["type"].isin(music_parts)]
    df = df.dropna(subset="filename")

    for filename in df["filename"].unique():
        file_df = df[df["filename"] == filename]
        filename = filename.replace(":", "：").replace("\"", "＂").replace("/", "⧸")
        if len(file_df) == 0:
            continue
        file_df["no"] = file_df.groupby(["type"]).cumcount()
        sample_path = os.path.join(folder_path, filename + ".wav")
        audio_1 = AudioSample(filepath=sample_path, temp_dir=temp_dir)
        for idx, row in file_df.iterrows():
            type = row["type"]
            no = row["no"]
            print(f"Saving {filename}_{type}_{no}")
            spec_1 = audio_1.to_mp3(output_dir=output_dir,
                                    name=f"{filename}_{type}_{no}" + ".wav",
                                    beginning=row["beginning_time"],
                                    end=row["end_time"])
