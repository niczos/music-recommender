import os

import yt_dlp
import tqdm
import pandas as pd

from sklearn.model_selection import train_test_split
from music_recommender.src.youtube_api import get_metadata, youtube_id_to_url, download_audio


def run_data_download(output_folder: str, df: pd.DataFrame):

    metadata_filename = os.path.join(output_folder, "metadata.csv")
    val_metadata_filename = os.path.join(output_folder, "val_metadata.csv")

    metadata = []
    limit = 100000
    for ind, row in tqdm.tqdm(df.iterrows()):
        if ind > limit:
            break
        youtube_id = row['youtube_id']
        salami_id = row['salami_id']
        salami_length = row['salami_length']
        youtube_url = youtube_id_to_url(id=youtube_id)
        try:
            filename = download_audio(youtube_url, output_folder)
            if filename is None:
                continue
            sample_metadata = get_metadata(salami_id=salami_id,
                                           salami_length=salami_length,
                                           youtube_id=youtube_id,
                                           filename=filename)
            metadata.append(sample_metadata)
        except yt_dlp.utils.DownloadError as e:
            continue
    metadata = [x for xs in metadata for x in xs]
    metadata_df = pd.DataFrame(metadata)
    # Get unique YouTube IDs
    unique_youtube_ids = metadata_df['youtube_id'].unique()

    # Split the unique IDs into training and validation sets
    train_ids, val_ids = train_test_split(unique_youtube_ids, test_size=0.2, random_state=42)

    # Create training and validation dataframes based on the split IDs
    train_df = metadata_df[metadata_df['youtube_id'].isin(train_ids)]
    val_df = metadata_df[metadata_df['youtube_id'].isin(val_ids)]

    train_df.to_csv(metadata_filename, index=False)
    val_df.to_csv(val_metadata_filename, index=False)


def run_parsing_test(df: pd.DataFrame):
    for _, row in df.iterrows():
        youtube_id = row['youtube_id']
        salami_id = row['salami_id']
        salami_length = row['salami_length']
        sample_metadata = get_metadata(salami_id=salami_id,
                                       salami_length=salami_length,
                                       youtube_id=youtube_id,
                                       filename="Sample")
        print(sample_metadata)
        break


if __name__ == '__main__':
    output_folder = r"C:\Users\skrzy\Music\sample_music"
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(os.path.join(output_folder, 'salami_youtube_pairings.csv'))
    output_folder = r"C:\Users\skrzy\Music\e2e_test"
    run_data_download(output_folder=output_folder, df=df)
    # run_parsing_test(df)
