import os
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from music_recommender.src.audio_dataset import RecommendationDataset
from music_recommender.src.image_utils import transforms
from music_recommender.src.model import ConvNextTinyEncoder
from music_recommender.src.utils import get_config, get_metric_by_name


def get_recommendations(query: torch.tensor, ds: RecommendationDataset, pool: list[torch.tensor], how_many: int,
                        norm: Callable):
    assert all(embedding.shape == query.shape for embedding in
               pool), f"Wrong shapes query={query.shape} others {[emb.shape for emb in pool]}"
    distances = [norm(embedding, query) for embedding in pool]
    max_dist = max(distances)
    best_indices = np.argsort(distances)[:how_many + 1]
    best_scores = [((max_dist - dist) / max_dist).item() * 100 for dist in np.sort(distances)[:how_many + 1]]
    best_songs = [ds.get_title(idx) for idx in best_indices]
    return best_songs[1:], best_scores[1:]


def main(config):
    ds = RecommendationDataset(annotations_file=config['annotations_file'],
                               music_dir=config['music_dir'],
                               music_parts=config['music_parts'],
                               transforms=transforms,
                               temp_dir=config['temp_dir'])
    dataloader = DataLoader(ds, batch_size=config['batch_size'], shuffle=False)

    sample_tensor, sample_idx = ds.get_sample_by_title(title=config['reference_music'])

    model = ConvNextTinyEncoder(pretrained=os.path.join(config["models_path"], "model_weights.pth"))
    model.eval()

    # one vector represents my chosen music file
    embeddings = model.get_embeddings(dataloader=dataloader)

    norm = get_metric_by_name(name=config["norm"])

    recommendations, scores = get_recommendations(query=embeddings[sample_idx], pool=embeddings,
                                                  how_many=config['how_many'], norm=norm, ds=ds)

    print(f"Your chosen song is {config['reference_music']}")
    print("Best recommendations are:")
    for song_name, score in zip(recommendations, scores):
        print(f"{round(score, 1)}%", song_name)


if __name__ == "__main__":
    config = get_config()
    main(config=config)
