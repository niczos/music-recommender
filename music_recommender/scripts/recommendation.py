import os
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from music_recommender.src.audio_dataset import RecommendationDataset
from music_recommender.src.image_utils import transforms
from music_recommender.src.model import ConvNextTinyEncoder
from music_recommender.src.utils import get_metric_by_name


def get_recommendations(
    query_emb: torch.Tensor,
    pool_emb: torch.Tensor,
    ds: RecommendationDataset,
    how_many: int,
    norm: Callable,
    query_idx: int,
) -> Tuple[List[str], List[float]]:
    """
    Zwraca listę nazw utworów i „procentowych” score’ów
    dla how_many najlepszych rekomendacji (z wyłączeniem samego utworu referencyjnego).

    Zakładamy, że `norm(a, b)` jest metryką odległości (im mniejsza, tym lepiej),
    np. odległość euklidesowa albo 1 - cosine_similarity.
    """

    # Upewniamy się, że query ma kształt (D,)
    if query_emb.ndim == 2 and query_emb.shape[0] == 1:
        query_emb = query_emb[0]

    distances = []
    for idx, emb in enumerate(pool_emb):
        d = norm(emb, query_emb)
        if isinstance(d, torch.Tensor):
            d = d.item()
        distances.append((idx, float(d)))

    # Wycinamy z listy sam utwór referencyjny
    distances_no_self = [(idx, d) for idx, d in distances if idx != query_idx]

    if not distances_no_self:
        raise ValueError("Nie znaleziono żadnych innych utworów w puli do rekomendacji.")

    # Sortujemy rosnąco po odległości (mniejsza = bliżej)
    distances_no_self.sort(key=lambda x: x[1])

    # Bierzemy top-K
    top = distances_no_self[:how_many]

    # Skalowanie do „procentowej podobności”
    # 100% ~ najbardziej podobny z puli, 0% ~ najdalszy z rozważanej top-K puli (lokalnie)
    # Można też skalować względem całej puli, ale to prosta, intuicyjna wersja.
    d_vals = [d for _, d in top]
    d_min = min(d_vals)
    d_max = max(d_vals)
    if d_max == d_min:
        # wszystkie takie same → dajemy 100% wszystkim
        scores = [100.0 for _ in top]
    else:
        scores = [100.0 * (d_max - d) / (d_max - d_min) for d in d_vals]

    song_names = [ds.get_title(idx) for idx, _ in top]

    return song_names, scores


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> ConvNextTinyEncoder:
    """
    Ładuje ConvNextTinyEncoder z podanego checkpointu.

    Obsługuje zarówno:
    - checkpoint zawierający dict z kluczem 'model_state_dict',
    - jak i czyste state_dict.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Nie znaleziono pliku modelu: {checkpoint_path}")

    model = ConvNextTinyEncoder(pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model


def compute_embeddings_with_model(
    model: ConvNextTinyEncoder,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Uniwersalny sposób na policzenie embeddingów,
    niezależnie od tego, czy model ma metodę get_embeddings, czy nie.
    """
    if hasattr(model, "get_embeddings"):
        # Jeśli w Twoim modelu jest już zaimplementowana metoda get_embeddings – korzystamy
        embeddings = model.get_embeddings(dataloader=dataloader)
        if isinstance(embeddings, torch.Tensor):
            return embeddings
        else:
            return torch.tensor(embeddings)

    # Wersja „ręczna”
    all_emb = []
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)
            emb = model(batch)
            all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)


def main(config: dict):
    """
    Główna funkcja rekomendacyjna.

    Zakładamy, że w configu masz m.in.:
    - 'annotations_file'
    - 'music_dir'
    - 'music_parts'
    - 'temp_dir'
    - 'batch_size'
    - 'reference_music'  (tytuł utworu, który wybiera osoba)
    - 'how_many'         (ile rekomendacji pokazać, np. 3)
    - 'norm'             (nazwa metryki, np. 'euclidean' / 'cosine_distance')
    - 'checkpoint_path'  lub 'models_path' + 'model_weights.pth'
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset do rekomendacji – wszystkie dostępne utwory / fragmenty
    ds = RecommendationDataset(
        annotations_file=config["annotations_file"],
        music_dir=config["music_dir"],
        music_parts=config["music_parts"],
        transforms=transforms,
        temp_dir=config["temp_dir"],
    )
    dataloader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False)

    # Znajdujemy embedding odpowiadający wybranemu tytułowi
    sample_tensor, sample_idx = ds.get_sample_by_title(title=config["reference_music"])

    # Wczytanie modelu
    checkpoint_path = config.get("checkpoint_path", None)
    if checkpoint_path is None:
        # fallback na starą logikę z models_path
        checkpoint_path = os.path.join(config["models_path"], "model_weights.pth")

    model = load_model_from_checkpoint(checkpoint_path, device=device)

    # Embeddingi dla całego datasetu
    embeddings = compute_embeddings_with_model(model, dataloader, device=device)

    if sample_idx < 0 or sample_idx >= embeddings.shape[0]:
        raise IndexError(
            f"Nieprawidłowy indeks próbki referencyjnej: {sample_idx} (embeddingów: {embeddings.shape[0]})"
        )

    # Metryka podobieństwa / odległości
    norm = get_metric_by_name(name=config["norm"])

    # Obliczamy rekomendacje
    query_emb = embeddings[sample_idx]
    recommendations, scores = get_recommendations(
        query_emb=query_emb,
        pool_emb=embeddings,
        ds=ds,
        how_many=config["how_many"],
        norm=norm,
        query_idx=sample_idx,
    )

    # Output – do ankiety
    print(f"\nWybrany utwór (referencyjny): {config['reference_music']}")
    print(f"Najlepsze {config['how_many']} rekomendacje (bez utworu referencyjnego):")
    for song_name, score in zip(recommendations, scores):
        print(f"  {round(score, 1):5.1f}%  –  {song_name}")

    # Zwracamy też dane, gdybyś chciała to zapisać do CSV / ankiety
    return {
        "reference_music": config["reference_music"],
        "recommendations": recommendations,
        "scores": scores,
    }


if __name__ == "__main__":
    raise RuntimeError(
        "Ten skrypt jest przeznaczony do użycia z notatnika: "
        "from music_recommender.scripts.recommendation import main; main(config)"
    )
