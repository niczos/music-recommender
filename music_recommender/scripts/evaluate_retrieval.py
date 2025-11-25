import os
import json
import warnings
from typing import Tuple, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from music_recommender.src.triplet_dataset import TripletRecommendationDataset
from music_recommender.src.image_utils import transforms
from music_recommender.src.model import ConvNextTinyEncoder

warnings.filterwarnings("ignore")


# =====================================================
#   TU UZUPEŁNIASZ ŚCIEŻKI DO KONKRETNYCH MODELI
# =====================================================

EXPERIMENTS: List[Dict[str, str]] = [
    {
        "name": "margin_0.01_lr1e-5_sched",
        "checkpoint": "/content/drive/MyDrive/music_recommender/results/triplet_ssrl_learning_20251124_115322/model_epoch_20.pth",
    },
    {
        "name": "margin_0.1_lr1e-5_sched",
        "checkpoint": "/content/drive/MyDrive/music_recommender/results/triplet_ssrl_learning_20251124_135949/model_epoch_20.pth",
    },
    {
        "name": "margin_0.1_lr1e-4_nosched_bs16",
        "checkpoint": "/content/drive/MyDrive/music_recommender/results/triplet_ssrl_learning_20251124_220244/model_epoch_30.pth",
    },
    # jeśli chcesz, możesz dodać kolejny, np. bs32:
    # {
    #     "name": "margin_0.1_lr1e-4_nosched_bs32",
    #     "checkpoint": "/content/drive/MyDrive/music_recommender/results/POPRWADZ_TUTAJ/model_epoch_30.pth",
    # },
]


# -----------------------------------------------------
#  Dataset do ewaluacji: pojedyncze fragmenty
# -----------------------------------------------------

class SegmentEvalDataset(Dataset):
    """
    Dataset do ewaluacji:
    - zamiast trójek zwraca pojedynczy fragment + id utworu,
    - korzysta z tych samych mechanizmów co TripletRecommendationDataset
      (w szczególności z _load_image i sanitizacji nazw).
    """

    def __init__(self, base_ds: TripletRecommendationDataset):
        self.base = base_ds
        self.samples = []  # lista (base_name, seg_name, track_id)

        df = self.base.img_labels
        columns = df.columns

        # track_id: wolimy salami_id, jeśli jest; inaczej filename
        if ("salami_id", "") in columns:
            track_key = ("salami_id", "")
        else:
            track_key = ("filename", "")

        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            track_id = row[track_key]
            base_name = str(row[("filename", "")])

            for col in columns:
                if not (isinstance(col, tuple) and len(col) == 2):
                    continue
                seg_name, time_attr = col
                if time_attr != "beginning_time":
                    continue
                val = row[col]
                # NaN check
                if isinstance(val, float) and np.isnan(val):
                    continue

                # próbujemy wczytać obrazek tym samym mechanizmem,
                # którego używa TripletRecommendationDataset
                try:
                    _ = self.base._load_image(base_name, seg_name)
                    self.samples.append((base_name, seg_name, track_id))
                except FileNotFoundError:
                    continue

        print(f"[EVAL] fragmentów dostępnych do ewaluacji: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_name, seg_name, track_id = self.samples[idx]
        img = self.base._load_image(base_name, seg_name)
        return img, track_id


# -----------------------------------------------------
#   Liczenie embeddingów
# -----------------------------------------------------

def compute_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_emb = []
    all_ids = []

    with torch.no_grad():
        for imgs, ids in loader:
            imgs = imgs.to(device)
            emb = model(imgs)
            # L2-normalizacja → cosine similarity = iloczyn skalarny
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_emb.append(emb.cpu())

            # ids może być tensorem / listą – uprośćmy do listy intów
            # zakładamy, że track_id jest numeryczne (np. salami_id)
            ids_list = [int(x) for x in ids]
            all_ids.extend(ids_list)

    embeddings = torch.cat(all_emb, dim=0)                    # (N, D)
    track_ids = torch.tensor(all_ids, dtype=torch.long)       # (N,)
    return embeddings, track_ids


# -----------------------------------------------------
#   Retrieval: Recall@K, Mean Rank, MRR
# -----------------------------------------------------

def retrieval_metrics(
    emb: torch.Tensor,
    ids: torch.Tensor,
    ks=(1, 5, 10)
) -> Dict[str, Any]:
    """
    Dla każdego fragmentu:
    - liczymy podobieństwo do wszystkich innych,
    - sprawdzamy, na jakiej pozycji jest pierwszy fragment z tej samej piosenki.
    """
    sim = emb @ emb.T  # cosine, bo emb są znormalizowane
    sim.fill_diagonal_(-1e9)  # wyłączamy podobieństwo z samym sobą
    N = sim.shape[0]

    ranks = []
    recalls = {k: 0 for k in ks}

    for i in range(N):
        # sortujemy od najbardziej podobnych
        order = torch.argsort(sim[i], descending=True)
        same = (ids[order] == ids[i])
        pos = torch.nonzero(same, as_tuple=False)

        if pos.numel() == 0:
            # brak innego fragmentu z tej samej piosenki
            continue

        r = int(pos[0].item()) + 1  # 1-based rank
        ranks.append(r)

        for k in ks:
            if r <= k:
                recalls[k] += 1

    if not ranks:
        return {
            "mean_rank": None,
            "MRR": None,
            "recall": {k: 0.0 for k in ks},
            "num_queries": N,
            "num_queries_with_pos": 0,
        }

    mean_rank = float(sum(ranks) / len(ranks))
    mrr = float(sum(1.0 / r for r in ranks) / len(ranks))
    recall = {k: float(recalls[k] / N) for k in ks}

    return {
        "mean_rank": mean_rank,
        "MRR": mrr,
        "recall": recall,
        "num_queries": N,
        "num_queries_with_pos": len(ranks),
    }


# -----------------------------------------------------
#   Same vs Different
# -----------------------------------------------------

def same_diff_metrics(
    emb: torch.Tensor,
    ids: torch.Tensor,
    max_neg: int = 200_000
) -> Dict[str, Any]:
    """
    Buduje pary (same / different), liczy similarity,
    szuka progu max accuracy, zwraca też surowe similarity
    do wykresu.
    """
    sim = emb @ emb.T
    N = sim.shape[0]

    pos_sims = []
    neg_sims = []

    for i in range(N):
        for j in range(i + 1, N):
            s = sim[i, j].item()
            if ids[i] == ids[j]:
                pos_sims.append(s)
            else:
                neg_sims.append(s)

    if not pos_sims or not neg_sims:
        return {
            "pos_mean": None,
            "neg_mean": None,
            "best_acc": None,
            "best_threshold": None,
            "num_pos_pairs": len(pos_sims),
            "num_neg_pairs": len(neg_sims),
            "pos_sims": np.array([]),
            "neg_sims": np.array([]),
        }

    # subsampling negatywnych par
    import random
    if len(neg_sims) > max_neg:
        neg_sims = random.sample(neg_sims, max_neg)

    pos = np.array(pos_sims, dtype=float)
    neg = np.array(neg_sims, dtype=float)

    all_s = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])

    thresholds = np.linspace(all_s.min(), all_s.max(), num=200)
    best_acc = 0.0
    best_thr = thresholds[0]

    for thr in thresholds:
        preds = (all_s >= thr).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return {
        "pos_mean": float(pos.mean()),
        "neg_mean": float(neg.mean()),
        "best_acc": float(best_acc),
        "best_threshold": float(best_thr),
        "num_pos_pairs": int(pos.size),
        "num_neg_pairs": int(neg.size),
        "pos_sims": pos,
        "neg_sims": neg,
    }


# -----------------------------------------------------
#   Wykresy
# -----------------------------------------------------

def plot_recall_bar(
    retrieval: Dict[str, Any],
    save_path: str
):
    recall = retrieval["recall"]
    ks = sorted(recall.keys())
    vals = [recall[k] for k in ks]

    plt.figure(figsize=(6, 4))
    plt.bar([str(k) for k in ks], vals)
    plt.ylim(0, 1.0)
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("Retrieval – Recall@K dla fragmentów\n(ta sama piosenka)")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_similarity_hist(
    pos_sims: np.ndarray,
    neg_sims: np.ndarray,
    save_path: str
):
    plt.figure(figsize=(7, 4))
    plt.hist(pos_sims, bins=40, density=True, alpha=0.6, label="Ta sama piosenka")
    plt.hist(neg_sims, bins=40, density=True, alpha=0.6, label="Różne piosenki")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Gęstość")
    plt.title("Rozkład podobieństw embeddingów\npar pozytywnych i negatywnych")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# -----------------------------------------------------
# MAIN – teraz przyjmuje config jako argument
# -----------------------------------------------------

def main(config: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # bazowy dataset walidacyjny – ten sam, co do treningu, ale używamy go do generowania fragmentów
    base_val = TripletRecommendationDataset(
        annotations_file=config["val_annotations_file"],
        music_dir=config["music_dir"],
        music_parts=config["music_parts"],
        transforms=transforms,
        temp_dir=config["temp_dir"],
    )

    eval_ds = SegmentEvalDataset(base_val)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    summary_rows = []

    for exp in EXPERIMENTS:
        name = exp["name"]
        model_path = exp["checkpoint"]

        if not os.path.isfile(model_path):
            print(f"[WARN] Pomijam {name} – brak pliku: {model_path}")
            continue

        print("\n==============================")
        print(f"   Ewaluacja modelu: {name}")
        print(f"   Checkpoint: {model_path}")
        print("==============================")

        # 1) wczytanie modelu
        model = ConvNextTinyEncoder(pretrained=False).to(device)
        state = torch.load(model_path, map_location=device)

        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

        # 2) embeddingi
        embeddings, track_ids = compute_embeddings(model, eval_loader, device)

        # 3) retrieval
        retrieval = retrieval_metrics(embeddings, track_ids, ks=(1, 5, 10))

        # 4) same vs different
        same_diff = same_diff_metrics(embeddings, track_ids)

        # katalog na wyniki dla tego modelu
        base_results_dir = config.get("eval_output_dir", None)
        if base_results_dir is None:
            # domyślnie: obok modelu
            base_results_dir = os.path.join(os.path.dirname(model_path), "evaluation")
        results_dir = os.path.join(base_results_dir, name)
        os.makedirs(results_dir, exist_ok=True)

        # zapis metryk do JSON
        metrics = {
            "retrieval": retrieval,
            "same_diff": {
                k: v for k, v in same_diff.items()
                if k not in ("pos_sims", "neg_sims")
            },
        }
        with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        # wykresy
        plot_recall_bar(
            retrieval,
            save_path=os.path.join(results_dir, "recall_at_k.png"),
        )

        if same_diff["pos_sims"].size > 0 and same_diff["neg_sims"].size > 0:
            plot_similarity_hist(
                same_diff["pos_sims"],
                same_diff["neg_sims"],
                save_path=os.path.join(results_dir, "similarity_hist.png"),
            )

        # print do konsoli
        print("\n=== RETRIEVAL ===")
        print(f"Liczba fragmentów (N): {retrieval['num_queries']}")
        print(f"Liczba zapytań z >=1 pozytywnym: {retrieval['num_queries_with_pos']}")
        for k, v in retrieval["recall"].items():
            print(f"Recall@{k}: {v:.3f}")
        print(f"Mean rank: {retrieval['mean_rank']:.2f}")
        print(f"MRR: {retrieval['MRR']:.3f}")

        print("\n=== SAME vs DIFFERENT ===")
        print(f"Średnie similarity (ta sama piosenka): {same_diff['pos_mean']}")
        print(f"Średnie similarity (różne piosenki):   {same_diff['neg_mean']}")
        print(f"Najlepsza accuracy: {same_diff['best_acc']}")
        print(f"Próg similarity:    {same_diff['best_threshold']}")
        print(f"Liczba pozytywnych par: {same_diff['num_pos_pairs']}")
        print(f"Liczba negatywnych par: {same_diff['num_neg_pairs']}")
        print(f"Wykresy i metryki zapisane w: {results_dir}")

        # dopisz wiersz do zbiorczego podsumowania
        summary_rows.append({
            "name": name,
            "checkpoint": model_path,
            "N_fragments": retrieval["num_queries"],
            "N_queries_with_pos": retrieval["num_queries_with_pos"],
            "Recall@1": retrieval["recall"][1],
            "Recall@5": retrieval["recall"][5],
            "Recall@10": retrieval["recall"][10],
            "Mean_rank": retrieval["mean_rank"],
            "MRR": retrieval["MRR"],
            "pos_mean_sim": same_diff["pos_mean"],
            "neg_mean_sim": same_diff["neg_mean"],
            "same_diff_best_acc": same_diff["best_acc"],
            "same_diff_best_thr": same_diff["best_threshold"],
            "num_pos_pairs": same_diff["num_pos_pairs"],
            "num_neg_pairs": same_diff["num_neg_pairs"],
        })

    # zapis zbiorczy – jak „excel” z porównaniem modeli
    if summary_rows:
        if base_results_dir is None:
            any_cp = EXPERIMENTS[0]["checkpoint"]
            base_results_dir = os.path.join(os.path.dirname(any_cp), "evaluation")
        os.makedirs(base_results_dir, exist_ok=True)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(base_results_dir, "eval_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nZbiorcze podsumowanie zapisane w: {summary_path}")
    else:
        print("Brak udanych ewaluacji – sprawdź ścieżki w EXPERIMENTS.")


if __name__ == "__main__":
    raise RuntimeError("Użyj tego skryptu z Colaba: from ... import main; main(config)")
