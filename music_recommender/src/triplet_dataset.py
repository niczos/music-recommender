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
    """
    Wersja: max 2 triplety na utwór
    - jeśli utwór ma ≥2 Chorus + ≥2 Verse -> 2 triplety (1 z Chorus, 1 z Verse)
    - jeśli ma tylko Chorus (≥2) -> 1 triplet (Chorus)
    - jeśli ma tylko Verse (≥2) -> 1 triplet (Verse)
    - jeśli nie ma żadnego typu z ≥2 segmentami -> 0 tripletów
    Negatyw jest z innej piosenki, preferencyjnie tego samego typu (Chorus/Verse).
    """

    def __init__(
        self,
        annotations_file: str,
        music_dir: str,
        temp_dir: str,
        music_parts: list[str],
        transforms=None,
    ):
        if transforms is None:
            transforms = default_transforms

        # to buduje self.img_labels (pivot) itd.
        super().__init__(
            annotations_file=annotations_file,
            music_dir=music_dir,
            temp_dir=temp_dir,
            music_parts=music_parts,
            transforms=transforms,
        )

        self.spectrogram_dir = os.path.join(music_dir, "spectrograms")

        # tu prekomputujemy wszystkie pary pozytywne
        self._build_pairs_index()

    # ------------ public ------------

    def __len__(self) -> int:
        # liczba *par* (anchor, positive) w całym zbiorze
        return len(self.pairs)

    def __getitem__(self, pair_idx: int):
        pair = self.pairs[pair_idx]
        row_idx = pair["row_idx"]          # indeks wiersza (utworu)
        part_type = pair["part_type"]      # "Chorus" albo "Verse"
        segA = pair["segA"]                # np. "Chorus_0"
        segB = pair["segB"]                # np. "Chorus_1"

        row = self._row_dict(row_idx)
        base_name = str(row[("filename", "")])
        sanitized_name = self._sanitize_filename(base_name)

        # anchor & positive
        images = []
        for seg in (segA, segB):
            images.append(self._load_image(sanitized_name, seg))

        # NEGATYW: inny utwór, prefer. ten sam typ części
        neg_row_idx = self._sample_other_row(row_idx)
        neg_row = self._row_dict(neg_row_idx)
        neg_base = str(neg_row[("filename", "")])
        neg_sanitized = self._sanitize_filename(neg_base)

        neg_seg = self._pick_negative_segment(neg_row_idx, prefer_type=part_type)
        neg_img = self._load_image(neg_sanitized, neg_seg)

        return [images[0], images[1], neg_img]

    # ------------ internals ------------

    def _build_pairs_index(self):
        """
        Dla każdego utworu:
        - wyciąga listę segmentów Chorus_* i Verse_*,
        - jeśli Chorus ma ≥2 segmenty -> tworzy 1 parę (losowo),
        - jeśli Verse ma ≥2 segmenty -> tworzy 1 parę (losowo),
        - max 2 pary na utwór, bez dobijania innymi typami.
        """
        self.pairs = []  # każdy element: {"row_idx", "part_type", "segA", "segB"}

        columns = self.img_labels.columns  # MultiIndex: (segment_name, time_attr)

        for row_idx in range(len(self.img_labels)):
            row = self._row_dict(row_idx)

            # zbierz segmenty Chorus_* i Verse_*, które faktycznie istnieją (nie NaN)
            segs_by_type = {"Chorus": [], "Verse": []}

            for col_name in columns:
                if not isinstance(col_name, tuple) or len(col_name) != 2:
                    continue
                seg_name, time_attr = col_name  # np. ("Chorus_0", "beginning_time")
                if time_attr != "beginning_time":
                    continue
                if pd.isna(row[(seg_name, "beginning_time")]):
                    continue

                base_type = seg_name.split("_")[0]
                if base_type in segs_by_type:
                    segs_by_type[base_type].append(seg_name)

            # 1) Chorus para, jeśli się da
            if len(segs_by_type["Chorus"]) >= 2:
                a, b = random.sample(segs_by_type["Chorus"], 2)
                self.pairs.append({
                    "row_idx": row_idx,
                    "part_type": "Chorus",
                    "segA": a,
                    "segB": b,
                })

            # 2) Verse para, jeśli się da
            if len(segs_by_type["Verse"]) >= 2:
                a, b = random.sample(segs_by_type["Verse"], 2)
                self.pairs.append({
                    "row_idx": row_idx,
                    "part_type": "Verse",
                    "segA": a,
                    "segB": b,
                })

            # Czyli: możliwe 0, 1, albo 2 pary z tego utworu.
            # Nie dobieramy nic więcej „na siłę”.

    def _row_dict(self, row_idx: int):
        column_names = self.img_labels.columns
        row_values = self.img_labels.iloc[row_idx].values
        return {col_name: val for col_name, val in zip(column_names, row_values)}

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        return (
            name
            .replace(":", "ďĽš")
            .replace("\"", "ďĽ‚")
            .replace("/", "â§¸")
        )

    def _load_image(self, sanitized_base: str, segment_name: str) -> torch.Tensor:
        image_file = f"{sanitized_base}_{segment_name}.wav.png"
        image_path = os.path.join(self.spectrogram_dir, image_file)
        img = Image.open(image_path).convert("RGB")
        return self.transform(img)

    def _sample_other_row(self, current_row_idx: int) -> int:
        n = len(self.img_labels)
        other_idx = random.randrange(n)
        while other_idx == current_row_idx:
            other_idx = random.randrange(n)
        return other_idx

    def _pick_negative_segment(self, row_idx: int, prefer_type: str | None):
        """
        Wybierz segment negatywu z wiersza row_idx.
        Preferuje ten sam typ (Chorus/Verse), jeśli dostępny.
        """
        row = self._row_dict(row_idx)
        columns = self.img_labels.columns

        def available_segments_of_type(t):
            segs = []
            for col_name in columns:
                if not isinstance(col_name, tuple) or len(col_name) != 2:
                    continue
                seg_name, time_attr = col_name
                if time_attr != "beginning_time":
                    continue
                if not seg_name.startswith(t + "_"):
                    continue
                if pd.isna(row[(seg_name, "beginning_time")]):
                    continue
                segs.append(seg_name)
            return segs

        # 1) spróbuj w tym samym typie (Chorus/Verse)
        if prefer_type is not None:
            segs = available_segments_of_type(prefer_type)
            if segs:
                return random.choice(segs)

        # 2) fallback: jakikolwiek segment dostępny
        any_segs = []
        for col_name in columns:
            if not isinstance(col_name, tuple) or len(col_name) != 2:
                continue
            seg_name, time_attr = col_name
            if time_attr != "beginning_time":
                continue
            if pd.isna(row[(seg_name, "beginning_time")]):
                continue
            any_segs.append(seg_name)

        if not any_segs:
            # po Twoim filtrze w read_annotations raczej się nie zdarzy
            # ale damy cokolwiek, żeby nie wywalić się na None
            return "Chorus_0"
        return random.choice(any_segs)

    # ---- bez zmian: wczytywanie anotacji ----
    def read_annotations(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotations_path)
        df = df[df["type"].isin(self.music_parts)]
        df = df.dropna(subset=["filename"])

        counts = df.groupby(["salami_id", "type"]).size().reset_index(name="count")
        valid_combos = counts[counts["count"] > 1][["salami_id", "type"]]
        df = valid_combos.merge(df, on=["salami_id", "type"], how="left")

        df["segment_index"] = df.groupby(["salami_id", "type"]).cumcount()
        df["counted_type"] = df["type"] + "_" + df["segment_index"].astype(str)

        df = df.pivot(
            index=["salami_id", "filename"],
            columns="counted_type",
            values=["beginning_time", "end_time"],
        )
        df.columns = df.columns.reorder_levels(order=[1, 0])
        df = df.reset_index()
        return df


if __name__ == "__main__":
    config = get_config()
    dataset = TripletRecommendationDataset(
        annotations_file=config["annotations_file"],
        music_dir=config["music_dir"],
        temp_dir=config["temp_dir"],
        music_parts=config["music_parts"],
    )
    print(f"Liczba tripletów (par A/P) w tej konfiguracji: {len(dataset)}")
    anchor_img, positive_img, negative_img = dataset[0]
    print(f"A shape: {anchor_img.shape}, P shape: {positive_img.shape}, N shape: {negative_img.shape}")
