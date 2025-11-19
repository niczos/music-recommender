import os
import argparse
from typing import List

import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def sanitize_filename(name: str) -> str:
    """
    Delikatne czyszczenie nazw, tak żeby:
    - nie rozwalać logiki,
    - ale usunąć znaki niebezpieczne dla systemu plików.
    To powinno być spójne z tym, co potem robisz przy spektrogramach.
    """
    return (
        name.replace(":", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("\"", "")
            .replace("?", "")
            .replace("*", "")
            .replace("<", "")
            .replace(">", "")
            .strip()
    )


def load_audio_for_track(audio_dir: str, base_name: str, try_exts=None, sr: int = 22050):
    """
    Próbuje znaleźć i załadować audio dla danego 'filename' z metadanych.
    Zakładamy, że plik audio nazywa się np. '<filename>.wav'.
    Możesz dopisać inne rozszerzenia w try_exts, jeśli używasz np. mp3.
    """
    if try_exts is None:
        try_exts = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    for ext in try_exts:
        candidate = os.path.join(audio_dir, base_name + ext)
        if os.path.exists(candidate):
            y, sr = librosa.load(candidate, sr=sr)
            return y, sr, candidate

    raise FileNotFoundError(
        f"Nie znaleziono pliku audio dla '{base_name}' z rozszerzeniami {try_exts} "
        f"w katalogu: {audio_dir}"
    )


def cut_segments_from_metadata(
    metadata_path: str,
    audio_dir: str,
    output_dir: str,
    types_to_keep: List[str],
    sr: int = 22050,
):
    """
    Tnie utwory na fragmenty na podstawie metadata.csv:
    - każdy wiersz opisuje segment (beginning_time, end_time, type),
    - segmenty są indeksowane per (salami_id, type) => Chorus_0, Chorus_1 itd,
    - zapisuje każdy fragment jako osobny .wav w output_dir.
    """

    print(f"[INFO] Wczytuję metadane z: {metadata_path}")
    df = pd.read_csv(metadata_path)

    # filtrujemy tylko wybrane typy (np. Chorus, Verse)
    df = df[df["type"].isin(types_to_keep)].copy()
    print(f"[INFO] Segmentów po filtrze type in {types_to_keep}: {len(df)}")

    # indeksowanie segmentów per (salami_id, type) tak jak w TripletRecommendationDataset
    df["segment_index"] = df.groupby(["salami_id", "type"]).cumcount()
    # np. "Chorus_0", "Verse_1"
    df["segment_name"] = df["type"] + "_" + df["segment_index"].astype(str)

    ensure_dir(output_dir)

    # dla oszczędności – ładowanie audio raz na (salami_id, filename)
    grouped = df.groupby(["salami_id", "filename"])

    for (salami_id, filename), group in tqdm(grouped, desc="Przetwarzanie utworów"):
        base_name = str(filename)
        sanitized = sanitize_filename(base_name)

        try:
            y, sr_loaded, audio_path = load_audio_for_track(audio_dir, base_name, sr=sr)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        total_duration = librosa.get_duration(y=y, sr=sr_loaded)

        for _, row in group.iterrows():
            seg_type = row["type"]
            seg_idx = int(row["segment_index"])
            seg_name = row["segment_name"]

            start_t = float(row["beginning_time"])
            end_t = float(row["end_time"])

            # lekkie zabezpieczenia, gdyby end_time wychodził poza długość
            start_t = max(0.0, start_t)
            end_t = min(total_duration, end_t)
            if end_t <= start_t:
                print(
                    f"[WARN] salami_id={salami_id}, segment={seg_name}: "
                    f"end_time <= beginning_time, pomijam."
                )
                continue

            start_idx = int(start_t * sr_loaded)
            end_idx = int(end_t * sr_loaded)

            fragment = y[start_idx:end_idx]

            out_filename = f"{sanitized}_{seg_name}.wav"
            out_path = os.path.join(output_dir, out_filename)

            try:
                sf.write(out_path, fragment, sr_loaded)
            except Exception as e:
                print(f"[ERROR] Nie udało się zapisać {out_path}: {e}")
                continue

    print(f"[DONE] Fragmenty zapisane w: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Wycinanie fragmentów na podstawie metadata.csv (SALAMI)."
    )
    parser.add_argument("--metadata", required=True, help="Ścieżka do metadata.csv lub val_metadata.csv")
    parser.add_argument("--audio_dir", required=True, help="Katalog z plikami audio (pełne utwory)")
    parser.add_argument("--output_dir", required=True, help="Katalog, gdzie zapisać pocięte fragmenty")
    parser.add_argument(
        "--types",
        nargs="+",
        default=["Chorus", "Verse"],
        help="Jakie typy segmentów zachować (np. Chorus Verse)",
    )
    parser.add_argument("--sr", type=int, default=22050, help="Docelowy sampling rate dla odczytu audio")

    args = parser.parse_args()

    cut_segments_from_metadata(
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        types_to_keep=args.types,
        sr=args.sr,
    )


if __name__ == "__main__":
    main()
