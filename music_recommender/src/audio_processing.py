import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mel_spectrogram_fixed(
    audio_path,
    sr=22050,
    n_mels=128,
    hop_length=512,
    max_frames=512,
):
    """
    Tworzy mel-spektrogram:
    - log-Mel,
    - przycięcie lub zero-padding do max_frames,
    - zwraca numpy array (n_mels, max_frames)
    """

    # ---- Wczytaj audio ----
    y, sr = librosa.load(audio_path, sr=sr)

    # ---- Mel-spektrogram ----
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # ---- Przycinanie / padding ----
    frames = S_db.shape[1]

    if frames > max_frames:
        # bierzemy środkowe okno – zwykle stabilniejsze niż początek/koniec
        start = (frames - max_frames) // 2
        S_db = S_db[:, start:start + max_frames]

    elif frames < max_frames:
        pad_width = max_frames - frames
        S_db = np.pad(
            S_db,
            pad_width=((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=-80.0  # reprezentuje ciszę
        )

    return S_db


def save_spectrogram_png(S_db, output_path, target_size=(224, 224)):
    """
    Zapisuje spektrogram jako PNG 224×224:
    - bez osi,
    - bez marginesów,
    - z resize po zapisaniu.
    """

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_db, cmap='magma')
    plt.axis("off")
    plt.tight_layout(pad=0)

    # Tymczasowy zapis
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Resize PNG do 224x224
    img = Image.open(output_path)
    img = img.resize(target_size, Image.BILINEAR)
    img.save(output_path)


def process_audio_folder(audio_dir, output_dir, max_frames=512):
    """
    Przechodzi po wszystkich .wav i generuje spektrogramy PNG
    z pad/crop → resize.
    """

    ensure_dir(output_dir)

    files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]

    for file in files:
        in_path = os.path.join(audio_dir, file)
        base = file.replace(".wav", "")
        out_path = os.path.join(output_dir, base + ".wav.png")

        try:
            S_db = mel_spectrogram_fixed(in_path, max_frames=max_frames)
            save_spectrogram_png(S_db, out_path)
        except Exception as e:
            print(f"[ERROR] {file}: {e}")


if __name__ == "__main__":
    AUDIO_DIR = "C:/Users/nikaj/source/repos/music-recommender-system/data/new/cutted"
    OUTPUT_DIR = "C:/Users/nikaj/source/repos/music-recommender-system/data/new/new_spectrograms"

    process_audio_folder(AUDIO_DIR, OUTPUT_DIR, max_frames=512)
