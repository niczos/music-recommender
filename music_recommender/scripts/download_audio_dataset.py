import os
from yt_dlp import YoutubeDL
import csv

# Ścieżki wejściowe (można zmienić w razie potrzeby lub przerobić na argumenty skryptu)
PAIRINGS_CSV = "salami_youtube_pairings.csv"
TRAIN_META_CSV = "metadata.csv"
VAL_META_CSV = "val_metadata.csv"

# Ścieżka wyjściowa dla danych (folder do którego zostaną pobrane pliki)
OUTPUT_DIR = "./data"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")

# Utwórz foldery wyjściowe jeśli nie istnieją
os.makedirs(AUDIO_DIR, exist_ok=True)

# Wczytaj mapowanie SALAMI -> YouTube z pliku pairings
import pandas as pd
pairings_df = pd.read_csv(PAIRINGS_CSV)
pairings_dict = {}
for _, row in pairings_df.iterrows():
    sid = int(row['salami_id'])
    pairings_dict[sid] = {
        'youtube_id': str(row['youtube_id']),
        'onset_in_salami': float(row['onset_in_salami']),
        'coverage': float(row['coverage'])
    }

# Wczytaj metadata i val_metadata (jako stringi, żeby zachować formatowanie dokładnie)
train_df_str = pd.read_csv(TRAIN_META_CSV, dtype=str)
val_df_str = pd.read_csv(VAL_META_CSV, dtype=str)
# Równolegle wczytaj te pliki również jako numeryczne (dla obliczeń filtrujących)
train_df = pd.read_csv(TRAIN_META_CSV)
val_df = pd.read_csv(VAL_META_CSV)

# Filtrowanie segmentów poza pokryciem audio
to_drop_train = []
for idx, row in train_df.iterrows():
    sid = int(row['salami_id'])
    end_time = float(row['end_time'])
    if sid in pairings_dict:
        salami_cov_end = pairings_dict[sid]['onset_in_salami'] + pairings_dict[sid]['coverage']
        if end_time > salami_cov_end:
            to_drop_train.append(idx)
to_drop_val = []
for idx, row in val_df.iterrows():
    sid = int(row['salami_id'])
    end_time = float(row['end_time'])
    if sid in pairings_dict:
        salami_cov_end = pairings_dict[sid]['onset_in_salami'] + pairings_dict[sid]['coverage']
        if end_time > salami_cov_end:
            to_drop_val.append(idx)
# Usuń wskazane wiersze z dataframów *tekstowych* (aby zachować format oryginalny dla pozostałych)
if to_drop_train:
    train_df_str.drop(index=to_drop_train, inplace=True)
if to_drop_val:
    val_df_str.drop(index=to_drop_val, inplace=True)
# Reset indexów po usunięciu (żeby zapisać poprawnie, bez przeskoków indeksów)
train_df_str.reset_index(drop=True, inplace=True)
val_df_str.reset_index(drop=True, inplace=True)

# Zidentyfikuj wszystkie unikalne utwory do pobrania (ze zbioru treningowego i walidacyjnego)
train_track_ids = set(train_df_str['salami_id'].astype(int).unique())
val_track_ids = set(val_df_str['salami_id'].astype(int).unique())
all_track_ids = train_track_ids.union(val_track_ids)

# Przygotuj listę utworów do pobrania wraz z ich YouTube ID i tytułem (filename)
tracks_to_download = []
# Sklej dane z train i val, grupując po utworze, aby wziąć po jednym przykładzie (tytuł/youtube_id) na utwór
combined_df = pd.concat([train_df_str[['salami_id', 'youtube_id', 'filename']],
                         val_df_str[['salami_id', 'youtube_id', 'filename']]])
grouped = combined_df.groupby('salami_id', as_index=False).first()
for _, row in grouped.iterrows():
    sid = int(row['salami_id'])
    if sid in all_track_ids:
        yt_id = str(row['youtube_id'])
        title = str(row['filename'])
        tracks_to_download.append({'salami_id': sid, 'youtube_id': yt_id, 'old_title': title})

# Konfiguracja yt_dlp do pobierania audio
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': os.path.join(AUDIO_DIR, '%(title)s.%(ext)s'),  # nazwa pliku wg tytułu video
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192'
    }],
    'quiet': False,  # Ustaw na True, by ukryć logi yt_dlp
    'nocheckcertificate': True
}

print(f"Pobieranie {len(tracks_to_download)} utworów do folderu '{AUDIO_DIR}'...")
failed_tracks = []

with YoutubeDL(ydl_opts) as ydl:
    for track in tracks_to_download:
        sid = track['salami_id']
        youtube_id = track['youtube_id']
        old_title = track['old_title']
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        try:
            info = ydl.extract_info(url, download=True)
        except Exception as e:
            print(f"** Błąd pobierania utworu {sid} (YouTube ID: {youtube_id}): {e}")
            failed_tracks.append(sid)
            continue
        # Uzyskaj aktualny tytuł video z informacji zwróconych przez yt_dlp
        new_title = info.get('title', None)
        if new_title is None:
            new_title = old_title  # w razie braku informacji, pozostań przy starym tytule
        # Sprawdź, czy tytuł uległ zmianie
        if new_title != old_title:
            # Zaktualizuj w strukturach metadata (train/val) nazwę pliku dla tego utworu
            train_mask = train_df_str['salami_id'].astype(int) == sid
            val_mask = val_df_str['salami_id'].astype(int) == sid
            if train_mask.any():
                train_df_str.loc[train_mask, 'filename'] = new_title
            if val_mask.any():
                val_df_str.loc[val_mask, 'filename'] = new_title
            print(f"(Uwaga: Zaktualizowano tytuł utworu {sid}: '{old_title}' -> '{new_title}')")

# Usuń z dataframów segmenty utworów, które nie zostały pobrane (np. film niedostępny)
if failed_tracks:
    print(f"Usuwanie z metadanych {len(failed_tracks)} utworów, których nie udało się pobrać...")
    train_df_str = train_df_str[~train_df_str['salami_id'].astype(int).isin(failed_tracks)].reset_index(drop=True)
    val_df_str = val_df_str[~val_df_str['salami_id'].astype(int).isin(failed_tracks)].reset_index(drop=True)

# Zapisz wyjściowe pliki CSV (metadata.csv i val_metadata.csv) do folderu wyjściowego
train_csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
val_csv_path = os.path.join(OUTPUT_DIR, "val_metadata.csv")
# Zapis z zachowaniem nagłówków i bez dodatkowego indeksu
train_df_str.to_csv(train_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
val_df_str.to_csv(val_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

print("Zakończono pobieranie i przygotowanie danych.")
print(f"Liczba segmentów (train): {len(train_df_str)}; liczba segmentów (val): {len(val_df_str)}.")
print(f"Pliki zapisane w folderze: {OUTPUT_DIR}")
