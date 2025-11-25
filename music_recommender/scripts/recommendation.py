"""
Music Recommendation System Survey Script

This script randomly selects a reference music fragment (e.g., the first verse of a song) and finds the top 3 most similar tracks using a recommendation model (here simulated with random similarities). It then prepares four audio clips (one reference + three recommendations, in random order) for the user to evaluate. The user listens to each clip and ranks them by personal preference. The code records the user's rankings along with the model's similarity scores into a CSV file for later analysis.

Requirements:
- A folder with audio files for each track (e.g., WAV files named by track ID).
- CSV files with metadata (track IDs, segment times, etc.).
- (Optional) A trained recommendation model or precomputed embeddings to determine similarity (the code currently uses random selection as a placeholder).

Instructions:
1. Place your audio files in `AUDIO_DIR` (default "./audio"). Adjust `AUDIO_EXT` if needed (e.g., ".wav" or ".mp3").
2. Ensure you have `metadata.csv` (and optionally `val_metadata.csv`) in the working directory, or update the paths below.
3. Run the script. It will output four audio clip file paths (in random order).
4. Open/listen to each clip, then input your ranking from favorite (1) to least favorite (4).
5. The results (user rankings + model similarities) will be appended to `survey_results.csv`.
6. Repeat for multiple recommendations or users as needed. (The CSV will accumulate results.)

Note: 
- If multiple users evaluate the *same* set of clips, you can set a fixed random seed or specify a particular reference track to ensure consistency between runs.
- If audio playback is not available, you could use spectrogram images instead of audio clips for evaluation (not implemented here).
"""

import os
import random
import csv
import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml

# Configuration: adjust paths if needed
AUDIO_DIR = "./audio"              # Directory containing audio files for tracks
AUDIO_EXT = ".wav"                # Audio file extension (e.g., ".wav" or ".mp3")
METADATA_CSV = "metadata.csv"     # Path to metadata CSV (segments for all tracks)
VAL_METADATA_CSV = "val_metadata.csv"  # Path to validation set metadata (if available)
CLIPS_DIR = "survey_clips"        # Directory to save the 4 clips for listening
RESULT_CSV = "survey_results.csv" # Output CSV file for survey results

# Load optional config from YAML if provided (for overriding paths, etc.)
config_path = "config.yaml"
if os.path.isfile(config_path):
    with open(config_path, 'r') as cf:
        config = yaml.safe_load(cf)
    AUDIO_DIR = config.get("audio_dir", AUDIO_DIR)
    METADATA_CSV = config.get("metadata_csv", METADATA_CSV)
    VAL_METADATA_CSV = config.get("val_metadata_csv", VAL_METADATA_CSV)
    # You may also include model parameters in the config if needed.

# Load metadata
try:
    meta_df = pd.read_csv(METADATA_CSV)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Metadata file not found: {METADATA_CSV}") from e

if os.path.isfile(VAL_METADATA_CSV):
    val_df = pd.read_csv(VAL_METADATA_CSV)
else:
    # If no separate val set, use the full metadata for choosing reference
    val_df = meta_df

# Define a helper function to get the first meaningful segment of a track (preferably a verse)
def get_first_content_segment(track_id: int, df: pd.DataFrame):
    """Return the first significant segment (preferably 'Verse', else 'Chorus', etc.) for the given track."""
    segments = df[df['salami_id'] == track_id]
    # Exclude non-content segments
    segments = segments[~segments['type'].isin(['Silence', 'End'])]
    if segments.empty:
        return None
    # Prefer a Verse segment if available
    verses = segments[segments['type'].str.contains('Verse', case=False, na=False)]
    if not verses.empty:
        first_verse = verses.loc[verses['beginning_time'].idxmin()]
        return first_verse
    # If no verse, try Chorus
    choruses = segments[segments['type'].str.contains('Chorus', case=False, na=False)]
    if not choruses.empty:
        first_chorus = choruses.loc[choruses['beginning_time'].idxmin()]
        return first_chorus
    # Otherwise, take the earliest segment of any type (e.g., Intro, Solo, etc.)
    first_segment = segments.loc[segments['beginning_time'].idxmin()]
    return first_segment

# Function to get top-N similar tracks using the recommendation model (placeholder uses random scores)
def get_top_similar_tracks(ref_segment, candidate_track_ids, top_n=3):
    """
    Compute similarity between the reference segment and candidate tracks.
    This is a placeholder implementation using random scores.
    Replace this with your model's actual similarity computation.
    """
    # TODO: Integrate your model here. For example:
    # ref_track_id = int(ref_segment['salami_id'])
    # Load or compute embedding for ref_segment (e.g., using the model and audio data).
    # Compute embeddings for each candidate track (e.g., for a representative segment of each).
    # Calculate similarity scores (or distances) between ref_embedding and each candidate_embedding.
    # Sort candidates by similarity (or distance) and return the top N.
    # The structure below assumes higher score = more similar.
    scores = np.random.rand(len(candidate_track_ids))
    top_indices = np.argsort(scores)[-top_n:][::-1]  # indices of top N scores (descending)
    top_track_ids = [candidate_track_ids[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]
    return top_track_ids, top_scores

# Prompt for user identifier (optional)
user_name = input("Enter your name or ID (leave blank if you prefer not to provide one): ").strip()

# Select a random reference track from the validation set
val_tracks = val_df['salami_id'].unique().tolist()
random.shuffle(val_tracks)
ref_track_id = None
ref_segment = None
for tid in val_tracks:
    seg = get_first_content_segment(tid, val_df)
    if seg is None:
        continue
    # Prefer a track that has a 'Verse' segment as reference (if available)
    if str(seg['type']).lower().startswith('verse'):
        ref_track_id = tid
        ref_segment = seg
        break
# If no track with a Verse was found (very unlikely), just pick the first available track
if ref_track_id is None:
    for tid in val_tracks:
        seg = get_first_content_segment(tid, val_df)
        if seg is not None:
            ref_track_id = tid
            ref_segment = seg
            break

if ref_track_id is None or ref_segment is None:
    raise RuntimeError("No valid reference track could be selected from the dataset.")

# Get top 3 recommended similar tracks using the model (currently random)
candidate_ids = [int(t) for t in meta_df['salami_id'].unique() if t != ref_track_id]
top_track_ids, top_scores = get_top_similar_tracks(ref_segment, candidate_ids, top_n=3)

# Build recommendation list with segment info for each recommended track
recommendations = []
for track_id, score in zip(top_track_ids, top_scores):
    seg = get_first_content_segment(track_id, meta_df)
    if seg is None:
        # If no meaningful segment (should not happen if data is good, but just in case, skip this track)
        continue
    rec_info = {
        "track_id": int(track_id),
        "track_name": seg['filename'],
        "segment_type": seg['type'],
        "start_time": float(seg['beginning_time']),
        "end_time": float(seg['end_time']),
        "similarity": float(score),
        "is_ref": False
    }
    recommendations.append(rec_info)

# (Optional) If fewer than 3 recommendations found (e.g., if some were skipped), fill from next best candidates
if len(recommendations) < 3:
    # Sort all candidates by score (descending) and pick next best until we have 3
    all_candidates_sorted = sorted(zip(candidate_ids, np.random.rand(len(candidate_ids))), key=lambda x: x[1], reverse=True)
    # (Using random scores here as placeholder; replace with actual model scores if available)
    for cand_id, cand_score in all_candidates_sorted:
        if cand_id == ref_track_id or any(rec['track_id'] == cand_id for rec in recommendations):
            continue
        seg = get_first_content_segment(cand_id, meta_df)
        if seg is None:
            continue
        rec_info = {
            "track_id": int(cand_id),
            "track_name": seg['filename'],
            "segment_type": seg['type'],
            "start_time": float(seg['beginning_time']),
            "end_time": float(seg['end_time']),
            "similarity": float(cand_score),
            "is_ref": False
        }
        recommendations.append(rec_info)
        if len(recommendations) == 3:
            break

# Prepare the reference track info
reference_info = {
    "track_id": int(ref_track_id),
    "track_name": ref_segment['filename'],
    "segment_type": ref_segment['type'],
    "start_time": float(ref_segment['beginning_time']),
    "end_time": float(ref_segment['end_time']),
    "similarity": None,
    "is_ref": True
}

# Combine reference and recommendations, then shuffle for blind evaluation
items = [reference_info] + recommendations
random.shuffle(items)

# Create directory for output clips and clear any previous clips
os.makedirs(CLIPS_DIR, exist_ok=True)
for f in os.listdir(CLIPS_DIR):
    if f.endswith(".wav"):
        os.remove(os.path.join(CLIPS_DIR, f))

# Generate audio clips for each item
option_map = {}  # map from option number to item info
for idx, item in enumerate(items, start=1):
    option_map[idx] = item
    track_id = item['track_id']
    # Construct audio file path for the track (try the specified extension, then an alternative if not found)
    audio_file = os.path.join(AUDIO_DIR, f"{track_id}{AUDIO_EXT}")
    if not os.path.isfile(audio_file):
        alt_ext = ".mp3" if AUDIO_EXT != ".mp3" else ".wav"
        alt_file = os.path.join(AUDIO_DIR, f"{track_id}{alt_ext}")
        if os.path.isfile(alt_file):
            audio_file = alt_file
        else:
            print(f"Warning: Audio file for track {track_id} not found. Skipping this item.")
            continue
    # Load full track audio
    waveform, sr = torchaudio.load(audio_file)
    # Compute sample indices for the segment snippet
    start_sample = int(item['start_time'] * sr)
    end_sample = int(item['end_time'] * sr)
    if end_sample > waveform.shape[1]:
        end_sample = waveform.shape[1]
    if start_sample > waveform.shape[1]:
        start_sample = 0
    snippet_waveform = waveform[:, start_sample:end_sample]
    # Save the snippet to a new WAV file
    out_path = os.path.join(CLIPS_DIR, f"option{idx}.wav")
    torchaudio.save(out_path, snippet_waveform, sr)
    # Print the option for the user
    print(f"{idx}. {out_path}")

# Instructions for the user to proceed with listening and ranking
print("\nPlease listen to the above 4 audio clips (they are listed in random order).")
print("After listening, rank the clips from 1 (most preferred) to 4 (least preferred).")
print("Enter the order of the option numbers from most liked to least liked, e.g., '2 1 3 4'.\n")

# Get user ranking input and validate it
order = None
while True:
    user_input = input("Your preferred order of the clips (e.g., 2 1 3 4): ")
    # Normalize the input by removing commas and extra spaces
    user_input = user_input.replace(",", " ").strip()
    parts = user_input.split()
    if len(parts) != len(items):
        print(f"Please enter exactly {len(items)} numbers separated by spaces.")
        continue
    try:
        order = [int(x) for x in parts]
    except ValueError:
        print("Invalid input. Please enter numeric option labels (e.g., 1 2 3 4).")
        continue
    if sorted(order) != list(range(1, len(items) + 1)):
        print(f"Invalid order. Please use each number from 1 to {len(items)} exactly once.")
        continue
    # If we reach here, input is valid
    break

# Map each option number to its rank (1 = most preferred)
rank_map = {opt: rank for rank, opt in enumerate(order, start=1)}

# Prepare data for CSV logging
# Find the reference item and its rank
ref_item = next((info for opt, info in option_map.items() if info.get('is_ref')), None)
ref_rank = None
if ref_item:
    # Find which option number was the reference
    ref_option_num = next((opt for opt, info in option_map.items() if info.get('is_ref')), None)
    if ref_option_num is not None:
        ref_rank = rank_map.get(ref_option_num)

# Prepare recommended items output (rec1, rec2, rec3) in order of model similarity (descending)
# We will use the original recommendations list (which is sorted by similarity descending)
rec_outputs = []
for rec in recommendations:
    rec_id = rec['track_id']
    rec_name = rec['track_name']
    rec_type = rec['segment_type']
    rec_sim = rec['similarity']
    # Determine which option number this recommendation was presented as (to get user rank)
    rec_option_num = next((opt for opt, info in option_map.items() 
                            if not info.get('is_ref') and info['track_id'] == rec_id), None)
    rec_rank = rank_map.get(rec_option_num)
    rec_outputs.append((rec_id, rec_name, rec_type, rec_sim, rec_rank))

# Define CSV header if creating file new
headers = [
    "user",
    "reference_track_id", "reference_track_name", "reference_segment_type", "reference_rank",
    "rec1_track_id", "rec1_track_name", "rec1_segment_type", "rec1_similarity", "rec1_rank",
    "rec2_track_id", "rec2_track_name", "rec2_segment_type", "rec2_similarity", "rec2_rank",
    "rec3_track_id", "rec3_track_name", "rec3_segment_type", "rec3_similarity", "rec3_rank"
]

# Assemble the row for this survey result
row = []
row.append(user_name if user_name else "")
if ref_item:
    row.extend([
        ref_item['track_id'], 
        ref_item['track_name'], 
        ref_item['segment_type'], 
        ref_rank
    ])
else:
    # If no reference item found (should not happen), fill with N/A
    row.extend(["N/A", "N/A", "N/A", ""])
# Now add each recommended track's info
for rec_id, rec_name, rec_type, rec_sim, rec_rank in rec_outputs:
    # Format similarity score to 3 decimal places (or use percentage as needed)
    sim_val = f"{rec_sim:.3f}" if rec_sim is not None else ""
    row.extend([rec_id, rec_name, rec_type, sim_val, rec_rank])

# Write to the CSV file (append mode). Include header if file is new.
file_exists = os.path.isfile(RESULT_CSV)
with open(RESULT_CSV, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(headers)
    writer.writerow(row)

print(f"\nYour preferences have been recorded. Thank you!")
print(f"Survey results have been saved to '{RESULT_CSV}'. You can open this file to view all responses.")
# (You can use pandas to read the results later: e.g., pd.read_csv('survey_results.csv') )
