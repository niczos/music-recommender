import os
import torchaudio

class AudioDataset:
    def __init__(self, csv_path: str, audio_dir: str, triplet: bool = False, transform=None):
        """
        Dataset audio dla modelu muzycznego.
        csv_path: ścieżka do pliku CSV z metadanymi segmentów (metadata.csv lub val_metadata.csv).
        audio_dir: ścieżka do katalogu z plikami audio.
        triplet: jeśli True, dataset zwraca triplet (anchor, positive, negative) dla treningu.
                 jeśli False, zwraca (audio, label) dla walidacji/testu.
        transform: opcjonalna transformacja do zastosowania na sygnał audio (np. MelSpectrogram).
        """
        self.audio_dir = audio_dir
        self.triplet = triplet
        self.transform = transform

        # Wczytanie metadanych segmentów z pliku CSV
        import pandas as pd
        data = pd.read_csv(csv_path)
        # Odfiltruj potencjalne segmenty typu "Silence" lub segmenty zerowej długości (jeśli jakimś cudem pozostały)
        # Konwertujemy DataFrame na listę segmentów
        self.segments = []
        for _, row in data.iterrows():
            seg_type = str(row['type'])
            # Pomijamy segmenty oznaczone jako cisza lub koniec utworu
            if seg_type.lower() == 'silence' or seg_type.lower() == 'end':
                continue
            # Obliczamy długość segmentu
            begin_time = float(row['beginning_time'])
            end_time = float(row['end_time'])
            if end_time - begin_time <= 0:
                # pomiń segmenty o zerowej lub ujemnej długości (nie powinny wystąpić po filtracji, ale na wszelki wypadek)
                continue
            # Zapisujemy informacje o segmencie
            segment_info = {
                'salami_id': int(row['salami_id']),
                'youtube_id': str(row['youtube_id']),
                'filename': str(row['filename']),
                'begin': begin_time,
                'end': end_time,
                'type': seg_type
            }
            self.segments.append(segment_info)
        # Grupowanie segmentów po utworach (salami_id) dla szybkiego dostępu podczas tworzenia tripletów
        self.segments_by_track = {}
        for idx, seg in enumerate(self.segments):
            track_id = seg['salami_id']
            if track_id not in self.segments_by_track:
                self.segments_by_track[track_id] = []
            self.segments_by_track[track_id].append(idx)
        # Usuń ewentualne utwory, które po filtracji mają mniej niż 2 segmenty (nieprzydatne do triplet loss)
        if self.triplet:
            tracks_to_remove = []
            for track_id, idx_list in self.segments_by_track.items():
                if len(idx_list) < 2:
                    # oznacz utwór do usunięcia z datasetu (brak par pozytywnych)
                    tracks_to_remove.append(track_id)
            # Usuń segmenty należące do utworów z jednym segmentem
            if tracks_to_remove:
                self.segments = [seg for seg in self.segments if seg['salami_id'] not in tracks_to_remove]
                self.segments_by_track = {tid: idx_list for tid, idx_list in self.segments_by_track.items() 
                                          if tid not in tracks_to_remove}

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        """
        Zwraca:
        - jeśli self.triplet: (anchor_audio, positive_audio, negative_audio)
        - jeśli nie: (audio, label)
        gdzie audio to tensor waveform (1 x N), a label to ID utworu (int).
        """
        if not self.triplet:
            # Tryb walidacyjny – pojedynczy segment i label
            seg = self.segments[idx]
            audio_tensor = self._load_audio_segment(seg)
            label = seg['salami_id']
            return audio_tensor, label
        else:
            # Tryb treningowy – triplet anchor, positive, negative
            anchor_seg = self.segments[idx]
            anchor_track = anchor_seg['salami_id']
            # Wybierz losowy positive z tego samego utworu (inny segment niż anchor)
            pos_idx = idx
            if len(self.segments_by_track[anchor_track]) > 1:
                import random
                candidates = self.segments_by_track[anchor_track]
                # wybieramy randomowo inny segment z tego samego utworu
                pos_idx = random.choice(candidates)
                while pos_idx == idx:
                    pos_idx = random.choice(candidates)
            positive_seg = self.segments[pos_idx]
            # Wybierz losowy negative z innego utworu
            import random
            neg_track = anchor_track
            neg_idx = idx
            while neg_track == anchor_track:
                neg_idx = random.randrange(len(self.segments))
                neg_track = self.segments[neg_idx]['salami_id']
            negative_seg = self.segments[neg_idx]

            # Wczytaj dane audio dla anchor, positive, negative
            anchor_audio = self._load_audio_segment(anchor_seg)
            positive_audio = self._load_audio_segment(positive_seg)
            negative_audio = self._load_audio_segment(negative_seg)
            return anchor_audio, positive_audio, negative_audio

    def _load_audio_segment(self, segment):
        """Wczytuje z pliku odpowiedni wycinek audio określony przez segment (słownik segmentu)."""
        # Ścieżka do pliku audio (zakładamy rozszerzenie .mp3 dla wszystkich pobranych plików)
        audio_filename = f"{segment['filename']}.mp3"
        audio_path = os.path.join(self.audio_dir, audio_filename)
        # Wczytaj cały plik audio
        waveform, sample_rate = torchaudio.load(audio_path)
        # Jeśli audio jest stereo (2 kanały), zredukuj do mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Wytnij żądany fragment według begin/end (czasy w sekundach)
        start_sample = int(segment['begin'] * sample_rate)
        end_sample = int(segment['end'] * sample_rate)
        # Upewnij się, że nie wykraczamy poza długość sygnału
        if end_sample > waveform.size(1):
            end_sample = waveform.size(1)
        if start_sample < 0:
            start_sample = 0
        segment_waveform = waveform[:, start_sample:end_sample]
        # Zastosuj opcjonalną transformację (np. MelSpectrogram), jeśli podano
        if self.transform:
            segment_waveform = self.transform(segment_waveform)
        return segment_waveform
