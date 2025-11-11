import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import librosa
import matplotlib.pyplot as plt
import numpy as np
import cv2
import soundfile as sf
import os


class AudioSample:
    def __init__(self, filepath: str, temp_dir: str, sample_rate: int | None = None):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No such file: {filepath}")
        self.filepath = filepath
        if sample_rate is None:
            self.y, self.sample_rate = librosa.load(path=self.filepath)
        else:
            self.y, _ = librosa.load(path=self.filepath, sr=sample_rate)
            self.sample_rate = sample_rate
        self.temp_dir = temp_dir

    def to_spectrogram(self, beginning: float | None = None, end: float | None = None) -> np.ndarray:
        if beginning is not None and end is not None:
            audio_sample = self.y[int(beginning * self.sample_rate): int(end * self.sample_rate)]
        else:
            audio_sample = self.y.copy()
        # Extract Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_sample, sr=self.sample_rate)

        # Convert Decibels (Log Scale)
        raw_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        librosa.display.specshow(raw_spectrogram, cmap='viridis')
        plt.axis('off')
        # output_filename = os.path.splitext(os.path.basename(self.filepath))[0]
        # output_filepath = os.path.join(self.temp_dir, output_filename)
        # plt.savefig(output_filepath)
        # return cv2.imread(output_filepath + ".png")
        figure = plt.gcf()
        figure.set_size_inches(12, 1)
        figure.set_dpi(50)

        figure.canvas.draw()

        b = figure.axes[0].get_window_extent()
        img = np.array(figure.canvas.buffer_rgba())
        img = img[int(b.y0):int(b.y1), int(b.x0):int(b.x1)]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def to_mp3(self, name:str, output_dir: str, beginning: float | None = None, end: float | None = None) -> np.ndarray:
        if beginning is not None and end is not None:
            audio_sample = self.y[int(beginning * self.sample_rate): int(end * self.sample_rate)]
        else:
            audio_sample = self.y.copy()
        filepath = os.path.join(output_dir, f'{name}.wav')
        sf.write(filepath, audio_sample, self.sample_rate, format='wav', subtype='PCM_24')

if __name__ == "__main__":
    folder_path = r"/home/nika/music-recommender-system/sample_music/"
    temp_dir = r"C:\Users\skrzy\Music\sample_music"
    folder_path = r"C:\Users\skrzy\Music\sample_music"
    filename = "Ave Verum Corpus.wav"
    sample_path = os.path.join(folder_path, filename)
    audio_1 = AudioSample(filepath=sample_path, temp_dir=temp_dir)
    spec_1 = audio_1.to_mp3(output_dir=r"C:\Users\skrzy\Music", beginning=0, end=15)


    # for filename in os.listdir(folder_path):
    #     sample_path = os.path.join(folder_path, filename)
    #     if os.path.isfile(sample_path):
    #         audio_1 = AudioSample(filepath=sample_path)
    #         spec_1 = audio_1.to_spectrogram()
    # image = cv2.imread(r"C:\Users\skrzy\Music\sample_music\01 - The Golden Age [BeckďĽš Sea Change].png", cv2.IMREAD_UNCHANGED)
    # print(image)
