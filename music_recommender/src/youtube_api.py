import os
from typing import Any

import requests
import yt_dlp


def get_metadata(salami_id: int, salami_length: float, youtube_id: str, filename: str) -> list[
    dict[str, float | str | int]]:
    info = {'salami_id': salami_id, 'salami_length': salami_length, 'youtube_id': youtube_id, 'filename': filename}
    part_info = get_part_info(salami_id=salami_id)
    rows = []
    for i, part in enumerate(part_info):
        info.update(part)
        info.update({"end_time": (part_info[i + 1]["beginning_time"] if i + 1 != len(part_info) else salami_length)})
        rows.append(info.copy())
    return rows


def parse_music_parts(text: str) -> list[dict[str, Any]]:
    rows = text.split("\n")
    result = []
    for row in rows:
        try:
            sec, type_ = row.split("\t")
            result.append({"beginning_time": float(sec), "type": type_})
        except:
            continue
    return result


def get_part_info(salami_id: int) -> list[dict[str, Any]]:
    url = f"https://raw.githubusercontent.com/DDMAL/salami-data-public/master/annotations/{salami_id}/parsed/textfile1_functions.txt"
    txt_string = requests.get(url=url).text
    return parse_music_parts(txt_string)


def youtube_id_to_url(id: str) -> str:
    return 'https://www.youtube.com/watch?v=' + id


def download_audio(url, output_folder: str):
    with yt_dlp.YoutubeDL({'extract_audio': True, 'format': 'bestaudio',
                           'outtmpl': os.path.join(output_folder, '%(title)s.wav')}) as video:
        info_dict = video.extract_info(url, download=True)
        video_title = info_dict['title']
        print(video_title)
        # jeĹ›li plik istnieje pomiĹ„
        if any(fname.startswith(os.path.basename(url)) for fname in os.listdir(output_folder)):
            return video_title
        try:
            video.download([url])
        except yt_dlp.utils.DownloadError as e:
            print(f"Nie udaĹ‚o siÄ™ pobraÄ‡ pliku WAV dla {url}: {e}")
            # TODO napraw
            return None
    return video_title
