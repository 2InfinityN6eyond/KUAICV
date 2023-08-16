import os
from glob import glob
from tqdm.notebook import tqdm
import pandas as pd
import json

#import youtube_dl
import yt_dlp as youtube_dl
import librosa
from pydub import AudioSegment

import concurrent.futures 
import threading



def download_video(url, opt) :
    with youtube_dl.YoutubeDL(opt) as ydl :
        try :
            ydl.download([url])
            return {"url": url, "status":True}
        except Exception as e :
            return {"url": url, "status":False, "error": str(e)}

def download_parallel(url_list, opt, max_workers = 10) :
    log_data = []
    
    with concurrent.futures.ThreadPoolExecutor(
        max_workers = max_workers
    ) as executor :
        future_to_url = {
            executor.submit(
                download_video, url, opt
            ) : url for url in url_list
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_url),
            total = len(url_list)
        ) :
            url = future_to_url[future]
            result = future.result()
            log_data.append(result)

    return log_data


AUDIOSET_DATASET_PATH = "./audioset/"
MP3_FILE_ROOT_PATH = "./audioset_tmp"
AUDIOSET_METADATA_PATH = "./audioset_meta_data.json"

audioset_data = pd.read_csv(
    "./eval_segments.csv",
    delimiter=', ',
    skiprows = [0, 1, 2],
    names = ["vid", "stt", "ett", "label"]
)

with open(AUDIOSET_METADATA_PATH, "r") as fp :
    audioset_meta_data = json.load(fp)    

audioset_data

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '320',
    }],
    'outtmpl': os.path.join(MP3_FILE_ROOT_PATH, "%(id)s.%(ext)s"),
    'quiet':True,
    'external_downloader_args': ['-loglevel', 'panic']
}
slink = "https://www.youtube.com/watch?v="

audioset_data["url"] = slink + audioset_data["vid"]



results = download_parallel(
    audioset_data["url"], ydl_opts, max_workers = 50
)

with open(AUDIOSET_METADATA_PATH, "w") as fp :
    json.dump(
        results, fp
    )