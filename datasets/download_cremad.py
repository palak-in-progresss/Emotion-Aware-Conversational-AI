import os
import sys
import json
import urllib.request
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import logger

EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

def download_single_file(item, target_dir, emotion):
    fname = item["name"]
    local_path = os.path.join(target_dir, fname)
    parts = fname.replace(".wav", "").split("_")
    actor_id = parts[0]

    if not os.path.exists(local_path):
        download_url = item["download_url"]
        try:
            req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as d_resp:
                content = d_resp.read()
            with open(local_path, "wb") as f:
                f.write(content)
        except Exception as e:
            return None

    return {
        "filepath": local_path,
        "actor_id": f"crema_{actor_id}",
        "emotion": emotion,
        "dataset": "crema_d"
    }

def download_crema_d_subset(target_dir="datasets/crema_d", max_per_emotion=200, max_workers=25):
    """
    Downloads a balanced subset of CREMA-D audio clips in parallel from GitHub.
    """
    os.makedirs(target_dir, exist_ok=True)
    metadata_csv = os.path.join(target_dir, "metadata.csv")

    if os.path.exists(metadata_csv):
        df = pd.read_csv(metadata_csv)
        if len(df) >= max_per_emotion * len(EMOTION_MAP):
            logger.info(f"CREMA-D subset already exists with {len(df)} files at {metadata_csv}")
            return df

    logger.info("Fetching CREMA-D file listing from GitHub API...")
    url = "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/contents/AudioWAV?per_page=1000"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(req) as resp:
            files_info = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to fetch CREMA-D file list: {e}")
        return pd.DataFrame()

    counts = {emo: 0 for emo in EMOTION_MAP.values()}
    to_download = []

    for item in files_info:
        fname = item["name"]
        if not fname.endswith(".wav"):
            continue

        parts = fname.replace(".wav", "").split("_")
        if len(parts) < 3:
            continue

        emo_code = parts[2]
        if emo_code not in EMOTION_MAP:
            continue

        emotion = EMOTION_MAP[emo_code]
        if counts[emotion] >= max_per_emotion:
            continue

        counts[emotion] += 1
        to_download.append((item, emotion))

        if all(c >= max_per_emotion for c in counts.values()):
            break

    logger.info(f"Downloading {len(to_download)} files in parallel ({max_workers} workers)...")
    downloaded = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_single_file, item, target_dir, emo) for item, emo in to_download]
        for future in as_completed(futures):
            res = future.result()
            if res:
                downloaded.append(res)

    df = pd.DataFrame(downloaded)
    df.to_csv(metadata_csv, index=False)
    logger.info(f"Successfully downloaded {len(df)} CREMA-D audio clips to {target_dir}")
    return df

if __name__ == "__main__":
    download_crema_d_subset()
