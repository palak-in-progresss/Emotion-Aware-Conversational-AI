import os
import sys
import time
import zipfile
import urllib.request

RAVDESS_URL = "https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content"
TARGET_DIR = os.path.abspath("datasets/real_ravdess")
ZIP_PATH = os.path.join(TARGET_DIR, "ravdess_speech.zip")

def download_file(url, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"Downloading real RAVDESS from:\n{url}\nto {target_path}...")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB chunks
        downloaded = 0
        last_print = time.time()
        
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)
            
            if time.time() - last_print > 1.5 or downloaded == total_size:
                percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                print(f"Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)")
                last_print = time.time()

def extract_ravdess(zip_path, target_dir):
    print(f"Extracting {zip_path} to {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction completed successfully!")

def verify_dataset(target_dir):
    wav_files = []
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if f.endswith('.wav') and f.startswith('03-01-'):
                wav_files.append(os.path.join(root, f))
                
    actors = set()
    emotions = set()
    for w in wav_files:
        parts = os.path.basename(w).split('-')
        if len(parts) == 7:
            emotions.add(parts[2])
            actors.add(parts[6].replace('.wav', ''))
            
    print("="*60)
    print("      REAL RAVDESS DATASET VERIFICATION      ")
    print("="*60)
    print(f"Total Real Audio WAV Files: {len(wav_files)}")
    print(f"Unique Actor IDs Count:     {len(actors)} (Actors: {sorted(list(actors))})")
    print(f"Unique Emotion Codes Count: {len(emotions)} (Codes: {sorted(list(emotions))})")
    print("="*60)
    return len(wav_files) == 1440 and len(actors) == 24

if __name__ == "__main__":
    if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) < 200000000:
        download_file(RAVDESS_URL, ZIP_PATH)
        
    extract_ravdess(ZIP_PATH, TARGET_DIR)
    is_valid = verify_dataset(TARGET_DIR)
    if is_valid:
        print("REAL RAVDESS DATASET FULLY VERIFIED (1,440 WAV files across 24 actors)!")
    else:
        print("WARNING: Dataset verification count mismatch!")
