import os
import sys
import zipfile
import urllib.request

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"
TARGET_DIR = os.path.abspath("datasets/audio")

def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = (downloaded / total_size) * 100 if total_size > 0 else 0
    sys.stdout.write(f"\rDownloading RAVDESS: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)")
    sys.stdout.flush()

def download_and_extract_ravdess():
    os.makedirs(TARGET_DIR, exist_ok=True)
    zip_path = os.path.join(TARGET_DIR, "ravdess_speech.zip")
    
    print(f"Downloading official RAVDESS dataset from:\n{RAVDESS_URL}")
    try:
        urllib.request.urlretrieve(RAVDESS_URL, zip_path, reporthook=download_progress_hook)
        print("\nDownload complete! Extracting audio files...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TARGET_DIR)
            
        print(f"Extraction complete! Audio files extracted to '{TARGET_DIR}'.")
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
    except Exception as e:
        print(f"\nFailed to download/extract RAVDESS dataset: {e}")

if __name__ == "__main__":
    download_and_extract_ravdess()
