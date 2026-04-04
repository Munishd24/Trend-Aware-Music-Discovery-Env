import os
import csv
import json
import random
import urllib.request

# Configuration
DATASET_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
DATASET_FILE = "dataset.csv"
OUTPUT_FILE = "server/catalog.json"
TARGET_SAMPLES = 1000
MIN_POPULARITY = 60

def get_vibe(energy, valence):
    """
    High energy (>0.7) + High valence (>0.6) = "party"
    High energy (>0.7) + Low valence (<=0.6) = "hyped"
    Low energy (<=0.5) + Low valence (<=0.4) = "sad"
    Low energy (<=0.5) + High valence (>0.4) = "relaxed"
    Else = "focus"
    """
    if energy > 0.7 and valence > 0.6:
        return "party"
    elif energy > 0.7 and valence <= 0.6:
        return "hyped"
    elif energy <= 0.5 and valence <= 0.4:
        return "sad"
    elif energy <= 0.5 and valence > 0.4:
        return "relaxed"
    else:
        return "focus"

def build_catalog():
    print(f"Downloading from {DATASET_URL}...")
    if not os.path.exists(DATASET_FILE):
        urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)
    
    print("Parsing CSV...")
    filtered_songs = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pop = float(row.get("popularity", 0))
                if pop > MIN_POPULARITY:
                    filtered_songs.append(row)
            except ValueError:
                pass
                
    print(f"Found {len(filtered_songs)} songs with popularity > {MIN_POPULARITY}")
    
    # Sample 1000 songs
    random.seed(42) # Replicable sampling
    sampled = random.sample(filtered_songs, min(TARGET_SAMPLES, len(filtered_songs)))
    
    # Format according to spec
    catalog = []
    for row in sampled:
        energy = float(row.get("energy", 0))
        valence = float(row.get("valence", 0))
        pop = float(row.get("popularity", 0))
        
        song = {
            "id": row.get("track_id", ""),
            "title": row.get("track_name", ""),
            "artist": row.get("artists", ""),
            "genre": row.get("track_genre", ""),
            "trend_velocity": round(pop / 100.0, 2),
            "vibe": get_vibe(energy, valence)
            # source_media and trend_age_days are injected dynamically at runtime
        }
        catalog.append(song)
        
    print(f"Writing {len(catalog)} songs to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
        
    print("Success!")

if __name__ == "__main__":
    build_catalog()
