"""
Cat vs Dog Dataset Populator
----------------------------
Fetches clean, high-resolution cat and dog photos:
  - Dogs: Dog CEO API
  - Cats: Wikimedia Commons CDN Thumbnails (iiurlwidth=256)
Populates:
  - data/train/cat/       (80 images)
  - data/train/dog/       (80 images)
  - data/validation/cat/  (20 images)
  - data/validation/dog/  (20 images)
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))

def download_image(url, target_path):
    if target_path.exists() and target_path.stat().st_size > 1500:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
            if len(data) > 1500:
                with open(target_path, "wb") as f:
                    f.write(data)
                return True
    except Exception:
        pass
    return False

def get_dog_urls(count):
    urls = set()
    while len(urls) < count:
        needed = min(50, count - len(urls) + 10)
        data = fetch_json(f"https://dog.ceo/api/breeds/image/random/{needed}")
        if data.get("status") == "success":
            for u in data.get("message", []):
                clean_u = u.split("?")[0].lower()
                if clean_u.endswith((".jpg", ".jpeg", ".png")):
                    urls.add(u)
    return list(urls)[:count]

def get_cat_urls(count):
    urls = set()
    gcmcontinue = ""
    while len(urls) < count:
        url = "https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category:Cats&gcmtype=file&gcmlimit=50&prop=imageinfo&iiprop=url&iiurlwidth=256&format=json"
        if gcmcontinue:
            url += f"&gcmcontinue={urllib.parse.quote(gcmcontinue)}"
        
        data = fetch_json(url)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [])
            if info:
                # Use thumburl which is served directly by the CDN
                thumb = info[0].get("thumburl") or info[0].get("url")
                if thumb:
                    clean_u = thumb.split("?")[0].lower()
                    if clean_u.endswith((".jpg", ".jpeg", ".png")):
                        urls.add(thumb)
        
        cont = data.get("continue", {})
        gcmcontinue = cont.get("gcmcontinue")
        if not gcmcontinue:
            break
            
    return list(urls)[:count]

def populate_folder(urls, target_dir, prefix):
    target_dir.mkdir(parents=True, exist_ok=True)
    placeholder = target_dir / "PLACE_IMAGES_HERE.txt"
    if placeholder.exists():
        try:
            placeholder.unlink()
        except Exception:
            pass

    success_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for idx, url in enumerate(urls, 1):
            clean_u = url.split("?")[0].lower()
            ext = ".jpg"
            if clean_u.endswith(".png"):
                ext = ".png"
            target_path = target_dir / f"{prefix}_{idx:03d}{ext}"
            futures[executor.submit(download_image, url, target_path)] = target_path

        for f in as_completed(futures):
            if f.result():
                success_count += 1
    return success_count

def main():
    print("=" * 60)
    print("🐾 Populating Cat & Dog Dataset...")
    print("=" * 60)

    train_cat_dir = DATA_DIR / "train" / "cat"
    train_dog_dir = DATA_DIR / "train" / "dog"
    val_cat_dir   = DATA_DIR / "validation" / "cat"
    val_dog_dir   = DATA_DIR / "validation" / "dog"

    print("Fetching image URLs...")
    dog_urls = get_dog_urls(100)
    print(f" -> Found {len(dog_urls)} dog images.")
    cat_urls = get_cat_urls(100)
    print(f" -> Found {len(cat_urls)} cat images.")

    print("\nDownloading Train Cats (80 images)...")
    c1 = populate_folder(cat_urls[:80], train_cat_dir, "cat")
    print(f" [OK] {c1} images saved to {train_cat_dir.relative_to(ROOT)}")

    print("\nDownloading Train Dogs (80 images)...")
    d1 = populate_folder(dog_urls[:80], train_dog_dir, "dog")
    print(f" [OK] {d1} images saved to {train_dog_dir.relative_to(ROOT)}")

    print("\nDownloading Validation Cats (20 images)...")
    c2 = populate_folder(cat_urls[80:100], val_cat_dir, "cat")
    print(f" [OK] {c2} images saved to {val_cat_dir.relative_to(ROOT)}")

    print("\nDownloading Validation Dogs (20 images)...")
    d2 = populate_folder(dog_urls[80:100], val_dog_dir, "dog")
    print(f" [OK] {d2} images saved to {val_dog_dir.relative_to(ROOT)}")

    print("\n" + "=" * 60)
    print("🎉 Dataset successfully ready!")
    print(f"  • Train Cats      : {len(list(train_cat_dir.glob('*.*')))}")
    print(f"  • Train Dogs      : {len(list(train_dog_dir.glob('*.*')))}")
    print(f"  • Validation Cats : {len(list(val_cat_dir.glob('*.*')))}")
    print(f"  • Validation Dogs : {len(list(val_dog_dir.glob('*.*')))}")
    print("=" * 60)

if __name__ == "__main__":
    main()
