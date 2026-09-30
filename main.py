import os
import sys
import io
import shutil
import pickle
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set, Any

import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

# Windows terminal encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from deepface import DeepFace

# ============================================================
# LOGGING SETUP
# ============================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
fh = logging.FileHandler('organizer.log', encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================
DETECTOR: str = "retinaface"       # Fast detector (Reverted for speed)
EMBEDDING_MODEL: str = "Facenet512" # Face embedding model
MIN_CONFIDENCE: float = 0.99       # Extremely strict to avoid clothes/arms
MIN_FACE_SIZE: int = 300           # Only clear visible faces


def scan_photos(folder_path: Path) -> List[Path]:
    """Folder me se saari photo files dhundho"""
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    photos = []
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in valid_extensions:
            photos.append(file)
    return sorted(photos)


def extract_faces_and_embeddings(folder_path: Path, photos: List[Path]) -> Tuple[List[Dict[str, Any]], List[Path]]:
    """
    Har photo se faces nikalo aur unka embedding banao.
    Cache me mtime (modified time) use kiya hai for robust invalidation.
    """
    face_data: List[Dict[str, Any]] = []
    no_face_photos: List[Path] = []
    
    db_folder = folder_path.parent / "database"
    db_file = db_folder / "embeddings_cache.pkl"
    db_folder.mkdir(exist_ok=True)
    
    cache: Dict[str, Dict[str, Any]] = {}
    if db_file.exists():
        try:
            with open(db_file, "rb") as f:
                cache = pickle.load(f)
            logger.info(f"[Database] Loaded {len(cache)} scanned photos from memory! 🚀")
        except Exception as e:
            logger.warning(f"[Database] Cache load failed, starting fresh. Error: {e}")
            cache = {}
            
    new_data_added = False
    
    for i, photo_path in enumerate(photos):
        photo_name = photo_path.name
        
        try:
            mtime = photo_path.stat().st_mtime
        except Exception:
            mtime = 0.0

        # Check robust cache (mtime check)
        if photo_name in cache and cache[photo_name].get("mtime") == mtime:
            cached_info = cache[photo_name].get("faces")
            if cached_info == "no_face":
                no_face_photos.append(photo_path)
                logger.info(f"[{i+1}/{len(photos)}] Skipping: {photo_name} (Cached: No face ⏩)")
            else:
                face_data.extend(cached_info)
                logger.info(f"[{i+1}/{len(photos)}] Skipping: {photo_name} (Cached: {len(cached_info)} faces ⏩)")
            continue
            
        logger.info(f"[{i+1}/{len(photos)}] Scanning: {photo_name}...")

        try:
            embeddings = DeepFace.represent(
                img_path=str(photo_path),
                model_name=EMBEDDING_MODEL,
                detector_backend=DETECTOR,
                enforce_detection=False
            )

            valid_faces = []
            for e in embeddings:
                confidence = e.get("face_confidence", 0.0)
                face_area = e.get("facial_area", {})
                face_w = face_area.get("w", 0)
                face_h = face_area.get("h", 0)
                
                if confidence > MIN_CONFIDENCE and face_w >= MIN_FACE_SIZE and face_h >= MIN_FACE_SIZE:
                    valid_faces.append(e)

            if not valid_faces:
                logger.info(f"[{i+1}/{len(photos)}] {photo_name} - No valid face")
                no_face_photos.append(photo_path)
                cache[photo_name] = {"mtime": mtime, "faces": "no_face"}
            else:
                logger.info(f"[{i+1}/{len(photos)}] {photo_name} - {len(valid_faces)} face(s) found! ✅")
                photo_face_data = []
                for j, face in enumerate(valid_faces):
                    data = {
                        "photo": photo_path,
                        "embedding": face["embedding"],
                        "facial_area": face.get("facial_area", {}),
                        "face_index": j
                    }
                    face_data.append(data)
                    photo_face_data.append(data)
                    
                cache[photo_name] = {"mtime": mtime, "faces": photo_face_data}
                
            new_data_added = True

        except Exception as e:
            logger.error(f"Error scanning {photo_name}: {e}")
            no_face_photos.append(photo_path)
            cache[photo_name] = {"mtime": mtime, "faces": "no_face"}
            new_data_added = True

    if new_data_added:
        logger.info("\n   [Database] Saving new scans to memory...")
        try:
            with open(db_file, "wb") as f:
                pickle.dump(cache, f)
            logger.info("   [Database] Memory updated successfully! 💾")
        except Exception as e:
            logger.error(f"   [Database] Failed to save memory: {e}")

    return face_data, no_face_photos


def cluster_faces(face_data: List[Dict[str, Any]]) -> Tuple[Dict[str, Set[Path]], Dict[str, List[Dict[str, Any]]]]:
    """Clustering using Cosine similarity. Limits mentioned in README regarding scalability."""
    if not face_data:
        return {}, {}

    embeddings = np.array([f["embedding"] for f in face_data])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / norms

    cosine_distances = squareform(pdist(embeddings_normalized, metric="cosine"))

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.35,
        metric="precomputed",
        linkage="average"
    ).fit(cosine_distances)

    labels = clustering.labels_

    person_photos: Dict[str, Set[Path]] = {}
    person_faces: Dict[str, List[Dict[str, Any]]] = {}

    for i, label in enumerate(labels):
        person_name = f"Person_{label + 1}"

        if person_name not in person_photos:
            person_photos[person_name] = set()
            person_faces[person_name] = []

        person_photos[person_name].add(face_data[i]["photo"])
        person_faces[person_name].append({
            "photo": face_data[i]["photo"],
            "facial_area": face_data[i]["facial_area"]
        })

    final_photos: Dict[str, Set[Path]] = {}
    final_faces: Dict[str, List[Dict[str, Any]]] = {}
    unknown_photos: Set[Path] = set()
    unknown_face_list: List[Dict[str, Any]] = []
    person_counter = 1

    sorted_groups = sorted(person_photos.items(), key=lambda x: len(x[1]), reverse=True)

    for old_name, photos_set in sorted_groups:
        if len(photos_set) >= 2:
            new_name = f"Person_{person_counter}"
            final_photos[new_name] = photos_set
            final_faces[new_name] = person_faces[old_name]
            person_counter += 1
        else:
            unknown_photos.update(photos_set)
            unknown_face_list.extend(person_faces[old_name])

    if unknown_photos:
        final_photos["Unknown"] = unknown_photos
        final_faces["Unknown"] = unknown_face_list

    return final_photos, final_faces


def save_face_preview(photo_path: Path, face_info: Dict[str, Any], output_path: Path) -> None:
    try:
        area = face_info["facial_area"]
        x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)
        
        with Image.open(photo_path) as img:
            padding_x = int(w * 0.2)
            padding_y = int(h * 0.2)
            
            left = max(0, x - padding_x)
            top = max(0, y - padding_y)
            right = min(img.width, x + w + padding_x)
            bottom = min(img.height, y + h + padding_y)
            
            face_img = img.crop((left, top, right, bottom))
            face_img.thumbnail((300, 300))
            face_img.save(output_path, "JPEG", quality=85)
            
    except Exception as e:
        logger.error(f"Error saving face preview for {photo_path.name}: {e}")


def create_link_or_copy(src: Path, dst: Path) -> None:
    """Uses hardlink to save disk space, falls back to copy2 if hardlink fails."""
    try:
        # Creating a hard link (Space efficient)
        os.link(str(src), str(dst))
    except Exception as e:
        # Fallback to normal copy if hardlink fails (e.g., crossing drives)
        logger.debug(f"Hardlink failed for {src.name} -> {dst.name} ({e}), falling back to copy2.")
        shutil.copy2(src, dst)


def organize_photos(input_folder: Path, output_folder: Path, 
                    person_photos: Dict[str, Set[Path]], 
                    person_faces: Dict[str, List[Dict[str, Any]]], 
                    no_face_photos: List[Path]) -> int:
    
    if output_folder.exists():
        shutil.rmtree(output_folder, ignore_errors=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    total_copies = 0

    for person, photos_set in person_photos.items():
        person_folder = output_folder / person
        person_folder.mkdir(exist_ok=True)

        if person in person_faces and len(person_faces[person]) > 0:
            best_face = person_faces[person][0]
            preview_path = person_folder / "_face_preview.jpg"
            save_face_preview(best_face["photo"], best_face, preview_path)

        for photo_path in photos_set:
            dest_path = person_folder / photo_path.name
            create_link_or_copy(photo_path, dest_path)
            total_copies += 1
            
        logger.info(f"   {person}: {len(photos_set)} photos processed + face preview saved")

    if no_face_photos:
        no_face_folder = output_folder / "No_Faces"
        no_face_folder.mkdir(exist_ok=True)
        for photo_path in no_face_photos:
            dest_path = no_face_folder / photo_path.name
            create_link_or_copy(photo_path, dest_path)
            total_copies += 1
        logger.info(f"   No_Faces: {len(no_face_photos)} photos processed")

    return total_copies


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Photo Organizer - Group photos by faces locally.")
    parser.add_argument("--input", type=str, default="input_photos", help="Path to input photos folder")
    parser.add_argument("--output", type=str, default="organized_photos", help="Path to output photos folder")
    args = parser.parse_args()

    input_folder = Path(args.input).resolve()
    output_folder = Path(args.output).resolve()

    logger.info("==================================================")
    logger.info("   AI Photo Organizer (Pro Version)")
    logger.info("==================================================")

    if not input_folder.exists():
        logger.error(f"Input folder '{input_folder}' nahi mila. Folder create kijiye aur usme photos daaliye.")
        return

    logger.info("\nStep 1: Scanning photos...")
    photos = scan_photos(input_folder)
    
    if not photos:
        logger.warning(f"Koi photos nahi mili '{input_folder}' me.")
        return
        
    logger.info(f"   Found {len(photos)} photos.")

    logger.info("\nStep 2: Detecting faces & generating embeddings...")
    logger.info(f"   (Using {DETECTOR.upper()} + {EMBEDDING_MODEL} for best accuracy)")
    
    face_data, no_face_photos = extract_faces_and_embeddings(input_folder, photos)

    logger.info(f"\n   Summary:")
    logger.info(f"   Total faces detected: {len(face_data)}")
    logger.info(f"   Photos without faces: {len(no_face_photos)}")

    if not face_data:
        logger.info("\n   No faces found in any photo. Nothing to organize.")
        return

    logger.info("\nStep 3: Grouping similar faces (AI clustering)...")
    person_photos, person_faces = cluster_faces(face_data)

    person_groups = [k for k in person_photos if k != "Unknown"]
    logger.info(f"   Found {len(person_groups)} person group(s)")
    for person, photos_set in sorted(person_photos.items()):
        logger.info(f"   - {person}: {len(photos_set)} photos")

    logger.info(f"\nStep 4: Creating hardlinks & saving face previews...")
    total = organize_photos(input_folder, output_folder, person_photos, person_faces, no_face_photos)

    logger.info(f"\n   Done! {total} photo links/copies created.")
    logger.info(f"   Output folder: {output_folder}")

    logger.info("\n" + "=" * 50)
    logger.info("   Photo Organization Complete!")
    logger.info("   Original photos are SAFE and UNTOUCHED (using Hardlinks).")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
