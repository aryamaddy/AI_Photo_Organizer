import os
import sys
import io
import shutil
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

# Windows terminal me emojis ke liye UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from deepface import DeepFace

# ============================================================
# SETTINGS - Yahan se aap apne paths change kar sakte hain
# ============================================================
INPUT_FOLDER = r"C:\Users\BIT PATNA\Desktop\AI_Photo_Organizer\input_photos"
OUTPUT_FOLDER = r"C:\Users\BIT PATNA\Desktop\AI_Photo_Organizer\organized_photos"

# AI Models - Best accuracy ke liye
DETECTOR = "retinaface"       # Face dhundhne ke liye (sabse accurate)
EMBEDDING_MODEL = "Facenet512" # Face ka code (embedding) banane ke liye

# Face confidence threshold - isse kam confidence wale faces ignore honge
MIN_CONFIDENCE = 0.95

# Minimum face size (pixels) - isse chota chehra ignore hoga
# Group photos me background ke chote chehre filter ho jayenge
MIN_FACE_SIZE = 250


# ============================================================
# STEP 1: Photos Dhundho
# ============================================================
def scan_photos(folder_path):
    """Folder me se saari photo files dhundho"""
    valid_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
    all_files = os.listdir(folder_path)
    photos = [f for f in all_files if f.lower().endswith(valid_extensions)]
    return sorted(photos)


# ============================================================
# STEP 2: Har Photo Me Faces Dhundho Aur Embedding Banao
# ============================================================
def extract_faces_and_embeddings(folder_path, photos):
    """
    Har photo se faces nikalo aur unka embedding (mathematical code) banao.
    
    Ye function 2 cheezein return karta hai:
    - face_data: List of dicts jisme har face ka embedding aur photo name hai
    - no_face_photos: Wo photos jinme koi chehra nahi mila
    """
    face_data = []
    no_face_photos = []

    for i, photo in enumerate(photos):
        photo_path = os.path.join(folder_path, photo)
        print(f"[{i+1}/{len(photos)}] Scanning: {photo}...", end=" ", flush=True)

        try:
            # DeepFace.represent ek hi baar me face detect + embedding dono karta hai
            embeddings = DeepFace.represent(
                img_path=photo_path,
                model_name=EMBEDDING_MODEL,
                detector_backend=DETECTOR,
                enforce_detection=False  # Agar face nahi mila toh crash mat karo
            )

            # Sirf confident AUR bade faces rakhna
            # Chote faces (background crowd) ko filter kar dena
            valid_faces = []
            for e in embeddings:
                confidence = e.get("face_confidence", 0)
                face_area = e.get("facial_area", {})
                face_w = face_area.get("w", 0)
                face_h = face_area.get("h", 0)
                
                # Face tabhi valid hai jab:
                # 1. Confidence 90% se zyada ho
                # 2. Face ka size minimum 80x80 pixels ho
                if confidence > MIN_CONFIDENCE and face_w >= MIN_FACE_SIZE and face_h >= MIN_FACE_SIZE:
                    valid_faces.append(e)

            if not valid_faces:
                print("No face")
                no_face_photos.append(photo)
            else:
                print(f"{len(valid_faces)} face(s) found!")
                for j, face in enumerate(valid_faces):
                    face_data.append({
                        "photo": photo,
                        "embedding": face["embedding"],
                        "facial_area": face.get("facial_area", {}),
                        "face_index": j
                    })

        except Exception as e:
            print(f"Error: {e}")
            no_face_photos.append(photo)

    return face_data, no_face_photos


# ============================================================
# STEP 3: Similar Faces Ko Group Karo (Clustering)
# ============================================================
def cluster_faces(face_data):
    """
    Agglomerative Clustering (Complete Linkage) use karke similar faces ko group karo.
    
    DBSCAN se ye BEHTAR hai kyunki:
    - Complete Linkage = group me tabhi daalega jab SAARE chehre match karein
    - Chaining problem nahi hogi (A matches B, B matches C, but A != C)
    - Google Photos bhi similar approach use karta hai
    
    Return: (person_photos dict, person_faces dict)
    """
    if not face_data:
        return {}, {}

    # Saare embeddings ko ek array me daalo
    embeddings = np.array([f["embedding"] for f in face_data])

    # Embeddings ko normalize karo
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / norms

    # Cosine distance matrix banao (har face ka har dusre face se distance)
    cosine_distances = squareform(pdist(embeddings_normalized, metric="cosine"))

    # Agglomerative Clustering with Average Linkage:
    # - distance_threshold = 0.55
    # - linkage = "average" matlab group me AVERAGE distance check hoga
    #   Complete linkage bahut strict tha (same person split ho raha tha)
    #   Average linkage balanced hai — na zyada strict, na zyada loose
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.55,
        metric="precomputed",
        linkage="average"
    ).fit(cosine_distances)

    labels = clustering.labels_

    # Ab har person ke liye photos aur face info collect karo
    person_photos = {}   # person_name -> set of photo names
    person_faces = {}    # person_name -> list of face info (for preview)

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

    # Sirf 1 photo wale groups ko "Unknown" me daal do
    # (Agar kisi insaan ki sirf 1 photo hai toh wo identify nahi ho sakta)
    final_photos = {}
    final_faces = {}
    unknown_photos = set()
    unknown_face_list = []
    person_counter = 1

    # Pehle groups ko size ke hisaab se sort karo (bade groups pehle)
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


# ============================================================
# STEP 4: Photos Ko Person-Wise Folders Me COPY Karo
# ============================================================
def save_face_preview(input_folder, output_folder, person_name, face_info):
    """
    Person ka chehra crop karke ek preview image save karo.
    Ye Google Photos jaisa face icon hai — taaki pata chale ki ye kaun hai.
    """
    try:
        # Pehle face ki info le lo
        photo_name = face_info["photo"]
        area = face_info["facial_area"]
        x = area.get("x", 0)
        y = area.get("y", 0)
        w = area.get("w", 100)
        h = area.get("h", 100)

        # Original photo open karo
        photo_path = os.path.join(input_folder, photo_name)
        with Image.open(photo_path) as img:
            # Thoda padding add karo taaki chehra acche se dikhe
            padding = int(max(w, h) * 0.3)
            left = max(0, x - padding)
            top = max(0, y - padding)
            right = min(img.width, x + w + padding)
            bottom = min(img.height, y + h + padding)

            # Chehra crop karo
            face_crop = img.crop((left, top, right, bottom))

            # 300x300 resize karo taaki sab previews ek size ke hon
            face_crop = face_crop.resize((300, 300), Image.LANCZOS)

            # Save karo
            person_folder = os.path.join(output_folder, person_name)
            preview_path = os.path.join(person_folder, "_face_preview.jpg")
            face_crop.save(preview_path, "JPEG", quality=95)

    except Exception as e:
        print(f"   Preview error for {person_name}: {e}")


def organize_photos(input_folder, output_folder, person_photos, person_faces, no_face_photos):
    """
    Photos ko organized folders me COPY karo.
    Har person folder me ek face preview bhi save hoga.
    
    IMPORTANT: Original photos KABHI move ya delete nahi hongi!
    Sirf copy hogi.
    """
    # Output folder banao (agar nahi hai toh)
    os.makedirs(output_folder, exist_ok=True)

    total_copies = 0

    # Har person ke liye folder banao aur photos copy karo
    for person, photos in sorted(person_photos.items()):
        person_folder = os.path.join(output_folder, person)
        os.makedirs(person_folder, exist_ok=True)

        for photo in sorted(photos):
            src = os.path.join(input_folder, photo)
            dst = os.path.join(person_folder, photo)
            shutil.copy2(src, dst)
            total_copies += 1

        # Face preview save karo (pehle face ka crop)
        if person in person_faces and person_faces[person]:
            save_face_preview(input_folder, output_folder, person, person_faces[person][0])

        print(f"   {person}: {len(photos)} photos copied + face preview saved")

    # Bina face wali photos ko "No_Faces" folder me daalo
    if no_face_photos:
        no_face_folder = os.path.join(output_folder, "No_Faces")
        os.makedirs(no_face_folder, exist_ok=True)

        for photo in no_face_photos:
            src = os.path.join(input_folder, photo)
            dst = os.path.join(no_face_folder, photo)
            shutil.copy2(src, dst)
            total_copies += 1

        print(f"   No_Faces: {len(no_face_photos)} photos copied")

    return total_copies


# ============================================================
# MAIN FUNCTION - Sab kuch yahan se start hota hai
# ============================================================
def main():
    print("=" * 50)
    print("   AI Photo Organizer")
    print("   Google Photos-Style Face Grouping")
    print("=" * 50)

    # Step 1: Photos dhundho
    print("\nStep 1: Scanning photos...")
    photos = scan_photos(INPUT_FOLDER)
    print(f"   Found {len(photos)} photos.\n")

    if not photos:
        print("No photos found! Check your input folder.")
        return

    # Step 2: Faces dhundho aur embeddings banao
    print("Step 2: Detecting faces & generating embeddings...")
    print("   (Using RetinaFace + Facenet512 for best accuracy)")
    print("   This may take a few minutes...\n")

    face_data, no_face_photos = extract_faces_and_embeddings(INPUT_FOLDER, photos)

    print(f"\n   Summary:")
    print(f"   Total faces detected: {len(face_data)}")
    print(f"   Photos without faces: {len(no_face_photos)}")

    if not face_data:
        print("\n   No faces found in any photo. Nothing to organize.")
        return

    # Step 3: Similar faces ko group karo
    print("\nStep 3: Grouping similar faces (AI clustering)...")
    person_photos, person_faces = cluster_faces(face_data)

    person_groups = [k for k in person_photos if k != "Unknown"]
    print(f"   Found {len(person_groups)} person group(s)")
    for person, photos_set in sorted(person_photos.items()):
        print(f"   - {person}: {len(photos_set)} photos")

    # Step 4: Photos organize karo + face previews banao
    print(f"\nStep 4: Copying photos + saving face previews...")
    total = organize_photos(INPUT_FOLDER, OUTPUT_FOLDER, person_photos, person_faces, no_face_photos)

    print(f"\n   Done! {total} photo copies created.")
    print(f"   Output folder: {OUTPUT_FOLDER}")

    print("\n" + "=" * 50)
    print("   Photo Organization Complete!")
    print("   Original photos are SAFE and UNTOUCHED.")
    print("=" * 50)


# Ye line ensure karti hai ki code tabhi chale jab directly run karein
if __name__ == "__main__":
    main()
