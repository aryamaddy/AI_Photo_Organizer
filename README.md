# 📸 AI Photo Organizer

Google Photos-style face-based photo organizer that runs **locally and offline** on your machine.

## ✨ Features

- 🔍 **Face Detection** using RetinaFace (most accurate detector)
- 🧠 **Face Embeddings** using Facenet512 (512-dimensional feature vectors)
- 🤖 **Smart Clustering** using Agglomerative Clustering with Average Linkage
- 👤 **Face Previews** - Each person folder has a cropped face thumbnail for easy identification
- 🔒 **Safe** - Original photos are NEVER moved or deleted, only copied

## 🚀 How It Works

1. **Scan** - Finds all photos in the input folder
2. **Detect** - Uses RetinaFace AI to find faces in each photo
3. **Embed** - Generates 512-dimensional mathematical representations using Facenet512
4. **Cluster** - Groups similar faces using Agglomerative Clustering (Average Linkage)
5. **Organize** - Copies photos into person-wise folders with face preview thumbnails

## 📁 Project Structure

```
AI_Photo_Organizer/
├── main.py              # Main script
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
├── input_photos/       # Put your photos here (not tracked by git)
└── organized_photos/   # Output folder (auto-generated)
    ├── Person_1/
    │   ├── _face_preview.jpg   # Face thumbnail
    │   ├── DSC_0001.JPG
    │   └── ...
    ├── Person_2/
    ├── Unknown/
    └── No_Faces/
```

## 🛠️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/aryamaddy/AI_Photo_Organizer.git
cd AI_Photo_Organizer

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Put your photos in input_photos/ folder
mkdir input_photos
# Copy your photos there

# 5. Run!
python main.py
```

## ⚙️ Configuration

Edit the settings at the top of `main.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `INPUT_FOLDER` | `input_photos/` | Where your photos are |
| `OUTPUT_FOLDER` | `organized_photos/` | Where organized photos go |
| `DETECTOR` | `retinaface` | Face detection model |
| `EMBEDDING_MODEL` | `Facenet512` | Face embedding model |
| `MIN_CONFIDENCE` | `0.95` | Minimum face detection confidence |
| `MIN_FACE_SIZE` | `250` | Minimum face size in pixels |

## 🧪 Tech Stack

- **Python 3.x**
- **DeepFace** - Face detection & recognition
- **RetinaFace** - Face detection backend
- **Facenet512** - Face embedding model
- **scikit-learn** - Agglomerative Clustering
- **SciPy** - Distance computation
- **Pillow** - Image processing

## 📝 License

MIT License

## 👨‍💻 Author

**Arya** - [@aryamaddy](https://github.com/aryamaddy)
