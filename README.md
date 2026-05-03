# 🎵 LyricLens
### Model-Driven Content Augmentation with Collaborative Filtering for Music Recommendation

> A hybrid music recommendation system that combines **audio feature analysis** with **lyric-based semantic similarity** using NLP — going beyond genre boundaries to find songs that truly *feel* alike.

---

## 📌 What is LyricLens?

Traditional recommendation systems match songs by audio features (tempo, energy, danceability) or user listening history. LyricLens adds a third signal — **what a song is actually *about*** — by analyzing lyrics using NLP.

**Example:**  
A sad Ed Sheeran ballad and a sad Hindi ghazal might share lyrical themes of heartbreak but sound completely different. LyricLens catches that connection. Spotify won't.

---

## 🏗️ System Architecture

```
Spotify Audio Dataset (170k songs)          Lyrics Dataset (18k songs)
        │                                           │
        └──────────── Merge on name + artist ───────┘
                              │
                    Data Preprocessing
                    (normalize, clean)
                    ┌─────────┴─────────┐
            Audio Pipeline          Lyrics NLP Pipeline
            (StandardScaler)        (tokenize → stopwords
            (K-Means k=20)           → lemmatize → TF-IDF)
                    │                       │
            Audio Cosine Sim        Lyrics Cosine Sim
                    └─────────┬─────────┘
                       Hybrid Score
                  (audio × w1 + lyrics × w2)
                              │
                    Top-N Recommendations
```

---

## 📂 Project Structure

```
LyricLens/
├── notebooks/
│   └── LyricLens_Main.ipynb      ← Main notebook (all phases)
├── data/
│   ├── raw/                      ← Original CSV files (not tracked in Git)
│   │   ├── data.csv
│   │   ├── data_by_genres.csv
│   │   ├── data_by_year.csv
│   │   └── spotify_songs.csv
│   └── processed/                ← Merged + cleaned data (not tracked)
│       └── lyriclens_merged.csv
├── models/                       ← Saved TF-IDF + scaler (not tracked)
│   └── lyriclens_models.pkl
├── src/                          ← (Future) modular Python scripts
├── assets/                       ← Screenshots, diagrams
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/jais001-sushant/LyricLens.git
cd LyricLens
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download datasets

| Dataset | Source | File |
|---------|--------|------|
| Spotify 1921–2020 (audio) | [Kaggle](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-160k-tracks) | `data.csv`, `data_by_genres.csv`, `data_by_year.csv` |
| Audio features + Lyrics | [Kaggle](https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs) | `spotify_songs.csv` |

Place all CSV files in `data/raw/`.

### 4. Run the notebook
```bash
jupyter notebook notebooks/LyricLens_Main.ipynb
```

---

## 🧪 How It Works

### Phase 1 — Audio Clustering
- 15 audio features (danceability, energy, valence, tempo, acousticness...)
- `StandardScaler` normalization
- `K-Means` clustering (k=20 for songs, k=10 for genres)
- `t-SNE` and `PCA` for 2D visualization
- `Cosine similarity` for recommendation

### Phase 2 — Lyrics NLP Pipeline
- Text cleaning: lowercase, remove special characters
- Tokenization with NLTK
- Stopword removal (English + music-specific noise words)
- Lemmatization (`loved → love`, `running → run`)
- `TF-IDF` vectorization (5000 features, unigrams + bigrams)
- Cosine similarity on lyric vectors

### Phase 3 — Hybrid Scoring
```python
hybrid_score = (audio_weight × audio_similarity) + (lyric_weight × lyric_similarity)
```
Default weights: `audio=0.5, lyric=0.5` (tunable)

---

## 📊 Results

| Metric | Phase 1 (Audio Only) | Phase 2 (Hybrid) |
|--------|---------------------|-----------------|
| Dataset size | 170,653 songs | ~15,000–18,000 songs |
| Cross-genre matching | Limited | Strong |
| Lyric-aware | ❌ | ✅ |
| Cold-start friendly | ✅ | ✅ |

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `NLTK` · `Plotly` · `Matplotlib` · `Seaborn`

---

## 👥 Team

| Name | Enrollment |
|------|-----------|
| Suvrat Joshi | 500124269 |
| Shivam Venkatesh | 500126674 |
| Satyam Khandkeshar | 500124823 |
| Sushant Jaiswal | 500123999 |

**Mentor:** Mr. Lalit Sachan  
**Institution:** UPES Dehradun — B.Tech CSE (AI/ML), 2023–2027

---

## 📄 License

This project is for academic purposes under UPES Dehradun minor project guidelines.