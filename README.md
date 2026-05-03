# 🎵 LyricLens
### Hybrid Music Recommendation using Audio, Lyrics & Collaborative Filtering

> A content-aware music recommendation system that goes beyond audio matching — it understands what songs *mean* lyrically and how humans *group* them together.

---

## 📌 What is LyricLens?

Traditional recommenders match songs by audio features like tempo and energy. LyricLens adds two more signals — **lyric semantics** (what a song is about) and **collaborative filtering** (what songs humans listen to together) — making recommendations far more accurate and meaningful.

**Example:** For "Love Me Like You Do" (Ellie Goulding), audio-only finds film scores and jazz. LyricLens finds "She Will Be Loved" (Maroon 5), "A Thousand Miles" (Vanessa Carlton) — thematically correct.

| Signal | What it captures |
|--------|-----------------|
| Audio features (C matrix) | How a song SOUNDS |
| Word2Vec lyrics (L matrix) | What a song MEANS |
| Playlist CF (R matrix) | What humans GROUP together |

---

## 🏗️ Architecture

```
Spotify Audio Dataset (170k songs) + Lyrics & Playlist Dataset (18k songs)
                    ↓ Merge on song name + artist
              NLP Preprocessing
        (tokenize → stopwords → lemmatize)
         ┌──────────┴──────────┐
   Audio Features          Lyrics NLP
   9 features              Word2Vec (100-dim)
   StandardScaler          Avg pooling → song vector
   K-Means (k=20)                 +
   Cosine Similarity       Playlist Co-occurrence CF
                           Confidence weighting
         └──────────┬──────────┘
           Adaptive Gated Fusion
        Weights adjust by song popularity
        Popular → trust CF more
        Niche   → trust content more
                    ↓
           Top-N Recommendations
           Recall@10 + NDCG@10
```

---

## 📂 Project Structure

```
LyricLens/
├── notebooks/
│   └── LyricLens_Main.ipynb
├── data/
│   ├── raw/                  ← CSVs (not tracked in Git)
│   │   ├── data.csv
│   │   ├── data_by_genres.csv
│   │   ├── data_by_year.csv
│   │   └── spotify_songs.csv
│   └── processed/            ← Auto-generated after running notebook
├── models/                   ← Auto-generated after running notebook
├── assets/
│   └── evaluation_dashboard.png
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

```bash
git clone https://github.com/jais001-sushant/Lyric-Lens.git
cd Lyric-Lens
pip install -r requirements.txt
```

**Download datasets and place in `data/raw/`:**
- [Spotify 1921-2020](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-160k-tracks) → `data.csv`, `data_by_genres.csv`, `data_by_year.csv`
- [Audio + Lyrics](https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs) → `spotify_songs.csv`

```bash
jupyter notebook notebooks/LyricLens_Main.ipynb
```

---

## 🧪 How It Works

**Phase 1 — Audio Clustering**
K-Means clustering on 15 audio features. Cosine similarity for recommendations. Visualized with t-SNE and PCA.

**Phase 2 — TF-IDF Hybrid**
NLP pipeline on raw lyrics. TF-IDF vectorization. Fixed 50/50 weighted hybrid scoring.

**Phase 3 — Full Hybrid (Best)**
Word2Vec semantic embeddings replace TF-IDF. Playlist co-occurrence builds the collaborative signal. Adaptive weights adjust automatically based on song popularity. Evaluated with Recall@10 and NDCG@10.

---

## 📊 Results

| Phase | Recall@10 | NDCG@10 | Notes |
|-------|-----------|---------|-------|
| Phase 1 — Audio only | 0.0000 | 0.0000 | Matches sound, ignores meaning |
| Phase 2 — TF-IDF | 0.0000 | 0.0000 | Word count, no semantics |
| **Phase 3 — Full Hybrid** | **0.1263** | **0.2816** | All 3 signals combined |

---

## ⚙️ Tech & Concepts Coverage

| Concept | Implementation |
|---------|---------------|
| Audio Feature Clustering | K-Means (k=20 songs, k=10 genres) |
| Dimensionality Reduction | t-SNE + PCA visualization |
| NLP Preprocessing | Tokenization, stopwords, lemmatization |
| Content Filtering | TF-IDF + cosine similarity (Phase 2) |
| Semantic Embeddings | Word2Vec skip-gram, 100-dim (Phase 3) |
| Collaborative Filtering | Playlist co-occurrence, item-based CF |
| Confidence Weighting | αui = 1 + 10·log(1 + rui) |
| Adaptive Gating | softmax weights based on song popularity |
| Evaluation | Recall@K, NDCG@K, Intra-list Diversity |
| Cold-start handling | Content fallback when CF unavailable |

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `NLTK` · `Gensim` · `SciPy` · `Plotly` · `Matplotlib` · `Streamlit`

---

## 🔮 Future Scope

- Real user play-count data for true collaborative filtering
- Deep neural recommendation with PyTorch (MLP encoders + BPR loss)
- Streamlit deployment with live demo interface
- Faiss ANN indexing for production-scale inference