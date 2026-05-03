# 🎵 LyricLens
### Model-Driven Content Augmentation with Collaborative Filtering for Music Recommendation

> **Minor Project — B.Tech CSE (AI/ML) | UPES Dehradun | 2022–2026**
> **Reference:** *A Hybrid Deep Recommendation Model for Music Personalization* — Lalit Sachan

---

## 📌 What is LyricLens?

A hybrid music recommendation system combining **three signals** to find songs that truly feel alike — not just sound alike.

| Signal | What it captures | PDF Section |
|--------|-----------------|-------------|
| Audio features (C matrix) | How a song SOUNDS | 2.3 |
| Word2Vec lyrics (L matrix) | What a song MEANS | 2.4 |
| Playlist CF (R matrix) | What humans GROUP together | 2.1 |

---

## 🏗️ Architecture

```
Spotify Audio (170k) + Lyrics/Playlist Dataset (18k)
              ↓ Merge on name + artist
         NLP Preprocessing
    ┌────────┴─────────┐
  C matrix           L matrix + R matrix
  Audio features     Word2Vec + CF (playlist)
  StandardScaler     Confidence: αui=1+10·log(1+rui)
    └────────┬─────────┘
    Adaptive Gated Fusion (Section 6.4)
    ρ = softmax(gate_logits(popularity))
              ↓
    sui = γi + Σ(ρk × signalk)
              ↓
    Top-N Recommendations
    Recall@K + NDCG@K Evaluation
```

---

## 📂 Project Structure

```
LyricLens/
├── notebooks/
│   └── LyricLens_FINAL.ipynb     ← Complete notebook (all cells)
├── data/
│   ├── raw/                      ← CSVs here (not tracked in Git)
│   │   ├── data.csv
│   │   ├── data_by_genres.csv
│   │   ├── data_by_year.csv
│   │   └── spotify_songs.csv
│   └── processed/                ← Auto-generated
├── models/                       ← Auto-generated
├── assets/
│   └── evaluation_dashboard.png
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

```bash
git clone https://github.com/jais001-sushant/LyricLens.git
cd LyricLens
pip install -r requirements.txt
```

**Datasets (place in data/raw/):**
- [Spotify 1921-2020](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-160k-tracks) → `data.csv`, `data_by_genres.csv`, `data_by_year.csv`
- [Audio + Lyrics](https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs) → `spotify_songs.csv`

```bash
jupyter notebook notebooks/LyricLens_FINAL.ipynb
```

---

## 📊 Evaluation Results

| Phase | Recall@10 | NDCG@10 |
|-------|-----------|---------|
| Phase 1 — Audio only | 0.0000 | 0.0000 |
| Phase 2 — TF-IDF Hybrid | 0.0000 | 0.0000 |
| **Phase 3 — Full PDF** | **0.1263** | **0.2816** |

Phase 1 & 2 score 0.0 because they ignore the collaborative signal entirely. Phase 3 catches real co-playlist songs via CF.

---

## 📋 PDF Coverage

| Section | Concept | Status |
|---------|---------|--------|
| 2.1 | R matrix | ✅ Playlist co-occurrence |
| 2.3 | C matrix | ✅ 9 audio features |
| 2.4 | L matrix | ✅ Word2Vec 100-dim |
| 3.1 | Confidence weighting | ✅ αui=1+10·log(1+rui) |
| 6.4 | Adaptive gating | ✅ softmax weights |
| 6.5 | Scoring + bias | ✅ γi + weighted signals |
| 11.2 | Recall@K | ✅ Evaluated |
| 11.3 | NDCG@K | ✅ Evaluated |
| 14 | PyTorch NN | ⬜ Future work |
| 8.8 | BPR loss | ⬜ Future work |
| 2.2 | User features | ⬜ Future work |

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `NLTK` · `Gensim` · `SciPy` · `Plotly` · `Matplotlib` · `Streamlit`

---

## 👥 Team

| Name | Enrollment |
|------|-----------|
| Sushant Jaiswal | 500123999 |
| Suvrat Joshi | 500124269 |
| Shivam Venkatesh | 500126674 |
| Satyam Khandkeshar | 500124823 |

**Mentor:** Mr. Lalit Sachan | **UPES Dehradun** | B.Tech CSE (AI/ML), 2022–2026