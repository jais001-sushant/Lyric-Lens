import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import math
import time
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from collections import defaultdict
from difflib import get_close_matches

# ── Page config (MUST be first) ──────────────────────────────────────────────
st.set_page_config(
    page_title="LyricLens",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Spotify-inspired CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@400;700;900&family=DM+Sans:wght@300;400;500;700&display=swap');

/* ── Root variables ── */
:root {
    --green:      #1DB954;
    --green-dim:  #1aa34a;
    --black:      #121212;
    --surface:    #181818;
    --surface2:   #282828;
    --surface3:   #3E3E3E;
    --text:       #FFFFFF;
    --text-dim:   #B3B3B3;
    --text-faint: #535353;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--black) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Main container ── */
.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--surface2) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hero header ── */
.lyriclens-hero {
    background: linear-gradient(135deg, #1DB954 0%, #0d7a37 40%, #121212 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.lyriclens-hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -2px;
    margin: 0;
    line-height: 1;
}
.hero-title span { color: #121212; }
.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.75);
    margin-top: 0.5rem;
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* ── Search bar ── */
.stTextInput input {
    background: var(--surface2) !important;
    border: 2px solid var(--surface3) !important;
    border-radius: 500px !important;
    color: var(--text) !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.5rem !important;
    line-height: 1.5 !important;
    transition: border-color 0.2s ease;
}
.stTextInput input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 3px rgba(29,185,84,0.15) !important;
}

.stTextInput > label {
    display: none !important;
}

.stTextInput input {
    padding: 0.75rem 1.5rem !important;
    line-height: 1.5 !important;
}

div[data-testid="stHorizontalBlock"] > div:last-child .stButton > button {
    margin-top: 0 !important;
    padding: 0.75rem 2rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--green) !important;
    color: #000 !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 500px !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #1ed760 !important;
    transform: scale(1.03) !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] { padding: 0.5rem 0 !important; }
.stSlider [data-testid="stMarkdownContainer"] p { color: var(--text-dim) !important; font-size: 0.85rem !important; }

/* ── Section headers ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}

/* ── Song card ── */
.song-card {
    background: var(--surface);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    border: 1px solid transparent;
    transition: all 0.2s ease;
    cursor: pointer;
}
.song-card:hover {
    background: var(--surface2);
    border-color: var(--surface3);
}
.song-rank {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-faint);
    width: 20px;
    text-align: center;
}
.song-disc {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--green), #0d5e2a);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.song-info { flex: 1; min-width: 0; }
.song-name {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.song-artist {
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-top: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.song-score {
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--green);
    text-align: right;
    flex-shrink: 0;
}
.song-year {
    font-size: 0.75rem;
    color: var(--text-faint);
    margin-top: 2px;
}

/* ── Phase badge ── */
.phase-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 500px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.phase1-badge { background: rgba(83,83,83,0.4); color: #B3B3B3; border: 1px solid #535353; }
.phase2-badge { background: rgba(30,215,96,0.15); color: #1DB954; border: 1px solid rgba(29,185,84,0.3); }
.phase3-badge { background: var(--green); color: #000; }

/* ── Metric card ── */
.metric-card {
    background: var(--surface);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid var(--surface2);
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--green);
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Found song banner ── */
.found-banner {
    background: linear-gradient(90deg, rgba(29,185,84,0.2) 0%, transparent 100%);
    border-left: 3px solid var(--green);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1.5rem;
}
.found-banner-title { font-weight: 700; font-size: 1.1rem; color: var(--text); }
.found-banner-sub { font-size: 0.85rem; color: var(--text-dim); margin-top: 2px; }

/* ── Weight display ── */
.weight-pill {
    display: inline-block;
    background: var(--surface2);
    border-radius: 500px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-dim);
    margin-right: 6px;
    border: 1px solid var(--surface3);
}
.weight-pill span { color: var(--green); }

/* ── Divider ── */
.spotify-divider {
    border: none;
    border-top: 1px solid var(--surface2);
    margin: 1.5rem 0;
}

/* ── Plotly chart background ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* ── Selectbox ── */
.stSelectbox [data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border: 1px solid var(--surface3) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--surface2) !important;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-dim) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    border-radius: 0 !important;
    padding: 0.75rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom-color: var(--green) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--green) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all saved models and datasets"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load merged dataset
    merged = pd.read_csv(os.path.join(base, 'data/processed/lyriclens_merged.csv'))

    # Load sklearn models
    with open(os.path.join(base, 'models/lyriclens_models.pkl'), 'rb') as f:
        models = pickle.load(f)

    # Load Word2Vec L matrix
    L_matrix = np.load(os.path.join(base, 'models/L_matrix.npy'))

    # Load CF similarity matrix
    cf_sim_df = pd.read_pickle(os.path.join(base, 'models/cf_sim_matrix.pkl'))

    return merged, models, L_matrix, cf_sim_df


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINES
# ══════════════════════════════════════════════════════════════════════════════

def get_song_index(song_name, data):
    exact = data[data['name'].str.lower() == song_name.lower()]
    if len(exact) > 0:
        return exact.sort_values('popularity', ascending=False).index[0]
    partial = data[data['name'].str.lower().str.contains(song_name.lower(), na=False)]
    if len(partial) > 0:
        return partial.sort_values('popularity', ascending=False).index[0]
    return None

def clean_artist(artists_str):
    """Remove brackets and quotes from artist string"""
    return re.sub(r"[\[\]']", '', str(artists_str)).strip()

def clean_results(result, input_idx, input_name, n):
    result = result.drop(index=input_idx, errors='ignore')
    result = result.sort_values('hybrid_score', ascending=False)
    result = result.drop_duplicates(subset=['name'], keep='first')
    result = result[~result['name'].str.lower().str.startswith(input_name.lower() + ' -')]
    return result.head(n)

def get_cf_scores(song_name, all_names, cf_sim_df):
    scores = np.zeros(len(all_names))
    if song_name not in cf_sim_df.index:
        return scores
    for i, name in enumerate(all_names):
        if name in cf_sim_df.index:
            scores[i] = cf_sim_df.loc[song_name, name]
    return scores

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def compute_gate_weights(popularity, cf_available):
    p = np.clip(popularity / 100.0, 0, 1)
    logit_audio = 0.6 - 0.2 * p
    logit_lyric = 0.5 - 0.1 * p
    logit_cf    = 0.1 + 0.5 * p if cf_available else -999.0
    w = softmax(np.array([logit_audio, logit_lyric, logit_cf]))
    return w[0], w[1], w[2]


def recommend_phase1(song_name, merged, audio_matrix, n=10):
    """Phase 1: Audio-only cosine similarity"""
    idx = get_song_index(song_name, merged)
    if idx is None:
        return None, None
    found = merged.loc[idx]
    audio_sim = cosine_similarity(audio_matrix[idx].reshape(1,-1), audio_matrix)[0]
    result = merged.copy()
    result['audio_sim']    = audio_sim
    result['lyric_sim']    = 0.0
    result['cf_sim']       = 0.0
    result['hybrid_score'] = audio_sim
    result = clean_results(result, idx, found['name'], n)
    return found, result


def recommend_phase2(song_name, merged, audio_matrix, tfidf_matrix,
                     audio_weight=0.5, lyric_weight=0.5, n=10):
    """Phase 2: Audio + TF-IDF lyrics"""
    idx = get_song_index(song_name, merged)
    if idx is None:
        return None, None
    found     = merged.loc[idx]
    audio_sim = cosine_similarity(audio_matrix[idx].reshape(1,-1), audio_matrix)[0]
    lyric_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    result = merged.copy()
    result['audio_sim']    = audio_sim
    result['lyric_sim']    = lyric_sim
    result['cf_sim']       = 0.0
    result['hybrid_score'] = audio_weight * audio_sim + lyric_weight * lyric_sim
    result = clean_results(result, idx, found['name'], n)
    return found, result


def recommend_phase3(song_name, merged, audio_matrix, L_matrix, cf_sim_df, n=10):
    """Phase 3: Audio + Word2Vec + CF + Adaptive Gating"""
    idx = get_song_index(song_name, merged)
    if idx is None:
        return None, None, None, None, None
    found      = merged.loc[idx]
    popularity = found.get('popularity', 50)
    cf_available = found['name'] in cf_sim_df.index

    audio_sim = cosine_similarity(audio_matrix[idx].reshape(1,-1), audio_matrix)[0]
    lyric_sim = cosine_similarity(L_matrix[idx].reshape(1,-1), L_matrix)[0]
    cf_sim    = get_cf_scores(found['name'], merged['name'].tolist(), cf_sim_df)

    audio_w, lyric_w, cf_w = compute_gate_weights(popularity, cf_available)

    result = merged.copy()
    result['audio_sim']    = audio_sim
    result['lyric_sim']    = lyric_sim
    result['cf_sim']       = cf_sim
    result['hybrid_score'] = (
        0.05 * (result['popularity'] / 100.0) +
        audio_w * audio_sim + lyric_w * lyric_sim + cf_w * cf_sim
    )
    result = clean_results(result, idx, found['name'], n)
    return found, result, audio_w, lyric_w, cf_w


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

RADAR_FEATURES = ['danceability','energy','valence','acousticness',
                  'speechiness','liveness','instrumentalness']
RADAR_LABELS   = ['Dance','Energy','Valence','Acoustic','Speech','Live','Instrumental']

PLOTLY_LAYOUT = dict(
    paper_bgcolor='#181818',
    plot_bgcolor='#181818',
    font=dict(color='#B3B3B3', family='DM Sans'),
    margin=dict(l=20, r=20, t=40, b=20),
)

def make_radar_chart(input_song, recs_df, merged):
    """Radar chart comparing input song vs top recommendations"""
    fig = go.Figure()

    def add_trace(row, name, color, fill_color, width=2):
        vals = [row.get(f, 0) for f in RADAR_FEATURES]
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=RADAR_LABELS + [RADAR_LABELS[0]],
            fill='toself', fillcolor=fill_color,
            line=dict(color=color, width=width),
            name=name, opacity=0.85
        ))

    # Input song
    add_trace(input_song, f"🎵 {input_song['name'][:25]}", '#1DB954', 'rgba(29,185,84,0.15)', width=3)

    # Top 3 recommendations
    colors = ['#FFFFFF', '#B3B3B3', '#535353']
    fills  = ['rgba(255,255,255,0.05)', 'rgba(179,179,179,0.05)', 'rgba(83,83,83,0.05)']
    for i, (_, row) in enumerate(recs_df.head(3).iterrows()):
        add_trace(row, row['name'][:20], colors[i], fills[i])

    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor='#282828',
            radialaxis=dict(visible=True, range=[0,1], showticklabels=False,
                           gridcolor='#3E3E3E', linecolor='#3E3E3E'),
            angularaxis=dict(gridcolor='#3E3E3E', linecolor='#3E3E3E',
                           tickfont=dict(size=11, color='#FFFFFF'))
        ),
        legend=dict(bgcolor='#282828', bordercolor='#3E3E3E', borderwidth=1,
                   font=dict(size=10)),
        title=dict(text='Audio Profile Comparison', font=dict(size=13, color='#FFFFFF'), x=0.5)
    )
    return fig


def make_phase_comparison_chart(p1_df, p2_df, p3_df):
    """Bar chart comparing top scores across 3 phases"""
    def top_scores(df, n=5):
        if df is None or len(df) == 0:
            return [], []
        top = df.head(n)
        names  = [n[:18]+'…' if len(n)>18 else n for n in top['name']]
        scores = top['hybrid_score'].tolist()
        return names, scores

    n1, s1 = top_scores(p1_df)
    n2, s2 = top_scores(p2_df)
    n3, s3 = top_scores(p3_df)

    fig = go.Figure()
    if s1: fig.add_trace(go.Bar(name='Phase 1 — Audio', x=n1, y=s1, marker_color='#535353', opacity=0.9))
    if s2: fig.add_trace(go.Bar(name='Phase 2 — TF-IDF', x=n2, y=s2, marker_color='#1aa34a', opacity=0.9))
    if s3: fig.add_trace(go.Bar(name='Phase 3 — Full Hybrid', x=n3, y=s3, marker_color='#1DB954', opacity=0.95))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode='group',
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#282828', title='Score', range=[0,1]),
        legend=dict(bgcolor='#282828', bordercolor='#3E3E3E', borderwidth=1),
        title=dict(text='Top-5 Score Comparison Across Phases', font=dict(size=13, color='#FFFFFF'), x=0.5)
    )
    return fig


def make_score_breakdown_chart(rec_df):
    """Stacked bar showing audio/lyric/cf contribution"""
    if rec_df is None or len(rec_df) == 0:
        return None
    top = rec_df.head(8)
    names = [n[:16]+'…' if len(n)>16 else n for n in top['name']]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Audio', x=names, y=top['audio_sim'], marker_color='#1DB954', opacity=0.9))
    fig.add_trace(go.Bar(name='Lyrics (W2V)', x=names, y=top['lyric_sim'], marker_color='#535353', opacity=0.9))
    fig.add_trace(go.Bar(name='CF', x=names, y=top['cf_sim'], marker_color='#B3B3B3', opacity=0.7))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode='group',
        xaxis=dict(showgrid=False, tickfont=dict(size=9), tickangle=-25),
        yaxis=dict(gridcolor='#282828', title='Similarity Score', range=[0,1.1]),
        legend=dict(bgcolor='#282828', bordercolor='#3E3E3E', borderwidth=1),
        title=dict(text='Signal Breakdown per Recommendation', font=dict(size=13, color='#FFFFFF'), x=0.5)
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SONG CARD HTML
# ══════════════════════════════════════════════════════════════════════════════

DISC_EMOJIS = ['🎵','🎶','🎸','🎹','🎺','🎻','🥁','🎤']

def render_song_card(rank, name, artist, year, score, score_label="Match"):
    emoji = DISC_EMOJIS[rank % len(DISC_EMOJIS)]
    artist_clean = clean_artist(artist)
    st.markdown(f"""
    <div class="song-card">
        <div class="song-rank">{rank}</div>
        <div class="song-disc">{emoji}</div>
        <div class="song-info">
            <div class="song-name">{name}</div>
            <div class="song-artist">{artist_clean}</div>
        </div>
        <div class="song-score">
            {score:.0%}<br>
            <span style="color:#535353;font-size:0.7rem;font-weight:400">{score_label}</span>
        </div>
        <div style="color:#535353;font-size:0.8rem;flex-shrink:0">{int(year)}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem 0">
        <div style="font-size:1.4rem;font-weight:700;letter-spacing:-0.5px">🎵 LyricLens</div>
        <div style="font-size:0.75rem;color:#B3B3B3;margin-top:4px">Hybrid Music Recommender</div>
    </div>
    <hr style="border-color:#282828;margin:0 0 1.5rem 0">
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Recommendations</div>', unsafe_allow_html=True)
    n_recs = st.slider("Number of songs", 5, 20, 10, 1)

    st.markdown('<hr style="border-color:#282828;margin:1.5rem 0">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Phase 2 Weights</div>', unsafe_allow_html=True)
    st.caption("Adjust audio vs lyrics balance")
    audio_w = st.slider("🔊 Audio weight", 0.0, 1.0, 0.5, 0.05)
    lyric_w = round(1.0 - audio_w, 2)
    st.markdown(f"""
    <div style="margin-top:0.5rem">
        <span class="weight-pill">Audio <span>{audio_w:.0%}</span></span>
        <span class="weight-pill">Lyrics <span>{lyric_w:.0%}</span></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#282828;margin:1.5rem 0">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem;color:#B3B3B3;line-height:1.6">
        <b style="color:#fff">Phase 1</b> — Audio features only<br>
        <b style="color:#1DB954">Phase 2</b> — Audio + TF-IDF lyrics<br>
        <b style="color:#1DB954">Phase 3</b> — Audio + Word2Vec + CF<br><br>
        <b style="color:#fff">10,960</b> songs with audio + lyrics<br>
        <b style="color:#fff">170k</b> songs audio-only database
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="lyriclens-hero">
    <div class="hero-title">Lyric<span>Lens</span></div>
    <div class="hero-sub">Music recommendation beyond audio — powered by semantics & collaborative intelligence</div>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("Loading models..."):
    try:
        merged, models, L_matrix, cf_sim_df = load_models()
        audio_matrix  = models['audio_matrix']
        tfidf_matrix  = models['tfidf_matrix']
        tfidf         = models['tfidf']
        MODEL_LOADED  = True
    except Exception as e:
        st.error(f"❌ Could not load models: {e}")
        st.info("Run the notebook first to generate model files in the `models/` folder.")
        MODEL_LOADED = False
        st.stop()

# Search
col_input, col_btn = st.columns([5, 1])
with col_input:
    song_query = st.text_input(
        "", placeholder="🔍  Search for a song... (e.g. Shape of You, Despacito, Blinding Lights)",
        label_visibility="collapsed"
    )
with col_btn:
    search_btn = st.button("Search")

# ── Quick picks ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin:0.75rem 0 1.5rem 0">
    <span style="font-size:0.75rem;color:#535353;margin-right:0.75rem">TRY:</span>
""", unsafe_allow_html=True)
quick_songs = ["Shape of You", "Despacito", "Blinding Lights", "Love Me Like You Do", "Watermelon Sugar"]
cols = st.columns(len(quick_songs))
for i, s in enumerate(quick_songs):
    with cols[i]:
        if st.button(s, key=f"quick_{i}"):
            song_query = s
            search_btn = True
st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if (search_btn or song_query) and song_query.strip():
    st.markdown('<hr class="spotify-divider">', unsafe_allow_html=True)

    with st.spinner(f'Finding songs similar to "{song_query}"...'):
        # Run all 3 phases
        found1, p1_recs            = recommend_phase1(song_query, merged, audio_matrix, n=n_recs)
        found2, p2_recs            = recommend_phase2(song_query, merged, audio_matrix, tfidf_matrix, audio_w, lyric_w, n=n_recs)
        found3, p3_recs, aw, lw, cw = recommend_phase3(song_query, merged, audio_matrix, L_matrix, cf_sim_df, n=n_recs)

    if found3 is None:
        st.warning(f'⚠️  "{song_query}" not found in the lyrics dataset (10,960 songs). Try a different song name.')
        # Show search suggestions
        suggestions = merged[merged['name'].str.contains(song_query[:4], case=False, na=False)]['name'].head(5).tolist()
        if suggestions:
            st.markdown("**Did you mean:**")
            for s in suggestions:
                st.markdown(f"- {s}")
        st.stop()

    # ── Found song banner ─────────────────────────────────────────────────────
    pop = found3.get('popularity', 50)
    st.markdown(f"""
    <div class="found-banner">
        <div class="found-banner-title">🎵 {found3['name']}</div>
        <div class="found-banner-sub">
            {clean_artist(found3['artists'])} · {int(found3['year'])} · 
            Popularity: {pop}/100 · 
            Gate: 🔊 {aw:.0%} audio · 📝 {lw:.0%} lyrics · 🤝 {cw:.0%} CF
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🏆 Phase 3 — Best Results", "📊 Phase Comparison", "📈 Charts"])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1 — Phase 3 Results (main)
    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-label">Top Recommendations — Full Hybrid</div>', unsafe_allow_html=True)
        st.markdown('<span class="phase-badge phase3-badge">Phase 3 · Audio + Word2Vec + CF</span>', unsafe_allow_html=True)

        col_songs, col_radar = st.columns([1.1, 1])

        with col_songs:
            if p3_recs is not None and len(p3_recs) > 0:
                for i, (_, row) in enumerate(p3_recs.iterrows(), 1):
                    render_song_card(i, row['name'], row['artists'], row['year'], row['hybrid_score'])
            else:
                st.info("No results found.")

        with col_radar:
            if p3_recs is not None and len(p3_recs) > 0:
                radar = make_radar_chart(found3, p3_recs, merged)
                st.plotly_chart(radar, use_container_width=True, config={'displayModeBar': False})

                # Score breakdown
                st.markdown('<div class="section-label" style="margin-top:1rem">Signal Breakdown</div>', unsafe_allow_html=True)
                breakdown = make_score_breakdown_chart(p3_recs)
                if breakdown:
                    st.plotly_chart(breakdown, use_container_width=True, config={'displayModeBar': False})

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2 — All 3 Phases side by side
    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-label">All 3 Phases — Side by Side</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<span class="phase-badge phase1-badge">Phase 1 · Audio Only</span>', unsafe_allow_html=True)
            if p1_recs is not None:
                for i, (_, row) in enumerate(p1_recs.head(8).iterrows(), 1):
                    render_song_card(i, row['name'], row['artists'], row['year'], row['hybrid_score'], "Audio")
        with c2:
            st.markdown('<span class="phase-badge phase2-badge">Phase 2 · TF-IDF Hybrid</span>', unsafe_allow_html=True)
            if p2_recs is not None:
                for i, (_, row) in enumerate(p2_recs.head(8).iterrows(), 1):
                    render_song_card(i, row['name'], row['artists'], row['year'], row['hybrid_score'], "Hybrid")
        with c3:
            st.markdown('<span class="phase-badge phase3-badge">Phase 3 · Full Hybrid</span>', unsafe_allow_html=True)
            if p3_recs is not None:
                for i, (_, row) in enumerate(p3_recs.head(8).iterrows(), 1):
                    render_song_card(i, row['name'], row['artists'], row['year'], row['hybrid_score'], "Score")

        # Comparison chart
        st.markdown('<hr class="spotify-divider">', unsafe_allow_html=True)
        cmp_chart = make_phase_comparison_chart(p1_recs, p2_recs, p3_recs)
        st.plotly_chart(cmp_chart, use_container_width=True, config={'displayModeBar': False})

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3 — Charts & Metrics
    # ────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-label">Evaluation Metrics</div>', unsafe_allow_html=True)

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">0.1263</div>
                <div class="metric-label">Recall@10</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">0.2816</div>
                <div class="metric-label">NDCG@10</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">10,960</div>
                <div class="metric-label">Songs (Hybrid)</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">170k</div>
                <div class="metric-label">Audio DB</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="spotify-divider">', unsafe_allow_html=True)

        # Input song audio profile
        st.markdown('<div class="section-label">Input Song — Audio Profile</div>', unsafe_allow_html=True)
        song_feats = {f: found3.get(f, 0) for f in RADAR_FEATURES}
        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            for feat, val in list(song_feats.items())[:4]:
                st.markdown(f"""
                <div style="margin-bottom:0.75rem">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="font-size:0.8rem;color:#B3B3B3;text-transform:capitalize">{feat}</span>
                        <span style="font-size:0.8rem;font-weight:700;color:#1DB954">{val:.2f}</span>
                    </div>
                    <div style="background:#282828;border-radius:500px;height:4px">
                        <div style="background:#1DB954;width:{val*100:.0f}%;height:4px;border-radius:500px"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with feat_col2:
            for feat, val in list(song_feats.items())[4:]:
                st.markdown(f"""
                <div style="margin-bottom:0.75rem">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="font-size:0.8rem;color:#B3B3B3;text-transform:capitalize">{feat}</span>
                        <span style="font-size:0.8rem;font-weight:700;color:#1DB954">{val:.2f}</span>
                    </div>
                    <div style="background:#282828;border-radius:500px;height:4px">
                        <div style="background:#1DB954;width:{val*100:.0f}%;height:4px;border-radius:500px"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#535353">
        <div style="font-size:4rem;margin-bottom:1rem">🎵</div>
        <div style="font-size:1.2rem;font-weight:600;color:#B3B3B3">Search for a song to get started</div>
        <div style="font-size:0.9rem;margin-top:0.5rem">
            Type any song name above or click one of the quick picks
        </div>
    </div>
    """, unsafe_allow_html=True)