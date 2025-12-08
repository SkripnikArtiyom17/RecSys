"""
Activity-Aware Podcast Recommender + LLM Review Summaries (Streamlit)

How to run
----------

1. Install dependencies:

    pip install streamlit scikit-learn pandas numpy together

2. Set Together API key (Linux/macOS):

    export TOGETHER_API_KEY="YOUR_API_KEY_HERE"

   On Windows PowerShell:

    setx TOGETHER_API_KEY "YOUR_API_KEY_HERE"

3. Run the app:

    streamlit run app.py
"""

import os
import csv
import json
import uuid
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from together import Together


# -----------------------------
# Together API key configuration
# -----------------------------
# Recommended: set TOGETHER_API_KEY as an environment variable.
# Optional (LOCAL ONLY): you can paste your key here for quick testing.
# !!! Do NOT commit a real key with this file to a public repo !!!
TOGETHER_API_KEY_FROM_CODE = "tgp_v1_DYc1X5IbJJq8f0UkShffPUCHLFKLz4THrvOaZfJepwY"  # e.g. "tg-XXXXXXXXXXXXXXXXXXXXXXXX"


def get_together_client():
    """
    Returns (client, error_message).

    If API key is not configured or client creation fails,
    returns (None, "error text").
    """
    api_key = os.environ.get("TOGETHER_API_KEY") or TOGETHER_API_KEY_FROM_CODE
    if not api_key:
        return None, (
            "Together API key is not configured. "
            "Set TOGETHER_API_KEY env var or fill TOGETHER_API_KEY_FROM_CODE in app.py."
        )
    try:
        client = Together(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"Couldn't create Together client: {e}"


# -----------------------------
# Paths & constants
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
PODCASTS_CSV = os.path.join(DATA_DIR, "sample_podcasts.csv")
REVIEWS_JSON = os.path.join(DATA_DIR, "reviews.json")

TELEMETRY_CSV = os.path.join(BASE_DIR, "telemetry.csv")

TOP_K = 10

SLEEP_TAGS = {"calm", "storytelling", "sleep", "soft", "soft_start", "relax"}
FOCUS_POS_TAGS = {"interview", "background", "deep", "low-energy", "focus", "study"}
FOCUS_NEG_TAGS = {"comedy", "energetic", "hype", "upbeat"}
WORKOUT_TAGS = {"energetic", "upbeat", "motivational", "high-intensity", "hype"}
COMMUTE_TAGS = {"narrative", "news", "daily brief", "daily-brief", "commute"}

INTEREST_CHIPS = [
    "Tech",
    "True Crime",
    "Comedy",
    "Self-help",
    "Business",
    "News",
    "History",
    "Mindfulness",
    "Football",
    "Sports",
    "AI",
    "Startups",
]

CHIP_TO_TERMS = {
    "Tech": ["tech", "technology", "software", "programming", "engineering"],
    "True Crime": ["crime", "murder", "investigation", "case"],
    "Comedy": ["comedy", "funny", "humor"],
    "Self-help": ["self-help", "mindfulness", "productivity", "habits"],
    "Business": ["business", "startup", "finance", "entrepreneurship"],
    "News": ["news", "daily brief", "current events"],
    "History": ["history", "historical", "war", "ancient"],
    "Mindfulness": ["mindfulness", "meditation", "calm"],
    "Football": ["football", "premier league", "soccer"],
    "Sports": ["sports", "workout", "training"],
    "AI": ["ai", "artificial intelligence", "machine learning"],
    "Startups": ["startup", "vc", "founder"],
}


# -----------------------------
# Data loading & TF-IDF
# -----------------------------

@st.cache_data
def load_podcasts(path=PODCASTS_CSV):
    if not os.path.exists(path):
        st.error(f"Missing data file: {path}")
        st.stop()

    df = pd.read_csv(path)

    # Basic sanitation
    for col in ["explicit", "soft_start"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
        else:
            df[col] = 0

    if "ep_duration_min" not in df.columns:
        st.error("sample_podcasts.csv must contain 'ep_duration_min' column.")
        st.stop()

    if "avg_len_min" not in df.columns:
        df["avg_len_min"] = df["ep_duration_min"]

    if "popularity_score" not in df.columns:
        df["popularity_score"] = 0.5

    # Parse timestamps as UTC-aware datetimes
    df["publish_ts"] = pd.to_datetime(
        df["publish_ts"], errors="coerce", utc=True
    )

    df = df.reset_index(drop=True)
    df["tfidf_index"] = np.arange(len(df))
    return df


@st.cache_data
def load_reviews(path=REVIEWS_JSON):
    reviews = []
    if not os.path.exists(path):
        return reviews

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                reviews.append(obj)
            except json.JSONDecodeError:
                continue
    return reviews


@st.cache_data
def build_tfidf(podcasts_df: pd.DataFrame):
    """Build TF-IDF model over ep_title + ep_desc + tags."""
    corpus = (
        podcasts_df["ep_title"].fillna("")
        + " "
        + podcasts_df["ep_desc"].fillna("")
        + " "
        + podcasts_df["tags"].fillna("")
    )
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


# -----------------------------
# Session state helpers
# -----------------------------

def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.session_start_ts = time.time()
        st.session_state.first_play_ts = None

    if "chip_interest_terms" not in st.session_state:
        st.session_state.chip_interest_terms = {}
    if "feedback_interest_terms" not in st.session_state:
        st.session_state.feedback_interest_terms = {}
    if "selected_chips" not in st.session_state:
        st.session_state.selected_chips = []

    if "liked_items" not in st.session_state:
        st.session_state.liked_items = set()
    if "saved_items" not in st.session_state:
        st.session_state.saved_items = set()
    if "disliked_items" not in st.session_state:
        st.session_state.disliked_items = set()
    if "heard_items" not in st.session_state:
        st.session_state.heard_items = set()
    if "liked_shows" not in st.session_state:
        st.session_state.liked_shows = set()
    if "disliked_shows" not in st.session_state:
        st.session_state.disliked_shows = set()

    if "activity_penalties" not in st.session_state:
        st.session_state.activity_penalties = {}  # activity_label -> {show_id: penalty}
    if "duration_bias" not in st.session_state:
        st.session_state.duration_bias = {}       # activity_label -> shift in minutes
    if "sleep_penalty_extra" not in st.session_state:
        st.session_state.sleep_penalty_extra = 0.0

    if "shown_items" not in st.session_state:
        st.session_state.shown_items = set()

    if "llm_summaries" not in st.session_state:
        st.session_state.llm_summaries = {}

    if "current_activity" not in st.session_state:
        st.session_state.current_activity = "Commute"
    if "current_activity_label" not in st.session_state:
        st.session_state.current_activity_label = "Commute"


def apply_chip_interests(chips):
    """Build base interest term weights from selected chips."""
    chip_terms = {}
    for chip in chips:
        for term in CHIP_TO_TERMS.get(chip, []):
            term = term.lower()
            chip_terms[term] = chip_terms.get(term, 0.0) + 1.0
    st.session_state.chip_interest_terms = chip_terms


def add_feedback_interest(terms, delta):
    """Adjust term weights based on feedback (like/dislike)."""
    fb = st.session_state.feedback_interest_terms
    for t in terms:
        t = t.lower()
        fb[t] = fb.get(t, 0.0) + float(delta)
    st.session_state.feedback_interest_terms = fb


def current_activity_label(activity, workout_mode):
    if activity == "Workout" and workout_mode:
        return f"{activity}:{workout_mode}"
    return activity


# -----------------------------
# Telemetry
# -----------------------------

def log_event(event_type, activity_label, item_id=None, extra=None):
    row = {
        "session_id": st.session_state.get("session_id", ""),
        "user_ts": datetime.now(timezone.utc).isoformat(),
        "activity": activity_label or "",
        "item_id": item_id or "",
        "event_type": event_type,
        "extra": json.dumps(extra or {}, ensure_ascii=False),
    }
    file_exists = os.path.exists(TELEMETRY_CSV)
    with open(TELEMETRY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_telemetry_df():
    if not os.path.exists(TELEMETRY_CSV):
        return pd.DataFrame(
            columns=["session_id", "user_ts", "activity", "item_id", "event_type", "extra"]
        )
    try:
        return pd.read_csv(TELEMETRY_CSV)
    except Exception:
        return pd.DataFrame(
            columns=["session_id", "user_ts", "activity", "item_id", "event_type", "extra"]
        )


def render_metrics():
    st.subheader("📊 Metrics (demo)")
    df = load_telemetry_df()
    if df.empty:
        st.write("No telemetry yet.")
        return

    total_impressions = (df["event_type"] == "impression").sum()
    plays = (df["event_type"] == "play").sum()
    review_opens = (df["event_type"] == "review_summary_opened").sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total impressions", int(total_impressions))
    with col2:
        st.metric("Total plays", int(plays))
    with col3:
        ctr = plays / total_impressions if total_impressions > 0 else 0.0
        st.metric("Play CTR", f"{ctr * 100:.1f}%")
    with col4:
        st.metric("Review summary opens", int(review_opens))

    session_id = st.session_state.get("session_id")
    df_sess = df[df["session_id"] == session_id].copy()
    df_sess["user_ts_dt"] = pd.to_datetime(df_sess["user_ts"], errors="coerce")
    plays_sess = df_sess[df_sess["event_type"] == "play"]
    if not plays_sess.empty:
        t0 = df_sess["user_ts_dt"].min()
        t_play = plays_sess["user_ts_dt"].min()
        if pd.notna(t0) and pd.notna(t_play):
            delta_sec = (t_play - t0).total_seconds()
            st.metric("Time to first play (this session)", f"{delta_sec:.1f} s")


# -----------------------------
# Ranking signals
# -----------------------------

def smooth_duration_score(d, target_min, target_max):
    if pd.isna(d):
        return 0.0
    d = float(d)
    if target_min <= d <= target_max:
        return 1.0
    if d < target_min:
        return max(0.0, 1.0 - (target_min - d) / max(target_min, 1.0))
    else:
        return max(0.0, 1.0 - (d - target_max) / max(90 - target_max, 1.0))


def compute_duration_fit(duration_min, slider_min, slider_max):
    if pd.isna(duration_min):
        return 0.0
    d = float(duration_min)
    if slider_min <= d <= slider_max:
        return 1.0
    if d < slider_min:
        return max(0.0, 1.0 - (slider_min - d) / max(slider_min, 1.0))
    else:
        return max(0.0, 1.0 - (d - slider_max) / max(90 - slider_max, 1.0))


def get_activity_target_window(activity, workout_mode, activity_label):
    if activity == "Sleep":
        base_min, base_max = 20, 60
    elif activity == "Focus":
        base_min, base_max = 20, 60
    elif activity == "Commute":
        base_min, base_max = 10, 30
    elif activity == "Workout":
        if workout_mode == "Run":
            base_min, base_max = 30, 60
        elif workout_mode == "Lift":
            base_min, base_max = 35, 75
        elif workout_mode == "Cardio":
            base_min, base_max = 25, 50
        else:
            base_min, base_max = 25, 50
    else:
        base_min, base_max = 15, 60

    bias = st.session_state.duration_bias.get(activity_label, 0.0)
    min_d = max(10, base_min + bias)
    max_d = max(min_d + 5, base_max + bias)
    return min_d, max_d


def compute_activity_fit(row, activity, workout_mode):
    activity_label = st.session_state.get("current_activity_label", activity)
    target_min, target_max = get_activity_target_window(
        activity, workout_mode, activity_label
    )
    d = row["ep_duration_min"]
    dur_score = smooth_duration_score(d, target_min, target_max)

    tags = [
        t.strip().lower()
        for t in str(row.get("tags", "")).split(",")
        if t.strip()
    ]
    tag_score = 0.0
    tag_set = set(tags)

    if activity == "Sleep":
        if int(row.get("soft_start", 0)) == 1:
            dur_score = min(1.0, dur_score + 0.2)
        matches = len(tag_set & SLEEP_TAGS)
        tag_score = min(1.0, matches / 2.0)
        penalty = 0.3 + float(st.session_state.get("sleep_penalty_extra", 0.0))
        if tag_set & FOCUS_NEG_TAGS or tag_set & WORKOUT_TAGS:
            tag_score -= penalty
    elif activity == "Focus":
        matches = len(tag_set & FOCUS_POS_TAGS)
        tag_score = min(1.0, matches / 2.0)
        if tag_set & FOCUS_NEG_TAGS:
            tag_score -= 0.3
    elif activity == "Workout":
        matches = len(tag_set & WORKOUT_TAGS)
        tag_score = min(1.0, matches / 2.0)
    elif activity == "Commute":
        matches = len(tag_set & COMMUTE_TAGS)
        tag_score = min(1.0, matches / 2.0)
    else:
        matches = len(tag_set)
        tag_score = min(1.0, matches / 5.0)

    score = 0.6 * max(0.0, min(1.0, dur_score)) + 0.4 * max(
        0.0, min(1.0, tag_score)
    )
    return float(max(0.0, min(1.0, score)))


def build_interest_vector(vectorizer):
    weights = {}
    for source in ("chip_interest_terms", "feedback_interest_terms"):
        for term, w in st.session_state.get(source, {}).items():
            weights[term] = weights.get(term, 0.0) + float(w)

    if not weights:
        return None

    vocab = vectorizer.vocabulary_
    vec = np.zeros(len(vocab), dtype=float)
    for term, w in weights.items():
        idx = vocab.get(term.lower())
        if idx is not None:
            vec[idx] += w

    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return (vec / norm).reshape(1, -1)


def compute_interest_fit(embeddings, interest_vec):
    if interest_vec is None:
        return np.zeros(embeddings.shape[0])
    sims = cosine_similarity(embeddings, interest_vec)[:, 0]
    return np.clip(sims, 0.0, 1.0)


def build_query_vector(vectorizer, query_text):
    if not query_text:
        return None
    return vectorizer.transform([query_text])


def compute_query_match(embeddings, query_vec):
    if query_vec is None:
        return np.zeros(embeddings.shape[0])
    sims = cosine_similarity(embeddings, query_vec)[:, 0]
    return np.clip(sims, 0.0, 1.0)


# -----------------------------
# MMR diversification
# -----------------------------

def mmr_select(candidates_df, relevance, sim_matrix, lambda_div):
    n = len(candidates_df)
    if n == 0:
        return []

    relevance = np.asarray(relevance, dtype=float)

    selected = []
    remaining = list(range(n))
    show_counts = {}
    shown_items = st.session_state.get("shown_items", set())

    while remaining and len(selected) < min(TOP_K, n):
        best_idx = None
        best_score = None

        for i in remaining:
            row = candidates_df.iloc[i]
            show_id = row["show_id"]
            ep_id = row["episode_id"]

            if len(selected) < 5 and show_counts.get(show_id, 0) >= 2:
                continue

            if not selected:
                score = relevance[i]
            else:
                max_sim = max(sim_matrix[i][j] for j in selected)
                score = lambda_div * relevance[i] - (1.0 - lambda_div) * max_sim

            if lambda_div >= 0.5 and ep_id in shown_items:
                score -= 0.02

            if best_score is None or score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            best_idx = max(remaining, key=lambda idx: relevance[idx])

        selected.append(best_idx)
        remaining.remove(best_idx)

        row_sel = candidates_df.iloc[best_idx]
        show_id = row_sel["show_id"]
        show_counts[show_id] = show_counts.get(show_id, 0) + 1

    return selected


# -----------------------------
# Explainability
# -----------------------------

def get_top_terms_for_row(row, vectorizer, X, top_k=15):
    idx = int(row["tfidf_index"])
    vec = X[idx].toarray().ravel()
    vocab = vectorizer.vocabulary_
    inv_vocab = {v: k for k, v in vocab.items()}
    top_idx = vec.argsort()[::-1][:top_k]
    terms = []
    for i in top_idx:
        if vec[i] <= 0:
            continue
        term = inv_vocab.get(i)
        if term:
            terms.append(term)
    return terms


def because_you_liked(row, podcasts_df):
    liked_ids = set(st.session_state.get("liked_items", set())) | set(
        st.session_state.get("saved_items", set())
    )
    if not liked_ids:
        return []

    row_tags = {
        t.strip().lower()
        for t in str(row.get("tags", "")).split(",")
        if t.strip()
    }
    liked = podcasts_df[podcasts_df["episode_id"].isin(liked_ids)]
    suggestions = []
    for _, liked_row in liked.iterrows():
        liked_tags = {
            t.strip().lower()
            for t in str(liked_row.get("tags", "")).split(",")
            if t.strip()
        }
        if row_tags & liked_tags:
            suggestions.append(liked_row["show_title"])
    suggestions = list(dict.fromkeys(suggestions))[:2]
    return suggestions


def build_explain_text(activity, interest_terms, activity_keywords, query_terms, because):
    parts = []
    if activity_keywords:
        parts.append(
            f"Fits your {activity.lower()} via {', '.join(activity_keywords[:2])}."
        )
    if interest_terms:
        parts.append(
            f"Overlaps with your interests in {', '.join(interest_terms[:2])}."
        )
    if query_terms:
        parts.append(
            f"Matches your search for {', '.join(query_terms[:2])}."
        )
    if because:
        parts.append(
            f"Similar to your liked shows: {', '.join(because[:2])}."
        )
    if not parts:
        parts.append("Recommended based on activity fit and popularity.")
    return " ".join(parts)


def build_explain_data(row, podcasts_df, vectorizer, X, query):
    activity_label = st.session_state.get("current_activity_label", "Commute")
    activity = activity_label.split(":")[0]

    top_terms = get_top_terms_for_row(row, vectorizer, X, top_k=20)

    interest_weights = {}
    for source in ("chip_interest_terms", "feedback_interest_terms"):
        for term, w in st.session_state.get(source, {}).items():
            interest_weights[term] = interest_weights.get(term, 0.0) + float(w)
    interest_terms = [t for t in top_terms if t in interest_weights][:5]

    tags = [
        t.strip().lower()
        for t in str(row.get("tags", "")).split(",")
        if t.strip()
    ]

    if activity == "Sleep":
        activity_keywords = [t for t in tags if t in SLEEP_TAGS][:5]
    elif activity == "Focus":
        activity_keywords = [t for t in tags if t in FOCUS_POS_TAGS][:5]
    elif activity == "Workout":
        activity_keywords = [t for t in tags if t in WORKOUT_TAGS][:5]
    elif activity == "Commute":
        activity_keywords = [t for t in tags if t in COMMUTE_TAGS][:5]
    else:
        activity_keywords = list(set(tags))[:5]

    query_terms = []
    if query:
        q_tokens = set(query.lower().split())
        query_terms = [t for t in top_terms if t in q_tokens][:5]

    because = because_you_liked(row, podcasts_df)
    text = build_explain_text(activity, interest_terms, activity_keywords, query_terms, because)

    return {
        "top_terms": top_terms,
        "interest_terms": interest_terms,
        "activity_keywords": activity_keywords,
        "query_terms": query_terms,
        "because": because,
        "text": text,
    }


# -----------------------------
# LLM review summarization
# -----------------------------

def summarize_reviews(show_id, show_title):
    """
    Делает краткое саммари отзывов по подкасту (show_id) на основе reviews.json
    с помощью Together LLM.

    Возвращает текст саммари и кеширует результат в session_state.
    """
    cache = st.session_state.llm_summaries
    if show_id in cache:
        return cache[show_id]

    all_reviews = load_reviews()

    # 1) Файл пустой или отсутствует
    if not all_reviews:
        summary = (
            "Review dataset is empty or data/reviews.json is missing.\n"
            "Place the Kaggle reviews.json file into the data/ folder."
        )
        cache[show_id] = summary
        st.session_state.llm_summaries = cache
        return summary

    show_id_str = str(show_id)
    reviews = [r for r in all_reviews if str(r.get("podcast_id")) == show_id_str]

    # 2) Нет ни одного отзыва для этого show_id
    if not reviews:
        summary = (
            f"No reviews found for this podcast (show_id={show_id_str}).\n"
            "Make sure `show_id` in sample_podcasts.csv matches `podcast_id` "
            "values in data/reviews.json."
        )
        cache[show_id] = summary
        st.session_state.llm_summaries = cache
        return summary

    def _get_created_at(r):
        return r.get("created_at") or ""

    reviews_sorted = sorted(reviews, key=_get_created_at)
    recent = reviews_sorted[-20:]

    snippets = []
    ratings = []
    for r in recent:
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()
        rating = r.get("rating", None)
        if rating is not None:
            try:
                ratings.append(float(rating))
            except Exception:
                pass
        parts = []
        if rating is not None:
            parts.append(f"Rating {rating}/5")
        if title:
            parts.append(title)
        if content:
            parts.append(content)
        snippets.append(" - ".join(parts))

    reviews_block = "\n".join(f"- {s}" for s in snippets)

    avg_rating_text = ""
    if ratings:
        avg_rating = sum(ratings) / len(ratings)
        avg_rating_text = f"Average rating in this sample: {avg_rating:.2f}/5.\n"

    prompt = f"""You are an assistant summarizing user reviews for a podcast.

Podcast title: {show_title}
Podcast id: {show_id_str}

{avg_rating_text}
Here are some user reviews (each bullet is one review; they may disagree):

{reviews_block}

Task:
- Summarize the overall listener opinion in 2–3 concise sentences.
- Mention both positive and negative aspects if they appear.
- Highlight recurring themes (e.g., pace, depth, host style, audio quality, bias, production).
- Do not include user names or any personal identifiers.
- Keep a neutral, analytical tone (not promotional)."""

    client, err = get_together_client()
    if err is not None:
        summary = err
        cache[show_id] = summary
        st.session_state.llm_summaries = cache
        return summary

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response.choices[0].message.content.strip()
        cache[show_id] = summary
        st.session_state.llm_summaries = cache
        return summary
    except Exception as e:
        log_event(
            "review_summary_error",
            st.session_state.get("current_activity_label", ""),
            item_id=show_id,
            extra={"error": str(e)},
        )
        return "Couldn't load review summary from LLM. Please try again later."


# -----------------------------
# Feedback handlers
# -----------------------------

def handle_like(row, activity_label, vectorizer, X):
    ep_id = row["episode_id"]
    show_id = row["show_id"]
    st.session_state.liked_items.add(ep_id)
    st.session_state.liked_shows.add(show_id)

    top_terms = get_top_terms_for_row(row, vectorizer, X, top_k=10)
    add_feedback_interest(top_terms, delta=0.7)
    log_event("like", activity_label, item_id=ep_id, extra={"show_id": show_id})


def handle_dislike(row, activity_label, reason, vectorizer, X):
    ep_id = row["episode_id"]
    show_id = row["show_id"]
    st.session_state.disliked_items.add(ep_id)

    top_terms = get_top_terms_for_row(row, vectorizer, X, top_k=10)

    if reason == "Too long":
        db = st.session_state.duration_bias
        db[activity_label] = db.get(activity_label, 0.0) - 5.0
        st.session_state.duration_bias = db
    elif reason == "Not my topic":
        add_feedback_interest(top_terms, delta=-0.7)
        st.session_state.disliked_shows.add(show_id)
    elif reason == "Too intense for sleep":
        st.session_state.sleep_penalty_extra = st.session_state.get(
            "sleep_penalty_extra", 0.0
        ) + 0.3
    elif reason == "I've heard it":
        st.session_state.heard_items.add(ep_id)

    log_event(
        "dislike_reason",
        activity_label,
        item_id=ep_id,
        extra={"show_id": show_id, "reason": reason},
    )


# -----------------------------
# Recommendation pipeline
# -----------------------------

def recommend(
    podcasts_df,
    vectorizer,
    X,
    activity,
    workout_mode,
    query,
    duration_range,
    language,
    is_new_only,
    lambda_div,
):
    now = pd.Timestamp.now(timezone.utc)

    candidates = podcasts_df.copy()

    if language != "Any":
        candidates = candidates[
            candidates["language"].str.lower() == language.lower()
        ]

    if activity in ("Sleep", "Focus"):
        candidates = candidates[candidates["explicit"] == 0]

    if is_new_only:
        cutoff = now - pd.Timedelta(days=30)
        candidates = candidates[candidates["publish_ts"] >= cutoff]

    if candidates.empty:
        candidates = podcasts_df.copy()
        if activity in ("Sleep", "Focus"):
            candidates = candidates[candidates["explicit"] == 0]

    candidates = candidates.reset_index(drop=True)

    tfidf_idxs = candidates["tfidf_index"].tolist()
    embeddings = X[tfidf_idxs]

    interest_vec = build_interest_vector(vectorizer)
    interest_scores = compute_interest_fit(embeddings, interest_vec)

    query_vec = build_query_vector(vectorizer, query)
    query_scores = compute_query_match(embeddings, query_vec)

    slider_min, slider_max = duration_range
    activity_scores = []
    duration_scores = []
    for _, row in candidates.iterrows():
        activity_scores.append(
            compute_activity_fit(row, activity, workout_mode)
        )
        duration_scores.append(
            compute_duration_fit(row["ep_duration_min"], slider_min, slider_max)
        )
    activity_scores = np.array(activity_scores, dtype=float)
    duration_scores = np.array(duration_scores, dtype=float)

    pop_scores = candidates["popularity_score"].astype(float).to_numpy()
    recent_mask = candidates["publish_ts"] >= (now - pd.Timedelta(days=30))
    pop_scores = np.clip(pop_scores + recent_mask.astype(float) * 0.05, 0.0, 1.0)

    wI, wA, wQ, wP, wD = 0.35, 0.25, 0.20, 0.15, 0.05
    if activity == "Sleep":
        wA += 0.05
        wI -= 0.05
    elif activity == "Workout":
        wA += 0.05
        wP += 0.05
        wQ -= 0.05
    elif activity == "Focus":
        wI += 0.05
        wQ += 0.05
        wP -= 0.05

    relevance = (
        wI * interest_scores
        + wA * activity_scores
        + wQ * query_scores
        + wP * pop_scores
        + wD * duration_scores
    )
    relevance = np.asarray(relevance, dtype=float)

    activity_label = st.session_state.get("current_activity_label", activity)
    penalties = st.session_state.activity_penalties.get(activity_label, {})

    liked_items = st.session_state.get("liked_items", set())
    disliked_items = st.session_state.get("disliked_items", set())
    heard_items = st.session_state.get("heard_items", set())
    liked_shows = st.session_state.get("liked_shows", set())
    disliked_shows = st.session_state.get("disliked_shows", set())

    adjust = np.zeros_like(relevance)

    for i, row in enumerate(candidates.itertuples()):
        show_id = row.show_id
        ep_id = row.episode_id

        if show_id in liked_shows:
            adjust[i] += 0.03
        if show_id in disliked_shows:
            adjust[i] -= 0.05
        if ep_id in liked_items:
            adjust[i] += 0.05
        if ep_id in disliked_items:
            adjust[i] -= 0.1
        if ep_id in heard_items:
            adjust[i] -= 0.2

        adjust[i] += penalties.get(show_id, 0.0)

    if lambda_div >= 0.5:
        shown_items = st.session_state.get("shown_items", set())
        for i, row in enumerate(candidates.itertuples()):
            if row.episode_id in shown_items:
                adjust[i] -= 0.02

    relevance = relevance + adjust

    sim_matrix = cosine_similarity(embeddings)
    selected_local_idx = mmr_select(candidates, relevance, sim_matrix, lambda_div)
    ranked = candidates.iloc[selected_local_idx].reset_index(drop=True)

    shown_items = st.session_state.get("shown_items", set())
    for _, row in ranked.iterrows():
        shown_items.add(row["episode_id"])
    st.session_state.shown_items = shown_items

    return ranked


# -----------------------------
# UI
# -----------------------------

def main():
    st.set_page_config(
        page_title="Activity-Aware Podcast Recommender",
        page_icon="🎧",
        layout="centered",
    )

    init_session_state()

    podcasts_df = load_podcasts()
    vectorizer, X = build_tfidf(podcasts_df)

    st.title("🎧 Activity-Aware Podcast Recommender")
    st.caption("Activity-aware ranking with explainability and LLM-based review summaries.")

    st.markdown("### What are you doing now?")
    activity = st.radio(
        "",
        options=["Commute", "Workout", "Focus", "Sleep"],
        horizontal=True,
    )

    workout_mode = None
    if activity == "Workout":
        workout_mode = st.radio(
            "Workout type",
            options=["Run", "Lift", "Cardio"],
            horizontal=True,
        )

    activity_label = current_activity_label(activity, workout_mode)
    st.session_state.current_activity = activity
    st.session_state.current_activity_label = activity_label

    query = st.text_input(
        "Optional topic or keywords",
        placeholder='e.g. "AI ethics", "Premier League", "Stoicism"',
    )

    target_min, target_max = get_activity_target_window(
        activity, workout_mode, activity_label
    )
    duration_range = st.slider(
        "Preferred duration (minutes)",
        min_value=10,
        max_value=90,
        value=(int(max(10, target_min)), int(min(90, target_max))),
        step=5,
    )

    col_lang, col_new, col_div = st.columns([1, 1, 2])
    with col_lang:
        language = st.selectbox("Language", ["Any", "en", "es", "ru"])
    with col_new:
        newness = st.radio(
            "Episode type",
            options=["Evergreen", "New (last 30 days)"],
            index=0,
        )
        is_new_only = newness.startswith("New")
    with col_div:
        lambda_div = st.slider(
            "More familiar  ↔  More exploratory",
            0.0,
            1.0,
            0.5,
            0.05,
            help="Controls MMR diversification: 0 = focus on relevance, 1 = more diversity.",
        )

    if not st.session_state.liked_items and not st.session_state.saved_items:
        st.info("No history yet — tuned for your activity. Pick a few topics to get started.")
    chips_default = st.session_state.get("selected_chips", [])
    chips = st.multiselect(
        "Quick interests (optional)",
        INTEREST_CHIPS,
        default=chips_default,
    )
    if chips != chips_default:
        st.session_state.selected_chips = chips
        apply_chip_interests(chips)

    st.markdown("---")

    ranked = recommend(
        podcasts_df,
        vectorizer,
        X,
        activity,
        workout_mode,
        query,
        duration_range,
        language,
        is_new_only,
        lambda_div,
    )

    if ranked.empty:
        st.warning("No episodes found for these filters.")
    else:
        st.subheader("Recommended episodes")

        for idx, row in ranked.iterrows():
            ep_id = row["episode_id"]
            show_id = row["show_id"]

            log_event(
                "impression",
                activity_label,
                item_id=ep_id,
                extra={"rank": idx + 1, "lambda": lambda_div},
            )

            explain = build_explain_data(row, podcasts_df, vectorizer, X, query)

            st.markdown("---")
            with st.container():
                col_icon, col_main = st.columns([1, 5])
                with col_icon:
                    st.markdown("### 🎙️")
                with col_main:
                    st.markdown(f"#### {row['ep_title']}")
                    st.caption(f"{row['show_title']} · {row['publisher']}")
                    st.write(
                        f"⏱ {row['ep_duration_min']} min · {row['freq']} · "
                        f"Popularity {row['popularity_score']:.2f}"
                    )

                    interest_label = (
                        explain["interest_terms"][0]
                        if explain["interest_terms"]
                        else "mixed"
                    )
                    activity_kw = (
                        explain["activity_keywords"][0]
                        if explain["activity_keywords"]
                        else activity
                    )
                    query_label = query if query else "none"

                    st.markdown(
                        f"`Interest: {interest_label}`  "
                        f"`Activity: {activity_kw}`  "
                        f"`Query: {query_label}`  "
                        f"`Pop: {row['popularity_score']:.2f}`"
                    )

                    st.write(f"**Why we think you'll like it:** {explain['text']}")

                col_a, col_b, col_c = st.columns([2, 3, 3])

                with col_a:
                    if st.button("▶ Play latest", key=f"play_{ep_id}"):
                        log_event(
                            "play",
                            activity_label,
                            item_id=ep_id,
                            extra={"show_id": show_id},
                        )
                        if st.session_state.first_play_ts is None:
                            st.session_state.first_play_ts = time.time()

                    saved_key = f"saved_flag_{ep_id}"
                    saved = st.session_state.get(saved_key, False)
                    label = "💾 Save" if not saved else "✅ Saved"
                    if st.button(label, key=f"save_{ep_id}"):
                        saved = not saved
                        st.session_state[saved_key] = saved
                        if saved:
                            st.session_state.saved_items.add(ep_id)
                            log_event(
                                "save",
                                activity_label,
                                item_id=ep_id,
                                extra={"show_id": show_id},
                            )
                        else:
                            if ep_id in st.session_state.saved_items:
                                st.session_state.saved_items.remove(ep_id)

                with col_b:
                    reason = st.selectbox(
                        "If you dislike, why?",
                        ["Too long", "Not my topic", "Too intense for sleep", "I've heard it", "Other"],
                        key=f"reason_{ep_id}",
                    )
                    col_like, col_dislike = st.columns(2)
                    with col_like:
                        if st.button("👍", key=f"like_{ep_id}"):
                            handle_like(row, activity_label, vectorizer, X)
                    with col_dislike:
                        if st.button("👎", key=f"dislike_{ep_id}"):
                            handle_dislike(row, activity_label, reason, vectorizer, X)

                with col_c:
                    if st.button("⋯ Not for this activity", key=f"na_{ep_id}"):
                        penalties = st.session_state.activity_penalties.get(activity_label, {})
                        penalties[show_id] = penalties.get(show_id, 0.0) - 0.5
                        st.session_state.activity_penalties[activity_label] = penalties
                        log_event(
                            "not_for_activity",
                            activity_label,
                            item_id=ep_id,
                            extra={"show_id": show_id},
                        )

                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        if st.button("50% listened", key=f"c50_{ep_id}"):
                            log_event(
                                "completion_50",
                                activity_label,
                                item_id=ep_id,
                                extra={"show_id": show_id},
                            )
                    with col_c2:
                        if st.button("Finished", key=f"c80_{ep_id}"):
                            log_event(
                                "completion_80",
                                activity_label,
                                item_id=ep_id,
                                extra={"show_id": show_id},
                            )

                    if st.button("Skip", key=f"skip_{ep_id}"):
                        log_event(
                            "skip",
                            activity_label,
                            item_id=ep_id,
                            extra={"show_id": show_id},
                        )

                    # Review summary (LLM over reviews.json)
                    open_key = f"open_summary_{ep_id}"

                    if st.button("Review summary", key=f"review_{ep_id}"):
                        log_event(
                            "review_summary_opened",
                            activity_label,
                            item_id=show_id,
                            extra={"episode_id": ep_id},
                        )
                        with st.spinner("Summarizing listener reviews..."):
                            summarize_reviews(show_id, row["show_title"])
                        st.session_state[open_key] = True

                    expanded = st.session_state.get(open_key, False)
                    if show_id in st.session_state.llm_summaries:
                        with st.expander("What other listeners say", expanded=expanded):
                            summary_text = st.session_state.llm_summaries[show_id]
                            st.write(summary_text)

                why_key = f"why_open_{ep_id}"
                show_why = st.session_state.get(why_key, False)
                if st.button("Why this?", key=f"why_btn_{ep_id}"):
                    show_why = not show_why
                    st.session_state[why_key] = show_why
                    if show_why:
                        log_event("why_opened", activity_label, item_id=ep_id)

                if show_why:
                    st.markdown("##### Why this recommendation?")
                    col_w1, col_w2 = st.columns(2)
                    with col_w1:
                        st.write("**Interest overlap terms**")
                        st.write(", ".join(explain["interest_terms"]) or "—")
                        st.write("**Activity match keywords**")
                        st.write(", ".join(explain["activity_keywords"]) or "—")
                    with col_w2:
                        st.write("**Query hits**")
                        st.write(", ".join(explain["query_terms"]) or "—")
                        st.write("**Because you liked**")
                        st.write(", ".join(explain["because"]) or "—")

    st.markdown("---")
    render_metrics()


if __name__ == "__main__":
    main()
