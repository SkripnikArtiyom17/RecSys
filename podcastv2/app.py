import os
import json
import time
import uuid
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# ==========
# Constants
# ==========

DATA_CSV_PATH = "podcastv2/Data/sample_podcasts.csv"
REVIEWS_JSONL_PATH = "podcastv2/Data/reviews.json"   # or .jsonl if that’s the real filename
TELEMETRY_PATH = "telemetry.csv"

# NOTE: For security, do NOT hardcode API keys in code.
# Put your key in:
# 1) .streamlit/secrets.toml as TOGETHER_API_KEY="..."
# OR 2) environment variable TOGETHER_API_KEY
def _get_together_api_key() -> Optional[str]:
    if "TOGETHER_API_KEY" in st.secrets:
        return str(st.secrets["TOGETHER_API_KEY"]).strip()
    return os.getenv("TOGETHER_API_KEY", "").strip() or None


# Activity taxonomy (deterministic, local)
ACTIVITY_TERMS = {
    "Commute": ["commute", "drive", "subway", "train", "short", "news", "catch up", "update"],
    "Workout": ["workout", "training", "run", "gym", "energy", "motivation", "pace", "intense"],
    "Focus / Deep work": ["focus", "deep work", "concentration", "learn", "analysis", "calm", "no distraction"],
    "Chores / Cleaning": ["chores", "cleaning", "house", "light", "background", "fun", "easy"],
    "Relax / Wind down": ["relax", "wind down", "sleep", "calm", "meditation", "slow", "soothing"],
}
WORKOUT_SUBMODES = {
    "Easy": ["easy", "steady", "zone 2", "low intensity", "calm"],
    "HIIT": ["hiit", "intervals", "sprint", "intense", "fast"],
    "Strength": ["strength", "lifting", "weights", "power", "sets"],
    "Run": ["run", "pace", "cadence", "endurance"],
}

# Default weights for the simple relevance model
DEFAULT_WEIGHTS = {
    "InterestFit": 0.28,
    "ActivityFit": 0.24,
    "QueryMatch": 0.24,
    "Popularity": 0.14,
    "DurationFit": 0.10,
}

# Columns we expect (do not modify the dataset)
REQ_COLS = [
    "show_id",
    "show_title",
    "publisher",
    "tags",
    "language",
    "explicit",
    "avg_len_min",
    "freq",
    "episode_id",
    "ep_title",
    "ep_desc",
    "ep_duration_min",
    "soft_start",
    "publish_ts",
    "popularity_score",
]

REVIEW_FIELDS = ["podcast_id", "title", "content", "rating", "author_id", "created_at"]

# Determinism
np.random.seed(0)


# ==================
# Utility / telemetry
# ==================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

def telemetry_log(event: str, payload: Dict):
    """
    Append-only CSV telemetry.
    """
    row = {
        "ts": now_iso(),
        "session_id": st.session_state.get("session_id", ""),
        "event": event,
        **payload,
    }
    df = pd.DataFrame([row])
    write_header = not os.path.exists(TELEMETRY_PATH)
    df.to_csv(TELEMETRY_PATH, mode="a", header=write_header, index=False)

def safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    s = safe_str(x).strip().lower()
    return s in ("true", "1", "yes", "y", "t")


# ==========
# Data Load
# ==========

@st.cache_data(show_spinner=False)
def load_episodes(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Keep data as-is; only light type coercions for computation
    df = df.copy()
    for c in ["avg_len_min", "ep_duration_min", "popularity_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # publish_ts might be numeric/str; store parsed seconds when possible
    df["publish_ts_num"] = pd.to_numeric(df["publish_ts"], errors="coerce")

    # normalized text fields for ranking
    df["tags"] = df["tags"].fillna("")
    df["ep_desc"] = df["ep_desc"].fillna("")
    df["ep_title"] = df["ep_title"].fillna("")
    df["show_title"] = df["show_title"].fillna("")
    df["publisher"] = df["publisher"].fillna("")
    df["language"] = df["language"].fillna("")

    # per-episode combined text for TF-IDF
    df["text"] = (
        df["show_title"].astype(str) + " "
        + df["publisher"].astype(str) + " "
        + df["tags"].astype(str) + " "
        + df["ep_title"].astype(str) + " "
        + df["ep_desc"].astype(str)
    )

    return df

@st.cache_data(show_spinner=False)
def load_reviews_jsonl(jsonl_path: str) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Keep only known fields (do not synthesize anything)
            rows.append({k: obj.get(k, None) for k in REVIEW_FIELDS})
    rdf = pd.DataFrame(rows)
    if len(rdf) == 0:
        return rdf
    rdf["rating"] = pd.to_numeric(rdf["rating"], errors="coerce")
    rdf["podcast_id"] = rdf["podcast_id"].astype(str)
    return rdf

@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(df: pd.DataFrame):
    # Deterministic TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=40000,
        ngram_range=(1, 2),
        lowercase=True,
    )
    X = vectorizer.fit_transform(df["text"].astype(str).tolist())
    return vectorizer, X


# ===========================
# Ranking model + diversification
# ===========================

def minmax_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo = np.nanmin(s.values) if np.isfinite(np.nanmin(s.values)) else 0.0
    hi = np.nanmax(s.values) if np.isfinite(np.nanmax(s.values)) else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)

def make_activity_prompt(activity: str, workout_submode: str) -> str:
    terms = ACTIVITY_TERMS.get(activity, [])
    if activity == "Workout" and workout_submode in WORKOUT_SUBMODES:
        terms = terms + WORKOUT_SUBMODES[workout_submode]
    return " ".join(terms)

def compute_query_sim(vectorizer: TfidfVectorizer, X, query: str) -> np.ndarray:
    if not query.strip():
        return np.zeros(X.shape[0], dtype=float)
    qv = vectorizer.transform([query])
    sims = cosine_similarity(X, qv).reshape(-1)
    return sims.astype(float)

def compute_interest_profile_sim(X, liked_idx: List[int]) -> np.ndarray:
    if not liked_idx:
        return np.zeros(X.shape[0], dtype=float)
    profile = X[liked_idx].mean(axis=0)
    sims = cosine_similarity(X, profile).reshape(-1)
    return np.asarray(sims).astype(float)

def duration_fit(ep_dur: pd.Series, target_min: int, target_max: int) -> np.ndarray:
    # 1.0 if inside range, else smooth decay by distance (minutes)
    d = ep_dur.to_numpy(dtype=float)
    out = np.zeros_like(d, dtype=float)
    for i, val in enumerate(d):
        if not np.isfinite(val):
            out[i] = 0.0
            continue
        if target_min <= val <= target_max:
            out[i] = 1.0
        else:
            dist = min(abs(val - target_min), abs(val - target_max))
            out[i] = math.exp(-dist / 12.0)  # deterministic smooth penalty
    return out

def is_evergreen(publish_ts_num: pd.Series, toggle: str) -> np.ndarray:
    """
    toggle: "New", "Evergreen", "Any"
    Uses a simple cutoff: last 60 days considered "New".
    """
    if toggle == "Any":
        return np.ones(len(publish_ts_num), dtype=bool)

    # publish_ts_num can be unix seconds or other numeric; we handle seconds-like.
    # If missing, treat as "Any"
    now_s = time.time()
    ts = publish_ts_num.to_numpy(dtype=float)
    is_new = np.zeros(len(ts), dtype=bool)
    for i, v in enumerate(ts):
        if not np.isfinite(v):
            is_new[i] = True  # don't exclude unknowns
        else:
            # If value looks like milliseconds, convert
            vv = v / 1000.0 if v > 1e11 else v
            is_new[i] = (now_s - vv) <= (60 * 24 * 3600)

    if toggle == "New":
        return is_new
    if toggle == "Evergreen":
        return ~is_new
    return np.ones(len(ts), dtype=bool)

def apply_feedback_rules(df: pd.DataFrame, base_scores: np.ndarray, activity: str) -> np.ndarray:
    """
    Down/up-weights based on session feedback.
    """
    scores = base_scores.copy()

    # per-activity show downweights
    not_for = st.session_state.get("not_for_activity", {})  # {activity: set(show_id)}
    blocked_shows = not_for.get(activity, set())

    if blocked_shows:
        mask = df["show_id"].astype(str).isin(list(blocked_shows)).to_numpy()
        scores[mask] *= 0.30  # strong down-weight for that activity

    # dislike reasons -> simple deterministic adjustments
    # counters stored as {reason: count}
    reasons = st.session_state.get("dislike_reason_counts", {})

    # If user often dislikes "Too long", penalize durations > target_max more
    if reasons.get("Too long", 0) > 0:
        target_min, target_max = st.session_state.get("duration_range", (10, 60))
        dur = pd.to_numeric(df["ep_duration_min"], errors="coerce").fillna(np.nan).to_numpy(float)
        long_mask = np.isfinite(dur) & (dur > target_max)
        scores[long_mask] *= 0.85 ** reasons.get("Too long", 1)

    # If user dislikes "Too intense", penalize intensity-ish tags
    if reasons.get("Too intense", 0) > 0:
        intense_terms = ["hiit", "intense", "hardcore", "extreme", "aggressive"]
        text = df["text"].astype(str).str.lower()
        mask = np.zeros(len(df), dtype=bool)
        for t in intense_terms:
            mask |= text.str.contains(t, regex=False).to_numpy()
        scores[mask] *= 0.88 ** reasons.get("Too intense", 1)

    # If user dislikes "Not my topics", downweight shows similar to most-disliked shows
    disliked_show_ids = st.session_state.get("disliked_show_ids", set())
    if disliked_show_ids:
        # mild penalty at show level
        mask = df["show_id"].astype(str).isin(list(disliked_show_ids)).to_numpy()
        scores[mask] *= 0.75

    return scores

def mmr_select(
    candidates_idx: np.ndarray,
    rel_scores: np.ndarray,
    X,
    lam: float,
    k: int
) -> List[int]:
    """
    Standard MMR: maximize lam*relevance - (1-lam)*max_sim_to_selected
    Deterministic: ties resolved by higher relevance then lower index.
    """
    lam = float(lam)
    selected: List[int] = []
    cand = list(candidates_idx.tolist())

    if not cand:
        return selected

    # Precompute candidate vectors
    # (Note: sparse matrix supports row slicing)
    while cand and len(selected) < k:
        best = None
        best_score = -1e18
        for idx in cand:
            rel = float(rel_scores[idx])
            if not selected:
                mmr_score = rel
            else:
                sims = cosine_similarity(X[idx], X[selected]).reshape(-1)
                max_sim = float(np.max(sims)) if sims.size else 0.0
                mmr_score = lam * rel - (1.0 - lam) * max_sim

            # deterministic tie-breaks
            if (mmr_score > best_score) or (
                abs(mmr_score - best_score) < 1e-12 and (best is None or rel > rel_scores[best] or (rel == rel_scores[best] and idx < best))
            ):
                best_score = mmr_score
                best = idx

        selected.append(best)
        cand.remove(best)

    return selected

def rank_episodes(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    X,
    activity: str,
    workout_submode: str,
    query: str,
    duration_range: Tuple[int, int],
    lang_filter: str,
    new_toggle: str,
    lam: float,
    top_n: int = 20
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Returns ranked df (top_n) and debug feature arrays for explainability.
    """
    # Filters (deterministic)
    fmask = np.ones(len(df), dtype=bool)

    # language filter
    if lang_filter != "Any":
        fmask &= (df["language"].astype(str) == lang_filter).to_numpy()

    # duration filter (hard filter: within +/- 2x range; still allows DurationFit scoring)
    dmin, dmax = duration_range
    dur = pd.to_numeric(df["ep_duration_min"], errors="coerce").to_numpy(dtype=float)
    soft = np.ones(len(df), dtype=bool)
    soft &= np.isfinite(dur)
    soft &= (dur >= max(0, dmin - (dmax - dmin))) & (dur <= dmax + (dmax - dmin))
    # If duration missing, keep
    missing_dur = ~np.isfinite(dur)
    fmask &= (soft | missing_dur)

    # New/Evergreen filter
    keep_new = is_evergreen(df["publish_ts_num"], new_toggle)
    fmask &= keep_new

    # Feature computation
    weights = st.session_state.get("weights", DEFAULT_WEIGHTS).copy()

    q_sim = compute_query_sim(vectorizer, X, query)
    act_prompt = make_activity_prompt(activity, workout_submode)
    act_sim = compute_query_sim(vectorizer, X, act_prompt)

    # interest profile from liked episodes (episode-level)
    liked_episode_ids = st.session_state.get("liked_episode_ids", set())
    if liked_episode_ids:
        liked_idx = df.index[df["episode_id"].astype(str).isin(list(liked_episode_ids))].tolist()
    else:
        liked_idx = []
    interest_sim = compute_interest_profile_sim(X, liked_idx)

    pop = minmax_series(pd.to_numeric(df["popularity_score"], errors="coerce").fillna(0.0)).to_numpy(dtype=float)
    dur_fit = duration_fit(df["ep_duration_min"], dmin, dmax)

    # Weighted sum
    raw = (
        weights["InterestFit"] * interest_sim
        + weights["ActivityFit"] * act_sim
        + weights["QueryMatch"] * q_sim
        + weights["Popularity"] * pop
        + weights["DurationFit"] * dur_fit
    )

    # Apply feedback rules
    scored = apply_feedback_rules(df, raw, activity)

    # Apply filter mask
    scored = np.where(fmask, scored, -1e9)

    # MMR over top candidates
    # Candidate pool: top 200 by scored (deterministic stable sort)
    order = np.argsort(-scored, kind="mergesort")
    pool = order[: min(200, len(order))]
    pool = pool[scored[pool] > -1e8]  # valid
    selected_idx = mmr_select(pool, scored, X, lam=lam, k=top_n)

    ranked = df.loc[selected_idx].copy()
    ranked["score"] = scored[selected_idx]
    ranked["QueryMatch"] = q_sim[selected_idx]
    ranked["ActivityFit"] = act_sim[selected_idx]
    ranked["InterestFit"] = interest_sim[selected_idx]
    ranked["Popularity"] = pop[selected_idx]
    ranked["DurationFit"] = dur_fit[selected_idx]

    debug = {
        "q_sim": q_sim,
        "act_sim": act_sim,
        "interest_sim": interest_sim,
    }
    return ranked, debug


# ==========================
# Together LLM summarization
# ==========================

def together_summarize_reviews(show_id: str, reviews_df: pd.DataFrame) -> str:
    """
    Calls Together ONLY on user click. Summarizes reviews for this show_id (join: podcast_id == show_id).
    Caches in st.session_state["review_summaries"][show_id].
    """
    show_id = str(show_id)

    cache = st.session_state.setdefault("review_summaries", {})
    if show_id in cache:
        return cache[show_id]

    api_key = _get_together_api_key()
    if not api_key:
        raise RuntimeError(
            "TOGETHER_API_KEY not set. Add it to .streamlit/secrets.toml or environment variable."
        )

    # Strict join key
    sub = reviews_df[reviews_df["podcast_id"].astype(str) == show_id].copy()
    if sub.empty:
        cache[show_id] = "No reviews found for this show."
        return cache[show_id]

    # Build a compact input (deterministic truncation)
    sub["title"] = sub["title"].fillna("")
    sub["content"] = sub["content"].fillna("")
    sub["rating"] = pd.to_numeric(sub["rating"], errors="coerce")

    avg_rating = sub["rating"].dropna().mean()
    n = len(sub)

    # Concatenate a limited amount of text
    pieces = []
    for _, r in sub.iterrows():
        t = str(r.get("title", "")).strip()
        c = str(r.get("content", "")).strip()
        if t:
            pieces.append(f"- {t}: {c}")
        else:
            pieces.append(f"- {c}")
    joined = "\n".join(pieces)
    joined = joined[:8000]  # hard cap to keep request small/deterministic

    system = (
        "You are a helpful assistant summarizing podcast reviews. "
        "Write a short, balanced, anonymized summary in 2–3 sentences. "
        "Mention both positives and negatives if present. Do not quote users verbatim. "
        "No names, no user identifiers."
    )

    rating_line = f"Average rating: {avg_rating:.2f}/5 from {n} reviews." if pd.notna(avg_rating) else f"{n} reviews (rating missing)."

    user = (
        f"Podcast show_id: {show_id}\n"
        f"{rating_line}\n\n"
        f"Reviews:\n{joined}\n\n"
        "Return only the 2–3 sentence summary."
    )

    # Import only when needed (strict: no background calls)
    from together import Together  # noqa
    client = Together(api_key=api_key)

    # Keep deterministic as possible: low temperature
    resp = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=160,
    )

    text = resp.choices[0].message.content.strip()
    cache[show_id] = text
    return text


# ==========
# UI helpers
# ==========

def init_state():
    ensure_session_id()
    st.session_state.setdefault("weights", DEFAULT_WEIGHTS.copy())

    st.session_state.setdefault("liked_episode_ids", set())     # episode-level
    st.session_state.setdefault("liked_show_ids", set())        # show-level (optional reference)
    st.session_state.setdefault("disliked_episode_ids", set())
    st.session_state.setdefault("disliked_show_ids", set())

    st.session_state.setdefault("not_for_activity", {})         # {activity: set(show_id)}
    st.session_state.setdefault("dislike_reason_counts", {})    # {reason: count}

    st.session_state.setdefault("review_summaries", {})         # {show_id: summary}
    st.session_state.setdefault("last_selected_episode_id", None)

    st.session_state.setdefault("duration_range", (10, 60))     # for feedback rules

def add_not_for_activity(activity: str, show_id: str):
    d = st.session_state.setdefault("not_for_activity", {})
    d.setdefault(activity, set()).add(str(show_id))

def bump_dislike_reason(reason: str):
    d = st.session_state.setdefault("dislike_reason_counts", {})
    d[reason] = int(d.get(reason, 0)) + 1

def concise_tags(tag_str: str) -> str:
    s = safe_str(tag_str)
    parts = [p.strip() for p in s.replace("|", ",").split(",") if p.strip()]
    return ", ".join(parts[:6]) + ("…" if len(parts) > 6 else "")

def explain_row(row: pd.Series, activity: str, workout_submode: str, query: str, df_all: pd.DataFrame) -> str:
    """
    Concise explainability: interest overlap, activity terms, query matches, reference to liked items.
    """
    parts = []
    parts.append(f"**Interest overlap:** {row.get('InterestFit', 0):.3f} (similar to what you liked this session).")
    parts.append(f"**Activity fit:** {row.get('ActivityFit', 0):.3f} (matches: `{make_activity_prompt(activity, workout_submode)}`).")
    if query.strip():
        parts.append(f"**Query match:** {row.get('QueryMatch', 0):.3f} (query: `{query}`).")

    liked_eps = st.session_state.get("liked_episode_ids", set())
    if liked_eps:
        # Reference one liked show title (deterministic pick)
        liked_rows = df_all[df_all["episode_id"].astype(str).isin(list(liked_eps))]
        if not liked_rows.empty:
            ref = liked_rows.iloc[0]
            parts.append(f"**Because you liked:** “{safe_str(ref.get('show_title'))} — {safe_str(ref.get('ep_title'))}”.")
    return "\n\n".join(parts)

def render_header():
    st.title("Activity-Aware Podcast Recommender")
    st.caption("Offline ranking (TF-IDF + weighted model + MMR). Review summaries call Together only when clicked.")


# ==============
# Main app
# ==============

def main():
    st.set_page_config(page_title="Podcast Recommender", layout="wide")
    init_state()

    # Load data (in-memory only, no dataset modifications)
    try:
        episodes = load_episodes(DATA_CSV_PATH)
    except Exception as e:
        st.error(f"Failed to load {DATA_CSV_PATH}: {e}")
        return

    try:
        reviews = load_reviews_jsonl(REVIEWS_JSONL_PATH)
    except Exception as e:
        st.error(f"Failed to load {REVIEWS_JSONL_PATH}: {e}")
        return

    vectorizer, X = build_vectorizer_and_matrix(episodes)

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        activity = st.selectbox("Activity", list(ACTIVITY_TERMS.keys()), index=1)
        workout_submode = "Easy"
        if activity == "Workout":
            workout_submode = st.selectbox("Workout submode", list(WORKOUT_SUBMODES.keys()), index=0)
        else:
            st.caption("Workout submode (disabled)")
            workout_submode = "Easy"

        query = st.text_input("Search query", value="", placeholder="e.g., productivity, football, AI, history")

        # Duration filter
        dmin, dmax = st.slider("Duration (minutes)", 5, 180, value=(10, 60), step=5)
        st.session_state["duration_range"] = (dmin, dmax)

        # Language filter (from data)
        langs = sorted([l for l in episodes["language"].dropna().astype(str).unique().tolist() if l.strip()])
        lang_filter = st.selectbox("Language", ["Any"] + langs, index=0)

        new_toggle = st.selectbox("New / Evergreen", ["Any", "New", "Evergreen"], index=0)

        lam = st.slider("Diversity (λ)", 0.0, 1.0, value=0.65, step=0.05)

        st.divider()

        # Feedback panel (sidebar only)
        st.subheader("Feedback")
        selected_episode_id = st.text_input(
            "Selected episode_id (paste from table row)",
            value=st.session_state.get("last_selected_episode_id") or "",
            placeholder="episode_id"
        ).strip()

        dislike_reason = st.selectbox(
            "Dislike reason",
            ["Too long", "Too intense", "Not my topics", "Low quality", "Other"],
            index=0
        )

        c1, c2, c3 = st.columns(3)
        like_clicked = c1.button("👍 Like", use_container_width=True)
        dislike_clicked = c2.button("👎 Dislike", use_container_width=True)
        not_for_act_clicked = c3.button("🚫 Not for this activity", use_container_width=True)

        # Apply feedback immediately
        if like_clicked or dislike_clicked or not_for_act_clicked:
            if not selected_episode_id:
                st.warning("Paste an episode_id from a row first.")
            else:
                # Find row
                m = episodes[episodes["episode_id"].astype(str) == selected_episode_id]
                if m.empty:
                    st.warning("episode_id not found in dataset.")
                else:
                    r = m.iloc[0]
                    show_id = str(r["show_id"])

                    st.session_state["last_selected_episode_id"] = selected_episode_id

                    if like_clicked:
                        st.session_state["liked_episode_ids"].add(str(selected_episode_id))
                        st.session_state["liked_show_ids"].add(show_id)
                        telemetry_log("like", {
                            "activity": activity,
                            "workout_submode": workout_submode,
                            "query": query,
                            "episode_id": selected_episode_id,
                            "show_id": show_id,
                        })
                        st.success("Liked. Ranking updated.")

                    if dislike_clicked:
                        st.session_state["disliked_episode_ids"].add(str(selected_episode_id))
                        st.session_state["disliked_show_ids"].add(show_id)
                        bump_dislike_reason(dislike_reason)
                        telemetry_log("dislike", {
                            "activity": activity,
                            "workout_submode": workout_submode,
                            "query": query,
                            "episode_id": selected_episode_id,
                            "show_id": show_id,
                            "dislike_reason": dislike_reason,
                        })
                        st.success("Disliked. Ranking updated.")

                    if not_for_act_clicked:
                        add_not_for_activity(activity, show_id)
                        telemetry_log("not_for_activity", {
                            "activity": activity,
                            "workout_submode": workout_submode,
                            "query": query,
                            "episode_id": selected_episode_id,
                            "show_id": show_id,
                        })
                        st.success("Down-weighted for this activity. Ranking updated.")

        st.divider()
        st.caption("Telemetry written to telemetry.csv (append-only).")

    # Main panel
    render_header()

    ranked, _debug = rank_episodes(
        episodes,
        vectorizer,
        X,
        activity=activity,
        workout_submode=workout_submode,
        query=query,
        duration_range=(dmin, dmax),
        lang_filter=lang_filter,
        new_toggle=new_toggle,
        lam=lam,
        top_n=20
    )

    # Telemetry impression for current result set (once per render)
    telemetry_log("impression", {
        "activity": activity,
        "workout_submode": workout_submode,
        "query": query,
        "n_results": len(ranked),
        "lambda": lam,
        "duration_min": dmin,
        "duration_max": dmax,
        "language": lang_filter,
        "new_toggle": new_toggle,
    })

    # Table-like renderer (minimal + mobile-friendly)
    st.subheader("Recommendations")
    st.caption("Each row is an episode. Copy the episode_id into the sidebar to give feedback.")

    # Header row
    h = st.columns([2.2, 2.2, 1.3, 0.9, 0.9, 0.9, 0.9])
    h[0].markdown("**Show / Publisher**")
    h[1].markdown("**Episode**")
    h[2].markdown("**Tags**")
    h[3].markdown("**Lang**")
    h[4].markdown("**Min**")
    h[5].markdown("**Play**")
    h[6].markdown("**Save**")

    for i, row in ranked.reset_index(drop=True).iterrows():
        episode_id = str(row["episode_id"])
        show_id = str(row["show_id"])

        cols = st.columns([2.2, 2.2, 1.3, 0.9, 0.9, 0.9, 0.9])
        cols[0].write(f"{safe_str(row['show_title'])}\n\n_{safe_str(row['publisher'])}_")
        cols[1].write(f"**{safe_str(row['ep_title'])}**\n\n`episode_id`: {episode_id}")
        cols[2].write(concise_tags(row["tags"]))
        cols[3].write(safe_str(row["language"]) or "—")
        cols[4].write("" if pd.isna(row["ep_duration_min"]) else f"{float(row['ep_duration_min']):.0f}")

        play_key = f"play_{episode_id}"
        save_key = f"save_{episode_id}"

        if cols[5].button("▶️", key=play_key, help="Play (dummy)"):
            telemetry_log("play", {
                "activity": activity,
                "workout_submode": workout_submode,
                "query": query,
                "episode_id": episode_id,
                "show_id": show_id,
            })
            st.toast("Play clicked (dummy). Logged.")

        if cols[6].button("💾", key=save_key, help="Save"):
            telemetry_log("save", {
                "activity": activity,
                "workout_submode": workout_submode,
                "query": query,
                "episode_id": episode_id,
                "show_id": show_id,
            })
            st.toast("Saved. Logged.")

        # Expandable row area
        with st.expander("Details (Why this? / Review summary)"):
            if st.button("Why this?", key=f"why_{episode_id}"):
                telemetry_log("why_opened", {
                    "activity": activity,
                    "workout_submode": workout_submode,
                    "query": query,
                    "episode_id": episode_id,
                    "show_id": show_id,
                })
            st.markdown(explain_row(row, activity, workout_submode, query, episodes))

            st.divider()

            # STRICT: LLM called ONLY on click
            if st.button("Review summary", key=f"sum_{show_id}_{episode_id}"):
                telemetry_log("review_summary_opened", {
                    "activity": activity,
                    "workout_submode": workout_submode,
                    "query": query,
                    "episode_id": episode_id,
                    "show_id": show_id,
                })
                try:
                    summary = together_summarize_reviews(show_id, reviews)
                    st.markdown(summary)
                except Exception as e:
                    telemetry_log("review_summary_error", {
                        "activity": activity,
                        "workout_submode": workout_submode,
                        "query": query,
                        "episode_id": episode_id,
                        "show_id": show_id,
                        "error": str(e),
                    })
                    st.error(f"Review summary error: {e}")

        st.markdown("---")

    st.caption("Join key enforced: reviews.podcast_id == episodes.show_id. No synthetic data; all local ranking.")


if __name__ == "__main__":
    main()
