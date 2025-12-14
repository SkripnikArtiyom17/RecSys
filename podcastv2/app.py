import os
import json
import time
import uuid
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ======================
# File resolution (FIX #1)
# ======================

def _repo_root() -> Path:
    # Streamlit runs from repo root in Cloud; this is still safe locally
    return Path.cwd()

def resolve_file(filename: str, extra_candidates: Optional[List[str]] = None) -> str:
    """
    Find filename across common project folders.
    Does NOT create anything; only resolves existing paths.
    """
    root = _repo_root()
    candidates = [
        root / filename,

        # Most common layouts in your repo
        root / "podcastv2" / filename,
        root / "podcastv2" / "Data" / filename,
        root / "podcastv2" / "data" / filename,

        root / "podcast" / filename,
        root / "podcast" / "Data" / filename,
        root / "podcast" / "data" / filename,

        root / "Midtermproject" / filename,
        root / "Midtermproject" / "Data" / filename,
        root / "Midtermproject" / "data" / filename,
    ]

    if extra_candidates:
        candidates.extend([root / p for p in extra_candidates])

    for p in candidates:
        if p.exists() and p.is_file():
            return str(p)

    # Debugging output in-app (helps on Streamlit Cloud)
    st.error(f"Could not find '{filename}' in expected locations.")
    st.write("CWD:", str(root))
    st.write("Root files:", sorted([p.name for p in root.iterdir()]))

    for folder in ["podcastv2", "podcast", "Midtermproject", "Data", "data"]:
        fp = root / folder
        st.write(f"{folder}/ exists:", fp.exists(), "is_dir:", fp.is_dir())
        if fp.exists() and fp.is_dir():
            st.write(f"{folder}/ files:", sorted([p.name for p in fp.iterdir()]))

    raise FileNotFoundError(f"{filename} not found. Checked: {len(candidates)} candidate paths.")


# ======================
# Together key (SECURE)
# ======================

def _get_together_api_key() -> Optional[str]:
    # Streamlit secrets is the recommended way on Streamlit Cloud
    if "TOGETHER_API_KEY" in st.secrets:
        v = str(st.secrets["TOGETHER_API_KEY"]).strip()
        return v or None
    v = os.getenv("TOGETHER_API_KEY", "").strip()
    return v or None


# ======================
# Expected schema
# ======================

REQ_COLS = [
    "show_id", "show_title", "publisher", "tags", "language", "explicit", "avg_len_min", "freq",
    "episode_id", "ep_title", "ep_desc", "ep_duration_min", "soft_start", "publish_ts", "popularity_score",
]

REVIEW_FIELDS = ["podcast_id", "title", "content", "rating", "author_id", "created_at"]

DEFAULT_WEIGHTS = {
    "InterestFit": 0.28,
    "ActivityFit": 0.24,
    "QueryMatch": 0.24,
    "Popularity": 0.14,
    "DurationFit": 0.10,
}

ACTIVITY_TERMS = {
    "Commute": ["commute", "drive", "subway", "train", "short", "news", "update"],
    "Workout": ["workout", "training", "run", "gym", "energy", "motivation"],
    "Focus / Deep work": ["focus", "deep work", "concentration", "learn", "calm"],
    "Chores / Cleaning": ["chores", "cleaning", "background", "fun", "easy"],
    "Relax / Wind down": ["relax", "wind down", "sleep", "calm", "soothing"],
}
WORKOUT_SUBMODES = {
    "Easy": ["easy", "steady", "low intensity", "calm"],
    "HIIT": ["hiit", "intervals", "intense", "fast"],
    "Strength": ["strength", "lifting", "weights", "power"],
    "Run": ["run", "pace", "cadence", "endurance"],
}

np.random.seed(0)

TELEMETRY_PATH = "telemetry.csv"


# ======================
# Telemetry
# ======================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

def telemetry_log(event: str, payload: Dict):
    row = {"ts": now_iso(), "session_id": st.session_state.get("session_id", ""), "event": event, **payload}
    df = pd.DataFrame([row])
    write_header = not os.path.exists(TELEMETRY_PATH)
    df.to_csv(TELEMETRY_PATH, mode="a", header=write_header, index=False)

def safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)


# ======================
# Data load (FIX #2: delimiter + types)
# ======================

@st.cache_data(show_spinner=False)
def load_episodes(csv_path: str) -> pd.DataFrame:
    # Robust delimiter detection
    df = pd.read_csv(csv_path, sep=None, engine="python")

    # If the CSV was parsed into a single column (common delimiter issue), fail loudly with hint
    if df.shape[1] == 1 and (REQ_COLS[0] not in df.columns):
        raise ValueError(
            "CSV appears to be parsed as a single column. "
            "Likely wrong delimiter/encoding. Ensure a proper header row and delimiter."
        )

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.copy()

    # Coerce numeric columns used in ranking
    for c in ["avg_len_min", "ep_duration_min", "popularity_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["publish_ts_num"] = pd.to_numeric(df["publish_ts"], errors="coerce")

    # Fill text fields safely
    for c in ["tags", "ep_desc", "ep_title", "show_title", "publisher", "language"]:
        df[c] = df[c].fillna("")

    # Combined text
    df["text"] = (
        df["show_title"].astype(str) + " " +
        df["publisher"].astype(str) + " " +
        df["tags"].astype(str) + " " +
        df["ep_title"].astype(str) + " " +
        df["ep_desc"].astype(str)
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
            rows.append({k: obj.get(k, None) for k in REVIEW_FIELDS})

    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return rdf

    rdf["podcast_id"] = rdf["podcast_id"].astype(str)
    rdf["rating"] = pd.to_numeric(rdf["rating"], errors="coerce")
    return rdf


# ======================
# TF-IDF (FIX #2: empty vocab fallback)
# ======================

@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(df: pd.DataFrame):
    docs = df["text"].astype(str).fillna("").tolist()

    # If docs are empty/blank, fail with actionable message
    non_empty = sum(1 for d in docs if d.strip())
    if non_empty == 0:
        raise ValueError("All episode text fields are empty after loading. Check CSV column mapping/content.")

    # Try with english stopwords first; if it yields empty vocab, retry without stopwords.
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            max_features=40000,
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        X = vec.fit_transform(docs)
        if X.shape[1] == 0:
            raise ValueError("Empty vocabulary")
        return vec, X
    except ValueError:
        # fallback: remove stopword filtering (still deterministic)
        vec = TfidfVectorizer(
            stop_words=None,
            max_features=40000,
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        X = vec.fit_transform(docs)
        if X.shape[1] == 0:
            raise ValueError("TF-IDF still produced empty vocabulary. Dataset text may be non-linguistic/too sparse.")
        return vec, X


# ======================
# Ranking utilities
# ======================

def minmax_series(s: pd.Series) -> np.ndarray:
    s = s.astype(float).to_numpy()
    if not np.isfinite(s).any():
        return np.zeros_like(s, dtype=float)
    lo = np.nanmin(s)
    hi = np.nanmax(s)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(s, dtype=float)
    return (s - lo) / (hi - lo)

def make_activity_prompt(activity: str, workout_submode: str) -> str:
    terms = ACTIVITY_TERMS.get(activity, [])
    if activity == "Workout":
        terms = terms + WORKOUT_SUBMODES.get(workout_submode, [])
    return " ".join(terms)

def compute_query_sim(vectorizer: TfidfVectorizer, X, query: str) -> np.ndarray:
    if not query.strip():
        return np.zeros(X.shape[0], dtype=float)
    qv = vectorizer.transform([query])
    return cosine_similarity(X, qv).reshape(-1).astype(float)

def compute_interest_profile_sim(X, liked_idx: List[int]) -> np.ndarray:
    if not liked_idx:
        return np.zeros(X.shape[0], dtype=float)
    profile = X[liked_idx].mean(axis=0)
    return cosine_similarity(X, profile).reshape(-1).astype(float)

def duration_fit(ep_dur: pd.Series, target_min: int, target_max: int) -> np.ndarray:
    d = pd.to_numeric(ep_dur, errors="coerce").to_numpy(dtype=float)
    out = np.zeros_like(d, dtype=float)
    for i, val in enumerate(d):
        if not np.isfinite(val):
            out[i] = 0.0
        elif target_min <= val <= target_max:
            out[i] = 1.0
        else:
            dist = min(abs(val - target_min), abs(val - target_max))
            out[i] = math.exp(-dist / 12.0)
    return out

def filter_new_evergreen(publish_ts_num: pd.Series, toggle: str) -> np.ndarray:
    if toggle == "Any":
        return np.ones(len(publish_ts_num), dtype=bool)
    now_s = time.time()
    ts = publish_ts_num.to_numpy(dtype=float)
    is_new = np.zeros(len(ts), dtype=bool)
    for i, v in enumerate(ts):
        if not np.isfinite(v):
            is_new[i] = True
        else:
            vv = v / 1000.0 if v > 1e11 else v
            is_new[i] = (now_s - vv) <= (60 * 24 * 3600)
    return is_new if toggle == "New" else ~is_new

def apply_feedback_rules(df: pd.DataFrame, scores: np.ndarray, activity: str) -> np.ndarray:
    out = scores.copy()

    # Not-for-activity show downweight
    not_for = st.session_state.get("not_for_activity", {})
    blocked = not_for.get(activity, set())
    if blocked:
        mask = df["show_id"].astype(str).isin(list(blocked)).to_numpy()
        out[mask] *= 0.30

    # dislike reasons influence
    reasons = st.session_state.get("dislike_reason_counts", {})

    if reasons.get("Too long", 0) > 0:
        dmin, dmax = st.session_state.get("duration_range", (10, 60))
        dur = pd.to_numeric(df["ep_duration_min"], errors="coerce").to_numpy(dtype=float)
        long_mask = np.isfinite(dur) & (dur > dmax)
        out[long_mask] *= 0.85 ** int(reasons.get("Too long", 1))

    if reasons.get("Too intense", 0) > 0:
        intense_terms = ["hiit", "intense", "hardcore", "extreme", "aggressive"]
        text = df["text"].astype(str).str.lower()
        mask = np.zeros(len(df), dtype=bool)
        for t in intense_terms:
            mask |= text.str.contains(t, regex=False).to_numpy()
        out[mask] *= 0.88 ** int(reasons.get("Too intense", 1))

    disliked_show_ids = st.session_state.get("disliked_show_ids", set())
    if disliked_show_ids:
        mask = df["show_id"].astype(str).isin(list(disliked_show_ids)).to_numpy()
        out[mask] *= 0.75

    return out

def mmr_select(candidates_idx: np.ndarray, rel_scores: np.ndarray, X, lam: float, k: int) -> List[int]:
    lam = float(lam)
    selected: List[int] = []
    cand = list(candidates_idx.tolist())

    while cand and len(selected) < k:
        best = None
        best_score = -1e18
        for idx in cand:
            rel = float(rel_scores[idx])
            if not selected:
                mmr = rel
            else:
                sims = cosine_similarity(X[idx], X[selected]).reshape(-1)
                max_sim = float(np.max(sims)) if sims.size else 0.0
                mmr = lam * rel - (1.0 - lam) * max_sim

            if (mmr > best_score) or (
                abs(mmr - best_score) < 1e-12 and (best is None or rel > rel_scores[best] or (rel == rel_scores[best] and idx < best))
            ):
                best_score = mmr
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
) -> pd.DataFrame:
    weights = st.session_state.get("weights", DEFAULT_WEIGHTS).copy()

    fmask = np.ones(len(df), dtype=bool)

    # language filter
    if lang_filter != "Any":
        fmask &= (df["language"].astype(str) == lang_filter).to_numpy()

    # duration soft filter (keep missing)
    dmin, dmax = duration_range
    dur = pd.to_numeric(df["ep_duration_min"], errors="coerce").to_numpy(dtype=float)
    span = max(5, dmax - dmin)
    soft = np.isfinite(dur) & (dur >= max(0, dmin - span)) & (dur <= dmax + span)
    missing = ~np.isfinite(dur)
    fmask &= (soft | missing)

    # new/evergreen
    fmask &= filter_new_evergreen(df["publish_ts_num"], new_toggle)

    q_sim = compute_query_sim(vectorizer, X, query)
    act_sim = compute_query_sim(vectorizer, X, make_activity_prompt(activity, workout_submode))

    liked_episode_ids = st.session_state.get("liked_episode_ids", set())
    liked_idx = df.index[df["episode_id"].astype(str).isin(list(liked_episode_ids))].tolist() if liked_episode_ids else []
    interest_sim = compute_interest_profile_sim(X, liked_idx)

    pop = minmax_series(pd.to_numeric(df["popularity_score"], errors="coerce").fillna(0.0))
    dfit = duration_fit(df["ep_duration_min"], dmin, dmax)

    raw = (
        weights["InterestFit"] * interest_sim
        + weights["ActivityFit"] * act_sim
        + weights["QueryMatch"] * q_sim
        + weights["Popularity"] * pop
        + weights["DurationFit"] * dfit
    )

    scored = apply_feedback_rules(df, raw, activity)
    scored = np.where(fmask, scored, -1e9)

    order = np.argsort(-scored, kind="mergesort")
    pool = order[: min(200, len(order))]
    pool = pool[scored[pool] > -1e8]

    selected = mmr_select(pool, scored, X, lam=lam, k=top_n)
    ranked = df.loc[selected].copy()

    ranked["score"] = scored[selected]
    ranked["QueryMatch"] = q_sim[selected]
    ranked["ActivityFit"] = act_sim[selected]
    ranked["InterestFit"] = interest_sim[selected]
    ranked["Popularity"] = pop[selected]
    ranked["DurationFit"] = dfit[selected]

    return ranked


# ======================
# LLM summary (on click only)
# ======================

def together_summarize_reviews(show_id: str, reviews_df: pd.DataFrame) -> str:
    show_id = str(show_id)
    cache = st.session_state.setdefault("review_summaries", {})
    if show_id in cache:
        return cache[show_id]

    api_key = _get_together_api_key()
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY not set in Streamlit secrets or env var.")

    sub = reviews_df[reviews_df["podcast_id"].astype(str) == show_id].copy()
    if sub.empty:
        cache[show_id] = "No reviews found for this show."
        return cache[show_id]

    sub["title"] = sub["title"].fillna("")
    sub["content"] = sub["content"].fillna("")
    sub["rating"] = pd.to_numeric(sub["rating"], errors="coerce")
    avg_rating = sub["rating"].dropna().mean()
    n = len(sub)

    pieces = []
    for _, r in sub.iterrows():
        t = str(r.get("title", "")).strip()
        c = str(r.get("content", "")).strip()
        pieces.append(f"- {t}: {c}" if t else f"- {c}")
    joined = "\n".join(pieces)[:8000]

    system = (
        "You summarize podcast reviews. "
        "Return 2–3 short sentences, balanced, anonymized, no quotes, no names, no user IDs."
    )

    rating_line = f"Average rating: {avg_rating:.2f}/5 from {n} reviews." if pd.notna(avg_rating) else f"{n} reviews."

    user = (
        f"Podcast show_id: {show_id}\n"
        f"{rating_line}\n\n"
        f"Reviews:\n{joined}\n\n"
        "Return only the 2–3 sentence summary."
    )

    from together import Together
    client = Together(api_key=api_key)

    resp = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=160,
    )
    text = resp.choices[0].message.content.strip()
    cache[show_id] = text
    return text


# ======================
# UI helpers
# ======================

def init_state():
    ensure_session_id()
    st.session_state.setdefault("weights", DEFAULT_WEIGHTS.copy())

    st.session_state.setdefault("liked_episode_ids", set())
    st.session_state.setdefault("liked_show_ids", set())
    st.session_state.setdefault("disliked_episode_ids", set())
    st.session_state.setdefault("disliked_show_ids", set())

    st.session_state.setdefault("not_for_activity", {})
    st.session_state.setdefault("dislike_reason_counts", {})

    st.session_state.setdefault("review_summaries", {})
    st.session_state.setdefault("last_selected_episode_id", None)
    st.session_state.setdefault("duration_range", (10, 60))

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
    parts = [
        f"**Interest overlap:** {row.get('InterestFit', 0):.3f}",
        f"**Activity fit:** {row.get('ActivityFit', 0):.3f} (terms: `{make_activity_prompt(activity, workout_submode)}`)",
    ]
    if query.strip():
        parts.append(f"**Query match:** {row.get('QueryMatch', 0):.3f} (query: `{query}`)")

    liked_eps = st.session_state.get("liked_episode_ids", set())
    if liked_eps:
        liked_rows = df_all[df_all["episode_id"].astype(str).isin(list(liked_eps))]
        if not liked_rows.empty:
            ref = liked_rows.iloc[0]
            parts.append(f"**Because you liked:** “{safe_str(ref.get('show_title'))} — {safe_str(ref.get('ep_title'))}”")

    return "\n\n".join(parts)


# ======================
# Main
# ======================

def main():
    st.set_page_config(page_title="Podcast Recommender", layout="wide")
    init_state()

    # Resolve actual dataset locations
    # Try exact filenames you specified; adjust if your actual names differ.
    csv_path = resolve_file("sample_podcasts.csv")
    # reviews might be .json or .jsonl; try both
    try:
        reviews_path = resolve_file("reviews.json")
    except FileNotFoundError:
        reviews_path = resolve_file("reviews.jsonl")

    # Load
    episodes = load_episodes(csv_path)
    reviews = load_reviews_jsonl(reviews_path)

    vectorizer, X = build_vectorizer_and_matrix(episodes)

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        activity = st.selectbox("Activity", list(ACTIVITY_TERMS.keys()), index=1)

        workout_submode = "Easy"
        if activity == "Workout":
            workout_submode = st.selectbox("Workout submode", list(WORKOUT_SUBMODES.keys()), index=0)

        query = st.text_input("Search query", value="", placeholder="e.g., productivity, football, AI, history")

        dmin, dmax = st.slider("Duration (minutes)", 5, 180, value=(10, 60), step=5)
        st.session_state["duration_range"] = (dmin, dmax)

        langs = sorted([l for l in episodes["language"].dropna().astype(str).unique().tolist() if l.strip()])
        lang_filter = st.selectbox("Language", ["Any"] + langs, index=0)

        new_toggle = st.selectbox("New / Evergreen", ["Any", "New", "Evergreen"], index=0)

        lam = st.slider("Diversity (λ)", 0.0, 1.0, value=0.65, step=0.05)

        st.divider()
        st.subheader("Feedback (sidebar-only)")

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

        if like_clicked or dislike_clicked or not_for_act_clicked:
            if not selected_episode_id:
                st.warning("Paste an episode_id from a row first.")
            else:
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
                        telemetry_log("like", {"activity": activity, "query": query, "episode_id": selected_episode_id, "show_id": show_id})
                        st.success("Liked. Ranking updated.")

                    if dislike_clicked:
                        st.session_state["disliked_episode_ids"].add(str(selected_episode_id))
                        st.session_state["disliked_show_ids"].add(show_id)
                        bump_dislike_reason(dislike_reason)
                        telemetry_log("dislike", {"activity": activity, "query": query, "episode_id": selected_episode_id, "show_id": show_id, "dislike_reason": dislike_reason})
                        st.success("Disliked. Ranking updated.")

                    if not_for_act_clicked:
                        add_not_for_activity(activity, show_id)
                        telemetry_log("not_for_activity", {"activity": activity, "query": query, "episode_id": selected_episode_id, "show_id": show_id})
                        st.success("Down-weighted for this activity. Ranking updated.")

        st.divider()
        st.caption("Telemetry: telemetry.csv (append-only)")

    # Main panel
    st.title("Activity-Aware Podcast Recommender")
    st.caption("Offline ranking (TF-IDF + weighted model + MMR). Together called ONLY on Review summary click.")

    ranked = rank_episodes(
        episodes, vectorizer, X,
        activity=activity,
        workout_submode=workout_submode,
        query=query,
        duration_range=(dmin, dmax),
        lang_filter=lang_filter,
        new_toggle=new_toggle,
        lam=lam,
        top_n=20
    )

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

    st.subheader("Recommendations")
    st.caption("Each row is an episode. Copy episode_id into the sidebar to give feedback.")

    header = st.columns([2.2, 2.2, 1.3, 0.9, 0.9, 0.9, 0.9])
    header[0].markdown("**Show / Publisher**")
    header[1].markdown("**Episode**")
    header[2].markdown("**Tags**")
    header[3].markdown("**Lang**")
    header[4].markdown("**Min**")
    header[5].markdown("**Play**")
    header[6].markdown("**Save**")

    for _, row in ranked.reset_index(drop=True).iterrows():
        episode_id = str(row["episode_id"])
        show_id = str(row["show_id"])

        cols = st.columns([2.2, 2.2, 1.3, 0.9, 0.9, 0.9, 0.9])
        cols[0].write(f"{safe_str(row['show_title'])}\n\n_{safe_str(row['publisher'])}_")
        cols[1].write(f"**{safe_str(row['ep_title'])}**\n\n`episode_id`: {episode_id}")
        cols[2].write(concise_tags(row["tags"]))
        cols[3].write(safe_str(row["language"]) or "—")
        cols[4].write("" if pd.isna(row["ep_duration_min"]) else f"{float(row['ep_duration_min']):.0f}")

        if cols[5].button("▶️", key=f"play_{episode_id}", help="Play (dummy)"):
            telemetry_log("play", {"activity": activity, "query": query, "episode_id": episode_id, "show_id": show_id})
            st.toast("Play clicked (dummy). Logged.")

        if cols[6].button("💾", key=f"save_{episode_id}", help="Save"):
            telemetry_log("save", {"activity": activity, "query": query, "episode_id": episode_id, "show_id": show_id})
            st.toast("Saved. Logged.")

        with st.expander("Details (Why this? / Review summary)"):
            if st.button("Why this?", key=f"why_{episode_id}"):
                telemetry_log("why_opened", {"activity": activity, "query": query, "episode_id": episode_id, "show_id": show_id})
            st.markdown(explain_row(row, activity, workout_submode, query, episodes))

            st.divider()

            # STRICT: Together call ONLY on click
            if st.button("Review summary", key=f"sum_{show_id}_{episode_id}"):
                telemetry_log("review_summary_opened", {"activity": activity, "query": query, "episode_id": episode_id, "show_id": show_id})
                try:
                    summary = together_summarize_reviews(show_id, reviews)
                    st.markdown(summary)
                except Exception as e:
                    telemetry_log("review_summary_error", {"activity": activity, "query": query, "episode_id": episode_id, "show_id": show_id, "error": str(e)})
                    st.error(f"Review summary error: {e}")

        st.markdown("---")

    st.caption("Join key enforced: reviews.podcast_id == episodes.show_id. No synthetic data; offline ranking.")


if __name__ == "__main__":
    main()
