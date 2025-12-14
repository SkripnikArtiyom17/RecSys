import os
import json
import hashlib
import time
import uuid
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from together import Together

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Hard-pinned dataset paths
# =========================
CSV_PATH = "podcastv2/Data/sample_podcasts.csv"
REVIEWS_PATH_JSON = "podcastv2/Data/reviews.json"
REVIEWS_PATH_JSONL = "podcastv2/Data/reviews.jsonl"

TELEMETRY_PATH = "telemetry.csv"

np.random.seed(0)


# ======================
# Activity configuration
# ======================
ACTIVITY_TERMS = {
    "Commute": ["commute", "drive", "subway", "train", "short", "news", "update"],
    "Workout": ["workout", "training", "run", "gym", "energy", "motivation", "pace"],
    "Focus / Deep work": ["focus", "deep work", "concentration", "learn", "analysis", "calm"],
    "Chores / Cleaning": ["chores", "cleaning", "house", "background", "fun", "easy"],
    "Relax / Wind down": ["relax", "wind down", "sleep", "calm", "soothing", "slow"],
}
WORKOUT_SUBMODES = {
    "Easy": ["easy", "steady", "low intensity", "calm"],
    "HIIT": ["hiit", "intervals", "intense", "fast", "sprint"],
    "Strength": ["strength", "lifting", "weights", "power", "sets"],
    "Run": ["run", "pace", "cadence", "endurance"],
}

DEFAULT_WEIGHTS = {
    "InterestFit": 0.28,
    "ActivityFit": 0.24,
    "QueryMatch": 0.24,
    "Popularity": 0.14,
    "DurationFit": 0.10,
}

REQ_COLS = [
    "show_id", "show_title", "publisher", "tags", "language", "explicit", "avg_len_min", "freq",
    "episode_id", "ep_title", "ep_desc", "ep_duration_min", "soft_start", "publish_ts", "popularity_score",
]

REVIEW_FIELDS = ["podcast_id", "title", "content", "rating", "author_id", "created_at"]


# ======================
# Together key (secure)
# ======================
def get_together_api_key() -> str:
    # Use Streamlit Secrets ONLY (prevents stale env keys)
    if "TOGETHER_API_KEY" not in st.secrets:
        raise RuntimeError("TOGETHER_API_KEY missing in Streamlit Secrets (Settings → Secrets).")

def show_key_debug():
    st.sidebar.caption(f"Running: {Path(__file__).resolve()}")
    try:
        k = get_together_api_key()
        sha16 = hashlib.sha256(k.encode("utf-8")).hexdigest()[:16]
        st.sidebar.caption(f"Together key ✅ | len={len(k)} | prefix={k[:6]} | sha16={sha16}")
    except Exception as e:
        st.sidebar.caption("Together key ❌")
        st.sidebar.error(str(e))


    k = str(st.secrets["TOGETHER_API_KEY"]).strip()

    # Guard against copy/paste issues (no string escape literals to avoid paste corruption)
    if 10 in (ord(ch) for ch in k) or 13 in (ord(ch) for ch in k):
        raise RuntimeError("TOGETHER_API_KEY contains newline characters. Re-paste it cleanly in Secrets.")

    return k


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
# Diagnostics helpers
# ======================
def file_must_exist(path: str):
    p = Path(path)
    if not p.exists():
        st.error(f"File not found: {path}")
        st.write("CWD:", str(Path.cwd()))
        st.write("Root files:", sorted([x.name for x in Path.cwd().iterdir()]))
        st.stop()

def stop_with_csv_debug(df: pd.DataFrame, title: str, missing: Optional[List[str]] = None):
    st.error(title)
    if missing:
        st.write("Missing columns:", missing)
    st.write("CSV shape:", df.shape)
    st.write("CSV columns:", list(df.columns))
    st.write("Head (first 3 rows):")
    st.write(df.head(3))
    st.stop()


# ======================
# Data loading (robust)
# ======================
@st.cache_data(show_spinner=False)
def load_episodes(csv_path: str) -> pd.DataFrame:
    # Robust delimiter inference
    df = pd.read_csv(csv_path, sep=None, engine="python")

    # Normalize column names (strip spaces + BOM)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        stop_with_csv_debug(df, "❌ CSV schema mismatch (required columns not found).", missing=missing)

    df = df.copy()

    # numeric
    for c in ["avg_len_min", "ep_duration_min", "popularity_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["publish_ts_num"] = pd.to_numeric(df["publish_ts"], errors="coerce")

    # text fields
    text_cols = ["tags", "ep_desc", "ep_title", "show_title", "publisher", "language"]
    for c in text_cols:
        df[c] = (
            df[c]
            .fillna("")
            .astype(str)
            .replace("nan", "", regex=False)
            .replace("None", "", regex=False)
            .str.strip()
        )

    # combined text for TF-IDF
    df["text"] = (
        df["show_title"] + " " +
        df["publisher"] + " " +
        df["tags"] + " " +
        df["ep_title"] + " " +
        df["ep_desc"]
    ).str.strip()

    non_empty = int((df["text"] != "").sum())
    if non_empty == 0:
        st.error("❌ Loaded CSV, but all text fields are empty.")
        counts = {c: int((df[c].str.strip() != "").sum()) for c in ["show_title", "publisher", "tags", "ep_title", "ep_desc"]}
        st.write("Non-empty counts per text column:", counts)
        st.write("First 5 rows (text columns only):")
        st.write(df[["show_title", "publisher", "tags", "ep_title", "ep_desc"]].head(5))
        st.stop()

    return df

@st.cache_data(show_spinner=False)
def load_reviews_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
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

def load_reviews_any() -> pd.DataFrame:
    if Path(REVIEWS_PATH_JSON).exists():
        return load_reviews_jsonl(REVIEWS_PATH_JSON)
    if Path(REVIEWS_PATH_JSONL).exists():
        return load_reviews_jsonl(REVIEWS_PATH_JSONL)
    st.error("Reviews file not found. Expected reviews.json or reviews.jsonl in podcastv2/Data/")
    st.stop()


# ======================
# TF-IDF (no crash)
# ======================
@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(df: pd.DataFrame):
    docs = df["text"].astype(str).tolist()

    # Use stop_words=None to avoid empty-vocabulary edge case
    vec = TfidfVectorizer(
        stop_words=None,
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
        max_features=40000,
    )
    X = vec.fit_transform(docs)

    if X.shape[1] == 0:
        st.error("❌ TF-IDF produced empty vocabulary (no usable tokens).")
        st.stop()

    return vec, X


# ======================
# Ranking helpers
# ======================
def minmax_series(s: pd.Series) -> np.ndarray:
    arr = s.astype(float).to_numpy()
    if not np.isfinite(arr).any():
        return np.zeros_like(arr, dtype=float)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)

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

    not_for = st.session_state.get("not_for_activity", {})
    blocked = not_for.get(activity, set())
    if blocked:
        mask = df["show_id"].astype(str).isin(list(blocked)).to_numpy()
        out[mask] *= 0.30

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

    if lang_filter != "Any":
        fmask &= (df["language"].astype(str) == lang_filter).to_numpy()

    dmin, dmax = duration_range
    dur = pd.to_numeric(df["ep_duration_min"], errors="coerce").to_numpy(dtype=float)
    span = max(5, dmax - dmin)
    soft = np.isfinite(dur) & (dur >= max(0, dmin - span)) & (dur <= dmax + span)
    missing = ~np.isfinite(dur)
    fmask &= (soft | missing)

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
# Explainability + UI
# ======================
def concise_tags(tag_str: str) -> str:
    s = safe_str(tag_str)
    parts = [p.strip() for p in s.replace("|", ",").split(",") if p.strip()]
    return ", ".join(parts[:6]) + ("…" if len(parts) > 6 else "")

def explain_row(row: pd.Series, activity: str, workout_submode: str, query: str, df_all: pd.DataFrame) -> str:
    parts = [
        f"**Interest overlap:** {row.get('InterestFit', 0):.3f} (similar to liked items this session)",
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
# Together summary (click only)
# ======================
def together_summarize_reviews(show_id: str, reviews_df: pd.DataFrame) -> str:
    show_id = str(show_id)
    cache = st.session_state.setdefault("review_summaries", {})
    if show_id in cache:
        return cache[show_id]

    api_key = get_together_api_key()

    # Join key enforced: podcast_id == show_id
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
        "Return 2–3 short sentences, balanced, anonymized. "
        "No quotes, no names, no user identifiers."
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
# Session state
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


# ======================
# Main app
# ======================
def main():
    st.set_page_config(page_title="Podcast Recommender", layout="wide")
    init_state()

    # Ensure files exist (clear error if not)
    file_must_exist(CSV_PATH)
    if not Path(REVIEWS_PATH_JSON).exists() and not Path(REVIEWS_PATH_JSONL).exists():
        st.error("Reviews file not found in podcastv2/Data/. Expected reviews.json or reviews.jsonl")
        st.stop()

    episodes = load_episodes(CSV_PATH)
    reviews = load_reviews_any()

    vectorizer, X = build_vectorizer_and_matrix(episodes)

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        activity = st.selectbox("Activity", list(ACTIVITY_TERMS.keys()), index=1)

        workout_submode = "Easy"
        if activity == "Workout":
            workout_submode = st.selectbox("Workout submode", list(WORKOUT_SUBMODES.keys()), index=0)

        query = st.text_input("Search query", value="", placeholder="e.g., productivity, football, AI")

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

        # Apply feedback immediately
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
                        telemetry_log("like", {"activity": activity, "workout_submode": workout_submode, "query": query, "episode_id": selected_episode_id, "show_id": show_id})
                        st.success("Liked. Ranking updated.")

                    if dislike_clicked:
                        st.session_state["disliked_episode_ids"].add(str(selected_episode_id))
                        st.session_state["disliked_show_ids"].add(show_id)
                        bump_dislike_reason(dislike_reason)
                        telemetry_log("dislike", {"activity": activity, "workout_submode": workout_submode, "query": query, "episode_id": selected_episode_id, "show_id": show_id, "dislike_reason": dislike_reason})
                        st.success("Disliked. Ranking updated.")

                    if not_for_act_clicked:
                        add_not_for_activity(activity, show_id)
                        telemetry_log("not_for_activity", {"activity": activity, "workout_submode": workout_submode, "query": query, "episode_id": selected_episode_id, "show_id": show_id})
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
            telemetry_log("play", {"activity": activity, "workout_submode": workout_submode, "query": query, "episode_id": episode_id, "show_id": show_id})
            st.toast("Play clicked (dummy). Logged.")

        if cols[6].button("💾", key=f"save_{episode_id}", help="Save"):
            telemetry_log("save", {"activity": activity, "workout_submode": workout_submode, "query": query, "episode_id": episode_id, "show_id": show_id})
            st.toast("Saved. Logged.")

        with st.expander("Details (Why this? / Review summary)"):
            if st.button("Why this?", key=f"why_{episode_id}"):
                telemetry_log("why_opened", {"activity": activity, "workout_submode": workout_submode, "query": query, "episode_id": episode_id, "show_id": show_id})
            st.markdown(explain_row(row, activity, workout_submode, query, episodes))

            st.divider()

            # STRICT: Together call ONLY on click
            if st.button("Review summary", key=f"sum_{show_id}_{episode_id}"):
                telemetry_log("review_summary_opened", {"activity": activity, "workout_submode": workout_submode, "query": query, "episode_id": episode_id, "show_id": show_id})
                try:
                    summary = together_summarize_reviews(show_id, reviews)
                    st.markdown(summary)
                except Exception as e:
                    telemetry_log("review_summary_error", {"activity": activity, "workout_submode": workout_submode, "query": query, "episode_id": episode_id, "show_id": show_id, "error": str(e)})
                    st.error(f"Review summary error: {e}")

        st.markdown("---")

    st.caption("Join key enforced: reviews.podcast_id == episodes.show_id. No synthetic data; offline ranking.")


if __name__ == "__main__":
    main()

