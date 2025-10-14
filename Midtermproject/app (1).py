# Activity-Aware Podcast Recommender (Streamlit)
# Runs fully offline. Implements: TF-IDF features, MMR diversification, explainability,
# session-level feedback loop, cold start interests, guardrails, and simple telemetry.
#
# How to run (also see chat message):
#   1) pip install streamlit scikit-learn pandas numpy
#   2) streamlit run app.py

import os
import time
import math
import uuid
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------- Page Setup ----------------------------
st.set_page_config(page_title="Activity-Aware Podcast Recommender", page_icon="🎧", layout="centered")

# Backward-compatible segmented control
segmented = getattr(st, "segmented_control", None)


# High-contrast style + mobile-first tweaks
st.markdown("""

<style>
:root{
  --bg-left:#c42a66;
  --bg-deep:#0f1116;
  --bg-teal:#17343a;
  --accent:#e15b8a;
  --text:#e8edf2;
  --muted:#b9c0c7;
}
/* Base layout and background gradients to mimic the slide */
.stApp {
  background:
    radial-gradient(1100px 750px at 95% 0%, rgba(225,91,138,0.35) 0%, rgba(225,91,138,0.08) 45%, rgba(0,0,0,0) 46%),
    linear-gradient(120deg, var(--bg-left) 0%, var(--bg-deep) 35%, var(--bg-teal) 70%, var(--bg-deep) 100%);
  color: var(--text);
  min-height: 100vh;
}
.block-container { padding-top: 1.25rem; padding-bottom: 4.5rem; }

/* Typography */
h1, h2, h3, h4 { letter-spacing: 0.2px; }
h1 { font-weight: 700; }
p, .stMarkdown { color: var(--text); }

/* Cards (glass) */
.card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 0.95rem;
  margin-bottom: 0.9rem;
  background: rgba(17,19,26,0.55);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
}

/* Badges */
.badge {
  display: inline-block;
  padding: 0.22rem 0.48rem;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.05);
  margin-right: 6px;
  margin-bottom: 6px;
  font-size: 0.85rem;
  color: var(--muted);
}

/* Buttons */
.stButton>button {
  border-radius: 12px;
  padding: 0.45rem 0.7rem;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
  color: var(--text);
}
.stButton>button:hover {
  border-color: rgba(255,255,255,0.28);
  background: rgba(255,255,255,0.12);
}

/* Inputs */
.stTextInput>div>div>input, .stSelectbox>div>div, .stSlider, .stRadio, .stSegmentedControl, .stMultiSelect {
  color: var(--text) !important;
}
/* Radio (fallback for segmented control) */
.stRadio > label { color: var(--muted); }
.stRadio div[role="radiogroup"] label { padding: 0.15rem 0.2rem; }

.reason { opacity: 0.9; font-size: 0.95rem; color: var(--muted); }

/* Footer bar like the slide baseline */
.app-footer {
  position: fixed;
  left: 0; right: 0; bottom: 18px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.95rem; color: var(--muted);
  pointer-events: none;
}
.app-footer .bar {
  position: absolute; left: 6%; right: 6%; top: -10px; height: 1.5px;
  background: rgba(255,255,255,0.5);
  border-radius: 1px;
}
.app-footer .left, .app-footer .center, .app-footer .right {
  pointer-events: auto;
}
.app-footer .left { position: absolute; left: 6%; }
.app-footer .right { position: absolute; right: 6%; }
.app-footer .center { opacity: 0.95; }
</style>

""", unsafe_allow_html=True)

# ---------------------------- Constants ----------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_podcasts.csv")
TELEMETRY_PATH = os.path.join(os.getcwd(), "telemetry.csv")

ACTIVITIES = ["Commute", "Workout", "Focus", "Sleep"]
WORKOUT_MODES = ["Run", "Lift", "Cardio"]
INTEREST_VOCAB = [
    "ai","ethics","research","markets","finance","strategy","stoicism","history","mystery","investigation",
    "narrative","calm","background","energetic","upbeat","storytelling","interview","news","football",
    "premier league","documentary","space","climate","psychology","software","comedy","humor"
]

# Default weights
DEFAULT_WEIGHTS = dict(wI=0.35, wA=0.25, wQ=0.20, wP=0.15, wD=0.05)

# Activity tag biases (simple lists)
ACTIVITY_TAG_BIASES = {
    "Sleep": {"pos":["calm","storytelling","background","soft"], "neg":["comedy","energetic","high-energy","banter"]},
    "Focus": {"pos":["interview","background","calm","research"], "neg":["comedy","energetic","high-energy","improv"]},
    "Workout": {"pos":["energetic","upbeat","analysis"], "neg":["calm","sleep","soft"]},
    "Commute": {"pos":["narrative","news","briefing"], "neg":[]},
}

# Duration windows per activity/sub-mode
def activity_duration_window(activity, submode):
    if activity == "Sleep":
        return (30, 90)
    if activity == "Focus":
        return (25, 60)
    if activity == "Workout":
        if submode == "Run":
            return (20, 45)
        if submode == "Lift":
            return (35, 80)
        return (25, 60)  # Cardio
    if activity == "Commute":
        return (15, 45)
    return (20, 60)

# Lambda tweaks defaults per activity for diversification
def activity_lambda_default(activity):
    return 0.4 if activity == "Sleep" else (0.6 if activity == "Workout" else 0.5)

# ---------------------------- Utils ----------------------------
def ensure_telemetry_file():
    if not os.path.exists(TELEMETRY_PATH):
        with open(TELEMETRY_PATH, "w", encoding="utf-8") as f:
            f.write("session_id,user_ts,event_type,activity,item_id,details\\n")

def log_event(session_id, event_type, activity, item_id="", details=None):
    ensure_telemetry_file()
    payload = "" if details is None else json.dumps(details, ensure_ascii=False)
    with open(TELEMETRY_PATH, "a", encoding="utf-8") as f:
        f.write(f"{session_id},{datetime.utcnow().isoformat()},{event_type},{activity},{item_id},{payload}\\n")

def soft_clip01(x):
    return max(0.0, min(1.0, float(x)))

def triangular_fit(x, low, high):
    # returns 1 inside window, linearly decays to 0 outside at 50% of window width
    if low <= x <= high:
        return 1.0
    width = max(1.0, high - low)
    if x < low:
        return max(0.0, 1 - (low - x) / (0.5*width))
    else:
        return max(0.0, 1 - (x - high) / (0.5*width))

def freshness_boost(ts_str):
    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        if datetime.now() - ts <= timedelta(days=30):
            return 0.05
    except Exception:
        pass
    return 0.0

def within_last_days(ts_str, days=30):
    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - ts) <= timedelta(days=days)
    except Exception:
        return False

# ---------------------------- Data Load & TF-IDF ----------------------------
@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    # Build combined text field
    df["text"] = (df["ep_title"].fillna("") + " " + df["ep_desc"].fillna("") + " " + df["tags"].fillna("")).str.lower()
    return df

@st.cache_data(show_spinner=False)
def build_tfidf(df):
    vec = TfidfVectorizer(stop_words="english", min_df=1, max_df=0.95)
    X = vec.fit_transform(df["text"].tolist())
    return vec, X

# ---------------------------- Session State ----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "interest_weights" not in st.session_state:
    st.session_state.interest_weights = {}  # term -> weight delta
if "liked_items" not in st.session_state:
    st.session_state.liked_items = []  # list of episode_id
if "saved_items" not in st.session_state:
    st.session_state.saved_items = set()
if "disliked_items" not in st.session_state:
    st.session_state.disliked_items = {}  # episode_id -> reason
if "activity_penalties" not in st.session_state:
    st.session_state.activity_penalties = {}  # (activity, show_id) -> penalty [0..0.4]
if "seen_item_ids" not in st.session_state:
    st.session_state.seen_item_ids = set()
if "first_impression_ts" not in st.session_state:
    st.session_state.first_impression_ts = None
if "first_play_ts" not in st.session_state:
    st.session_state.first_play_ts = None
if "len_bias" not in st.session_state:
    st.session_state.len_bias = 0.0  # positive favors shorter after "Too long"

# ---------------------------- App Header ----------------------------
st.title("🎧 Activity‑Aware Podcast Recommender")
st.caption("Deterministic offline prototype with explainability, diversity (MMR), and in‑session learning.")

df = load_data(DATA_PATH)
vectorizer, X = build_tfidf(df)

# Map episode_id -> idx
ep_index = {eid: i for i, eid in enumerate(df["episode_id"])}

# ---------------------------- Controls ----------------------------
with st.container():
    activity = segmented("What are you doing now?", ACTIVITIES, default=ACTIVITIES[0]) if segmented else st.radio("What are you doing now?", ACTIVITIES, index=0)
    # Workout sub-mode
    submode = None
    if activity == "Workout":
        submode = segmented("Workout mode", WORKOUT_MODES, default=WORKOUT_MODES[0]) if segmented else st.radio("Workout mode", WORKOUT_MODES, index=0)

    # Search + Filters
    query = st.text_input("Search topics (e.g., “AI ethics”, “Premier League”, “Stoicism”)", value="", help="Free-text search across titles, descriptions, and tags.")
    colf1, colf2 = st.columns(2)
    with colf1:
        dur_min, dur_max = st.slider("Duration range (min)", 10, 90, value=activity_duration_window(activity, submode), step=5)
    with colf2:
        language = st.selectbox("Language", options=["any","en","es","ru"], index=0)

    colf3, colf4 = st.columns(2)
    with colf3:
        new_toggle = st.selectbox("Freshness", options=["Any","New (≤30d)","Evergreen (>30d)"], index=0)
    with colf4:
        lam = st.slider("Diversity: More familiar ↔ More exploratory (λ)", 0.0, 1.0, value=activity_lambda_default(activity), step=0.05)

# Safety defaults
hide_explicit = activity in ["Sleep","Focus"]

# Cold start: pick interests if no likes/saves and no selected interest weights
if not st.session_state.liked_items and not st.session_state.saved_items and not st.session_state.interest_weights:
    st.info("No history yet — tuned for your activity. Pick a few interests to get started.")
    picked = st.multiselect("Pick interests (multi-select)", options=INTEREST_VOCAB, default=["calm","interview"] if activity in ["Sleep","Focus"] else ["energetic","analysis"])
    # Create a light positive weight for selected terms
    for t in picked:
        st.session_state.interest_weights[t] = st.session_state.interest_weights.get(t, 0.0) + 0.5

# ---------------------------- Build User Interest Vector ----------------------------
def interest_vector(vectorizer, X, df):
    # Build a pseudo-document using weighted terms from interest_weights and liked items
    terms = []
    for term, w in st.session_state.interest_weights.items():
        if w > 0:
            terms.extend([term] * int(1 + min(5, w*2)))  # replicate by weight
    # Add liked items' texts
    for eid in st.session_state.liked_items[-10:]:
        idx = ep_index.get(eid)
        if idx is not None:
            terms.append(df.iloc[idx]["text"])
    if not terms:
        return None
    vec = vectorizer.transform([" ".join(terms)])
    return vec

user_vec = interest_vector(vectorizer, X, df)

# ---------------------------- Compute Signals ----------------------------
def tag_list(s):
    return [t.strip().lower() for t in str(s or "").split(",")]

def tag_score(tags, pos, neg):
    score = 0.0
    for t in tags:
        if t in pos: score += 0.15
        if t in neg: score -= 0.15
    return score

def activity_fit(row, activity, submode):
    # Duration window for activity
    a_low, a_high = activity_duration_window(activity, submode)
    dur = float(row["ep_duration_min"])
    # Base fit from triangular window (+ user bias favoring shorter if set)
    fit = triangular_fit(dur, a_low, a_high)
    if st.session_state.len_bias > 0:
        # amplify preference for shorter durations
        if dur > (a_low + a_high)/2:
            fit *= max(0.5, 1.0 - 0.3*st.session_state.len_bias)
        else:
            fit = min(1.0, fit + 0.15*st.session_state.len_bias)
    # Tag bias
    biases = ACTIVITY_TAG_BIASES.get(activity, {"pos":[], "neg":[]})
    tlist = tag_list(row["tags"])
    fit += tag_score(tlist, biases.get("pos",[]), biases.get("neg",[]))
    # Soft start preference for Sleep
    if activity == "Sleep" and int(row.get("soft_start",0)) == 1:
        fit += 0.1
    return soft_clip01(fit)

def duration_fit(row, low, high):
    dur = float(row["ep_duration_min"])
    return triangular_fit(dur, low, high)

def popularity_signal(row):
    base = float(row["popularity_score"])
    boost = freshness_boost(str(row["publish_ts"]))
    return soft_clip01(base + boost)

def query_vector(q):
    if not q.strip():
        return None
    return vectorizer.transform([q.strip().lower()])

def cosine_row(vec, row_idx):
    # Cosine similarity between vector and X[row_idx]
    if vec is None: return 0.0
    sim = cosine_similarity(vec, X[row_idx])
    return float(sim[0,0])

# ---------------------------- Filter Items ----------------------------
def apply_filters(df):
    mask = pd.Series([True]*len(df), index=df.index)
    if language != "any":
        mask &= (df["language"] == language)
    if hide_explicit:
        mask &= (df["explicit"] == 0)
    if new_toggle == "New (≤30d)":
        mask &= df["publish_ts"].apply(lambda s: within_last_days(s, 30))
    elif new_toggle == "Evergreen (>30d)":
        mask &= ~df["publish_ts"].apply(lambda s: within_last_days(s, 30))
    # Duration slider filter
    mask &= df["ep_duration_min"].between(dur_min, dur_max, inclusive="both")
    return df[mask].copy()

filtered = apply_filters(df)

# ---------------------------- Precompute Similarities ----------------------------
q_vec = query_vector(query)
if user_vec is not None:
    interest_sims = cosine_similarity(user_vec, X[filtered.index])  # shape (1, n)
    interest_sims = interest_sims.flatten()
else:
    interest_sims = np.zeros(len(filtered))

if q_vec is not None:
    query_sims = cosine_similarity(q_vec, X[filtered.index]).flatten()
else:
    query_sims = np.zeros(len(filtered))

# Activity fit and Duration fit
activity_fits = []
duration_fits = []
popularities = []
penalties = []  # per-show penalty for "Not for this activity"

for i, (idx, row) in enumerate(filtered.iterrows()):
    activity_fits.append(activity_fit(row, activity, submode))
    duration_fits.append(duration_fit(row, dur_min, dur_max))
    popularities.append(popularity_signal(row))
    penalties.append(st.session_state.activity_penalties.get((activity, str(row["show_id"])), 0.0))

activity_fits = np.array(activity_fits)
duration_fits = np.array(duration_fits)
popularities = np.array(popularities)
penalties = np.array(penalties)

# ---------------------------- Relevance Score ----------------------------
# Potential weight tweaks by activity
weights = DEFAULT_WEIGHTS.copy()
if activity == "Sleep":
    weights["wA"] += 0.05; weights["wI"] += 0.05; weights["wQ"] -= 0.05
if activity == "Workout":
    weights["wA"] += 0.05; weights["wP"] += 0.05; weights["wI"] -= 0.05

relevance = (
    weights["wI"]*interest_sims +
    weights["wA"]*activity_fits +
    weights["wQ"]*query_sims +
    weights["wP"]*popularities +
    weights["wD"]*duration_fits
) - penalties  # subtract per-show penalty

# Clip 0..1 for display
rel_display = np.clip(relevance, 0, 1)

# ---------------------------- MMR Diversification ----------------------------
# Greedy selection with guardrails
def mmr_select(k, lam, candidate_indices, tfidf_indices, relevance_scores):
    selected = []
    selected_set = set()
    same_show_counts = {}
    seen = st.session_state.seen_item_ids

    # Precompute cosine sim matrix among candidates (sparse via vector ops on demand)
    def sim(i, j):
        # cosine between X[cand_i] and X[cand_j]
        return float(cosine_similarity(X[[tfidf_indices[i]]], X[[tfidf_indices[j]]])[0,0])

    # ensure at least one novel if lambda >= 0.5 (novel = not in seen set)
    need_novel = lam >= 0.5

    while len(selected) < k and candidate_indices:
        best_idx = None
        best_score = -1e9
        for local_idx, cand in enumerate(candidate_indices):
            ridx = tfidf_indices[cand]
            row = df.iloc[ridx]
            # Guardrail: no more than 2 eps from same show in Top-5
            if len(selected) < 5:
                scount = same_show_counts.get(row["show_id"], 0)
                if scount >= 2:
                    continue
            # MMR score
            if not selected:
                score = relevance_scores[cand]
            else:
                max_sim_to_S = 0.0
                for s_local in selected:
                    max_sim_to_S = max(max_sim_to_S, sim(cand, s_local))
                score = lam*relevance_scores[cand] - (1-lam)*max_sim_to_S
            # Favor novel if needed: add small epsilon if not seen
            if need_novel and (row["episode_id"] not in seen):
                score += 0.02
            if score > best_score:
                best_score = score
                best_idx = local_idx
        if best_idx is None:
            break
        chosen_local = best_idx
        chosen = candidate_indices.pop(chosen_local)
        selected.append(chosen)
        row = df.iloc[tfidf_indices[chosen]]
        same_show_counts[row["show_id"]] = same_show_counts.get(row["show_id"], 0) + 1

    return selected

# Candidate arrays
candidate_local_indices = list(range(len(filtered)))
tfidf_indices = filtered.index.to_list()  # indices into df/X
selected_locals = mmr_select(k=10, lam=lam, candidate_indices=candidate_local_indices, tfidf_indices=tfidf_indices, relevance_scores=relevance)
# Ensure at least 1 novel item if λ ≥ 0.5
if lam >= 0.5 and selected_locals:
    seen = st.session_state.seen_item_ids
    selected_eids = [df.iloc[tfidf_indices[i]]["episode_id"] for i in selected_locals]
    has_novel = any(eid not in seen for eid in selected_eids)
    if not has_novel:
        # Find best novel candidate by relevance (desc) that isn't already selected and respects Top-5 guardrail
        rel_order = sorted(range(len(filtered)), key=lambda i: relevance[i], reverse=True)
        # Compute show counts in Top-5
        show_counts = {}
        for i, loc in enumerate(selected_locals[:5]):
            sid = df.iloc[tfidf_indices[loc]]["show_id"]
            show_counts[sid] = show_counts.get(sid, 0) + 1
        replacement_idx = None
        candidate_new = None
        for loc in rel_order:
            eid = df.iloc[tfidf_indices[loc]]["episode_id"]
            if loc in selected_locals or eid in seen:
                continue
            sid = df.iloc[tfidf_indices[loc]]["show_id"]
            # Respect "≤2 episodes from same show in Top-5"
            if show_counts.get(sid,0) >= 2 and len(selected_locals) >= 5:
                continue
            candidate_new = loc
            break
        if candidate_new is not None:
            # Replace the last item to inject novelty
            selected_locals[-1] = candidate_new


# Mark seen items (impressions)
if selected_locals and st.session_state.first_impression_ts is None:
    st.session_state.first_impression_ts = time.time()

# Prepare display data
disp_rows = []
for local_idx in selected_locals:
    global_idx = tfidf_indices[local_idx]
    row = df.iloc[global_idx].copy()
    row["_InterestFit"] = float(interest_sims[local_idx]) if len(interest_sims)>0 else 0.0
    row["_ActivityFit"] = float(activity_fits[local_idx]) if len(activity_fits)>0 else 0.0
    row["_QueryMatch"] = float(query_sims[local_idx]) if len(query_sims)>0 else 0.0
    row["_Popularity"] = float(popularities[local_idx]) if len(popularities)>0 else 0.0
    row["_DurationFit"] = float(duration_fits[local_idx]) if len(duration_fits)>0 else 0.0
    row["_Relevance"] = float(rel_display[local_idx]) if len(rel_display)>0 else 0.0
    disp_rows.append(row)

# Log impressions and update seen set
for r in disp_rows:
    if r["episode_id"] not in st.session_state.seen_item_ids:
        st.session_state.seen_item_ids.add(r["episode_id"])
    log_event(st.session_state.session_id, "impression", activity, r["episode_id"], {"show_id": r["show_id"]})

# ---------------------------- Metrics Panel ----------------------------
with st.expander("📈 Metrics (session)", expanded=False):
    ensure_telemetry_file()
    try:
        tdf = pd.read_csv(TELEMETRY_PATH)
        sess = tdf[tdf["session_id"] == st.session_state.session_id]
        plays = (sess["event_type"] == "play").sum()
        clicks = (sess["event_type"] == "click").sum()
        impressions = (sess["event_type"] == "impression").sum()
        saves = (sess["event_type"] == "save").sum()
        dislikes = (sess["event_type"] == "dislike").sum()
        ctr = (plays / impressions)*100 if impressions else 0.0
        st.write(f"Impressions: **{impressions}**  |  Plays: **{plays}**  |  Saves: **{saves}**  |  Dislikes: **{dislikes}**  |  CTR to play: **{ctr:.1f}%**")
        # time-to-first-play
        if st.session_state.first_impression_ts and st.session_state.first_play_ts:
            ttfp = st.session_state.first_play_ts - st.session_state.first_impression_ts
            st.write(f"Time‑to‑first‑play: **{ttfp:.1f}s**")
    except Exception as e:
        st.write("No telemetry yet.")

# ---------------------------- Helper: Explainability ----------------------------
def top_overlap_terms(item_idx, ref_vec, k=5):
    if ref_vec is None: return []
    # contribution approximated by elementwise product of TF-IDF values
    item_vec = X[item_idx]
    # Convert sparse to arrays
    iv = item_vec.toarray().flatten()
    rv = ref_vec.toarray().flatten()
    contrib = iv * rv
    top_ids = np.argsort(contrib)[::-1][:k]
    terms = vectorizer.get_feature_names_out()
    return [terms[i] for i in top_ids if contrib[i] > 0][:k]

def query_hit_tokens(item_idx, q_vec, k=5):
    if q_vec is None: return []
    return top_overlap_terms(item_idx, q_vec, k)

def activity_keywords(activity):
    b = ACTIVITY_TAG_BIASES.get(activity, {"pos":[], "neg":[]})
    return list(dict.fromkeys(b.get("pos", []) + b.get("neg", [])))[:5]

# ---------------------------- Render Cards ----------------------------
st.subheader("Top Picks")
if hide_explicit and activity in ["Sleep","Focus"]:
    st.caption("Explicit content hidden by default for this activity.")

if len(disp_rows) == 0:
    st.warning("No results with current filters. Try widening the duration range or changing freshness.")
else:
    for r in disp_rows:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2 = st.columns([0.18, 0.82])
            with c1:
                # Emoji art placeholder based on tags
                tagstr = str(r["tags"]).lower()
                emoji = "🎧"
                if "mystery" in tagstr or "investigation" in tagstr: emoji = "🕵️"
                elif "comedy" in tagstr or "humor" in tagstr: emoji = "😂"
                elif "finance" in tagstr or "business" in tagstr: emoji = "💼"
                elif "ai" in tagstr or "technology" in tagstr or "software" in tagstr: emoji = "🤖"
                elif "news" in tagstr or "briefing" in tagstr: emoji = "📰"
                elif "sports" in tagstr or "football" in tagstr: emoji = "⚽️"
                elif "calm" in tagstr or "stoicism" in tagstr: emoji = "🌿"
                elif "history" in tagstr or "documentary" in tagstr: emoji = "🏛️"
                st.markdown(f"<div style='font-size:2.4rem;line-height:1'>{emoji}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{r['ep_title']}**")
                st.caption(f"{r['show_title']} — {r['publisher']} • avg {int(r['avg_len_min'])}m • {r['freq']} • {r['language'].upper()}" + (" • 🚫 explicit" if int(r['explicit']) else ""))
                # Badges
                bcols = st.columns(4)
                bcols[0].markdown(f"<span class='badge'>Interest: {r['_InterestFit']:.2f}</span>", unsafe_allow_html=True)
                bcols[1].markdown(f"<span class='badge'>Activity: {r['_ActivityFit']:.2f}</span>", unsafe_allow_html=True)
                bcols[2].markdown(f"<span class='badge'>Query: {r['_QueryMatch']:.2f}</span>", unsafe_allow_html=True)
                bcols[3].markdown(f"<span class='badge'>Popularity: {r['_Popularity']:.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='reason'>Why: relevance {r['_Relevance']:.2f}. "
                            f"{'Prefers soft start; ' if (activity=='Sleep' and int(r.get('soft_start',0))==1) else ''}"
                            f"{'matches your interests' if r['_InterestFit']>=0.15 else 'fits activity window'}.</div>", unsafe_allow_html=True)

                # Actions
                a1, a2, a3, a4, a5 = st.columns(5, gap="small")
                if a1.button("▶ Play latest", key=f"play_{r['episode_id']}"):
                    log_event(st.session_state.session_id, "play", activity, r["episode_id"], {"show_id": r["show_id"]})
                    if st.session_state.first_play_ts is None:
                        st.session_state.first_play_ts = time.time()
                    st.success("Pretend playing… (dummy)")
                if a2.button("Save", key=f"save_{r['episode_id']}"):
                    st.session_state.saved_items.add(r["episode_id"])
                    log_event(st.session_state.session_id, "save", activity, r["episode_id"])
                    st.toast("Saved")
                if a3.button("👍", key=f"like_{r['episode_id']}"):
                    # Increase interest weights around top terms and similar items
                    idx = ep_index[r["episode_id"]]
                    top_terms = top_overlap_terms(idx, user_vec if user_vec is not None else X[[idx]], k=5)
                    for t in top_terms:
                        st.session_state.interest_weights[t] = st.session_state.interest_weights.get(t, 0.0) + 0.6
                    st.session_state.liked_items.append(r["episode_id"])
                    log_event(st.session_state.session_id, "like", activity, r["episode_id"], {"terms": top_terms})
                    st.rerun()
                if a4.button("👎", key=f"dislike_{r['episode_id']}"):
                    st.session_state.disliked_items[r["episode_id"]] = "unspecified"
                    log_event(st.session_state.session_id, "dislike", activity, r["episode_id"])
                    st.rerun()
                if a5.button("⋯ Not for this activity", key=f"nfta_{r['episode_id']}"):
                    # Down-weight ActivityFit for that show in current activity
                    key = (activity, str(r["show_id"]))
                    st.session_state.activity_penalties[key] = min(0.4, st.session_state.activity_penalties.get(key, 0.0) + 0.2)
                    log_event(st.session_state.session_id, "not_for_activity", activity, r["episode_id"], {"show_id": r["show_id"]})
                    st.rerun()

                # Explainability panel
                with st.expander("Why this?"):
                    idx = ep_index[r["episode_id"]]
                    iterms = top_overlap_terms(idx, user_vec, k=5)
                    qterms = query_hit_tokens(idx, q_vec, k=5)
                    ak = activity_keywords(activity)
                    liked_refs = [e for e in st.session_state.liked_items[-3:]]
                    st.write({
                        "Interest overlap": iterms,
                        "Activity keywords": ak,
                        "Query hits": qterms,
                        "Because you liked": liked_refs if liked_refs else None
                    })

                # Dislike reasons (if disliked, show controls)
                if r["episode_id"] in st.session_state.disliked_items:
                    st.markdown("---")
                    reason = st.selectbox("Tell us why (updates suggestions instantly)", ["Too long","Not my topic","Too intense for sleep","I've heard it"], key=f"reason_{r['episode_id']}")
                    if st.button("Apply", key=f"apply_{r['episode_id']}"):
                        st.session_state.disliked_items[r["episode_id"]] = reason
                        # Update biases
                        if reason == "Too long":
                            st.session_state.len_bias = min(1.0, st.session_state.len_bias + 0.7)
                        elif reason == "Not my topic":
                            # Penalize top terms of this item
                            idx = ep_index[r["episode_id"]]
                            top_terms = top_overlap_terms(idx, X[[idx]], k=5)
                            for t in top_terms:
                                st.session_state.interest_weights[t] = st.session_state.interest_weights.get(t, 0.0) - 0.8
                        elif reason == "Too intense for sleep" and activity == "Sleep":
                            # Penalize energetic/high-energy tags for this session
                            for t in ["energetic","high-energy","comedy","banter"]:
                                st.session_state.interest_weights[t] = st.session_state.interest_weights.get(t, 0.0) - 0.6
                        elif reason == "I've heard it":
                            # Light penalty for this show's future appearances
                            key = (activity, str(r["show_id"]))
                            st.session_state.activity_penalties[key] = min(0.4, st.session_state.activity_penalties.get(key, 0.0) + 0.1)
                        log_event(st.session_state.session_id, "dislike_reason", activity, r["episode_id"], {"reason": reason})
                        st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------- Footer ----------------------------
st.markdown(
    """
    <div class="app-footer">
      <div class="bar"></div>
      <div class="left">13</div>
      <div class="center">Podcast Recommender | Idea deck</div>
      <div class="right">2025</div>
    </div>
    """, unsafe_allow_html=True
)
st.caption("All signals are computed locally. No external services are used.")

