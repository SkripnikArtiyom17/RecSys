[technicaltask.md](https://github.com/user-attachments/files/24057720/technicaltask.md)
Prototype Prompt — Activity-Aware Podcast Recommender (Streamlit)

Role: You are a senior Python engineer and Streamlit expert.
Goal: Build a self-contained prototype of an activity-aware podcast recommender with explainability, cold start, diversity control, and feedback loop — optimized for a live demo.

Scope (build only what follows; exclude production infra)

Include: Streamlit UI, in-memory dataset, TF-IDF + simple embeddings, MMR diversification, “Why this?” panel, feedback that updates ranking in session state, basic telemetry to CSV.

Exclude: Auth, payments, external APIs, real audio playback, DBs, background jobs, microservices, cloud deploy.

Deliverables

Code files:

app.py (Streamlit app; single file is OK)

data/sample_podcasts.csv (≈40–60 rows; columns: show_id,show_title,publisher,tags,language,explicit,avg_len_min,freq,episode_id,ep_title,ep_desc,ep_duration_min,soft_start,publish_ts,popularity_score)

How to run: clear 3-step instructions (pip install streamlit scikit-learn pandas numpy, streamlit run app.py).

No placeholders — the app must run end-to-end offline.

Data & Features

Generate a small, varied CSV (EN) across topics (True Crime, Comedy, Business, Tech, News, Sports, Self-help, History).

Fill tags with helpful tokens (e.g., “calm, interview, background, energetic, storytelling”).

Ensure a mix of durations (10–90 min), explicit flags, languages (en, es, ru), and popularity scores (0–1).

UI Requirements (mobile-first in Streamlit)

Entry point: “What are you doing now?” segmented control: Commute • Workout • Focus • Sleep.

Search: text input with examples (“AI ethics”, “Premier League”, “Stoicism”).

Filters: duration range slider; language select; “New/Evergreen” toggle (recent = last 30 days).

Diversity slider: “More familiar ↔ More exploratory” → controls MMR λ (0..1).

Result cards (Top-10): art placeholder (emoji), title, publisher, avg length, frequency, badges (Interest • Activity • Query • Popularity numeric 0–1), 1-line reason, “▶ Play latest” (dummy), “Save”, “👍/👎”, “⋯ Not for this activity”.

Explainability: small chevron → panel with:

Interest overlap terms (top 5),

Activity match keywords,

Query hit tokens,

“Because you liked X/Y” (use session history).

Cold start: if no history, show 10–15s chips to pick interests (multi-select), then immediate activity-only list with high diversity. Microcopy: “No history yet — tuned for your activity.”

Safety & comfort:

Sleep/Focus: hide explicit by default; prefer soft_start=true.

Focus: down-weight comedy/high-energy tags.

Workout: allow sub-modes (Run, Lift, Cardio) that adjust target duration/energy.

Accessibility: keyboard-first, aria labels via Streamlit components where possible, high contrast.

Ranking Model (implement simply, in code)

Text features: TF-IDF over ep_title + ep_desc + tags.

Signals (0..1):

InterestFit: cosine similarity to user interest vector (from chips + liked items).

ActivityFit: windowed duration fitness + tag bias per activity (e.g., Sleep→calm/storytelling; Focus→interview/background; Workout→energetic/uplifting; Commute→narrative/news).

QueryMatch: BM25-style or TF-IDF cosine vs. query (fallback 0 if empty).

Popularity: normalized popularity_score with freshness boost (recent episodes +0.05 clipped).

DurationFit: penalty for being outside selected duration range (smooth).

Pre-MMR relevance:

relevance = wI*InterestFit + wA*ActivityFit + wQ*QueryMatch + wP*Popularity + wD*DurationFit


Default weights: wI=0.35, wA=0.25, wQ=0.20, wP=0.15, wD=0.05 (tweak per activity: Sleep λ lower, Workout λ higher).

MMR diversification:

final_score(item) = λ*relevance(item) - (1-λ)*max_sim(item, S)


where S is the selected set; similarity is cosine over TF-IDF vectors; λ comes from the UI slider (default 0.5).

Guardrails: no more than 2 episodes from the same show in Top-5; ensure ≥1 novel item if λ ≥ 0.5.

Feedback & Learning (session-level)

👍 increases user interest weights around this item’s top terms; 👎 decreases.

“Not for this activity” down-weights ActivityFit for that show in the current activity only.

Reason codes after 👎: “Too long”, “Not my topic”, “Too intense for sleep”, “I’ve heard it” → update duration/window or tag penalties.

Update recommendations immediately without page reload.

Telemetry (write to CSV in /tmp or project root)

Log impression, click, play, completion_50, completion_80, skip, save, dislike_reason, why_opened with session_id, user_ts, activity, item_id.

Provide a small metrics panel (counts + CTR to play, time-to-first-play).

Explainability Text (1–2 lines max)

Generate a concise reason per card from the top contributing signals/terms:

Template examples:

“Calm interview format matches Focus; overlaps with ‘ethics’, ‘research’.”

“Upbeat storytelling fits Workout (30–45m); similar to your saved Business shows.”

Implementation Notes

Use scikit-learn for TF-IDF and cosine similarity; cache matrices with st.cache_data.

Keep everything deterministic; no external network calls.

Provide clear inline comments explaining: TF-IDF build, signal calculations, MMR, feedback updates, and logging.

Acceptance Criteria (must pass)

Sleep + empty query → Top-10 has no explicit content, includes soft-start items, and at least 1 novel item when λ ≥ 0.5.

Workout + duration [20,45] → median of returned durations falls inside the window.

Clicking 👎 with “Too long” shifts subsequent suggestions to shorter episodes in the same session.

“Why this?” shows interest terms, activity keywords, and query hits for each item without errors.

Telemetry CSV grows with interactions and displays CTR and time-to-first-play in the metrics panel.

Deliver the complete code and the sample CSV, followed by run instructions.
