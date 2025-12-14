Role
You are a senior Python engineer and Streamlit expert responsible for implementing an Activity-Aware Podcast Recommender with on-demand LLM review summarization.

Core Requirements
1. Data
The app must use only the existing local CSV and JSON files provided by the user.


Do NOT generate or modify any dataset.


Load data fully in-memory.


2. UI
Build a simple hybrid UI:
Left sidebar:
Activity selection


Workout submodes


Search query


Duration filter


Language filter


“New/Evergreen” toggle


Diversity slider (λ)


Feedback panel (Like, Dislike + reasons, Not for this activity)


Right panel:
Table-based recommendations (primary UI)


Each episode = one row


Columns: basic metadata + action buttons


Row can expand to show:


“Why this?” explanation


“Review summary” output


No card grid.
 UI must stay minimal and mobile-friendly.

3. Ranking
Implement a simple weighted relevance model (InterestFit, ActivityFit, QueryMatch, Popularity, DurationFit).
 Use TF-IDF for text similarity.
 Use MMR diversification with λ from UI.
 Keep all computation local and deterministic.

4. Feedback Loop
Feedback lives only in the sidebar, not inline.


Like/Dislike must immediately adjust the ranking in session state.


Dislike reasons affect relevance rules (shorter, calmer, different tags, etc.).


“Not for this activity” down-weights the show for the selected activity only.



5. LLM Review Summaries (Strict Rules)
LLM provider
Use Together API with this hardcoded key:
TOGETHER_API_KEY = "tgp_v1_DYc1X5IbJJq8f0UkShffPUCHLFKLz4THrvOaZfJepwY"

Integration
Use the standard client:
from together import Together
client = Together(api_key=TOGETHER_API_KEY)

Trigger
LLM must be called ONLY when the user clicks “Review summary”.


No background calls.


No preloading.


No summary generation during ranking or rendering.


Scope
Summarize only reviews belonging to the selected show
 (podcast_id == show_id).


Summary must be short (2–3 sentences), balanced, anonymized.


Caching
Store summary in st.session_state to avoid repeated LLM calls.

6. Explainability
Expandable row shows:
interest overlap


activity match terms


query matches


reference to liked items


Keep explanations concise.

7. Telemetry
Log user interactions into a local CSV file:
impression


play


save


like / dislike


dislike_reason


not_for_activity


why_opened


review_summary_opened / error



8. Constraints
Do not generate any synthetic content or data rows.


All recommendations must useCSV and JSON files.


The prototype must run offline except for Together API calls.


No auth, no cloud deployment, no background workers.




Data Files and Join Keys

The prototype uses two local data files:

sample_podcasts.csv

Format: CSV with header row, one row per episode.

Key columns:

show_id – podcast/show identifier (used to group episodes into a show)

show_title

publisher

tags

language

explicit

avg_len_min

freq

episode_id

ep_title

ep_desc

ep_duration_min

soft_start

publish_ts

popularity_score

reviews.json

Format: line-delimited JSON, one review object per line.

Fields per review:

podcast_id – identifier of the show being reviewed

title – short review title

content – full review text

rating – numeric rating (e.g., 1–5)

author_id – anonymized reviewer identifier

created_at – timestamp string

Join rule (hard requirement):
Reviews must be linked to podcast metadata by matching reviews.podcast_id to sample_podcasts.show_id. All review-based features and LLM review summaries must use this join key to fetch the correct set of reviews for each show.
