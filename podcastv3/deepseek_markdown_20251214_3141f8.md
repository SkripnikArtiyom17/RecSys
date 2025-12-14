# Activity-Aware Podcast Recommender - Technical Documentation

## Overview
A hybrid recommender system that suggests podcasts based on user activity context, with on-demand LLM-powered review summarization.

## Architecture

### Data Flow
1. **Data Loading**: CSV and JSON files loaded into Pandas DataFrames
2. **Preprocessing**: Text normalization, TF-IDF vectorization
3. **Ranking**: Weighted relevance scoring with 5 factors
4. **Diversification**: MMR algorithm for diversity control
5. **UI Rendering**: Streamlit-based interactive interface
6. **Feedback**: Real-time relevance adjustments

### Core Components

#### 1. DataLoader (`DataLoader` class)
- Loads `sample_podcasts.csv` and `reviews.json`
- Preprocesses text fields for TF-IDF
- Groups reviews by show_id for LLM summarization
- Maintains in-memory data structures

#### 2. Recommender (`Recommender` class)
**Weighted Relevance Model:**
- InterestFit (25%): User interest profile vs podcast tags
- ActivityFit (30%): Activity-specific keyword matching
- QueryMatch (20%): TF-IDF similarity to search query
- Popularity (15%): Normalized popularity scores
- DurationFit (10%): Match to duration preferences

**Diversification:**
- Maximal Marginal Relevance (MMR) algorithm
- λ parameter controls relevance vs diversity trade-off
- Uses TF-IDF cosine similarity for content diversity

#### 3. LLM Integration (`LLMReviewSummarizer` class)
- Together API with Mixtral-8x7B model
- On-demand triggering only
- Reviews filtered by `podcast_id == show_id`
- Caching in session state
- Error handling for API failures

#### 4. Telemetry (`TelemetryLogger` class)
- Local CSV logging (`telemetry_log.csv`)
- Tracks all user interactions
- Session-based tracking
- No PII collection

### Key Algorithms

#### TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2)
)