"""
Activity-Aware Podcast Recommender with LLM Review Summarization
Main application file
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import hashlib
import re

# Vectorization and similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# LLM integration
from together import Together

# Set page config
st.set_page_config(
    page_title="Podcast Recommender",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
TOGETHER_API_KEY = "tgp_v1_DYc1X5IbJJq8f0UkShffPUCHLFKLz4THrvOaZfJepwY"
ACTIVITIES = ["Running", "Cycling", "Gym/Weightlifting", "Yoga", "Commuting", 
              "Cooking", "Cleaning", "Relaxing", "Studying", "Working"]
WORKOUT_SUBMODES = {
    "Running": ["Interval Training", "Long Run", "Recovery Run", "Trail Running"],
    "Cycling": ["Indoor Trainer", "Road Cycling", "Mountain Biking", "Recovery Ride"],
    "Gym/Weightlifting": ["Heavy Lifting", "Circuit Training", "Accessory Work", "Stretching"],
    "Yoga": ["Vinyasa", "Hatha", "Restorative", "Power Yoga"]
}
LANGUAGES = ["English", "Spanish", "French", "German", "Japanese", "Chinese", "All"]
TELEMETRY_FILE = "telemetry_log.csv"

class DataLoader:
    """Loads and manages podcast data"""
    
    def __init__(self):
        self.podcasts_df = None
        self.reviews_df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.show_reviews = {}
        
    def load_data(self):
        """Load CSV and JSON data into memory"""
        try:
            # Load podcasts data
            self.podcasts_df = pd.read_csv("sample_podcasts.csv")
            
            # Load reviews data
            reviews_data = []
            with open("reviews.json", 'r') as f:
                for line in f:
                    if line.strip():
                        reviews_data.append(json.loads(line))
            self.reviews_df = pd.DataFrame(reviews_data)
            
            # Group reviews by show
            for show_id, group in self.reviews_df.groupby('podcast_id'):
                self.show_reviews[show_id] = group.to_dict('records')
            
            # Preprocess text data
            self._preprocess_text()
            
            # Initialize TF-IDF
            self._initialize_tfidf()
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def _preprocess_text(self):
        """Preprocess text columns for better matching"""
        self.podcasts_df['combined_text'] = (
            self.podcasts_df['show_title'].fillna('') + ' ' +
            self.podcasts_df['tags'].fillna('') + ' ' +
            self.podcasts_df['ep_title'].fillna('') + ' ' +
            self.podcasts_df['ep_desc'].fillna('')
        ).str.lower()
        
        # Extract tags as list
        self.podcasts_df['tags_list'] = self.podcasts_df['tags'].fillna('').apply(
            lambda x: [tag.strip() for tag in x.split(',') if tag.strip()]
        )
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.podcasts_df['combined_text'])

class Recommender:
    """Main recommendation engine"""
    
    def __init__(self, data_loader: DataLoader):
        self.dl = data_loader
        self.activity_keywords = {
            "Running": ["energy", "motivation", "upbeat", "fast", "pace", "endurance", "cardio"],
            "Cycling": ["endurance", "scenic", "road", "outdoor", "indoor", "training"],
            "Gym/Weightlifting": ["intense", "power", "strength", "motivation", "aggressive"],
            "Yoga": ["calm", "mindful", "peaceful", "breathing", "meditation", "stretch"],
            "Commuting": ["news", "current events", "educational", "story", "conversation"],
            "Cooking": ["light", "entertaining", "storytelling", "casual", "educational"],
            "Cleaning": ["engaging", "story", "conversation", "entertaining", "background"],
            "Relaxing": ["calm", "soothing", "meditative", "slow", "peaceful", "ambient"],
            "Studying": ["educational", "technical", "detailed", "informative", "focused"],
            "Working": ["background", "instrumental", "ambient", "minimal", "focus"]
        }
        
        # User interest profiles (simulated)
        self.user_interests = {
            "technology": 0.8,
            "science": 0.7,
            "business": 0.6,
            "history": 0.9,
            "comedy": 0.5,
            "health": 0.7,
            "news": 0.6,
            "sports": 0.4
        }
    
    def calculate_relevance_scores(self, 
                                  selected_activity: str,
                                  search_query: str,
                                  min_duration: int,
                                  max_duration: int,
                                  language: str,
                                  evergreen: bool,
                                  user_feedback: dict) -> pd.DataFrame:
        """Calculate relevance scores for all podcasts"""
        df = self.dl.podcasts_df.copy()
        
        # 1. Interest Fit
        df['interest_fit'] = df['tags_list'].apply(
            lambda tags: self._calculate_interest_fit(tags)
        )
        
        # 2. Activity Fit
        df['activity_fit'] = df.apply(
            lambda row: self._calculate_activity_fit(row, selected_activity),
            axis=1
        )
        
        # 3. Query Match
        if search_query:
            df['query_match'] = self._calculate_query_match(search_query, df)
        else:
            df['query_match'] = 0.5  # Neutral score when no query
        
        # 4. Popularity (normalized)
        df['popularity_score_norm'] = df['popularity_score'] / df['popularity_score'].max()
        
        # 5. Duration Fit
        df['duration_fit'] = df['ep_duration_min'].apply(
            lambda x: self._calculate_duration_fit(x, min_duration, max_duration)
        )
        
        # 6. Language Filter
        if language != "All":
            df = df[df['language'] == language].copy()
        
        # 7. Evergreen filter
        if evergreen:
            # Filter for podcasts published in last 90 days
            df['publish_ts'] = pd.to_datetime(df['publish_ts'])
            cutoff = datetime.now() - pd.Timedelta(days=90)
            df = df[df['publish_ts'] > cutoff].copy()
        
        # 8. Apply feedback adjustments
        df = self._apply_feedback_adjustments(df, user_feedback, selected_activity)
        
        # Calculate weighted score
        weights = {
            'interest_fit': 0.25,
            'activity_fit': 0.30,
            'query_match': 0.20,
            'popularity_score_norm': 0.15,
            'duration_fit': 0.10
        }
        
        df['relevance_score'] = (
            df['interest_fit'] * weights['interest_fit'] +
            df['activity_fit'] * weights['activity_fit'] +
            df['query_match'] * weights['query_match'] +
            df['popularity_score_norm'] * weights['popularity_score_norm'] +
            df['duration_fit'] * weights['duration_fit']
        )
        
        return df.sort_values('relevance_score', ascending=False)
    
    def _calculate_interest_fit(self, tags: List[str]) -> float:
        """Calculate match between podcast tags and user interests"""
        if not tags:
            return 0.5
        
        total_score = 0
        matched_tags = 0
        
        for tag in tags:
            tag_lower = tag.lower()
            for interest, score in self.user_interests.items():
                if interest in tag_lower or tag_lower in interest:
                    total_score += score
                    matched_tags += 1
        
        if matched_tags == 0:
            return 0.5
        
        return total_score / matched_tags
    
    def _calculate_activity_fit(self, row: pd.Series, activity: str) -> float:
        """Calculate how well podcast fits the selected activity"""
        if activity not in self.activity_keywords:
            return 0.5
        
        keywords = self.activity_keywords[activity]
        
        # Check tags
        tag_score = 0
        tags = row['tags_list']
        for tag in tags:
            if any(keyword in tag.lower() for keyword in keywords):
                tag_score += 1
        
        # Check description
        text = row['combined_text'].lower()
        text_score = sum(1 for keyword in keywords if keyword in text)
        
        # Check soft_start (for workouts)
        if activity in ["Running", "Cycling", "Gym/Weightlifting", "Yoga"]:
            if row['soft_start'] == 1:
                activity_score = 0.8
            else:
                activity_score = 0.5
        else:
            activity_score = 0.5
        
        # Normalize scores
        tag_norm = min(tag_score / 3, 1.0)
        text_norm = min(text_score / 5, 1.0)
        
        return (tag_norm * 0.3 + text_norm * 0.3 + activity_score * 0.4)
    
    def _calculate_query_match(self, query: str, df: pd.DataFrame) -> pd.Series:
        """Calculate query match using TF-IDF"""
        query_vec = self.dl.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vec, self.dl.tfidf_matrix).flatten()
        
        # Get indices matching current dataframe
        indices = df.index
        return pd.Series(similarities[indices], index=df.index)
    
    def _calculate_duration_fit(self, duration: float, min_dur: int, max_dur: int) -> float:
        """Calculate how well duration fits the range"""
        if min_dur <= duration <= max_dur:
            return 1.0
        elif duration < min_dur:
            return 0.3
        else:
            # Exponential decay for longer durations
            diff = duration - max_dur
            return max(0.1, 1.0 / (1.0 + diff/30))
    
    def _apply_feedback_adjustments(self, df: pd.DataFrame, feedback: dict, activity: str) -> pd.DataFrame:
        """Adjust scores based on user feedback"""
        df = df.copy()
        
        for show_id, fb_data in feedback.items():
            if show_id in df['show_id'].values:
                mask = df['show_id'] == show_id
                
                if fb_data.get('liked'):
                    df.loc[mask, 'relevance_score'] *= 1.2
                elif fb_data.get('disliked'):
                    reason = fb_data.get('dislike_reason', '')
                    if reason == 'too_long':
                        df.loc[mask, 'duration_fit'] *= 0.5
                    elif reason == 'too_intense':
                        df.loc[mask, 'activity_fit'] *= 0.3
                    elif reason == 'wrong_topics':
                        df.loc[mask, 'interest_fit'] *= 0.4
                    else:
                        df.loc[mask, 'relevance_score'] *= 0.6
                
                if fb_data.get('not_for_activity') == activity:
                    df.loc[mask, 'activity_fit'] *= 0.2
        
        return df
    
    def apply_mmr_diversification(self, df: pd.DataFrame, lambda_param: float, top_n: int = 20) -> pd.DataFrame:
        """Apply Maximal Marginal Relevance diversification"""
        if len(df) == 0:
            return df
        
        selected = []
        remaining = df.copy()
        
        # Start with highest relevance
        first_idx = remaining.iloc[0:1].index[0]
        selected.append(first_idx)
        remaining = remaining.drop(first_idx)
        
        while len(selected) < min(top_n, len(df)):
            if len(remaining) == 0:
                break
            
            # Calculate similarity between remaining and selected
            sim_matrix = cosine_similarity(
                self.dl.tfidf_matrix[remaining.index],
                self.dl.tfidf_matrix[selected]
            )
            
            # Max similarity for each remaining item
            max_sim = sim_matrix.max(axis=1)
            
            # MMR score: lambda * relevance - (1-lambda) * similarity
            mmr_scores = (
                lambda_param * remaining['relevance_score'].values -
                (1 - lambda_param) * max_sim
            )
            
            next_idx = remaining.iloc[np.argmax(mmr_scores)].name
            selected.append(next_idx)
            remaining = remaining.drop(next_idx)
        
        return df.loc[selected]

class LLMReviewSummarizer:
    """Handles LLM-powered review summarization"""
    
    def __init__(self):
        self.client = Together(api_key=TOGETHER_API_KEY)
        self.model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    def summarize_reviews(self, reviews: List[Dict], show_title: str) -> str:
        """Generate a summary of reviews for a show"""
        if not reviews:
            return "No reviews available for this show."
        
        # Prepare review text
        review_texts = []
        for review in reviews[:10]:  # Limit to 10 reviews
            content = review.get('content', '')
            rating = review.get('rating', 3)
            if content:
                review_texts.append(f"Rating: {rating}/5 - {content[:200]}...")
        
        if not review_texts:
            return "No detailed reviews available."
        
        reviews_text = "\n\n".join(review_texts)
        
        prompt = f"""Please provide a concise, balanced 2-3 sentence summary of these podcast reviews.

Podcast: {show_title}

Reviews:
{reviews_text}

Instructions:
1. Summarize overall sentiment and key themes
2. Mention both positive and negative points if present
3. Keep it anonymous (don't mention specific reviewers)
4. Be objective and factual
5. Maximum 3 sentences

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
                top_p=0.9
            )
            
            summary = response.choices[0].message.content.strip()
            # Clean up any extra formatting
            summary = re.sub(r'\n+', ' ', summary)
            return summary
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"

class TelemetryLogger:
    """Logs user interactions to CSV"""
    
    def __init__(self, filename: str = TELEMETRY_FILE):
        self.filename = filename
        self._initialize_log()
    
    def _initialize_log(self):
        """Create log file with headers if it doesn't exist"""
        try:
            pd.read_csv(self.filename)
        except FileNotFoundError:
            log_df = pd.DataFrame(columns=[
                'timestamp', 'session_id', 'interaction_type',
                'episode_id', 'show_id', 'activity', 'submode',
                'search_query', 'additional_info'
            ])
            log_df.to_csv(self.filename, index=False)
    
    def log_interaction(self, interaction_type: str, **kwargs):
        """Log an interaction"""
        import uuid
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': st.session_state.get('session_id', str(uuid.uuid4())[:8]),
            'interaction_type': interaction_type,
            'episode_id': kwargs.get('episode_id', ''),
            'show_id': kwargs.get('show_id', ''),
            'activity': kwargs.get('activity', ''),
            'submode': kwargs.get('submode', ''),
            'search_query': kwargs.get('search_query', ''),
            'additional_info': json.dumps(kwargs.get('additional_info', {}))
        }
        
        try:
            log_df = pd.read_csv(self.filename)
            log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
            log_df.to_csv(self.filename, index=False)
        except Exception as e:
            st.error(f"Failed to log interaction: {e}")

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    if 'recommender' not in st.session_state:
        st.session_state.recommender = None
    
    if 'llm_summarizer' not in st.session_state:
        st.session_state.llm_summarizer = LLMReviewSummarizer()
    
    if 'telemetry' not in st.session_state:
        st.session_state.telemetry = TelemetryLogger()
    
    if 'user_feedback' not in st.session_state:
        st.session_state.user_feedback = {}
    
    if 'review_summaries' not in st.session_state:
        st.session_state.review_summaries = {}
    
    if 'expanded_row' not in st.session_state:
        st.session_state.expanded_row = None
    
    if 'selected_activity' not in st.session_state:
        st.session_state.selected_activity = ACTIVITIES[0]
    
    if 'lambda_param' not in st.session_state:
        st.session_state.lambda_param = 0.7

# Main application
def main():
    """Main application function"""
    
    # Initialize session state
    init_session_state()
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading podcast data..."):
            if st.session_state.data_loader.load_data():
                st.session_state.data_loaded = True
                st.session_state.recommender = Recommender(st.session_state.data_loader)
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to load data. Please check data files.")
                return
    
    # Sidebar
    with st.sidebar:
        st.title("🎧 Activity-Aware Podcast Recommender")
        
        # Activity selection
        st.subheader("Activity Selection")
        selected_activity = st.selectbox(
            "Select Activity",
            ACTIVITIES,
            index=ACTIVITIES.index(st.session_state.selected_activity)
        )
        st.session_state.selected_activity = selected_activity
        
        # Submode selection (if applicable)
        if selected_activity in WORKOUT_SUBMODES:
            submode = st.selectbox(
                "Workout Submode",
                WORKOUT_SUBMODES[selected_activity]
            )
        else:
            submode = ""
        
        # Search
        st.subheader("Search & Filters")
        search_query = st.text_input("Search podcasts", "")
        
        # Duration filter
        min_duration, max_duration = st.slider(
            "Duration (minutes)",
            min_value=0,
            max_value=240,
            value=(10, 90),
            step=5
        )
        
        # Language filter
        language = st.selectbox("Language", LANGUAGES)
        
        # Evergreen toggle
        evergreen = st.toggle("New Content Only (Last 90 days)", value=False)
        
        # Diversity slider
        st.subheader("Recommendation Diversity")
        lambda_param = st.slider(
            "Similarity vs Diversity (λ)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.lambda_param,
            step=0.1,
            help="Higher values prioritize relevance, lower values increase diversity"
        )
        st.session_state.lambda_param = lambda_param
        
        # Feedback panel
        st.subheader("Feedback Panel")
        
        with st.form("feedback_form"):
            show_to_rate = st.selectbox(
                "Select show to rate",
                options=sorted(st.session_state.data_loader.podcasts_df['show_title'].unique())
            )
            
            feedback_type = st.radio(
                "Feedback type",
                ["Like", "Dislike", "Not for this activity"]
            )
            
            if feedback_type == "Dislike":
                dislike_reason = st.selectbox(
                    "Reason",
                    ["", "too_long", "too_intense", "wrong_topics", "not_interesting"]
                )
            else:
                dislike_reason = ""
            
            if st.form_submit_button("Submit Feedback"):
                show_id = st.session_state.data_loader.podcasts_df[
                    st.session_state.data_loader.podcasts_df['show_title'] == show_to_rate
                ]['show_id'].iloc[0]
                
                if feedback_type == "Like":
                    st.session_state.user_feedback[show_id] = {
                        'liked': True,
                        'disliked': False
                    }
                    st.success(f"Liked '{show_to_rate}'")
                    
                    # Log interaction
                    st.session_state.telemetry.log_interaction(
                        'like',
                        show_id=show_id,
                        activity=selected_activity,
                        search_query=search_query
                    )
                    
                elif feedback_type == "Dislike" and dislike_reason:
                    st.session_state.user_feedback[show_id] = {
                        'liked': False,
                        'disliked': True,
                        'dislike_reason': dislike_reason
                    }
                    st.success(f"Disliked '{show_to_rate}' for reason: {dislike_reason}")
                    
                    # Log interaction
                    st.session_state.telemetry.log_interaction(
                        'dislike',
                        show_id=show_id,
                        activity=selected_activity,
                        search_query=search_query,
                        additional_info={'reason': dislike_reason}
                    )
                    
                elif feedback_type == "Not for this activity":
                    if show_id not in st.session_state.user_feedback:
                        st.session_state.user_feedback[show_id] = {}
                    st.session_state.user_feedback[show_id]['not_for_activity'] = selected_activity
                    st.success(f"Marked '{show_to_rate}' as not suitable for {selected_activity}")
                    
                    # Log interaction
                    st.session_state.telemetry.log_interaction(
                        'not_for_activity',
                        show_id=show_id,
                        activity=selected_activity,
                        search_query=search_query
                    )
        
        st.markdown("---")
        st.caption(f"Loaded {len(st.session_state.data_loader.podcasts_df)} episodes")
        st.caption(f"Loaded {len(st.session_state.data_loader.reviews_df)} reviews")
    
    # Main content area
    st.title(f"Podcasts for {selected_activity}")
    if submode:
        st.caption(f"Submode: {submode}")
    
    # Calculate recommendations
    recommendations = st.session_state.recommender.calculate_relevance_scores(
        selected_activity=selected_activity,
        search_query=search_query,
        min_duration=min_duration,
        max_duration=max_duration,
        language=language,
        evergreen=evergreen,
        user_feedback=st.session_state.user_feedback
    )
    
    # Apply MMR diversification
    if len(recommendations) > 0:
        recommendations = st.session_state.recommender.apply_mmr_diversification(
            recommendations,
            lambda_param=lambda_param,
            top_n=20
        )
    
    # Display recommendations table
    if len(recommendations) == 0:
        st.warning("No podcasts match your criteria. Try adjusting filters.")
    else:
        st.subheader(f"Top {len(recommendations)} Recommendations")
        
        # Display table
        for idx, row in recommendations.iterrows():
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{row['ep_title']}**")
                    st.caption(f"Show: {row['show_title']} | Publisher: {row['publisher']}")
                
                with col2:
                    tags_display = ', '.join(row['tags_list'][:3])
                    st.caption(f"Tags: {tags_display}")
                
                with col3:
                    st.metric("Duration", f"{row['ep_duration_min']:.0f} min")
                
                with col4:
                    st.metric("Score", f"{row['relevance_score']:.2f}")
                
                with col5:
                    if st.button("📖", key=f"expand_{row['episode_id']}"):
                        if st.session_state.expanded_row == row['episode_id']:
                            st.session_state.expanded_row = None
                        else:
                            st.session_state.expanded_row = row['episode_id']
                            
                            # Log interaction
                            st.session_state.telemetry.log_interaction(
                                'why_opened',
                                episode_id=row['episode_id'],
                                show_id=row['show_id'],
                                activity=selected_activity
                            )
                
                # Action buttons
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    if st.button("▶️ Play", key=f"play_{row['episode_id']}"):
                        st.session_state.telemetry.log_interaction(
                            'play',
                            episode_id=row['episode_id'],
                            show_id=row['show_id'],
                            activity=selected_activity
                        )
                        st.success(f"Playing: {row['ep_title']}")
                
                with action_col2:
                    if st.button("💾 Save", key=f"save_{row['episode_id']}"):
                        st.session_state.telemetry.log_interaction(
                            'save',
                            episode_id=row['episode_id'],
                            show_id=row['show_id'],
                            activity=selected_activity
                        )
                        st.success(f"Saved: {row['ep_title']}")
                
                with action_col3:
                    if st.button("📝 Reviews", key=f"reviews_{row['episode_id']}"):
                        # Generate or retrieve review summary
                        show_id = row['show_id']
                        
                        if show_id in st.session_state.review_summaries:
                            summary = st.session_state.review_summaries[show_id]
                        else:
                            with st.spinner("Generating review summary..."):
                                reviews = st.session_state.data_loader.show_reviews.get(show_id, [])
                                summary = st.session_state.llm_summarizer.summarize_reviews(
                                    reviews, row['show_title']
                                )
                                st.session_state.review_summaries[show_id] = summary
                                
                                # Log interaction
                                st.session_state.telemetry.log_interaction(
                                    'review_summary_opened',
                                    episode_id=row['episode_id'],
                                    show_id=show_id,
                                    activity=selected_activity
                                )
                        
                        st.info(f"**Review Summary for {row['show_title']}:**\n\n{summary}")
                
                # Expanded row content
                if st.session_state.expanded_row == row['episode_id']:
                    with st.expander("Why this recommendation?", expanded=True):
                        st.write("**Explanation:**")
                        
                        # Interest overlap
                        user_interests = st.session_state.recommender.user_interests
                        matched_interests = [
                            interest for interest in user_interests
                            if any(interest in tag.lower() for tag in row['tags_list'])
                        ]
                        
                        if matched_interests:
                            st.write(f"✅ Matches your interests: {', '.join(matched_interests)}")
                        
                        # Activity match terms
                        activity_keywords = st.session_state.recommender.activity_keywords.get(selected_activity, [])
                        matched_keywords = [
                            kw for kw in activity_keywords
                            if kw in row['combined_text']
                        ]
                        
                        if matched_keywords:
                            st.write(f"✅ Fits {selected_activity}: Contains terms like {', '.join(matched_keywords[:3])}")
                        
                        # Query match (if any)
                        if search_query:
                            st.write(f"✅ Matches your search: '{search_query}'")
                        
                        # Reference to liked items
                        liked_shows = [
                            show_id for show_id, fb in st.session_state.user_feedback.items()
                            if fb.get('liked')
                        ]
                        
                        if row['show_id'] in liked_shows:
                            st.write("✅ You previously liked this show")
                        
                        # Score breakdown
                        st.write("**Score Breakdown:**")
                        score_data = {
                            "Interest Fit": row['interest_fit'],
                            "Activity Fit": row['activity_fit'],
                            "Query Match": row['query_match'],
                            "Popularity": row['popularity_score_norm'],
                            "Duration Fit": row['duration_fit']
                        }
                        
                        for metric, score in score_data.items():
                            st.progress(score, text=f"{metric}: {score:.2f}")
                
                st.markdown("---")
        
        # Log impressions
        episode_ids = recommendations['episode_id'].tolist()[:10]
        for ep_id in episode_ids:
            st.session_state.telemetry.log_interaction(
                'impression',
                episode_id=ep_id,
                activity=selected_activity,
                search_query=search_query
            )

if __name__ == "__main__":
    main()