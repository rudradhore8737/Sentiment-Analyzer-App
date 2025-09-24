# -*- coding: utf-8 -*-
"""
Feedback Sentiment Analyzer Dashboard

This script creates an interactive web dashboard using Streamlit for analyzing
customer feedback. This version includes a robust, self-contained NLTK setup
and a fix for handling small datasets with the sampling feature.
"""

import streamlit as st
import pandas as pd
import nltk
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# --- Final Robust NLTK Data Setup ---
# This block handles the initial download of NLTK packages to a local folder.
def setup_nltk_data():
    """
    Downloads and configures NLTK data to a local project folder.
    This function runs on every app start to ensure reliability.
    """
    # 1. Define the absolute path for a local 'nltk_data' folder
    local_nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')

    # 2. Create the directory if it doesn't exist.
    os.makedirs(local_nltk_data_path, exist_ok=True)

    # 3. Add this local path to the FRONT of NLTK's search path
    if local_nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_data_path)

    # 4. Check for required packages and download if missing
    required_packages = {
        'tokenizers/punkt_tab': 'punkt_tab',
        'sentiment/vader_lexicon.zip': 'vader_lexicon',
        'corpora/stopwords': 'stopwords'
    }

    for path, package_id in required_packages.items():
        # Check if the specific unzipped folder/file exists
        if not os.path.exists(os.path.join(local_nltk_data_path, path)):
            print(f"NLTK: Package '{package_id}' not found. Downloading to {local_nltk_data_path}")
            try:
                nltk.download(package_id, download_dir=local_nltk_data_path)
                print(f"NLTK: Successfully downloaded '{package_id}'.")
            except Exception as e:
                st.error(f"FATAL ERROR: Failed to download NLTK package '{package_id}'. The app cannot continue. Please check your internet connection and restart. Error: {e}")
                st.stop()

# Run the setup function on every app start.
setup_nltk_data()
# --- End of NLTK Setup ---


# --- Core Functions ---
@st.cache_data
def analyze_sentiment(text):
    """Analyzes the sentiment of a given text using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(str(text))
    compound_score = score['compound']
    
    if compound_score >= 0.05:
        return 'Positive', compound_score
    elif compound_score <= -0.05:
        return 'Negative', compound_score
    else:
        return 'Neutral', compound_score

@st.cache_data
def extract_keywords(text_series):
    """Extracts and ranks keywords from a series of text."""
    # --- DEFINITIVE FIX FOR LOOKUPERROR ---
    # This block re-asserts the correct NLTK data path inside the cached function.
    # This prevents Streamlit's execution context from losing the path.
    local_nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
    if local_nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_data_path)
    # --- END OF FIX ---

    stop_words = set(stopwords.words('english'))
    custom_stopwords = ['customer', 'service', 'product', 'experience']
    stop_words.update(custom_stopwords)
    
    all_words = []
    for text in text_series.astype(str):
        text = re.sub(r'\W+', ' ', text.lower())
        words = word_tokenize(text)
        all_words.extend([word for word in words if word.isalpha() and word not in stop_words and len(word) > 1])
        
    word_counts = Counter(all_words)
    return pd.DataFrame(word_counts.most_common(20), columns=['Keyword', 'Frequency'])

def generate_wordcloud(keywords_df):
    """Generates a word cloud from a keywords DataFrame."""
    word_freq = dict(zip(keywords_df['Keyword'], keywords_df['Frequency']))
    if not word_freq:
        return None
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(word_freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Feedback Sentiment Analyzer", page_icon="📊")
st.title("📊 Customer Feedback Sentiment Analyzer")
st.markdown("""
Welcome! This tool analyzes customer feedback from CSV/Excel files. Upload your data, select the columns, and the dashboard will reveal insights through sentiment analysis and keyword extraction.
""")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False

    if uploaded_file:
        try:
            # Added a fallback encoding for robustness
            df = pd.read_csv(uploaded_file, encoding='utf-8') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.df = None
            uploaded_file = None

        if uploaded_file:
            st.session_state.df = df
            st.success(f"File uploaded successfully! ({len(df):,} rows)")
            
            available_columns = df.columns.tolist()
            text_column = st.selectbox("Select column with feedback text", available_columns)
            date_column = st.selectbox("Select date column (optional)", ["None"] + available_columns)
            
            st.markdown("---")
            st.subheader("Performance Options")
            use_sampling = st.checkbox("Analyze a sample for faster results", True)
            
            total_rows = len(st.session_state.df)
            if use_sampling:
                # This logic now correctly handles small files.
                sample_size = st.number_input(
                    label="Sample size",
                    min_value=1,
                    max_value=total_rows,
                    value=min(5000, total_rows), # Default to 5000 or total rows if smaller
                    step=100
                )

            if st.button("🚀 Run Analysis", type="primary"):
                st.session_state.run_analysis = True
                st.session_state.text_column = text_column
                st.session_state.date_column = date_column
                st.session_state.use_sampling = use_sampling
                st.session_state.sample_size = sample_size if use_sampling else total_rows
    else:
        st.info("Please upload a file to get started.")
        st.session_state.df = None

if st.session_state.get('run_analysis') and st.session_state.df is not None:
    df_to_analyze = st.session_state.df.copy()
    original_row_count = len(df_to_analyze)

    if st.session_state.use_sampling and original_row_count > st.session_state.sample_size:
        sample_size = min(st.session_state.sample_size, original_row_count)
        df_to_analyze = df_to_analyze.sample(n=sample_size, random_state=42)

    with st.spinner('Analyzing feedback... This may take a moment.'):
        df_to_analyze[['sentiment_label', 'sentiment_score']] = df_to_analyze[st.session_state.text_column].apply(lambda x: pd.Series(analyze_sentiment(x)))
        keywords_df = extract_keywords(df_to_analyze[st.session_state.text_column])

    st.success("Analysis Complete!")
    if st.session_state.use_sampling and original_row_count > len(df_to_analyze):
        st.info(f"Results are based on a random sample of **{len(df_to_analyze):,}** rows (out of **{original_row_count:,}**).")

    st.header("📈 Dashboard Overview")
    sentiment_counts = df_to_analyze['sentiment_label'].value_counts()
    pos_count = sentiment_counts.get('Positive', 0)
    neg_count = sentiment_counts.get('Negative', 0)
    neu_count = sentiment_counts.get('Neutral', 0)
    total_feedback = len(df_to_analyze)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Analyzed Feedback", f"{total_feedback:,}")
    kpi2.metric("Positive 😊", f"{pos_count:,} ({(pos_count/total_feedback):.1%})")
    kpi3.metric("Negative 😠", f"{neg_count:,} ({(neg_count/total_feedback):.1%})")
    kpi4.metric("Neutral 😐", f"{neu_count:,} ({(neu_count/total_feedback):.1%})")

    st.header("📊 Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, hole=0.3, color=sentiment_counts.index, color_discrete_map={'Positive':'#4CAF50', 'Negative':'#F44336', 'Neutral':'#FFC107'})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Feedback Count by Sentiment")
        fig_bar = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values, labels={'x': 'Sentiment', 'y': 'Count'}, color=sentiment_counts.index, color_discrete_map={'Positive':'#4CAF50', 'Negative':'#F44336', 'Neutral':'#FFC107'})
        st.plotly_chart(fig_bar, use_container_width=True)

    if st.session_state.date_column != "None":
        st.subheader("Sentiment Over Time")
        try:
            df_trend = df_to_analyze.copy()
            df_trend[st.session_state.date_column] = pd.to_datetime(df_trend[st.session_state.date_column])
            df_trend = df_trend.set_index(st.session_state.date_column).groupby([pd.Grouper(freq='D'), 'sentiment_label']).size().unstack(fill_value=0)
            st.line_chart(df_trend, color=['#4CAF50', '#F44336', '#FFC107'])
        except Exception:
            st.warning("Could not process the date column. Please ensure it's in a valid date format.")

    st.header("🔑 Top Keywords Analysis")
    col3, col4 = st.columns([1, 2])
    with col3:
        st.subheader("Top 20 Keywords")
        st.dataframe(keywords_df)
    with col4:
        st.subheader("Keywords Word Cloud")
        wordcloud_fig = generate_wordcloud(keywords_df)
        if wordcloud_fig:
            st.pyplot(wordcloud_fig)

    st.header("📄 Detailed Feedback Data")
    filter_options = st.multiselect("Filter by sentiment:", df_to_analyze['sentiment_label'].unique(), default=df_to_analyze['sentiment_label'].unique())
    st.dataframe(df_to_analyze[df_to_analyze['sentiment_label'].isin(filter_options)], use_container_width=True)
    st.download_button("📥 Download Analyzed Report (CSV)", convert_df_to_csv(df_to_analyze), 'sentiment_analysis_report.csv', 'text/csv')

