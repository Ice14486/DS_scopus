import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import seaborn as sns          


plt.style.use('ggplot')

# ==========================================
# 0. GLOBAL CONFIGURATION & PATHS
# ==========================================
st.set_page_config(page_title="Academic Recommender", layout="wide")

FILE_PATH = 'info.csv'
COSINE_SIM_PATH = 'cosine_sim_matrix.joblib'
INDICES_PATH = 'indices_series.joblib'

# ==========================================
# 1. SHARED DATA LOADING (Cached for Speed)
# ==========================================

@st.cache_data
def load_data_from_csv(file_path):
    """
    Loads data and cleans ALL requested columns.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}. Please make sure the CSV is in the same folder.")
        st.stop()

    df = pd.read_csv(file_path)
    

    # 2. Clean Data (Fill Missing Values)
    text_cols = ['Title', 'Abstract', 'Keywords', 'Publication_name', 'Authors', 'Link', 'Subject_Fields', 'Date']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')

    if 'Year' in df.columns:
        df['Year'] = df['Year'].fillna(0).astype(int)
    
    if 'Author_count' in df.columns:
        df['Author_count'] = df['Author_count'].fillna(0).astype(int)

    # 3. Create 'full_text' for AI Model (Model needs Title + Abstract + Keywords)
    df['full_text'] = (
        df.get('Title', '') + ' ' + 
        df.get('Abstract', '') + ' ' + 
        df.get('Keywords', '')
    )

    df = df[df['full_text'].str.len() > 10].reset_index(drop=True)

    # Generate keywords list for filters
    all_keywords = set()
    if 'Keywords' in df.columns:
        for keywords_string in df['Keywords']:
            if isinstance(keywords_string, str):
                keys = [k.strip() for k in keywords_string.split(',') if k.strip()]
                all_keywords.update(keys)

    keyword_options = ["No Filter"] + sorted(list(all_keywords))
    
    return df, keyword_options

# ==========================================
# 2. AI MODEL TRAINING (Cached & Persistent)
# ==========================================

@st.cache_resource
def train_model(data_frame):
    """
    Trains the TF-IDF model and calculates cosine similarity.
    """
    # --- 1. Check if saved models exist ---
    if os.path.exists(COSINE_SIM_PATH) and os.path.exists(INDICES_PATH):
        try:
            cosine_sim = joblib.load(COSINE_SIM_PATH)
            indices = joblib.load(INDICES_PATH)
            return cosine_sim, indices
        except Exception as e:
            st.warning(f"Error loading saved model: {e}. Retraining...")
    
    # --- 2. TRAIN and SAVE ---
    tfidf = TfidfVectorizer(stop_words='english', min_df=2)
    tfidf_matrix = tfidf.fit_transform(data_frame['full_text'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    indices = pd.Series(data_frame.index, index=data_frame['Title']).drop_duplicates()
    
    try:
        joblib.dump(cosine_sim, COSINE_SIM_PATH)
        joblib.dump(indices, INDICES_PATH)
        st.toast("AI Model trained and saved successfully!")
    except Exception as e:
        st.error(f"Could not save model files to disk: {e}")
        
    return cosine_sim, indices

# ==========================================
# 3. PAGE FUNCTIONS
# ==========================================

def show_recommender_page(df, cosine_sim, indices, keyword_options):
    st.title("AI Research Paper Recommender")
    st.markdown("Select a paper to find similar research based on content similarity.")

    st.sidebar.header("1. Select Paper")
    selected_paper = st.sidebar.selectbox("Choose a paper:", df['Title'].unique())

    st.sidebar.header("2. Filter Options")
    selected_keyword = st.sidebar.selectbox("Filter by keyword:", keyword_options)

    if st.sidebar.button('Show Recommendations'):
        try:
            idx = indices[selected_paper]
            if isinstance(idx, pd.Series): idx = idx.iloc[0]
                
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            sim_scores_candidates = sim_scores[1:50]
            candidate_indices = [i[0] for i in sim_scores_candidates]
            candidate_df = df.iloc[candidate_indices].copy()
            
            if selected_keyword != "No Filter":
                candidate_df = candidate_df[candidate_df['Keywords'].str.contains(selected_keyword, case=False, na=False)]
            
            recommendations = candidate_df.head(5)

            st.subheader("Selected Paper:")
            st.markdown(f"### {selected_paper}")
            with st.expander("View Abstract & Details", expanded=True):
                if 'Authors' in df.columns: st.write(f"**Authors:** {df.iloc[idx]['Authors']}")
                if 'Year' in df.columns: st.write(f"**Year:** {df.iloc[idx]['Year']}")
                st.write(f"**Abstract:** {df.iloc[idx]['Abstract'][:]}...")

            if recommendations.empty:
                st.warning(f"No similar papers found with the keyword '{selected_keyword}'.")
            else:
                st.subheader("Top Recommendations:")
                for i, row in recommendations.iterrows():
                    with st.container():
                        st.markdown(f"### {row['Title']}")
                        col1, col2 = st.columns([1, 4])
                                                
                        with col1:
                            if 'Year' in row: st.caption(f"Year: {row['Year']}")
                            if 'Authors' in row and str(row['Authors']).strip() != "":
                                authors = str(row['Authors'])
                                if len(authors) > 50: 
                                    authors = authors[:50] + "..."
                                st.caption(f"Author: {authors}")
                            else:
                                st.caption("Author: Unknown")
                            if 'Publication_name' in row: st.caption(f"Source: {row['Publication_name']}")
                        
                        with col2:
                            abstract_preview = row['Abstract'][:250] + "..." if len(row['Abstract']) > 250 else row['Abstract']
                            st.write(abstract_preview)
                            
                            with st.expander("Read full Abstract"):
                                st.write(row['Abstract'])
                            
                            raw_link = str(row.get('Link', '')).strip()
                            if raw_link and raw_link != 'No Link':
                                if not raw_link.startswith('http'):
                                    final_link = f"https://{raw_link}"
                                else:
                                    final_link = raw_link
                                
                                st.markdown(f"[Read Full Paper]({final_link})")
                            else:
                                st.caption("No Link Available")
                                
                        st.divider()

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Select a paper from the sidebar to start!")

def show_insights_page(df):
    st.title("Data Insights & Statistics")
    
    # --- 1. Key Metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Papers", f"{len(df):,}")
    with col2:
        if 'Year' in df.columns:
            st.metric("Date Range", f"{df['Year'].min()} - {df['Year'].max()}")
    with col3:
        if 'Author_count' in df.columns:
            st.metric("Avg. Authors per Paper", f"{df['Author_count'].mean():.1f}")

    st.divider()

    # --- 2. Publications by Year ---
    if 'Year' in df.columns:
        st.subheader("Publications by Year")
        yearly_counts = df[df['Year'] > 1900]['Year'].value_counts().sort_index()
        if not yearly_counts.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            yearly_counts.plot(kind='bar', ax=ax, color='#3498db')
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("View Publications Years Table"):
                st.dataframe(pd.DataFrame({'Year': yearly_counts.index, 'Count': yearly_counts.values}))

    # --- 3. Top Authors ---
    if 'Authors' in df.columns:
        st.subheader("Top Primary Authors")
        primary_authors = df['Authors'].apply(lambda x: x.split(',')[0] if isinstance(x, str) and x else "Unknown")
        top_authors = primary_authors.value_counts().head(10)
        
        if not top_authors.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.barplot(x=top_authors.values, y=top_authors.index, ax=ax2, palette='magma')
            ax2.set_xlabel("Number of Papers")
            st.pyplot(fig2)
            plt.close(fig2)

            with st.expander("View Top Author Frequency Table"):
                st.dataframe(pd.DataFrame({'Authors': top_authors.index, 'Count': top_authors.values}))

    # --- 4. Subject Field Analysis (NEW) ---

    if 'Subject_Fields' in df.columns:
        st.divider()
        st.subheader("Subject Area Distribution")
        
        # 1. Process the subjects (Split by semicolon or comma to handle multiple fields)
        all_subjects = []
        for item in df['Subject_Fields'].dropna():
            if isinstance(item, str):
                clean_text = item.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
                
                parts = clean_text.split(',')
                
                parts = [p.strip() for p in parts if p.strip()]
                
                all_subjects.extend(parts)

        # 2. Count top subjects
        if all_subjects:
            subject_counts = pd.Series(all_subjects).value_counts().head(10)
            
            # 3. Create Plot
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            sns.barplot(x=subject_counts.values, y=subject_counts.index, ax=ax3, palette='viridis')
            ax3.set_xlabel("Count")
            ax3.set_title(f"Top 10 Subject Fields")
            
            st.pyplot(fig3)
            plt.close(fig3)
            
            # Show raw data in an expander
            with st.expander("View Subject Frequency Table"):
                st.dataframe(pd.DataFrame({'Subject_Fields': subject_counts.index, 'Count': subject_counts.values}))
def show_explore_page(df):
    """
    NEW PAGE: Displays the raw data with all the requested columns.
    """
    st.title("Data Explorer")
    st.markdown("Browse the complete dataset. You can sort by clicking column headers.")

    if 'Year' in df.columns:
        years = sorted(df['Year'].unique(), reverse=True)
        selected_year = st.selectbox("Filter by Year (Optional):", ["All"] + list(years))
        
        if selected_year != "All":
            df_display = df[df['Year'] == selected_year]
        else:
            df_display = df
    else:
        df_display = df

    column_config = {
        "Link": st.column_config.LinkColumn("Paper Link"),
        "full_text": None, 
        "Year": st.column_config.NumberColumn("Year", format="%d"),
        "Author_count": st.column_config.NumberColumn("Authors", format="%d"),
    }

    st.dataframe(
        df_display,
        use_container_width=True,
        column_config=column_config,
        height=600
    )

    st.caption(f"Showing {len(df_display)} rows.")

# ==========================================
# 4. MAIN APPLICATION LOGIC
# ==========================================

try:
    with st.spinner('Loading data...'):
        df, keyword_options = load_data_from_csv(FILE_PATH)
    with st.spinner('Checking model...'):
        cosine_sim, indices = train_model(df)
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to:",
    ["Paper Recommender", "Data Insights & Stats", "Raw Data Explorer"]
)

st.sidebar.divider()

if selected_page == "Paper Recommender":
    show_recommender_page(df, cosine_sim, indices, keyword_options)
elif selected_page == "Data Insights & Stats":
    show_insights_page(df)
elif selected_page == "Raw Data Explorer":
    show_explore_page(df) 

# --- Reset Button ---
st.sidebar.subheader("Settings")
if st.sidebar.button("Reset Model Cache"):
    try:
        if os.path.exists(COSINE_SIM_PATH): os.remove(COSINE_SIM_PATH)
        if os.path.exists(INDICES_PATH): os.remove(INDICES_PATH)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.sidebar.success("Cache cleared! Refresh page.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error: {e}")