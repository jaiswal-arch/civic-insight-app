import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Civic Insight Analyzer",
    page_icon="🏛️",
    layout="wide"
)

# ─── Category Detection ──────────────────────────────────────────────────────
def categorize_issue(description):
    categories = {
        "Environment":    ["environment", "climate", "sustainability", "green"],
        "Education":      ["school", "education", "students", "learning"],
        "Infrastructure": ["road", "bridge", "infrastructure", "construction", "transit"],
        "Zoning":         ["zone", "zoning", "residential", "commercial", "land use"],
        "Safety":         ["safety", "crime", "police", "fire", "emergency"],
        "Housing":        ["housing", "affordable", "rent", "shelter"],
    }
    desc_lower = description.lower()
    for category, keywords in categories.items():
        if any(kw in desc_lower for kw in keywords):
            return category
    return "Other"

# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        feedback_df = pd.read_excel("City of Portland 2023.xlsx", sheet_name="Sheet2")
        council_df  = pd.read_excel("City of Portland 2023.xlsx", sheet_name="Sheet1")
    except FileNotFoundError:
        st.error("Data file not found. Make sure 'City of Portland 2023.xlsx' is in the same folder as app.py.")
        st.stop()

    feedback_df.columns = feedback_df.columns.str.strip()
    council_df.columns  = council_df.columns.str.strip()

    council_df['Issue Category'] = council_df['Description'].fillna('').apply(categorize_issue)

    feedback_df['combined_text'] = (
        "Interest: "       + feedback_df['Interest'].astype(str) +
        " | Address: "     + feedback_df['Address'].astype(str) +
        " | Environmental: "+ feedback_df['Enviornmental'].astype(str) +
        " | Infrastructure: "+ feedback_df['Infastructure'].astype(str) +
        " | Education: "   + feedback_df['Education'].astype(str) +
        " | Zoning: "      + feedback_df['Zoning'].astype(str) +
        " | Safety: "      + feedback_df['Safety'].astype(str)
    )
    return feedback_df, council_df

# ─── Model Loading ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    embed_model    = SentenceTransformer('all-MiniLM-L6-v2')
    summarizer = pipeline("text2text-generation", model="google/flan-t5-small")
    return embed_model, summarizer

# ─── Summarization ───────────────────────────────────────────────────────────
def summarize_texts(texts, summarizer, max_input_chars=800):
    if not texts:
        return "No matching data found for this query."

    # Clean and deduplicate
    cleaned = list({str(t).strip() for t in texts if str(t).strip() and str(t).strip() != 'nan'})
    if not cleaned:
        return "No meaningful data found."

    combined = " ".join(cleaned)[:max_input_chars]
    word_count = len(combined.split())

    if word_count < 15:
        return combined

    try:
        result = summarizer(
            combined,
            max_length=80,
            min_length=20,
            do_sample=False,
            no_repeat_ngram_size=3   # prevents repetition loops
        )
        return result[0]['summary_text']
    except Exception:
        # Fallback: return first 3 cleaned entries as bullet points
        return "\n".join([f"• {t[:150]}" for t in cleaned[:3]])

# ─── Semantic Search ─────────────────────────────────────────────────────────
def semantic_search(query, texts, embed_model, top_k=5):
    if not texts:
        return []
    embeddings  = embed_model.encode(texts)
    query_vec   = embed_model.encode([query])
    similarities = cosine_similarity(query_vec, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices.tolist()

# ─── Feedback Summary Generator ─────────────────────────────────────────────
def generate_feedback_summary(df, interest, area):
    if df.empty:
        return f"No feedback found for {interest} concerns in {area}."

    count = len(df)
    score_cols = {
        'Enviornmental': 'environmental',
        'Infastructure': 'infrastructure',
        'Education': 'education',
        'Zoning': 'zoning',
        'Safety': 'safety'
    }

    # Find highest scoring concern
    avg_scores = {}
    for col, label in score_cols.items():
        if col in df.columns:
            avg = pd.to_numeric(df[col], errors='coerce').mean()
            if not pd.isna(avg):
                avg_scores[label] = round(avg, 1)

    if avg_scores:
        top_concern = max(avg_scores, key=avg_scores.get)
        top_score = avg_scores[top_concern]
        summary = (
            f"{count} residents in {area} have expressed concerns related to {interest}. "
            f"The highest rated concern is {top_concern} with an average score of {top_score}/5. "
        )
        low_concerns = [k for k, v in avg_scores.items() if v < 2.0]
        if low_concerns:
            summary += f"Areas rated lower in priority include {', '.join(low_concerns)}."
    else:
        summary = f"{count} residents in {area} have shared feedback related to {interest}."

    return summary

def generate_council_summary(df, query):
    if df.empty:
        return f"No council orders found matching '{query}'."
    count = len(df)
    categories = df['Issue Category'].value_counts()
    top_category = categories.index[0] if not categories.empty else "General"
    top_count = categories.iloc[0] if not categories.empty else 0
    dates = pd.to_datetime(df['Passage Date'], errors='coerce').dropna()
    date_range = f" between {dates.min().year} and {dates.max().year}" if len(dates) > 0 else ""
    return (
        f"{count} council orders match '{query}'{date_range}. "
        f"The most common category is {top_category} ({top_count} orders). "
        f"All results are classified as: {', '.join(categories.index.tolist())}."
    )

# ─── Load Everything ─────────────────────────────────────────────────────────
feedback_df, council_df = load_data()
embed_model, summarizer = load_models()

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("🏛️ Civic Insight Analyzer")
st.markdown("**Connecting Portland's community voices with government action.**")
st.divider()

# ─── Top Metrics ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Feedback Entries",  f"{len(feedback_df):,}")
col2.metric("Total Council Orders",    f"{len(council_df):,}")
col3.metric("Issue Categories",        council_df['Issue Category'].nunique())
col4.metric("Unique Neighborhoods",    feedback_df['Address'].dropna().nunique())

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Citizen Feedback", "🏛️ Council Orders", "⚖️ Compare & Connect"])

# ── Tab 1: Citizen Feedback ──────────────────────────────────────────────────
with tab1:
    st.subheader("What are residents saying?")
    st.caption("Search across all citizen feedback using natural language.")

    feedback_query = st.text_input(
        "Ask a question about feedback",
        placeholder="e.g. 'Zoning concerns in East Bayside' or 'infrastructure issues'"
    )

    if feedback_query:
        with st.spinner("Searching feedback..."):
            texts   = feedback_df['combined_text'].tolist()
            top_idx = semantic_search(feedback_query, texts, embed_model)
            top_df  = feedback_df.iloc[top_idx]
            summary = generate_feedback_summary(top_df, feedback_query, "search results")

        st.markdown("#### 📝 Summary")
        st.info(summary)

        st.markdown("#### 📋 Top Matching Feedback Entries")
        display_cols = [c for c in top_df.columns if c != 'combined_text']
        st.dataframe(top_df[display_cols].reset_index(drop=True), use_container_width=True)

# ── Tab 2: Council Orders ────────────────────────────────────────────────────
with tab2:
    st.subheader("What actions has the council taken?")
    st.caption("Search council records and orders by topic or keyword.")

    council_query = st.text_input(
        "Ask about council actions",
        placeholder="e.g. 'affordable housing' or 'road construction 2023'"
    )

    if council_query:
        with st.spinner("Searching council records..."):
            descriptions = council_df['Description'].fillna('').astype(str).tolist()
            top_idx      = semantic_search(council_query, descriptions, embed_model)
            top_df       = council_df.iloc[top_idx][['Type', 'Description', 'Issue Category', 'Passage Date']]
            summary      = generate_council_summary(top_df, council_query)

        st.markdown("#### 📝 Summary")
        st.info(summary)

        st.markdown("#### 📋 Top Matching Council Orders")
        st.dataframe(top_df.reset_index(drop=True), use_container_width=True)

    # Category breakdown chart
    st.divider()
    st.markdown("#### 📊 Council Orders by Issue Category")
    category_counts = council_df['Issue Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    category_counts = category_counts[category_counts['Count'] > 0]
    st.bar_chart(category_counts.set_index('Category'), y='Count')

# ── Tab 3: Compare & Connect ─────────────────────────────────────────────────
with tab3:
    st.subheader("Where do citizen concerns and council actions align?")
    st.caption("Select a category and neighborhood to compare community voice with government response.")

    col1, col2 = st.columns(2)
    with col1:
        selected_interest = st.selectbox(
            "Select Issue Category:",
            sorted(feedback_df['Interest'].dropna().unique())
        )
    with col2:
        selected_area = st.selectbox(
            "Select Neighborhood / Area:",
            sorted(feedback_df['Address'].dropna().unique())
        )

    if st.button("🔍 Compare", use_container_width=True):
        with st.spinner("Connecting the dots..."):
            # Filter feedback
            filtered_feedback = feedback_df[
                (feedback_df['Interest'] == selected_interest) &
                (feedback_df['Address'].str.contains(selected_area, case=False, na=False))
            ]

            # Filter council
            council_filtered = council_df[
                council_df['Issue Category'].str.contains(selected_interest, case=False, na=False)
            ]

            # Generate summary directly from data without AI model
            feedback_summary = generate_feedback_summary(filtered_feedback, selected_interest, selected_area)
            council_summary = summarize_texts(
                council_filtered['Description'].dropna().tolist(), summarizer
            )

        # Metrics
        m1, m2 = st.columns(2)
        m1.metric("Matching Feedback Entries", len(filtered_feedback))
        m2.metric("Matching Council Orders",   len(council_filtered))

        st.divider()

        # Summaries side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 🗣️ Community Voice")
            st.markdown(f"*{selected_interest} concerns in {selected_area}*")
            st.info(feedback_summary if feedback_summary else "No matching feedback found.")

        with col2:
            st.markdown(f"#### 🏛️ Government Response")
            st.markdown(f"*Council actions on {selected_interest}*")
            st.success(council_summary if council_summary else "No matching council orders found.")

        st.divider()

        # Raw data
        with st.expander("📋 View Feedback Entries"):
            display_cols = [c for c in filtered_feedback.columns if c != 'combined_text']
            st.dataframe(filtered_feedback[display_cols].reset_index(drop=True), use_container_width=True)

        with st.expander("📋 View Council Orders"):
            st.dataframe(council_filtered.reset_index(drop=True), use_container_width=True)