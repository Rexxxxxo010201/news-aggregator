import streamlit as st
import pandas as pd
import ast
from news_visualizer import NewsVisualizer
import os

# --- Streamlit Page Setup ---
st.set_page_config(page_title="📰 News Insights Dashboard", layout="wide")
st.title("📰 News Insights Aggregator")
st.markdown("Explore trends, sources, and keywords in the latest enriched news data.")

# --- CSV Path ---
CSV_PATH = "news_dataset.csv"

if not os.path.exists(CSV_PATH):
    st.error("❌ CSV file not found. Please ensure 'news_dataset.csv' exists in this directory.")
    st.stop()

# --- Load Visualizer ---
visualizer = NewsVisualizer(CSV_PATH)

if visualizer.df.empty:
    st.warning("⚠️ Data could not be loaded. No visualizations will be shown.")
    st.stop()

# --- Sidebar: User Inputs ---
st.sidebar.header("🔧 Filters")

category_list = visualizer.categories
selected_category = st.sidebar.selectbox("Select a Category", category_list, index=0)

# Search + Reset
search_query = st.sidebar.text_input("Search in title / summary / content:")
if st.sidebar.button("🔄 Reset Search"):
    search_query = ""

max_rows = st.sidebar.slider("Max Rows to Display", 5, 100, 20, 5)

# --- Apply Filter to Data Table (Not Visuals) ---
df_filtered = visualizer.df.copy()

if search_query:
    q = search_query.lower()
    df_filtered = df_filtered[
        df_filtered["title"].str.lower().str.contains(q, na=False) |
        df_filtered["summary"].str.lower().str.contains(q, na=False) |
        df_filtered["content"].str.lower().str.contains(q, na=False)
    ]

# Filter by selected category
df_filtered = df_filtered[df_filtered["category"] == selected_category]

df_filtered["published_date"] = pd.to_datetime(df_filtered["published_date"], errors="coerce").dt.date


# --- Visualizations ---
st.subheader("🔗 Source to Category Flow")
visualizer.show_sankey()

st.subheader(f"☁️ Word Cloud – {selected_category.capitalize()}")
visualizer.show_wordcloud(selected_category)

st.subheader(f"📈 Article Volume – {selected_category.capitalize()}")
visualizer.show_scatter(selected_category)

# --- Summary and Table ---
st.subheader("📋 Filtered Article Table")

article_count = len(df_filtered)
summary_text = f"Showing {article_count} article{'s' if article_count != 1 else ''}"
if search_query:
    summary_text += f" matching “{search_query}”"
st.markdown(f"**{summary_text}**")

table_cols = ["title", "source", "category", "keywords", "published_date", "author"]

with st.expander("📄 Show Data Table"):
    st.dataframe(df_filtered[table_cols].head(max_rows))



st.subheader("🧾 Full Article View")

for idx, row in df_filtered.head(max_rows).iterrows():
    with st.expander(f"📰 {row['title']}"):
        st.markdown(f"**Source:** {row['source']}")
        st.markdown(f"**Published on:** {row['published_date']}")
        st.markdown(f"**Author:** {row.get('author', 'Unknown')}")
        st.markdown(f"**Keywords:** {', '.join(row.get('keywords', []))}")
        st.markdown("---")
        st.markdown(row.get('content', 'No content available'))


# --- Download Button ---
csv = df_filtered.to_csv(index=False)
st.download_button("📥 Download Filtered Data", csv, "filtered_news.csv", "text/csv")
