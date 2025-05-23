import streamlit as st
import pandas as pd
import ast
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objects as go
from collections import Counter

class NewsVisualizer:
    def __init__(self, csv_path: str):
        try:
            self.df = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
            self.df = pd.DataFrame()
            return

        try:
            self.df["date"] = pd.to_datetime(self.df["published_date"], errors="coerce").dt.date
            self.df = self.df.dropna(subset=["source", "category", "date"])
            self.df["keywords"] = self.df["keywords"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            self.categories = sorted(self.df["category"].dropna().unique().tolist())
        except Exception as e:
            st.error(f"❌ Data preprocessing failed: {e}")
            self.df = pd.DataFrame()

    def show_sankey(self):
        if self.df.empty:
            st.warning("⚠️ No data available for Sankey diagram.")
            return

        try:
            top_sources = self.df["source"].value_counts().head(10).index
            df_filtered = self.df[self.df["source"].isin(top_sources)]

            top_links = (
                df_filtered.groupby(["source", "category"])
                .size()
                .reset_index(name="count")
            )

            sources = top_links["source"].unique().tolist()
            categories = top_links["category"].unique().tolist()
            labels = sources + categories
            label_to_index = {label: idx for idx, label in enumerate(labels)}

            category_colors = {
                cat: f"hsl({i * 360 // len(categories)}, 70%, 60%)" for i, cat in enumerate(categories)
            }

            colors = [("#4B8BBE" if label in sources else category_colors[label]) for label in labels]
            link_colors = top_links["category"].map(category_colors).tolist()

            source_indices = top_links["source"].map(label_to_index).tolist()
            target_indices = top_links["category"].map(label_to_index).tolist()
            values = top_links["count"].tolist()

            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=colors
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                    color=link_colors
                )
            )])

            fig.update_layout(title_text="Sankey Diagram: Top 10 Sources → Category", font_size=13)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Failed to render Sankey diagram: {e}")

    def show_wordcloud(self, category_name: str):
        if self.df.empty:
            st.warning("⚠️ Data not loaded. Word cloud skipped.")
            return

        try:
            subset = self.df[self.df["category"] == category_name]
            all_keywords = [kw for keywords in subset["keywords"] for kw in keywords]

            custom_stopwords = set(STOPWORDS)
            custom_stopwords.update(["said", "also", "like"])

            # ❗ Filter BEFORE building Counter
            filtered_keywords = [kw for kw in all_keywords if kw.lower() not in custom_stopwords]

            if not filtered_keywords:
                st.warning(f"⚠️ No usable keywords found for category: {category_name}")
                return

            keyword_freq = Counter(filtered_keywords)

            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                stopwords=custom_stopwords
            ).generate_from_frequencies(keyword_freq)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            ax.set_title(f"Keyword Word Cloud – {category_name}", fontsize=16)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Failed to generate word cloud: {e}")



    def show_scatter(self, category_name: str):
        if self.df.empty:
            st.warning("⚠️ Data not loaded. Scatter plot skipped.")
            return

        try:
            df_cat = self.df[self.df["category"] == category_name]
            ts = df_cat.groupby("date").size().reset_index(name="article_count")
            ts["date"] = pd.to_datetime(ts["date"])
            ts = ts.sort_values("date")

            if ts.empty:
                st.warning(f"⚠️ No article data for category: {category_name}")
                return

            # Convert dates to string labels for categorical x-axis
            ts["date_str"] = ts["date"].dt.strftime('%Y-%m-%d')

            fig, ax = plt.subplots(figsize=(6, 5))

            # Plot points
            ax.scatter(ts["date_str"], ts["article_count"], color="#4B8BBE", s=40)

            # Connect the dots
            ax.plot(ts["date_str"], ts["article_count"], color="#4B8BBE", linestyle='-', linewidth=1)

            # Axis and title settings
            ax.set_title(f"Scatter Plot – Articles per Day ({category_name.capitalize()})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Articles")
            ax.set_xticks(ts["date_str"])
            ax.set_xticklabels(ts["date_str"], rotation=45)
            ax.grid(True, linestyle="--", alpha=0.3)

            # Tight layout and plot
            plt.tight_layout()
            st.pyplot(fig)


        except Exception as e:
            st.error(f"❌ Failed to render scatter plot: {e}")
