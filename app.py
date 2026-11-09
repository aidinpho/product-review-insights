import pandas as pd
import streamlit as st

from insights import analyze_reviews, example_data

st.set_page_config(page_title="Product Review Insights", page_icon="🔎", layout="wide")
st.title("Product Review Insights")
st.caption("Explore customer topics, sentiment over time, and the reviews behind the trends.")

with st.sidebar:
    st.header("Review data")
    uploaded = st.file_uploader("Upload a CSV", type="csv")
    st.caption("Required: review/text column. Optional: date, rating.")
    topic_count = st.slider("Discovered topics", min_value=2, max_value=10, value=5,
                            help="Themes are discovered from the review text.")

try:
    data = pd.read_csv(uploaded) if uploaded else example_data()
    result = analyze_reviews(data, n_topics=topic_count)
except Exception as exc:
    st.error(f"Could not analyze this file: {exc}")
    st.stop()

if not uploaded:
    st.info("Example dataset shown. Upload your own CSV to explore it.")
if result.empty:
    st.warning("No topic mentions found. Check the review column or expand topic keywords.")
    st.stop()

topics = sorted(result.topic.unique())
selected = st.multiselect("Topics", topics, default=topics)
view = result[result.topic.isin(selected)]
first, second, third = st.columns(3)
first.metric("Reviews analyzed", f"{len(data):,}")
second.metric("Topic mentions", f"{len(view):,}")
third.metric("Topics found", len(selected))

st.subheader("Topic sentiment")
summary = view.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)
summary = summary.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
st.bar_chart(summary)

dated = view.dropna(subset=["date"])
if not dated.empty:
    st.subheader("Sentiment over time")
    dated = dated.assign(month=dated.date.dt.to_period("M").dt.to_timestamp())
    trend = dated.groupby(["month", "topic", "sentiment"]).size().unstack(["topic", "sentiment"], fill_value=0)
    trend.columns = [f"{topic} · {sentiment}" for topic, sentiment in trend.columns]
    st.line_chart(trend.sort_index())

st.subheader("Review evidence")
topic_filter = st.selectbox("Show evidence for", ["All topics", *selected])
evidence = view if topic_filter == "All topics" else view[view.topic == topic_filter]
st.dataframe(evidence.sort_values("date", ascending=False, na_position="last"), use_container_width=True, hide_index=True)
