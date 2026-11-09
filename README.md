# Product Review Insights

A standalone Streamlit MVP for analyzing product reviews by topic, sentiment, time, and direct customer evidence. It does not reuse the existing sentiment-analysis projects.

## Run locally

```bash
cd product-review-insights
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload a UTF-8 CSV with a `review` column and optional `date` and `rating` columns. `text`, `review_text`, `content`, `review_date`, and `created_at` are also recognized. Try `sample_reviews.csv` first.

## Features and limits

- Discovers themes from review text and gives recognizable themes concise labels such as Battery life, Build quality, Price & value, and Shipping & delivery.
- Assigns English sentiment with VADER, which accounts for negation, emphasis, and common informal phrasing.
- Charts sentiment counts by topic and month; shows original review text as evidence.
- Reviews mentioning multiple topics appear once per matching topic.

Topic discovery is unsupervised, so labels and assignments are suggestions and can be imperfect. VADER is a general English lexicon, not a product-specific model; use the evidence table to review borderline results. Install updates with `pip install -r requirements.txt`.
