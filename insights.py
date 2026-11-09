import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

TOPICS = {
    "Battery": {"battery", "charge", "charging", "power", "lasts", "life"},
    "Quality": {"quality", "build", "broken", "durable", "defect", "material"},
    "Price": {"price", "cost", "expensive", "cheap", "value", "worth"},
    "Delivery": {"delivery", "shipping", "arrived", "late", "package", "courier"},
}
POSITIVE = {"great", "good", "excellent", "love", "fast", "lasting", "durable", "perfect", "easy", "worth", "quick", "impressed"}
NEGATIVE = {"bad", "poor", "broken", "slow", "late", "expensive", "short", "dead", "weak", "disappointed", "waste", "defective"}
NEGATIONS = {"not", "never", "no", "n't"}


def example_data():
    return pd.DataFrame([
        ["Battery lasts all day and charges fast", "2025-01-10", 5],
        ["Great build quality, feels durable", "2025-01-18", 5],
        ["Shipping was late and package was damaged", "2025-02-02", 2],
        ["Good value for the price", "2025-02-11", 4],
        ["Battery life is short; disappointed", "2025-03-03", 2],
        ["Excellent quality and worth the cost", "2025-03-15", 5],
        ["Delivery arrived quickly", "2025-04-04", 4],
        ["Too expensive for such poor quality", "2025-04-20", 1],
    ], columns=["review", "date", "rating"])


def analyze_reviews(frame, n_topics=5):
    columns = {str(col).strip().lower(): col for col in frame.columns}
    text_col = next((columns[key] for key in ("review", "text", "review_text", "reviewtext", "content") if key in columns), None)
    if text_col is None:
        raise ValueError("Add a 'review' or 'text' column.")
    date_col = next((columns[key] for key in ("date", "review_date", "reviewtime", "created_at") if key in columns), None)
    rating_col = columns.get("rating", columns.get("overall"))
    reviews = []
    for _, row in frame.iterrows():
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        if text.strip():
            reviews.append((row, text))
    if not reviews:
        return pd.DataFrame(columns=["date", "topic", "sentiment", "rating", "review"])

    texts = [text for _, text in reviews]
    # Product and review boilerplate often dominate small product-specific datasets.
    corpus_stop_words = {
        "great", "good", "excellent", "love", "product", "item", "buy", "bought",
        "購入", "amazon", "work", "works", "working", "thing", "things", "use", "used",
        "using", "got", "just", "really", "like", "nice", "one", "well", "best",
        "samsung", "galaxy", "phone", "card", "cards", "micro", "sd",
    }
    vectorizer = TfidfVectorizer(stop_words=list(TfidfVectorizer(stop_words="english").get_stop_words() or set()) + list(corpus_stop_words), max_features=3000, ngram_range=(1, 2),
                                 token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b", min_df=1)
    matrix = vectorizer.fit_transform(texts)
    actual_topics = min(max(1, int(n_topics)), len(texts), matrix.shape[1])
    if actual_topics == 1:
        assignments = [0] * len(texts)
        components = matrix.mean(axis=0)
    else:
        model = NMF(n_components=actual_topics, random_state=42, init="nndsvda", max_iter=300)
        weights = model.fit_transform(matrix)
        assignments = weights.argmax(axis=1)
        components = model.components_
    terms = vectorizer.get_feature_names_out()
    labels = {}
    topic_hints = {
        "Battery & charging": {"battery", "charge", "charging", "power"},
        "Build & reliability": {"quality", "build", "broken", "durable", "defect", "material", "reliable"},
        "Price & value": {"price", "cost", "expensive", "cheap", "value", "worth"},
        "Shipping & delivery": {"delivery", "shipping", "arrived", "late", "package", "courier"},
        "Speed & performance": {"speed", "fast", "slow", "transfer", "performance", "read", "write"},
        "Compatibility": {"compatible", "compatibility", "device", "camera", "tablet", "laptop"},
        "Storage capacity": {"capacity", "storage", "gb", "space", "memory"},
    }
    for index in range(actual_topics):
        component = components if actual_topics == 1 else components[index]
        top_terms = [terms[j] for j in component.argsort()[::-1] if len(terms[j]) > 2][:12]
        matched = [(name, len(set(top_terms).intersection(vocabulary))) for name, vocabulary in topic_hints.items()]
        best_name, matches = max(matched, key=lambda item: item[1])
        labels[index] = best_name if matches >= 2 else " · ".join(term.title() for term in top_terms[:2]) or f"Topic {index + 1}"

    sentiment_analyzer = SentimentIntensityAnalyzer()
    rows = []
    for (row, text), topic_index in zip(reviews, assignments):
        score = sentiment_analyzer.polarity_scores(text)["compound"]
        sentiment = "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
        rating = row[rating_col] if rating_col else None
        # Ratings are useful context and help resolve text that is sentimentally ambiguous.
        try:
            numeric_rating = float(rating)
            if score == 0 and numeric_rating >= 4:
                sentiment = "Positive"
            elif score == 0 and numeric_rating <= 2:
                sentiment = "Negative"
        except (TypeError, ValueError):
            pass
        rows.append({
            "date": pd.to_datetime(row[date_col], errors="coerce") if date_col else pd.NaT,
            "topic": labels[topic_index],
            "sentiment": sentiment,
            "rating": rating,
            "review": text,
        })
    return pd.DataFrame(rows, columns=["date", "topic", "sentiment", "rating", "review"])
