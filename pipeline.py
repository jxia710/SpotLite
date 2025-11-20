import json
import re
from pathlib import Path
from collections import defaultdict, Counter

import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# ================================================
# 0. Ensure VADER lexicon exists
# ================================================
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")


# ================================================
# 1. Aspect Lexicon
# ================================================
ASPECT_LEXICON = {
    "taste": ["taste", "flavor", "spicy", "fresh", "broth", "ingredient", "delicious", "yummy"],
    "service": ["service", "staff", "waiter", "waitress", "friendly", "rude", "helpful", "attitude"],
    "environment": ["clean", "space", "ambience", "noisy", "quiet", "crowded", "atmosphere", "environment", "patio"],
    "waiting_time": ["wait", "waiting", "queue", "line", "slow", "fast", "minutes", "minute"],
    "price": ["price", "expensive", "cheap", "value", "overpriced", "cost", "worth"],
}

BAD_SINGLE_WORDS = {
    "i", "it", "we", "you", "they", "he", "she",
    "person", "people", "that", "this", "there"
}

ASPECT_SINGLE_WORD_DELETE = {
    "taste": {"taste", "flavor", "food"},
    "service": {"service", "staff"},
    "environment": {"environment", "place", "location"},
    "waiting_time": {"wait", "waiting", "time", "line", "queue", "minutes", "minute"},
    "price": {"price", "value", "cost"},
}

WAIT_TERMS = {"wait", "waiting", "line", "queue", "minutes", "minute"}

# Generic food terms to exclude
GENERIC_FOOD = {
    "food", "rice", "soup", "seafood", "meat", "dish", "meal", "plate", "noodles"
}

# Food cue words
FOOD_CUE_WORDS = {
    "soup", "noodle", "noodles", "fish", "seafood", "chicken", "pork", "beef",
    "shrimp", "taco", "tacos", "burrito", "ramen", "dumpling", "dumplings",
    "egg roll", "roll", "rolls", "tofu", "salad", "lobster", "sashimi", "sushi",
    "ceviche", "mariscos", "pho", "hot pot", "malatang", "bbq", "burger",
    "fries", "pasta", "steak", "wings", "pizza", "quesadilla", "birria"
}

BANNED_PHRASES = {"dine", "dine-in", "dine in", "takeout", "take-out"}

OTHER_ASPECT_WORDS = {}
for asp in ASPECT_LEXICON:
    other = set()
    for b, kws in ASPECT_LEXICON.items():
        if b != asp:
            other.update(kws)
    OTHER_ASPECT_WORDS[asp] = other


# ============================================================
# Clean review text
# ============================================================
def clean_text(raw: str) -> str:
    patterns = [
        r"meal type[:：].*",
        r"price[:：].*",
        r"service[:：].*",
        r"atmosphere[:：].*",
        r"noise[:：].*",
        r"noise level[:：].*",
        r"parking[:：].*",
        r"special events[:：].*",
        r"\d+(\s*stars?)?",
        r"\$\S+",
    ]
    for p in patterns:
        raw = re.sub(p, "", raw, flags=re.IGNORECASE)

    raw = re.sub(r"…\s*More$", "", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


# ============================================================
# Load reviews from JSON file
# ============================================================
def load_reviews(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    for d in data:
        if "text" in d:
            t = clean_text(d["text"])
            if t:
                texts.append(t)
    return texts


# ============================================================
# Assign sentences to aspect buckets
# ============================================================
def bucket_sentences(sentences):
    buckets = defaultdict(list)
    for sent in sentences:
        s = sent.lower()
        scores = {asp: sum(kw in s for kw in kws)
                  for asp, kws in ASPECT_LEXICON.items()}
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            buckets[best].append(sent)
    return buckets


# ============================================================
# Check if phrase is a potential dish name
# ============================================================
def is_potential_dish(phrase: str) -> bool:
    phrase = phrase.lower().strip()

    if phrase.startswith("the "):
        phrase = phrase[4:]

    if any(ch.isdigit() for ch in phrase):
        return False

    if phrase in GENERIC_FOOD:
        return False

    if "price" in phrase:
        return False

    if len(phrase) < 4 or len(phrase) > 40:
        return False

    for cue in FOOD_CUE_WORDS:
        if cue in phrase:
            return True

    return False


# ============================================================
# Check if phrase should be dropped for aspect
# ============================================================
def should_drop_phrase_for_aspect(phrase: str, aspect: str) -> bool:
    phrase = phrase.lower().strip()

    if is_potential_dish(phrase):
        return True
    if any(b in phrase for b in BANNED_PHRASES):
        return True
    if any(ch.isdigit() for ch in phrase):
        return True
    if phrase in BAD_SINGLE_WORDS:
        return True

    words = phrase.split()

    if len(words) == 1:
        return True
    if words[0] == "the" and len(words) <= 3:
        return True

    for other_kw in OTHER_ASPECT_WORDS[aspect]:
        if other_kw in phrase:
            return True

    if phrase in ASPECT_SINGLE_WORD_DELETE[aspect]:
        return True

    if len(phrase) < 4 or len(phrase) > 40:
        return True

    return False


# ============================================================
# Extract and clean phrases for aspect
# ============================================================
def extract_clean_phrases_for_aspect(doc, aspect):
    phrases = []

    # noun chunks
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        if aspect == "waiting_time" and not any(t in phrase for t in WAIT_TERMS):
            continue
        if not should_drop_phrase_for_aspect(phrase, aspect):
            phrases.append(phrase)

    # ADJ + NOUN
    for i in range(len(doc) - 1):
        if doc[i].pos_ == "ADJ" and doc[i+1].pos_ == "NOUN":
            phrase = f"{doc[i].text.lower()} {doc[i+1].text.lower()}"
            if not should_drop_phrase_for_aspect(phrase, aspect):
                phrases.append(phrase)

    return phrases


# ============================================================
# Extract wait time in minutes
# ============================================================
def parse_wait_minutes(sentence: str):
    s = sentence.lower()

    m = re.search(r"(\d+)\s*[-]?\s*ish\s*(min|mins|minute|minutes)", s)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d+)\s*(min|mins|minute|minutes)\b", s)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d+)\s*[-]?\s*minute", s)
    if m:
        return int(m.group(1))

    return None


def wait_time_label(minutes: int):
    if minutes <= 10:
        return "very short wait (<10 min)"
    elif minutes <= 20:
        return "short wait (10–20 min)"
    elif minutes <= 35:
        return "moderate wait (20–35 min)"
    elif minutes <= 50:
        return "long wait (35–50 min)"
    else:
        return "very long wait (>50 min)"


# ============================================================
# Cluster phrases to get representative ones
# ============================================================
def cluster_phrases(phrases, num_clusters=4):
    phrases = list(set(phrases))
    if len(phrases) <= num_clusters:
        return phrases

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(phrases)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(X)

    grouped = []
    for c in range(num_clusters):
        group = [phrases[i] for i in range(len(phrases)) if labels[i] == c]
        if group:
            grouped.append(Counter(group).most_common(1)[0][0])
    return grouped


# ============================================================
# Summarize aspect
# ============================================================
def summarize_aspect(aspect, sentences, nlp, sia):
    if not sentences:
        return "neutral", [], None

    all_phrases = []
    sentiment_scores = []
    wait_times = []

    for doc in nlp.pipe(sentences):
        ps = extract_clean_phrases_for_aspect(doc, aspect)
        all_phrases.extend(ps)
        sentiment_scores.append(sia.polarity_scores(doc.text)["compound"])

        if aspect == "waiting_time":
            minutes = parse_wait_minutes(doc.text)
            if minutes:
                wait_times.append(minutes)

    freq = Counter(all_phrases)
    top_phrases = [p for p, c in freq.most_common(20)]
    representative = cluster_phrases(top_phrases, num_clusters=4)
    representative = [p for p in representative if len(p) >= 4]

    avg = sum(sentiment_scores) / len(sentiment_scores)
    polarity = ("positive" if avg > 0.1 else "negative" if avg < -0.1 else "neutral")

    if aspect == "waiting_time" and wait_times:
        import statistics
        median_wait = statistics.median(wait_times)
        wait_category = wait_time_label(median_wait)
    else:
        wait_category = None

    return polarity, representative, wait_category


# ============================================================
# Extract dish candidates from doc
# ============================================================
def extract_dish_candidates_from_doc(doc):
    candidates = []
    for chunk in doc.noun_chunks:
        txt = chunk.text.lower().strip()
        if is_potential_dish(txt):
            candidates.append(txt)
    return candidates


def extract_top_dishes(texts, nlp, top_k=5):
    all_dishes = []

    for doc in nlp.pipe(texts):
        ds = extract_dish_candidates_from_doc(doc)
        all_dishes.extend(ds)

    cleaned = {}
    for dish, cnt in Counter(all_dishes).items():
        d = dish.replace("the ", "")
        if d in GENERIC_FOOD:
            continue
        cleaned[d] = cnt

    sorted_dishes = sorted(cleaned.items(), key=lambda x: x[1], reverse=True)
    return [d for d, c in sorted_dishes[:top_k]]


# ============================================================
# Core analysis function for single file
# ============================================================
def analyze_single_file(path: Path, output_dir: Path, nlp, sia):

    texts = load_reviews(path)
    if not texts:
        return

    # Sentence segmentation
    sentences = []
    for doc in nlp.pipe(texts, disable=["ner"]):
        for s in doc.sents:
            st = s.text.strip()
            if st:
                sentences.append(st)

    buckets = bucket_sentences(sentences)

    # Output results
    result = []
    result.append(f"Analysis for {path.name}")

    for aspect in ["taste", "service", "environment", "waiting_time", "price"]:
        sents = buckets.get(aspect, [])
        pol, reps, wait_label = summarize_aspect(aspect, sents, nlp, sia)

        result.append(f"\n{aspect.upper()}")
        result.append(f"Sentiment: {pol}")

        if aspect == "waiting_time" and wait_label:
            result.append(f"Estimated wait time: {wait_label}")

        result.append("Keywords / phrases:")
        if reps:
            for r in reps:
                result.append(f"{r}")
        else:
            result.append("  (no strong keywords found)")

    # Dish highlights
    result.append("\nDISH HIGHLIGHTS")
    top_dishes = extract_top_dishes(texts, nlp, top_k=5)
    if not top_dishes:
        result.append("  (no strong dish names detected)")
    else:
        for d in top_dishes:
            result.append(f"{d}")

    # Save to output file
    output_path = output_dir / f"{path.stem}_analysis.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result))

    print(f"Saved analysis → {output_path}")


# ============================================================
# Run all data/*.json files
# ============================================================
def main():
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    sia = SentimentIntensityAnalyzer()

    data_dir = Path("data")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    json_files = list(data_dir.glob("*_reviews.json"))

    if not json_files:
        print("No JSON files found in data/")
        return

    print(f"Found {len(json_files)} JSON review files.\n")

    for path in json_files:
        analyze_single_file(path, output_dir, nlp, sia)


if __name__ == "__main__":
    main()
