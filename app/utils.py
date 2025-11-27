import re
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Paths & data loading

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "unified_courses_v1.csv"
EMB_PATH = BASE_DIR / "course_embeddings.npy"
INDEX_PATH = BASE_DIR / "faiss_index.bin"

df = pd.read_csv(DATA_PATH)

# Basic cleaning
for col in ["course_title", "instructor_title", "category", "subcategory",
           "course_url", "course_mapping_keywords", "topic_title"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

df["enrollments"] = df["enrollments"].fillna(0).astype(float)
df["avg_rating_90d"] = df["avg_rating_90d"].fillna(0).astype(float)

# Load main course embeddings + FAISS index
course_embeddings = np.load(EMB_PATH)
index = faiss.read_index(str(INDEX_PATH))


# Sentence transformer model

model = SentenceTransformer("clairenp/miniLM_finetuned_udemy")


#1. Domain detection with anchors + course-topic

DOMAIN_ANCHORS = {
    "data_science": "machine learning, data science, python, analytics, statistics",
    "software_engineering": "software engineering, backend, frontend, coding, programming",
    "cloud": "cloud computing, aws, azure, gcp, kubernetes, devops, containers",
    "hr": "human resources, recruiting, hiring, talent acquisition, HR analytics",
    "marketing": "marketing, branding, seo, social media, digital advertising",
    "finance": "finance, accounting, investing, trading, financial analysis",
    "product_management": "product management, product owner, roadmap, discovery, user research",
    "design": "ux design, ui design, figma, adobe, user interface, user experience",
    "leadership": "leadership, management, communication for managers, coaching, feedback",
}

domain_names = list(DOMAIN_ANCHORS.keys())
anchor_texts = list(DOMAIN_ANCHORS.values())

# Normalized anchor embeddings (used both for query domain detection
# and for computing similarity between courses and the chosen domain).
anchor_embeddings = model.encode(anchor_texts, normalize_embeddings=True)


# Build a course-level domain text
df["domain_text"] = (
    df["category"]
    + " | "
    + df["subcategory"]
    + " | "
    + df["topic_title"]
    + " | "
    + df["course_mapping_keywords"]
)

course_domain_embeddings = model.encode(
    df["domain_text"].tolist(),
    normalize_embeddings=True,
)


def detect_query_domain(query: str):
    """
    Detects the most likely learning domain for a user query.

    Returns a dict: {"domain_key": str, "domain_label": str, "score": float}
    """
    if not query or not query.strip():
        return {"domain_key": None, "domain_label": None, "score": 0.0}

    query_vec = model.encode([query], normalize_embeddings=True)[0]
    # cosine similarities with each anchor
    sims = anchor_embeddings @ query_vec
    best_idx = int(np.argmax(sims))
    best_key = domain_names[best_idx]
    best_score = float(sims[best_idx])

    pretty_labels = {
        "data_science": "Data Science & Analytics",
        "software_engineering": "Software Engineering",
        "cloud": "Cloud & DevOps",
        "hr": "Human Resources & People",
        "marketing": "Marketing & Growth",
        "finance": "Finance & Accounting",
        "product_management": "Product Management",
        "design": "UX/UI & Design",
        "leadership": "Leadership & Management",
    }

    return {
        "domain_key": best_key,
        "domain_label": pretty_labels.get(best_key, best_key.title()),
        "score": best_score,
    }


# 2. Keyword / TF-IDF style boost preparation

def _tokenize(text: str):
    tokens = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    # tiny, manual stopword list – good enough for this project
    stopwords = {"and", "for", "the", "with", "from", "into", "your", "you"}
    return [t for t in tokens if len(t) > 2 and t not in stopwords]


# Build a search text
df["search_text"] = (
    df["course_title"]
    + " "
    + df["topic_title"]
    + " "
    + df["course_mapping_keywords"]
    + " "
    + df["category"]
    + " "
    + df["subcategory"]
)

search_texts = df["search_text"].tolist()

# Pre-compute token sets for each course
search_tokens_list = [set(_tokenize(t)) for t in search_texts]

# Global token frequencies
word_counter = Counter()
for toks in search_tokens_list:
    word_counter.update(toks)

N_docs = len(search_tokens_list)
idf = {}
for word, freq in word_counter.items():
    # smooth IDF – this is intentionally simple
    idf[word] = np.log(1.0 + N_docs / (1.0 + freq))


# Category tokens
category_tokens_list = [
    set(_tokenize(c + " " + s))
    for c, s in zip(df["category"].tolist(), df["subcategory"].tolist())
]


# 3. Popularity boost

_enroll = df["enrollments"].values.astype(float)
_pop_raw = np.log10(1.0 + _enroll)
# normalize to [0,1]
if _pop_raw.max() > _pop_raw.min():
    popularity_norm = (_pop_raw - _pop_raw.min()) / (_pop_raw.max() - _pop_raw.min())
else:
    popularity_norm = np.zeros_like(_pop_raw)


# Helper: score normalization

def _normalize_scores(x: np.ndarray):
    if len(x) == 0:
        return x
    mn = float(x.min())
    mx = float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype="float32")
    return (x - mn) / (mx - mn)


# 4. Hybrid semantic search

def semantic_hybrid_search(query: str, top_k: int = 20, candidate_k: int = 200):
    """
    Main search function.

    Combines:
      - semantic similarity (FAISS over course_embeddings)
      - domain similarity (query anchor vs course domain_embeddings)
      - keyword / TF-IDF-style boost
      - category/subcategory matching boost
      - popularity boost (enrollments)

    It also applies a simple rejection rule to drop courses that are
    both semantically weak AND keyword-irrelevant.
    """

    query = (query or "").strip()
    if not query:
        return df.head(0).copy()  # empty frame, same schema

    # --- Embedding for query ---
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    # --- 4.1 semantic similarity via FAISS ---
    distances, indices = index.search(query_vec, candidate_k)
    indices = indices[0]
    semantic_scores = distances[0]  # already cosine-like if index built that way
    sem_norm = _normalize_scores(semantic_scores)

    # --- 4.2 domain similarity (anchor + course-topic embeddings) ---
    domain_info = detect_query_domain(query)
    domain_key = domain_info["domain_key"]
    domain_score = domain_info["score"]

    domain_component = np.zeros_like(sem_norm, dtype="float32")

    if domain_key is not None and domain_score > 0.25:  # small threshold
        dom_idx = domain_names.index(domain_key)
        dom_vec = anchor_embeddings[dom_idx]  # (dim,)
        # cosine similarity between course domain embedding and the chosen domain
        domain_sims = course_domain_embeddings[indices] @ dom_vec
        domain_component = _normalize_scores(domain_sims)

    # --- 4.3 keyword / TF-IDF boost ---
    query_tokens = _tokenize(query)
    keyword_scores = []

    for idx in indices:
        course_tokens = search_tokens_list[idx]
        score = 0.0
        for t in query_tokens:
            if t in course_tokens:
                score += idf.get(t, 0.0)
        keyword_scores.append(score)

    keyword_scores = np.array(keyword_scores, dtype="float32")
    keyword_norm = _normalize_scores(keyword_scores)

    # --- 4.4 category/subcategory boost ---
    cat_scores = []
    q_token_set = set(query_tokens)

    for idx in indices:
        cat_tokens = category_tokens_list[idx]
        overlap = q_token_set & cat_tokens
        # 0 if no overlap, 1 if there is at least one shared token
        cat_scores.append(1.0 if overlap else 0.0)

    cat_scores = np.array(cat_scores, dtype="float32")
    cat_norm = _normalize_scores(cat_scores)

    # --- 4.5 popularity boost (already normalized globally) ---
    pop_component = popularity_norm[indices].astype("float32")
    pop_norm = _normalize_scores(pop_component)

    # ---------- Combine scores ----------
    total_score = (
        1.0 * sem_norm
        + 0.25 * domain_component
        + 0.30 * keyword_norm
        + 0.15 * cat_norm
        + 0.10 * pop_norm
    )

    # ---------- Rejection rule ----------
    # Drop courses that are weak both semantically and in keyword overlap.
    mask_keep = ~((sem_norm < 0.35) & (keyword_norm < 0.10))
    if mask_keep.sum() == 0:
        # fallback: keep semantic top_k
        mask_keep = np.ones_like(mask_keep, dtype=bool)

    indices = indices[mask_keep]
    total_score = total_score[mask_keep]

    # ---------- Build result DataFrame ----------
    results = df.iloc[indices].copy()
    results["score"] = total_score

    # Deduplicate by course title to avoid repeated levels, keep best scoring
    results = (
        results.sort_values("score", ascending=False)
        .drop_duplicates(subset=["course_title"])
        .head(top_k)
    )

    return results


# For backward compatibility with the old app
def semantic_search(query: str, top_k: int = 20):
    """
    Thin wrapper around semantic_hybrid_search so the Streamlit app
    can keep calling `semantic_search` if it wants to.
    """
    return semantic_hybrid_search(query=query, top_k=top_k)
