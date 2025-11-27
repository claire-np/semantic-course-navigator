import os
import sys
import math
from typing import List, Dict, Any

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


# Path Setup
CURRENT_DIR = os.path.dirname(__file__)                     # /pages
ROOT_DIR = os.path.dirname(CURRENT_DIR)                     # /learning_path_recommender

# Allow importing modules
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "app"))
sys.path.append(os.path.join(ROOT_DIR, "search_engine"))

from app.utils import detect_query_domain
from search_engine.search_engine_v2 import (
    SemanticSearchEngineV2,
    enrich_courses_with_search_text,
)

# Page config & Styling


st.set_page_config(
    page_title="Semantic Retrieval Engine",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #1f2933 0,
                                        #020617 55%, #000000 100%);
            color: #e5e7eb;
        }
        .page-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .page-sub {
            font-size: 0.95rem;
            color: #9ca3af;
            max-width: 780px;
            margin-bottom: 1.1rem;
        }
        .detected-domain {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            background: rgba(59,130,246,0.18);
            color: #bfdbfe;
            font-size: 0.78rem;
            margin-bottom: 0.5rem;
        }
        .open-btn {
            font-weight: 600;
            color: #93c5fd;
            text-decoration: none;
            font-size: 0.82rem;
        }
        .open-btn:hover {
            text-decoration: underline;
            color: #bfdbfe;
        }
        th {
            text-align: center !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="page-title">Semantic Retrieval Engine</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="page-sub">
        Search the curriculum using a Hybrid Retrieval System combining Vector Similarity,
        BM25 lexical search, domain/skill boosting, and a cross-encoder re-ranker.
    </div>
    """,
    unsafe_allow_html=True,
)



# Engine Loader


@st.cache_resource(show_spinner="Loading retrieval engine...")
def load_search_engine():
    csv_path = os.path.join(ROOT_DIR, "unified_courses_v1.csv")
    emb_path = os.path.join(ROOT_DIR, "course_embeddings.npy")
    index_path = os.path.join(ROOT_DIR, "faiss_index.bin")

    df = pd.read_csv(csv_path)
    df = enrich_courses_with_search_text(df)

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("clairenp/miniLM_finetuned_udemy")


    if os.path.exists(emb_path) and os.path.exists(index_path):
        embeddings = np.load(emb_path)
        index = faiss.read_index(index_path)
    else:
        texts = df["search_text"].tolist()
        embeddings = embed_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        np.save(emb_path, embeddings)
        faiss.write_index(index, index_path)

    return SemanticSearchEngineV2(
        df_courses=df,
        faiss_index=index,
        embedding_model=embed_model,
        course_embeddings=embeddings,
        use_bm25=True,
    )

engine = load_search_engine()



# Session State


if "search_results" not in st.session_state:
    st.session_state["search_results"] = pd.DataFrame()
if "domain_info" not in st.session_state:
    st.session_state["domain_info"] = {}
if "current_page" not in st.session_state:
    st.session_state["current_page"] = 1



# Query Input


def clear_input():
    st.session_state["query"] = ""

query = st.text_input(
    "Enter a skill, topic, or career goal:",
    key="query",
    placeholder="e.g. data analysis, HR analytics, AWS cloud, communication skills...",
)

col1, col2 = st.columns(2)
with col1:
    use_rerank = st.checkbox("Use re-ranking", value=True)
with col2:
    use_expand = st.checkbox("Expand query (synonyms)", value=True)



# Search Handler


def execute_search():
    q = st.session_state["query"]

    if not q.strip():
        st.session_state["search_results"] = pd.DataFrame()
        return

    with st.spinner("Retrieving courses..."):
        domain_info = detect_query_domain(q)

        records = engine.search(
            query=q,
            top_k=200,
            use_query_expansion=use_expand,
            use_llm_expansion=False,
            use_rerank=use_rerank,
            domain_hint=domain_info,
            faiss_k=300,
            bm25_k=200,
        )

    if not records:
        st.session_state["search_results"] = pd.DataFrame()
        st.session_state["domain_info"] = {}
        return

    df = pd.DataFrame(records).drop_duplicates(subset=["course_url"])

    st.session_state["search_results"] = df
    st.session_state["domain_info"] = domain_info
    st.session_state["current_page"] = 1

    clear_input()


st.button("Search", on_click=execute_search)



# Display Results


results = st.session_state["search_results"]

if not results.empty:

    domain_info = st.session_state["domain_info"]

    st.markdown(
        f'<div class="detected-domain">Domain detected: {domain_info["domain_label"]} (confidence {domain_info["score"]:.2f})</div>',
        unsafe_allow_html=True,
    )

    st.write(f"**{len(results)} results found.**")

    # Pagination
    per_page = 10
    total_pages = math.ceil(len(results) / per_page)
    page = st.session_state["current_page"]

    start = (page - 1) * per_page
    end = start + per_page
    page_df = results.iloc[start:end].copy()

    # Build table
    display_df = pd.DataFrame({
        "Udemy Course": page_df["course_title"],
        "Instructor": page_df["instructor_title"],
        "Category": page_df["category"],
        "Subcategory": page_df["subcategory"],
        "Rating": page_df["avg_rating_90d"].astype(float).map(lambda x: f"{x:.2f}"),
        "Enrollment": page_df["enrollments"].astype(int).map("{:,}".format),
        "URL": page_df["course_url"].apply(lambda x: f'<a class="open-btn" target="_blank" href="{x}">Open</a>')
    })

    display_df = display_df.reset_index(drop=True)
    display_df.index = display_df.index + 1  # ensure sequential numbering

    styler = display_df.style.set_properties(
        subset=["Rating", "Enrollment", "URL"],
        **{"text-align": "center"}
    )

    st.write(styler.to_html(escape=False), unsafe_allow_html=True)

    
    # Pagination Buttons (bottom)
    
    colA, colB, colC, colD, colE = st.columns(5)

    with colA:
        if st.button("⏮ First"):
            st.session_state["current_page"] = 1

    with colB:
        if st.button("◀ Prev") and page > 1:
            st.session_state["current_page"] -= 1

    with colC:
        st.write(f"Page **{page}** / **{total_pages}**")

    with colD:
        if st.button("Next ▶") and page < total_pages:
            st.session_state["current_page"] += 1

    with colE:
        if st.button("Last ⏭"):
            st.session_state["current_page"] = total_pages

else:
    st.info("Enter a query and click Search to explore the catalog.")
