# app/Home.py

import streamlit as st


# PAGE CONFIG

st.set_page_config(
    page_title="Udemy Business Catalog Search",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM CSS

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #1f2933 0%, #020617 55%, #000000 100%);
            color: #e5e7eb;
            font-family: "Inter", system-ui, sans-serif;
        }

        .home-title {
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            letter-spacing: -0.01em;
        }

        .home-subtitle {
            font-size: 1.05rem;
            color: #9ca3af;
            line-height: 1.55;
            max-width: 900px;
            margin-bottom: 1.4rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 1.8rem 0 0.6rem 0;
        }

        .feature-card {
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.28);
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.65);
            height: 100%;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .feature-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        .feature-tag {
            display: inline-block;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #93a3b8;
            margin-bottom: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# HEADER

st.markdown(
    '<div class="home-title">Udemy Business Catalog Search</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="home-subtitle">
        This project is built by <strong>Linh Phuong Nguyen (Claire Nguyen)</strong>.  
        Thank you for visiting.  
        <br/><br/>
        It grew out of a simple question: <strong>What does the hidden architecture of an online learning platform actually look like?</strong>  
        The goal was to peel back those layers and surface the data science core — the NLP pipeline that reshapes unstructured text,  
        the vector search engine that drives discovery, and the ranking logic that determines relevance.  
        This application represents the first stage of that exploration.
    </div>
    """,
    unsafe_allow_html=True,
)


# PROJECT GOALS

st.markdown('<div class="section-title">🎯 Project Goals</div>', unsafe_allow_html=True)

st.markdown(
    """
    - Implement an end-to-end NLP pipeline to clean, normalize, and extract structured features from thousands of unstructured course descriptions.  
    - Design a hybrid retrieval system (semantic search + keyword fallback) powered by SBERT/GTE embeddings for natural-language, relevance-aware search.  
    - Build a rule-based Learning Path Generator that groups skills, calibrates difficulty, and organizes courses from foundational to advanced.  
    """
)


# TECH STACK

st.markdown('<div class="section-title">🛠️ Technical Stack</div>', unsafe_allow_html=True)

st.markdown(
    """
    This system is implemented in Python with a focus on efficient search and clean, interpretable logic.  
    **Streamlit**, **SBERT/GTE embeddings**, **FAISS**, and **Pandas** form the backbone of the experience.
    """
)

# FEATURE SECTION
st.markdown('<div class="section-title">Explore the Catalog Search</div>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="feature-tag">Page 1</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Semantic Course Search</div>', unsafe_allow_html=True)
        st.markdown(
            """
            A hybrid retrieval engine combining:
            - **SBERT/GTE embeddings**
            - **Lexical/semantic expansion**
            - **FAISS vector search**
            
            """
        )

    with col2:
        st.markdown('<div class="feature-tag">Page 2</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Career Learning Paths</div>', unsafe_allow_html=True)
        st.markdown(
            """
            Generates structured, four-stage learning paths for roles such as **Data Analyst**, **BI Analyst**, **Data Scientist**, **DevOps Engineer**, and more.
            
            Organized into four stages:
            
            - **Foundations -> Core Skills -> Specialization -> Portfolio**
            """ 
        )

# FOOTNOTE

st.markdown(
    """
    <br/><div style='color:#9ca3af; font-size:0.85rem;'>
        Use the sidebar to navigate between features.
    </div>
    """,
    unsafe_allow_html=True,
)
