# 🚀 Semantic Course Navigator  
### End-to-End Semantic Retrieval, Learning Path Generation & Modern Data Stack Integration

This project transforms a **raw Udemy Business marketplace dump** into a **robust, multi-layer retrieval system**, with a complete pipeline from:

`Ingestion → Transformation → ML Embeddings → FAISS Indexing → Semantic Retrieval → Learning Path Engine → Streamlit Application`

The system unifies **NLP embeddings**, **hybrid information retrieval**, **skill-based curriculum generation**, and a modern **Data Engineering + Orchestration + Search Architecture** behind a multi-page UI.

---

## 📸 Demo Screenshots

### **Semantic Search Interface**

### **Learning Path Generator**

---

## 🏗️ System Architecture Diagram


---

# 1. System Architecture Overview

A production-oriented, pipeline-first design consistent with modern DS/ML systems.

---

## **1 — Data Transformation (dbt Project + Local Warehouse)**  
`learning_dbt/`

- Raw input: `raw_marketplace.csv`
- Transformations:
  - Schema normalization & typing
  - Text cleaning & NA handling
  - Category/topic normalization
  - Derived fields (`search_text`, `domain_text`)
- Output:
  - `unified_courses_v1.csv` — **analytics-ready model stored in a local warehouse**

Implements the **Medallion Architecture (Bronze → Silver → Gold)** for course data.

---

## **2 — Orchestration (Prefect Pipelines)**  
`orchestration/prefect/`

Automates the entire lifecycle:

- dbt run → validate → export
- Batch embedding inference
- FAISS index build & persistence
- Scheduled refreshes & observability

Ensures full reproducibility and identity of pipelines.

---

## **3 — ML / Search Engine Layer**  
`model/` + `search_engine/`

### Components
- MiniLM SBERT encoder (`all-MiniLM-L6-v2`)
- Fine-tuned model for Udemy domain
- FAISS `IndexFlatIP` (cosine similarity)
- IDF-weighted lexical score
- Domain anchors to boost relevance (Data, Python, Cloud, PM, Finance)
- Popularity and metadata normalization

### Artifacts
`unified_courses_v1.csv`
`course_embeddings.npy`
`faiss_index.bin`

Retrieval latency: **< 10 ms / query** on CPU.

---

## **4 — Learning Path Engine**  
`app/utils_learning_path.py`

Generates 4-stage role-based upskilling paths:

- Foundations  
- Core Skills  
- Specialization  
- Portfolio & Certification  

Mechanisms:

- Role ontology (DA, BI, DS, DE, PM, DevOps, AI Engineer…)
- Hybrid search for each skill
- Select top-5 courses per skill
- Output: structured JSON-like dict → rendered in UI

---

## **5 — Streamlit Application Layer**  
`app/` + `pages/`

Two major modules:

### **1. Semantic Search Explorer**
- Fast vector search + hybrid ranking
- Domain detection + scoring breakdown
- Modern UI & CSS styling

### **2. Personalized Learning Path Generator**
- Auto-curated upskilling paths
- Expandable skill blocks
- Instructor, rating, enrollment metadata

Includes:

- Cached artifact loading (`st.cache_resource`)
- Defensive fallbacks for empty searches
- Modular, maintainable UI

---

# 2. Repository Structure

....


---

# 3. Technical Highlights

### **NLP / ML**
- SBERT MiniLM encoder  
- Domain-specific fine-tuning  
- Batch inference pipeline  
- Cosine similarity search  
- Hybrid scoring function

### **Information Retrieval**
- FAISS vector index  
- IDF lexical weighting  
- Domain anchor boosting  
- Category/subcategory signal  
- Popularity normalization  

### **Data Engineering**
- dbt transformations  
- Local warehouse modeling (DuckDB-like workflow)  
- Prefect orchestration  
- Reproducible, versioned artifacts  

### **Learning Path Generation**
- 4-phase roadmap logic  
- Skill → top-5 course retrieval  
- Structured curriculum output  

### **Application Layer**
- Streamlit multi-page design  
- Cached artifact loading  
- Scalable, modular utilities  

---

# 4. Quickstart

....


