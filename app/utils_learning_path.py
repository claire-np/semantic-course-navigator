import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# Path Setup
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "app"))

df = pd.read_csv("unified_courses_v1.csv")
embeddings = np.load("course_embeddings.npy")
index = faiss.read_index("faiss_index.bin")

model = SentenceTransformer("model/miniLM_finetuned_udemy")


# Role definitions and skill stages


ROADMAP = {
    "data_analyst": {
        "Stage 1 – Foundations": [
            "Excel foundations",
            "SQL fundamentals",
            "Basic statistics",
            "Intro to data visualization",
        ],
        "Stage 2 – Core Skills": [
            "Power BI essentials",
            "Data cleaning & wrangling",
            "Exploratory data analysis",
            "Dashboard building",
        ],
        "Stage 3 – Specialization": [
            "Advanced SQL",
            "Advanced visualization",
            "Business analytics",
            "Automation for analysts",
        ],
        "Stage 4 – Portfolio & Certification": [
            "Build capstone dashboard",
            "Real datasets analytics",
            "DA certification prep",
        ],
    },

    "bi_analyst": {
        "Stage 1 – Foundations": [
            "Excel analytics",
            "SQL basics",
            "Data modeling concepts",
        ],
        "Stage 2 – Core Skills": [
            "Power BI fundamentals",
            "DAX for analytics",
            "Star schema modeling",
        ],
        "Stage 3 – Specialization": [
            "Enterprise BI modeling",
            "Governance & KPI frameworks",
            "BI automation",
        ],
        "Stage 4 – Portfolio & Certification": [
            "BI capstone",
            "Power BI certification prep",
        ],
    },

    "data_engineer": {
        "Stage 1 – Foundations": [
            "Python foundations",
            "SQL for data engineering",
            "Data modeling",
        ],
        "Stage 2 – Core Skills": [
            "ETL pipelines",
            "Apache Airflow",
            "Spark foundations",
        ],
        "Stage 3 – Specialization": [
            "Cloud data engineering (AWS/GCP/Azure)",
            "Distributed systems",
            "Data lakes & warehouses",
        ],
        "Stage 4 – Portfolio & Certification": [
            "DE capstone project",
            "Cloud certifications",
        ],
    },

    "data_scientist": {
        "Stage 1 – Foundations": [
            "Python for DS",
            "Statistics for ML",
            "Data preprocessing",
        ],
        "Stage 2 – Core Skills": [
            "Supervised ML",
            "Unsupervised ML",
            "Model evaluation",
        ],
        "Stage 3 – Specialization": [
            "Deep learning",
            "NLP",
            "MLOps",
        ],
        "Stage 4 – Portfolio & Certification": [
            "DS capstone",
            "Model deployment",
            "ML certifications",
        ],
    },

    "ai_engineer": {
        "Stage 1 – Foundations": [
            "Python foundations",
            "Math for AI",
            "Machine learning essentials",
        ],
        "Stage 2 – Core Skills": [
            "Neural networks",
            "Computer vision",
            "NLP",
        ],
        "Stage 3 – Specialization": [
            "LLMs & Transformers",
            "Generative AI",
            "AI deployment",
        ],
        "Stage 4 – Portfolio & Certification": [
            "AI capstone",
            "Cloud AI certifications",
        ],
    },

    "devops": {
        "Stage 1 – Foundations": [
            "Linux fundamentals",
            "Cloud basics",
            "Networking basics",
        ],
        "Stage 2 – Core Skills": [
            "CI/CD",
            "Docker & Kubernetes",
            "Terraform",
        ],
        "Stage 3 – Specialization": [
            "Monitoring & Observability",
            "Scaling infrastructure",
            "DevSecOps",
        ],
        "Stage 4 – Portfolio & Certification": [
            "DevOps capstone",
            "DevOps certifications",
        ],
    },

    "full_stack": {
        "Stage 1 – Foundations": [
            "HTML CSS JS",
            "Python or JS backend basics",
            "Git & APIs",
        ],
        "Stage 2 – Core Skills": [
            "React fundamentals",
            "Node.js backend",
            "Databases & ORMs",
        ],
        "Stage 3 – Specialization": [
            "Advanced React",
            "Cloud deployment",
            "Architecture patterns",
        ],
        "Stage 4 – Portfolio & Certification": [
            "Full-stack project",
            "Cloud certification",
        ],
    },

    "cybersecurity": {
        "Stage 1 – Foundations": [
            "Networking fundamentals",
            "Operating systems",
            "Security fundamentals",
        ],
        "Stage 2 – Core Skills": [
            "Threat analysis",
            "Security operations",
            "Identity access management",
        ],
        "Stage 3 – Specialization": [
            "Cloud security",
            "Penetration testing",
            "Incident response",
        ],
        "Stage 4 – Portfolio & Certification": [
            "Cyber lab projects",
            "Security certifications",
        ],
    },

    "software_architect": {
        "Stage 1 – Foundations": [
            "Programming fundamentals",
            "OOP & clean code",
            "Databases",
        ],
        "Stage 2 – Core Skills": [
            "Design patterns",
            "System design",
            "Cloud architecture",
        ],
        "Stage 3 – Specialization": [
            "Microservices",
            "Scalable systems",
            "Distributed architecture",
        ],
        "Stage 4 – Portfolio & Certification": [
            "Architecture capstone",
            "Cloud architect certifications",
        ],
    },
}



# Hybrid retrieval: semantic vector search + keyword weighting


def hybrid_search(query: str, top_n: int = 40):
    """Run semantic search through FAISS, then apply keyword and engagement signals."""

    # Semantic similarity
    q_vec = model.encode([query])
    distances, idx = index.search(q_vec, top_n)

    results = df.iloc[idx[0]].copy()
    results["score_semantic"] = distances[0]

    # Simple keyword match score
    q_lower = query.lower()
    results["score_keyword"] = results["course_title"].str.lower().apply(
        lambda title: sum(w in title for w in q_lower.split())
    )

    # Engagement (rating + enrollments)
    results["score_engagement"] = (
        results["avg_rating_90d"] * 0.6
        + np.log1p(results["enrollments"]) * 0.4
    )

    # Weighted final score
    results["hybrid_score"] = (
        results["score_semantic"] * 0.50
        + results["score_keyword"] * 0.15
        + results["score_engagement"] * 0.35
    )

    results = results.sort_values("hybrid_score", ascending=False)
    results = results.drop_duplicates(subset="course_title")

    return results.head(top_n)



# Retrieve top-5 courses per skill


def get_top_courses_for_skill(skill: str):
    """Return top courses for a given skill label."""
    courses = hybrid_search(skill, top_n=40)
    selected = courses.head(5)

    return selected[
        [
            "courseid",
            "course_title",
            "instructor_title",
            "category",
            "subcategory",
            "avg_rating_90d",
            "enrollments",
            "course_url",
        ]
    ].to_dict(orient="records")



# Build the final 4-stage learning path


def generate_learning_path(role_name: str):
    """Construct structured learning path for a given role."""

    role = role_name.lower().replace(" ", "_")
    if role not in ROADMAP:
        return {"error": "Role not found"}

    stages = ROADMAP[role]
    output = {}

    for stage, skills in stages.items():
        blocks = []
        for skill in skills:
            blocks.append({
                "skill": skill,
                "courses": get_top_courses_for_skill(skill),
            })
        output[stage] = blocks

    return output
