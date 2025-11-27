from prefect import flow, task
import subprocess
import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# HELPERS
# -----------------------------

OUTPUT_DIR = "data"
MODEL_NAME = "clairenp/miniLM_finetuned_udemy"
DUCKDB_PATH = "learning_dbt/warehouse.duckdb"


@task
def run_dbt():
    subprocess.run(["dbt", "build"], check=True, cwd="learning_dbt")


@task
def load_transformed_data():
    import duckdb
    con = duckdb.connect(DUCKDB_PATH)
    df = con.execute("SELECT * FROM main.dim_unified_courses").df()
    return df


@task
def generate_embeddings(df):
    # Ensure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        df["course_title"].tolist(),
        show_progress_bar=True
    )

    np.save(f"{OUTPUT_DIR}/course_embeddings_prefect.npy", embeddings)
    return embeddings


@task
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, f"{OUTPUT_DIR}/faiss_index_prefect.bin")


@flow
def full_pipeline():
    run_dbt()
    df = load_transformed_data()
    embeddings = generate_embeddings(df)
    build_faiss_index(embeddings)


if __name__ == "__main__":
    full_pipeline()
