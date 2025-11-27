# app/search_engine_v2.py
#

from __future__ import annotations
from typing import List, Dict, Any
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None



# 1. TEXT NORMALIZATION


def _safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_level(raw: Any) -> str:
    if pd.isna(raw):
        return ""
    s = str(raw).strip().lower()
    if any(k in s for k in ["beginner", "introduct", "newbie", "for beginners",
                            "foundation", "fundamental", "beginning"]):
        return "beginner"
    if any(k in s for k in ["advanced", "expert", "pro"]):
        return "advanced"
    return "intermediate" if s else ""


def enrich_courses_with_search_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["normalized_level"] = df["level"].apply(normalize_level) if "level" in df else ""

    if "what_you_will_learn" in df.columns:
        def join_wyl(x):
            if isinstance(x, (list, tuple)):
                return " ; ".join(str(i) for i in x)
            return _safe_str(x)
        df["wyl_joined"] = df["what_you_will_learn"].apply(join_wyl)
    else:
        df["wyl_joined"] = ""

    def build(row: pd.Series):
        parts = [
            _safe_str(row.get("course_title")),
            _safe_str(row.get("headline")),
            _safe_str(row.get("description")),
            _safe_str(row.get("category")),
            _safe_str(row.get("subcategory")),
            _safe_str(row.get("normalized_level")),
            _safe_str(row.get("wyl_joined")),
        ]
        txt = " [SEP] ".join([p for p in parts if p])
        return re.sub(r"\s+", " ", txt).strip()

    if "search_text" not in df.columns:
        df["search_text"] = df.apply(build, axis=1)

    return df



# 2. SIMPLE LEXICAL EXPANSION


_BASIC_SYNONYM_MAP = {
    "data analyst": ["data analytics", "business intelligence", "bi analyst"],
    "data analysis": ["analytics", "data insights"],
    "sql": ["structured query language", "databases"],
    "power bi": ["business intelligence", "dashboards"],
    "tableau": ["dashboard", "data visualization"],
    "excel": ["spreadsheets"],
    "hr analytics": ["people analytics", "workforce analytics"],
    "product management": ["product manager", "pm", "product strategy"],
    "beginner": ["introduction", "for beginners"],
    "advanced": ["expert", "pro"],
}

def simple_lexical_expansion(q: str) -> str:
    base = q.strip()
    q_lower = base.lower()
    expansions = []
    for key, syns in _BASIC_SYNONYM_MAP.items():
        if key in q_lower:
            expansions.extend(syns)
    if expansions:
        exp = " ".join(sorted(set(expansions)))
        return f"{base} {exp}"
    return base



# 3. SKILL TOKENS


_SKILL_VOCAB = [
    "sql", "excel", "power bi", "tableau", "python", "r",
    "statistics", "data analysis", "data analytics", "business intelligence",
    "dashboard", "machine learning",
    "product management", "product manager", "agile", "scrum", "kanban", "jira",
    "aws", "azure", "gcp", "docker", "kubernetes",
    "hr analytics", "people analytics", "recruitment", "talent acquisition", "payroll",
    "communication", "presentation", "public speaking", "storytelling",
]

def _normalize_skill_token(s: str) -> str:
    return s.lower().strip().replace(" ", "_")

def extract_skill_tokens(text: str) -> List[str]:
    if not text:
        return []
    t = text.lower()
    found = []
    for ph in _SKILL_VOCAB:
        if ph in t:
            tok = _normalize_skill_token(ph)
            if len(tok) > 2:  # remove junk tokens
                found.append(tok)
    return sorted(set(found))



# 4. BM25 TOKENIZER


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")

def tokenize_for_bm25(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text or "")]



# 5. SEARCH ENGINE CORE


class SemanticSearchEngineV2:

    def __init__(
        self,
        df_courses: pd.DataFrame,
        faiss_index,
        embedding_model: SentenceTransformer,
        device=None,
        course_embeddings=None,
        ce_batch_size=16,
        use_bm25=True,
    ):
        self.df_courses = df_courses.reset_index(drop=True)
        self.df_courses["course_idx"] = np.arange(len(self.df_courses))
        self.index = faiss_index
        self.embedding_model = embedding_model
        self.course_embeddings = course_embeddings

        # normalizing text
        self.df_courses = enrich_courses_with_search_text(self.df_courses)
        self.df_courses["skill_tokens"] = self.df_courses["search_text"].apply(extract_skill_tokens)

        # device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # BM25
        self.bm25 = None
        if use_bm25 and BM25Okapi is not None:
            toks = [tokenize_for_bm25(x) for x in self.df_courses["search_text"]]
            self.bm25 = BM25Okapi(toks)

        # CE model removed for performance
        self.cross_encoder = None
        self.ce_batch_size = ce_batch_size

    
    # Query encoders
    
    def _encode_query_vec(self, text: str):
        v = self.embedding_model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return v.astype("float32")

    def _faiss_retrieve(self, q_vec, k):
        D, I = self.index.search(q_vec, k)
        out = {int(idx): float(score) for idx, score in zip(I[0], D[0]) if idx >= 0}
        return out

    def _bm25_retrieve(self, query, k):
        if self.bm25 is None:
            return {}
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return {}
        scores = self.bm25.get_scores(tokens)
        tops = np.argsort(-scores)[:k]
        return {int(i): float(scores[i]) for i in tops if scores[i] > 0}

    @staticmethod
    def _skill_overlap(q_skills, c_skills):
        if not q_skills or not c_skills:
            return 0.0
        inter = len(set(q_skills).intersection(c_skills))
        return inter / max(1, len(q_skills))

    @staticmethod
    def _domain_score(domain_hint, cat, subcat, txt):
        if not domain_hint:
            return 0.0
        label = (domain_hint.get("domain_label") or "").lower()
        toks = [t for t in re.findall(r"[a-zA-Z]+", label) if len(t) > 3]
        if not toks:
            return 0.0
        full = f"{cat} {subcat} {txt}".lower()
        hits = sum(1 for t in toks if t in full)
        return hits / max(1, len(toks))

    
    # PUBLIC SEARCH
    
    def search(
        self,
        query: str,
        top_k=20,
        use_query_expansion=True,
        use_llm_expansion=False,
        llm_expand_fn=None,
        use_rerank=True,    # kept for compatibility, ignored
        domain_hint=None,
        faiss_k=300,
        bm25_k=200,
    ):
        q0 = query.strip()
        if not q0:
            return []

        # 1. Expansion
        q_exp = simple_lexical_expansion(q0) if use_query_expansion else q0

        if use_llm_expansion and llm_expand_fn:
            try:
                add = llm_expand_fn(q_exp).strip()
                if add:
                    q_exp = f"{q_exp} {add}"
            except:
                pass

        # 2. Semantic candidates
        q_vec = self._encode_query_vec(q_exp)
        sem_dict = self._faiss_retrieve(q_vec, max(faiss_k, top_k))

        # 3. BM25 candidates
        bm25_dict = self._bm25_retrieve(q_exp, max(bm25_k, top_k))

        # 4. Merge
        cand_idx = sorted(set(sem_dict.keys()) | set(bm25_dict.keys()))
        if not cand_idx:
            return []

        dfc = self.df_courses.iloc[cand_idx].copy()

        # Scores
        dfc["bm25_score_norm"] = (
            np.array([bm25_dict.get(i, 0.0) for i in dfc["course_idx"]])
        )
        max_bm = dfc["bm25_score_norm"].max()
        if max_bm > 0:
            dfc["bm25_score_norm"] /= max_bm

        dfc["semantic_score"] = np.array([sem_dict.get(i, 0.0) for i in dfc["course_idx"]])
        dfc["skill_score"] = [
            self._skill_overlap(extract_skill_tokens(q_exp), sk or [])
            for sk in dfc["skill_tokens"]
        ]
        dfc["domain_score"] = [
            self._domain_score(
                domain_hint,
                _safe_str(row["category"]),
                _safe_str(row["subcategory"]),
                _safe_str(row["search_text"]),
            )
            for _, row in dfc.iterrows()
        ]

        # 5. NO CE rerank — replaced with clean weighted fusion
        final = (
            0.60 * dfc["semantic_score"].values +
            0.20 * dfc["bm25_score_norm"].values +
            0.12 * dfc["skill_score"].values +
            0.08 * dfc["domain_score"].values
        )

        dfc["final_score"] = final
        dfc = dfc.sort_values("final_score", ascending=False).head(top_k)

        return dfc.to_dict(orient="records")
