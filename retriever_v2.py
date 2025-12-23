# retriever.py — Robust retrieval with numeric-aware reranking & filter cleaning
from pathlib import Path
import os, ssl, re
from typing import List, Tuple, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & SSL BYPASS (Dev Only)
# -----------------------------------------------------------------------------
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

ssl._create_default_https_context = ssl._create_unverified_context

# Paths (Adjust defaults if needed, but these match your provided structure)
INDEX_DIR = Path(os.environ.get("INDEX_DIR", r"C:\Users\pnanda\LLM-Insurance-Agent\indexes\policies_faiss"))
EMBED_MODEL_LOCAL = os.environ.get("EMBED_MODEL_LOCAL", r"C:\Users\pnanda\LLM\all-MiniLM-L6-v2")

# HuggingFace Offline Mode settings
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


# -----------------------------------------------------------------------------
# CORE LOADERS
# -----------------------------------------------------------------------------
def get_embedding_model():
    """Load local embedding model (CPU-friendly)."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_LOCAL, model_kwargs={"device": "cpu"})

def load_vectorstore():
    """Safely load the FAISS index."""
    if not INDEX_DIR.exists():
        # Fallback empty or raise, depending on preference. Here we raise to alert setup issues.
        raise FileNotFoundError(f"[INDEX] Directory not found: {INDEX_DIR}")
    
    emb = get_embedding_model()
    # allow_dangerous_deserialization=True is required for local pickle files in newer LangChain
    vs = FAISS.load_local(str(INDEX_DIR), embeddings=emb, allow_dangerous_deserialization=True)
    
    try:
        # Debug info
        ntotal = getattr(vs.index, "ntotal", None)
        doc_count = len(getattr(vs.docstore, "_dict", {}))
        # print(f"[INDEX] Loaded {INDEX_DIR} · ntotal={ntotal}, docs={doc_count}")
    except Exception:
        pass
    return vs


# -----------------------------------------------------------------------------
# FILTER LOGIC (Enhanced for Agent Support)
# -----------------------------------------------------------------------------
def _normalize_filter(fd):
    """
    Cleans up the filter dictionary. 
    Removes keys with values like '(All)', 'None', or empty strings so FAISS doesn't crash.
    """
    if not isinstance(fd, dict): return None
    out = {}
    for k, v in fd.items():
        if v is None: continue
        
        # Convert to string to check for UI placeholders
        v_str = str(v).strip()
        
        # If value is effectively empty or a UI placeholder, skip it
        if not v_str or v_str.lower() in ["(all)", "none", "unknown", ""]:
            continue
        
        # Keep the cleaned value
        out[k] = v
        
    return out or None

def _apply_source_contains(docs: List[Any], substr: Optional[str]):
    """Post-retrieval filtering for specific source URLs/filenames."""
    if not substr or substr.lower() in ["(all)", "none", ""]: 
        return docs
        
    sl = substr.lower()
    out = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        src = md.get("source") or md.get("source_url") or ""
        # Loose match: if substring is in source path
        if sl in (src or "").lower(): 
            out.append(d)
    return out or docs


# -----------------------------------------------------------------------------
# RERANKER LOGIC
# -----------------------------------------------------------------------------
class NumericAwareRetriever:
    """
    Wraps a standard retriever to boost documents that contain numbers/currency
    when the query implies a numeric question (e.g. 'how much', 'limit', 'price').
    """
    NUMERIC_TOKENS = ["percent","percentage","%","sum insured","limit","amount","benefit",
                      "deductible","copay","co-pay","co pay","₹","rs.","room rent"]
    
    CONTEXT_TOKENS = ["silver","gold","platinum","early","minor","major","advanced","cancer",
                      "option","stage","room","private","icu","per day"]

    def __init__(self, base_retriever, k: int = 8, source_contains: Optional[str] = None):
        self.base = base_retriever
        self.k = k
        self.source_contains = source_contains

    def invoke(self, query: str):
        docs = self._safe_call_base(query)
        return self._rerank_if_needed(query, docs)

    # LangChain compatibility alias
    def get_relevant_documents(self, query: str):
        return self.invoke(query)

    def _safe_call_base(self, query: str):
        if hasattr(self.base, "get_relevant_documents"):
            docs = self.base.get_relevant_documents(query)
        elif hasattr(self.base, "invoke"):
            docs = self.base.invoke(query)
        else:
            docs = []
        
        # Apply strict source filtering after base retrieval
        docs = _apply_source_contains(docs, self.source_contains)
        return docs

    def _looks_numeric_query(self, q: str) -> bool:
        ql = q.lower()
        return any(tok in ql for tok in self.NUMERIC_TOKENS)

    def _numeric_score(self, text: str) -> int:
        """Heuristic score: more numbers/money symbols = higher relevance for quantitative questions."""
        pct = len(re.findall(r"\b\d{1,3}\s*%\b", text))
        nums = len(re.findall(r"\b\d[\d,.]*\b", text))
        money = len(re.findall(r"(₹|rs\.?|inr)\s*\d", text.lower()))
        return (pct * 3) + (money * 2) + nums

    def _context_boost(self, query: str, text: str) -> int:
        """Boost if context keywords (e.g., 'Gold Plan') appear in both query and text."""
        ql = query.lower(); tl = text.lower()
        return sum(tok in tl for tok in self.CONTEXT_TOKENS if tok in ql)

    def _rerank_if_needed(self, query: str, docs: List[Any]):
        # If query isn't numeric, just return original order (usually vector similarity)
        if not docs or not self._looks_numeric_query(query): 
            return docs
            
        scored: List[Tuple[int, Any]] = []
        for d in docs:
            txt = getattr(d, "page_content", "") or ""
            # Score = Heuristics + Context Match
            s = self._numeric_score(txt) + self._context_boost(query, txt)
            scored.append((s, d))
        
        # If no heuristic signal found, return original docs
        if not any(s > 0 for s, _ in scored): 
            return docs
            
        # Sort high score to low
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:self.k]]


# -----------------------------------------------------------------------------
# FACTORY FUNCTION (API)
# -----------------------------------------------------------------------------
def get_retriever(
    k: int = 12,
    search_type: str = "mmr",
    filter_dict: dict | None = None,
    score_threshold: float | None = None,
    numeric_boost: bool = True,
    fetch_k: int | None = None,
    lambda_mult: float = 0.25,
    source_contains: Optional[str] = None
):
    """
    Main entry point to get a configured retriever.
    
    Args:
        filter_dict: Dictionary for metadata filtering (e.g. {'company': 'HDFC'}).
                     Automatically cleans '(All)' values.
        source_contains: String to strict-filter sources (post-retrieval).
    """
    vs = load_vectorstore()
    kwargs = {"k": k}
    
    # Clean the filter dict (removes "(All)", "None", etc.)
    filt = _normalize_filter(filter_dict)
    if filt: 
        kwargs["filter"] = filt
    
    st = search_type
    if st == "mmr":
        if fetch_k is None: fetch_k = max(k * 4, 40)
        kwargs.update({"fetch_k": fetch_k, "lambda_mult": float(lambda_mult)})
    elif st == "similarity_score_threshold":
        kwargs["score_threshold"] = float(score_threshold or 0.35)
    else:
        st = "similarity"
        
    base = vs.as_retriever(search_type=st, search_kwargs=kwargs)
    
    if numeric_boost:
        return NumericAwareRetriever(base, k=k, source_contains=source_contains)
    
    return base

# Helper for other modules to check intent
def numeric_intent(q: str) -> bool:
    return any(t in q.lower() for t in NumericAwareRetriever.NUMERIC_TOKENS)
