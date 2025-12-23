
# retriever.py (supports filters: type, company, policy_name; numeric-aware reranker)
from pathlib import Path
import os, ssl, re
from typing import List, Tuple, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
# Optional: silence urllib3 warnings if any HTTPS client is used indirectly
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass


# ====== PORTABLE PATH CONFIGURATION ======
# 1. Dynamically find the project root (where this file is located)
CURRENT_DIR = Path(__file__).resolve().parent

# 2. Define the index directory relative to the project root
# This ensures it works on your machine (User 91720) automatically
INDEX_DIR = Path(os.environ.get("INDEX_DIR", CURRENT_DIR / "indexes" / "policies_faiss"))

# 3. Use the Model Name directly
# This forces the library to download the model if missing, instead of looking for a hardcoded folder
EMBED_MODEL_LOCAL = "all-MiniLM-L6-v2"

# HuggingFace offline hints (safe to keep)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
ssl._create_default_https_context = ssl._create_unverified_context

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_LOCAL, model_kwargs={"device": "cpu"})

def load_vectorstore():
    if not INDEX_DIR.exists(): raise FileNotFoundError(f"[INDEX] Directory not found: {INDEX_DIR}")
    emb = get_embedding_model()
    vs = FAISS.load_local(str(INDEX_DIR), embeddings=emb, allow_dangerous_deserialization=True)
    try:
        ntotal = getattr(vs.index, "ntotal", None)
        doc_count = len(getattr(vs.docstore, "_dict", {}))
        print(f"[INDEX] Loaded {INDEX_DIR} · ntotal={ntotal}, docs={doc_count}")
    except Exception:
        pass
    return vs

def _normalize_filter(fd):
    if not isinstance(fd, dict): return None
    out = {}
    for k, v in fd.items():
        if v is None: continue
        if isinstance(v, str):
            v2 = v.strip()
            if not v2: continue
            out[k] = v2
        else:
            out[k] = v
    return out or None

def _apply_source_contains(docs: List[Any], substr: Optional[str]):
    if not substr: return docs
    sl = substr.lower()
    out = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        src = md.get("source") or md.get("source_url") or ""
        if sl in (src or "").lower(): out.append(d)
    return out or docs

class NumericAwareRetriever:
    NUMERIC_TOKENS = ["percent","percentage","%","sum insured","limit","amount","benefit",
                      "deductible","copay","co-pay","co pay","₹","rs.","room rent"]
    CONTEXT_TOKENS = ["silver","gold","platinum","early","minor","major","advanced","cancer",
                      "option","stage","room","private","icu","per day"]

    def __init__(self, base_retriever, k: int = 8, source_contains: Optional[str] = None):
        self.base = base_retriever; self.k = k; self.source_contains = source_contains

    def invoke(self, query: str):
        docs = self._safe_call_base(query)
        return self._rerank_if_needed(query, docs)

    def get_relevant_documents(self, query: str):
        docs = self._safe_call_base(query)
        return self._rerank_if_needed(query, docs)

    def __call__(self, query: str): return self.invoke(query)

    def _safe_call_base(self, query: str):
        if hasattr(self.base, "get_relevant_documents"):
            docs = self.base.get_relevant_documents(query)
        elif hasattr(self.base, "invoke"):
            docs = self.base.invoke(query)
        else:
            docs = []
        docs = _apply_source_contains(docs, self.source_contains)
        return docs

    def _looks_numeric_query(self, q: str) -> bool:
        ql = q.lower(); return any(tok in ql for tok in self.NUMERIC_TOKENS)

    def _numeric_score(self, text: str) -> int:
        pct = len(re.findall(r"\b\d{1,3}\s*%\b", text))
        nums = len(re.findall(r"\b\d[\d,.]*\b", text))
        money = len(re.findall(r"(₹|rs\.?|inr)\s*\d", text.lower()))
        return pct * 3 + money * 2 + nums

    def _context_boost(self, query: str, text: str) -> int:
        ql = query.lower(); tl = text.lower()
        return sum(tok in tl for tok in self.CONTEXT_TOKENS if tok in ql)

    def _rerank_if_needed(self, query: str, docs: List[Any]):
        if not docs or not self._looks_numeric_query(query): return docs
        scored: List[Tuple[int, Any]] = []
        for d in docs:
            txt = getattr(d, "page_content", "") or ""
            s = self._numeric_score(txt) + self._context_boost(query, txt)
            scored.append((s, d))
        if not any(s > 0 for s, _ in scored): return docs
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:self.k]]

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
    vs = load_vectorstore()
    kwargs = {"k": k}
    filt = _normalize_filter(filter_dict)
    if filt: kwargs["filter"] = filt
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
