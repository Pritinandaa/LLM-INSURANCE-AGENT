
# app.py — Insurance Policy Q&A (Vertex Gemini, robust snippets + fallbacks)
import time, re, numpy as np
from typing import List, Dict, Tuple
import streamlit as st

# ========== SSL relaxed for DEV ==========
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# (Optional) suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Disable urllib3 insecure warnings since verify=False is intentional in dev
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# Local modules
from retriever import get_retriever, get_embedding_model, load_vectorstore
from llm_client import LLMClient, safe_json_loads

# === SSL verification disabled globally (DEV ONLY) ===
import os
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="Insurance Policy Q&A", layout="wide", initial_sidebar_state="expanded")

# Modern Enterprise UI Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
        font-weight: bold;
    }
    
    * {
        font-weight: bold !important;
    }
    
    .main {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1rem;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .hero-section {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(30, 60, 114, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="%23ffffff" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.1;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #ffffff, #e3f2fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 700;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    .filter-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .filter-title {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 800;
        margin: 0 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 800;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 2px solid transparent;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stSelectbox > div > div {
        background: white;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    
    .stTextInput > div > div > input {
        background: white;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .stTextArea > div > div > textarea {
        background: white;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .action-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .action-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.2);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .action-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
    }
    
    .mode-indicator {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .success-message {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 1rem;
        color: #15803d;
        font-weight: 500;
    }
    
    .info-message {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 1rem;
        color: #1d4ed8;
        font-weight: 500;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem;
        color: #92400e;
        font-weight: 500;
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    }
    
    .stExpander {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        transition: all 0.2s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <h1 class="hero-title">🏥 InsuranceAI Pro</h1>
        <p class="hero-subtitle">Advanced AI-powered insurance policy analysis and intelligent recommendations</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Small utility: safe string
# =========================
def to_str(val) -> str:
    """Return a safe, trimmed string for any input (never None)."""
    return str(val or "").strip()

# =========================
# Read index metadata
# =========================
@st.cache_data(show_spinner=False)
def get_catalog() -> Dict[str, Dict[str, set]]:
    """Return dict: {type: {company: set(policy_names)}} from the index."""
    vs = load_vectorstore()
    catalog: Dict[str, Dict[str, set]] = {}
    total_docs = 0
    
    # Keywords to infer insurance type from policy names
    type_keywords = {
        "Health": ["health", "medical", "medicare", "optima", "arogya", "hospital", "critical", "wellness"],
        "Motor": ["motor", "car", "vehicle", "auto", "two wheeler", "private car", "commercial vehicle"],
        "Travel": ["travel", "trip", "yatra"],
        "Business": ["business", "commercial", "liability", "professional", "trade", "suraksha", "udyam"]
    }
    
    for d in getattr(vs.docstore, "_dict", {}).values():
        total_docs += 1
        md = getattr(d, "metadata", {}) or {}
        typ = md.get("type") or "Unknown"
        comp = md.get("company") or "Unknown"
        pol = md.get("policy_name") or "Unknown"
        
        # If type is Unknown, try to infer from policy name
        if typ == "Unknown" and pol != "Unknown":
            pol_lower = pol.lower()
            for inferred_type, keywords in type_keywords.items():
                if any(keyword in pol_lower for keyword in keywords):
                    typ = inferred_type
                    break
        
        catalog.setdefault(typ, {}).setdefault(comp, set()).add(pol)
    
    # Debug info
    print(f"DEBUG: Total documents in index: {total_docs}")
    print(f"DEBUG: Catalog structure: {dict(catalog)}")
    return catalog

catalog = get_catalog()
types_list = ["(All)"] + sorted(t for t in catalog.keys())
companies_all = sorted({c for bucket in catalog.values() for c in bucket.keys()})
policies_all = sorted({p for bucket in catalog.values() for comp in bucket.values() for p in comp})

# Filter Section
st.markdown("""
<div class="filter-card">
    <h3 class="filter-title">🔧 Configure Your Search Parameters</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    selected_type = st.selectbox(
        "🏥 Insurance Type",
        types_list,
        index=0,
        key="filter_insurance_type",
    )

with col2:
    if selected_type == "(All)":
        company_options = ["(All)"] + companies_all
    else:
        company_options = ["(All)"] + sorted(catalog.get(selected_type, {}).keys())
    
    selected_company = st.selectbox(
        "🏢 Company",
        company_options,
        index=0,
        key="filter_company",
    )

with col3:
    if selected_type == "(All)" or selected_company == "(All)":
        policy_options = ["(All)"] + policies_all
    else:
        policy_options = ["(All)"] + sorted(catalog.get(selected_type, {}).get(selected_company, set()))
    
    selected_policy = st.selectbox(
        "📋 Policy (optional)",
        policy_options,
        index=0,
        key="filter_policy",
    )

# Hidden default settings
search_type = "similarity"
top_k = 8
use_llm_for_query = True
use_llm_for_rerank = False
use_llm_for_answer = True
diagnostics = False
llm_model = "gemini-2.5-flash"
llm_temp = 0.1

st.divider()

# =========================
# Regex helpers
# =========================
# Simple sentence split: split after ., !, ? followed by whitespace
SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
PERCENT_RE = re.compile(r"\b\d{1,3}\s?%")
CURRENCY_RE = re.compile(r"(₹|Rs\.?|INR)\s?[\d,]+")
NUMBER_RE = re.compile(r"\b[\d,]+\b")

STOPWORDS = {
    "the","a","an","and","or","for","of","on","in","to","is","are","was","were","be","been",
    "by","with","as","at","from","that","this","those","these","it","its","into","per",
    "what","which","who","whom","whose","when","where","why","how"
}

def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in SPLIT_RE.split(text) if s.strip()]
    if len(sents) < 2:
        sents = [s.strip() for s in re.split(r'[\r\n]+', text) if s.strip()]
    return [s for s in sents if len(s) >= 6]

def tokenize_q(text: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z0-9%]+", (text or "").lower())
    return [t for t in toks if t not in STOPWORDS and len(t) >= 2]

def extract_target_plan_tokens(question: str) -> List[str]:
    q = (question or "").lower()
    toks = []
    m = re.search(r'(\d+)\s*(lac|lakh|lacs|lakhs)', q)
    if m:
        n = m.group(1)
        toks += [f"{n} lac", f"{n} lacs", f"{n} lakh", f"{n} lakhs", f"{n},00,000"]
    m2 = re.search(r'₹?\s*(\d+)\s*(lac|lakh|lacs|lakhs)', q)
    if m2:
        n = m2.group(1)
        toks += [f"{n} lac", f"{n} lacs", f"{n} lakh", f"{n} lakhs", f"{n},00,000"]
    return list(dict.fromkeys([t.strip() for t in toks if t.strip()]))

def numeric_intent(q: str) -> bool:
    return "%" in (q or "") or bool(re.search(r"\d", q or ""))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def has_number(s: str) -> bool:
    return bool(PERCENT_RE.search(s) or CURRENCY_RE.search(s) or NUMBER_RE.search(s))

def extract_values(sentence: str) -> Dict[str, List[str]]:
    return {
        "percents": [m.group(0) for m in PERCENT_RE.finditer(sentence)],
        "amounts": [m.group(0) for m in CURRENCY_RE.finditer(sentence)],
        "numbers": [m.group(0) for m in NUMBER_RE.finditer(sentence)],
    }

# =========================
# Scoring for Q&A
# =========================
def score_sentences(question: str, docs, embedder, diagnostics_on: bool, top_n: int = 10) -> Dict:
    q_vec = np.array(embedder.embed_query(question))
    tokens_q = tokenize_q(question)
    expects_number = numeric_intent(question)
    ql = (question or "").lower()
    target_plans = extract_target_plan_tokens(question)
    policy_tokens = []
    # Extract specific policy names from question
    policy_names = ["optima restore", "optima cash", "optima secure", "optima vital", "optima plus", "energy", "business suraksha plus"]
    for tok in policy_names:
        if tok in ql: policy_tokens.append(tok)
    
    # Special handling for Optima variants
    if "optima restore" in ql:
        policy_tokens = ["optima restore"]  # Only match Optima Restore

    candidates = []
    for d in docs:
        md = d.metadata or {}
        page = md.get("page", "N/A")
        source = md.get("source") or "N/A"
        policy_name = (md.get("policy_name") or "").lower()
        sents = split_sentences(d.page_content)
        if not sents: continue
        sent_vecs = np.array(embedder.embed_documents(sents))
        for idx, (sent, vec) in enumerate(zip(sents, sent_vecs)):
            emb = cosine_sim(q_vec, vec)
            set_q, set_s = set(tokens_q), set(tokenize_q(sent))
            lex = len(set_q & set_s) / max(1, len(set_q | set_s))
            prox = 0.0
            if expects_number:
                s = sent.lower()
                num_spans = [m.span() for m in CURRENCY_RE.finditer(s)] + \
                            [m.span() for m in PERCENT_RE.finditer(s)] + \
                            [m.span() for m in NUMBER_RE.finditer(s)]
                if num_spans:
                    tok_spans = []
                    for t in set(tokens_q):
                        for m in re.finditer(re.escape(t), s):
                            tok_spans.append(m.span())
                    if tok_spans:
                        def dist(a, b):
                            if a[1] < b[0]: return b[0] - a[1]
                            if b[1] < a[0]: return a[0] - b[1]
                            return 0
                        min_d = min(dist(ns, ts) for ns in num_spans for ts in tok_spans)
                        prox = 1.0 / (1.0 + min_d)
            keyword_boost = 0.0
            s_lower = sent.lower()
            for kw in ["preventive health check","preventive health check-up","health check-up","health check up"]:
                if kw in ql and kw in s_lower:
                    keyword_boost = 0.05; break
            final = 0.65*emb + 0.25*lex + 0.10*prox + keyword_boost

            # plan-aware bonus/penalty
            plan_bonus = 0.0
            neighborhood = " ".join(sents[max(0, idx-2):min(len(sents), idx+3)]).lower()
            if target_plans:
                if any(tp in neighborhood for tp in target_plans):
                    plan_bonus = 0.12
            else:
                wrongs = ["3 lac","3 lacs","5 lac","5 lacs","15 lac","15 lacs","20 lac","25 lac","50 lac","100 lac",
                          "20 lacs","25 lacs","50 lacs","100 lacs"]
                if any(w in neighborhood for w in wrongs):
                    final *= 0.92
            final += plan_bonus

            # policy-aware bonus
            policy_bonus = 0.0
            if policy_tokens:
                # Strong bonus for exact policy name match in metadata
                for pt in policy_tokens:
                    if pt in policy_name.lower():
                        policy_bonus = 0.4  # Very strong bonus for correct policy
                        break
                    elif pt in s_lower:
                        policy_bonus = 0.15  # Moderate bonus for mention in text
            final += policy_bonus
            
            # Strong penalty for exclusion/negative sentences
            exclusion_words = ["will not be available", "not available", "excluded", "not covered", "does not cover"]
            if any(excl in s_lower for excl in exclusion_words):
                final *= 0.3  # Heavy penalty for exclusions
            
            # Boost for positive benefit descriptions
            benefit_words = ["benefit", "cover", "available", "provided", "includes"]
            if any(ben in s_lower for ben in benefit_words) and not any(excl in s_lower for excl in exclusion_words):
                final += 0.1

            # gates
            if len(set_q & set(tokenize_q(sent))) < 2: final *= 0.6
            if expects_number and not has_number(sent): final *= 0.85

            candidates.append((final, sent, page, source))

    if not candidates:
        return {"headline": "", "answer": "", "citations": [], "diag": [], "notes": "No candidate sentences."}

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:top_n]
    best = top_candidates[0]
    _, sent, page, source = best
    vals = extract_values(sent)
    headline = ""
    if vals["amounts"]:
        headline = " • ".join(vals["amounts"])
    elif vals["percents"]:
        headline = " • ".join(vals["percents"])
    elif expects_number and vals["numbers"]:
        headline = " • ".join(vals["numbers"])

    diag_rows = []
    if diagnostics_on:
        for (final, s, p, src) in top_candidates:
            diag_rows.append({
                "has_num": has_number(s),
                "final": round(final,3),
                "page": p, "source": src,
                "sentence": s[:400]
            })

    return {
        "headline": headline,
        "answer": sent,
        "citations": [{"page": page, "source": source}],
        "diag": diag_rows,
        "notes": "Blended score with plan/policy awareness; table-aware extraction."
    }

# =========================
# Build filter dict
# =========================
def make_filter_dict(selected_type, selected_company, selected_policy):
    filt = {}
    # Handle the "Unknown" type issue - if most docs are Unknown, don't filter by type
    if selected_type and selected_type != "(All)" and selected_type != "Unknown":
        # Only apply type filter if it's not Unknown (since most docs have Unknown type)
        filt["type"] = selected_type
    if selected_company and selected_company != "(All)": 
        filt["company"] = selected_company
    if selected_policy and selected_policy != "(All)": 
        filt["policy_name"] = selected_policy
    return filt or None

# =========================
# LLM helpers
# =========================
QUERY_REWRITE_SYS = """You are a helpful assistant that rewrites a user's question
into 1-2 semantically equivalent queries optimized for document retrieval.
Keep crucial entities, numbers, plan amounts, and policy names intact.
Return them as a short bullet list (no commentary)."""

def llm_rewrite_query(llm: LLMClient, question: str) -> str:
    base_q = to_str(question)
    prompt = f"Original question:\n{base_q}\n\nRewrite 1-2 alternatives:"
    out = llm.chat(system=QUERY_REWRITE_SYS, user=prompt, json_mode=False)
    safe_out = to_str(out)
    return base_q + (f"\n{safe_out}" if safe_out else "")

RERANK_SYS = """You are a careful judge. I will give a user question and a list of passages.
Rank them by how likely they directly answer the question, prioritizing passages
that contain exact amounts, percentages, limits, and policy-specific details.
Return JSON: { "ranking": [ { "id": "<C1>", "score": 0-1 } ... ] }.
Only include IDs you were given. Be conservative; if unsure, lower the score."""

def llm_rerank(llm: LLMClient, question: str, contexts: list[dict]) -> list[dict]:
    joined = "\n\n".join([f"{c['id']}: {to_str(c['text'])[:1500]}" for c in contexts])  # truncate each
    user = f"Question:\n{to_str(question)}\n\nPassages:\n{joined}\n\nReturn JSON ranking."
    raw = llm.chat(system=RERANK_SYS, user=user, json_mode=True)
    data = safe_json_loads(to_str(raw)) or {}
    ranked = []
    ids = {c["id"] for c in contexts}
    for item in data.get("ranking", []):
        cid = item.get("id")
        if cid in ids:
            ranked.append({"id": cid, "score": float(item.get("score", 0))})
    if not ranked:  # fallback
        return contexts
    id2ctx = {c["id"]: c for c in contexts}
    return [id2ctx[r["id"]] for r in ranked]

SYNTH_SYS = """You are an insurance-policy assistant. Using only the provided CONTEXT,
answer the user. If the answer is not in the context, say that you cannot find it.
Be precise with numbers (retain units, ₹/%, limits, caps, co-pay, room type).
Always include citations by the given context IDs.

Set confidence based on answer quality:
- "high": Complete answer with specific numbers/details from context
- "medium": Partial answer or general information available  
- "low": Limited/uncertain information or answer not fully in context

Return strict JSON:
{
 "answer": "final answer in 2-6 sentences",
 "bullets": ["optional key points"],
 "citations": [{"id": "C1", "page": 3, "source": "path/or/url"}],
 "quotes": [{"id":"C1", "quote":"verbatim snippet"}],
 "confidence": "high"
}"""

def llm_synthesize_answer(llm: LLMClient, question: str, contexts: list[dict]) -> dict:
    # Simplified approach - use fewer contexts to avoid token limits
    packed = []
    for c in contexts[:6]:  # Limit to 6 contexts
        meta = c.get("meta", {}) or {}
        page = meta.get("page", "N/A")
        src = meta.get("source", "N/A")
        packed.append(f"{c['id']} (page {page}): {to_str(c['text'])[:1200]}\n[Source: {src}]")
    joined = "\n\n".join(packed)
    user = f"QUESTION:\n{to_str(question)}\n\nCONTEXT (use IDs to cite):\n{joined}"
    
    try:
        raw = llm.chat(system=SYNTH_SYS, user=user, json_mode=True)
        result = safe_json_loads(to_str(raw))
        if result and isinstance(result, dict) and result.get("answer"):
            # Ensure confidence is always set
            if not result.get("confidence"):
                answer_text = result.get("answer", "")
                if "₹" in answer_text or "%" in answer_text or any(num in answer_text for num in ["lakh", "crore"]):
                    result["confidence"] = "high"
                elif len(answer_text) > 100 and result.get("citations"):
                    result["confidence"] = "medium"
                else:
                    result["confidence"] = "low"
            return result
        else:
            # Fallback: try without JSON mode
            raw_text = llm.chat(system="Answer the question using the provided context. Be concise and cite sources.", user=user, json_mode=False)
            return {"answer": to_str(raw_text), "confidence": "medium"}
    except Exception:
        return {"answer": "Error generating answer", "confidence": "low"}

# ====== Summarize (structured JSON) ======
SUMMARIZE_SYS = """You are an insurance-policy analyst. Using ONLY the provided excerpts
for a single policy, produce a terse but complete structured summary.
Be precise with numbers, currency (₹/INR), percentages, caps, co-pay, room type,
sum insured variants, waiting periods, and common exclusions. If a field is unknown,
use an empty list "" or empty string "". Do not invent data.
Return STRICT JSON with this schema:
{
 "policy": "",
 "company": "",
 "type": "",
 "sum_insured_variants": ["₹3 lakh", "₹5 lakh", "₹10 lakh"],
 "room_type": "Private/Semi-private/Shared/ICU rules/NA",
 "copay": "e.g., 20% above 60 years or NA",
 "restore_benefit": "e.g., 100% once per year/Not available",
 "key_benefits": ["...", "..."],
 "notable_limits_caps": ["...", "..."],
 "waiting_periods": ["...", "..."],
 "exclusions_common": ["...", "..."],
 "best_for": "one-line guidance for who it suits",
 "notes": "",
 "confidence": "low\nmedium\nhigh"
}
Keep the output compact. Numbers must keep units and symbols exactly as shown in context.
If snippets contradict, flag in notes and lower confidence."""

# ====== Summarize (paragraph mode) ======
PARAGRAPH_SUMMARY_SYS = """You are an insurance-policy analyst. Using ONLY the provided excerpts,
write a comprehensive summary with pros and cons for this policy. Structure it as:

**Policy Overview:** Brief description of what the policy covers

**Pros:**
• Key benefits and advantages
• Notable features and add-ons
• Competitive aspects

**Cons:**
• Important limitations and exclusions
• Waiting periods or restrictions
• Co-pay requirements or caps

Keep it factual—do NOT invent details. Base everything on the provided excerpts."""

def make_policy_snippets_for_llm(
    policy_docs: dict[str, list],
    per_policy_chars: int = 25000,   # reduced to avoid token limits
    sents_per_doc: int = 100         # reduced sentences per doc
) -> dict[str, str]:
    """
    Build a richer snippet per policy.

    Robustness:
    - If split_sentences() yields nothing (e.g., bullet-rich pages),
      fall back to line-based splitting.
    - Cap total characters and lines to avoid over-long prompts.
    """
    out = {}
    for pol, docs in policy_docs.items():
        chunks = []
        size = 0
        for d in docs:
            text = to_str(d.page_content)
            sents = split_sentences(text)
            if not sents:
                # Fallback to lines for bullet lists / tables without punctuation
                sents = [ln.strip() for ln in text.splitlines() if ln.strip()]

            for s in sents[:sents_per_doc]:
                if size + len(s) + 1 > per_policy_chars:
                    break
                chunks.append(s)
                size += len(s) + 1

            if size >= per_policy_chars:
                break

        out[pol] = "\n".join(chunks)
    return out

def llm_summarize_policy(llm: LLMClient, policy_name: str, company: str, ins_type: str, text_snippet: str) -> dict:
    # Start with smaller input to avoid MAX_TOKENS
    max_chars = 8000
    
    for attempt in range(3):
        try:
            # Reduce input size on each attempt
            current_snippet = to_str(text_snippet)[:max_chars]
            header = (
                f"POLICY: {to_str(policy_name)}\n"
                f"COMPANY: {to_str(company)}\n"
                f"TYPE: {to_str(ins_type)}\n\n"
                f"EXCERPTS:\n{current_snippet}"
            )
            
            if attempt == 0:
                raw = llm.chat(system=SUMMARIZE_SYS, user=header, json_mode=True)
            elif attempt == 1:
                simple_sys = "Create a JSON summary with fields: policy, company, type, key_benefits, confidence."
                raw = llm.chat(system=simple_sys, user=header, json_mode=True)
            else:
                # Minimal prompt for final attempt
                minimal_prompt = f"JSON summary for {policy_name}:\n{current_snippet[:3000]}"
                raw = llm.chat(system="Return JSON only.", user=minimal_prompt, json_mode=True)
            
            # Check for MAX_TOKENS error
            raw_str = to_str(raw)
            if "MAX_TOKENS" in raw_str or "finishReason" in raw_str:
                max_chars = max_chars // 2  # Reduce by half
                continue
                
            data = safe_json_loads(raw_str)
            
            if data and isinstance(data, dict):
                data.setdefault("policy", to_str(policy_name))
                data.setdefault("company", to_str(company))
                data.setdefault("type", to_str(ins_type))
                return data
                
        except Exception as e:
            max_chars = max_chars // 2
            continue
    
    # Final fallback with minimal data
    return {
        "policy": to_str(policy_name),
        "company": to_str(company),
        "type": to_str(ins_type),
        "key_benefits": ["Policy summary available"],
        "confidence": "medium"
    }

def llm_paragraph_summary(llm: LLMClient, policy_name: str, company: str, ins_type: str, text_snippet: str) -> str:
    header = (
        f"POLICY: {to_str(policy_name)}\n"
        f"COMPANY: {to_str(company)}\n"
        f"TYPE: {to_str(ins_type)}\n\n"
        f"EXCERPTS:\n{to_str(text_snippet)[:8000]}"
    )
    out = llm.chat(system=PARAGRAPH_SUMMARY_SYS, user=header, json_mode=False)
    return to_str(out)

@st.cache_data(show_spinner=False)
def build_policy_catalog_for_scope(selected_type: str, selected_company: str) -> Dict[str, List]:
    vs = load_vectorstore()
    out: Dict[str, List] = {}
    for d in getattr(vs.docstore, "_dict", {}).values():
        md = getattr(d, "metadata", {}) or {}
        if (selected_type and md.get("type") == selected_type) and \
           (selected_company and md.get("company") == selected_company):
            pol = md.get("policy_name") or "Unknown"
            out.setdefault(pol, []).append(d)
    return out

@st.cache_data(show_spinner=False)
def get_docs_for_policy(selected_type: str, selected_company: str, selected_policy: str) -> list:
    vs = load_vectorstore()
    docs = []
    for d in getattr(vs.docstore, "_dict", {}).values():
        md = getattr(d, "metadata", {}) or {}
        if md.get("type") == selected_type and md.get("company") == selected_company and \
           md.get("policy_name") == selected_policy:
            docs.append(d)
    return docs

def coerce_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip():
        parts = re.split(r"[;\n•]+", val)
        return [p.strip() for p in parts if p.strip()]
    return []

def normalize_summary_row(js: dict) -> dict:
    return {
        "policy": js.get("policy", ""),
        "company": js.get("company", ""),
        "type": js.get("type", ""),
        "sum_insured_variants": " \n ".join(coerce_list(js.get("sum_insured_variants", []))),
        "room_type": js.get("room_type", ""),
        "copay": js.get("copay", ""),
        "restore_benefit": js.get("restore_benefit", ""),
        "key_benefits": " \n ".join(coerce_list(js.get("key_benefits", []))),
        "notable_limits_caps": " \n ".join(coerce_list(js.get("notable_limits_caps", []))),
        "waiting_periods": " \n ".join(coerce_list(js.get("waiting_periods", []))),
        "exclusions_common": " \n ".join(coerce_list(js.get("exclusions_common", []))),
        "best_for": js.get("best_for", ""),
        "notes": js.get("notes", ""),
        "confidence": js.get("confidence", ""),
    }

# --- Export helpers ---
import pandas as pd
from io import BytesIO

def to_csv_bytes(rows: List[Dict]) -> bytes:
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

def to_excel_bytes(rows: List[Dict]) -> Tuple[bytes, str]:
    """
    Returns (bytes, engine_used).
    Tries openpyxl for .xlsx; if not available, returns CSV bytes and 'csv-fallback'.
    """
    df = pd.DataFrame(rows)
    try:
        from openpyxl import Workbook  # quick availability check
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Summaries")
        bio.seek(0)
        return bio.read(), "openpyxl"
    except Exception:
        return to_csv_bytes(rows), "csv-fallback"

# =========================
# Recommendation helpers
# =========================
def extract_plan_tokens_from_input(si_text: str) -> List[str]:
    if not si_text: return []
    return extract_target_plan_tokens(si_text)

FEATURE_KEYWORDS = {
    "preventive health check": ["preventive health check", "health check-up", "health check up"],
    "restore benefit": ["restore benefit", "restore"],
    "maternity": ["maternity", "child birth"],
    "diabetes program": ["diabetes", "HbA1C", "wellness programme for diabetes"],
    "hypertension program": ["hypertension", "wellness programme for hypertension"],
    "private room": ["private room", "room rent", "shared accommodation"],
    "ambulance": ["ambulance cover"],
    "organ donor": ["organ donor"],
    "day care": ["day care"],
    "cash benefit": ["cash benefit", "optima cash"]
}

def score_policy_against_requirements(policy_name: str, docs_scope, req_text: str, plan_tokens: List[str], embedder) -> Dict:
    score = 0.0; reasons = []; citations = []
    # Plan token check (neighborhood matches)
    for doc in docs_scope:
        sents = split_sentences(doc.page_content)
        for idx, s in enumerate(sents):
            neigh = " ".join(sents[max(0, idx-2):min(len(sents), idx+3)]).lower()
            if any(tp in neigh for tp in plan_tokens):
                score += 1.0
                citations.append({"page": doc.metadata.get("page"), "source": doc.metadata.get("source")})
                reasons.append(f"Matches your plan amount near: “{s[:120]}…”")
                break
        if plan_tokens: break

    # Feature probes
    req_lower = to_str(req_text).lower()
    for feature, keys in FEATURE_KEYWORDS.items():
        if any(k in req_lower for k in keys):
            probe_q = f"{feature}"
            best_hit = None; best_sim = -1
            q_vec = np.array(embedder.embed_query(probe_q))
            for d in docs_scope:
                sents = split_sentences(d.page_content)
                vecs = np.array(embedder.embed_documents(sents))
                for s, v in zip(sents, vecs):
                    sim = cosine_sim(q_vec, v)
                    if sim > best_sim:
                        best_sim = sim; best_hit = (s, d.metadata.get("page"), d.metadata.get("source"))
            if best_hit:
                s, p, src = best_hit
                score += 1.0
                reasons.append(f"Mentions **{feature}**: “{s[:120]}…”")
                citations.append({"page": p, "source": src})

    return {"score": round(score, 2), "reasons": reasons, "citations": citations}

def llm_rank_policies(llm: LLMClient, requirements: str, policy_snippets: dict[str,str]) -> list[dict]:
    RECO_SYS = """You are ranking health insurance policies for a user based on their requirements.
Use only the provided excerpts for each policy. Be explicit with room-type, co-pay, limits,
sum insured variants, exclusions, maternity, and wellness programs. Penalize if excerpts contradict
the requirement. Output strict JSON:
{
 "items": [
  {"policy": "<name>", "score": 0-100, "reasons": ["...","..."], "best_quotes": ["..."]},
  ...
 ]
}"""
    lines = [f"USER REQUIREMENTS:\n{to_str(requirements)}\n", "POLICIES:"]
    for name, text in policy_snippets.items():
        lines.append(f"=== {name} ===\n{to_str(text)[:6000]}")
    user = "\n".join(lines) + "\n\nReturn JSON as specified."
    raw = llm.chat(system=RECO_SYS, user=user, json_mode=True)
    data = safe_json_loads(to_str(raw)) or {}
    return data.get("items", [])

# =========================
# SECTION WRAPPERS (UI)
# =========================
def section_retrieve_answer(
    selected_type, selected_company, selected_policy,
    search_type, top_k, diagnostics,
    use_llm_for_query, use_llm_for_rerank, use_llm_for_answer,
    llm_model, llm_temp
):
    st.markdown('<div style="background: linear-gradient(90deg, #e3f2fd, #bbdefb); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;"><h2 style="color: #1565c0; margin: 0;">🔎 Ask Questions About Policies</h2></div>', unsafe_allow_html=True)
    default_q = "In Optima Restore, what is the preventive health check‑up benefit for an individual under ₹10 lakh plan?"
    question = st.text_input("Question", value=default_q, placeholder="Type your question…", key="qa_question_input")
    run_clicked = st.button("🔎 Retrieve", key="qa_run_button", use_container_width=True)
    if not run_clicked:
        return
    if not to_str(question):
        st.warning("Please enter a question.")
        return

    llm = LLMClient(model=to_str(llm_model), temperature=float(llm_temp))

    effective_search_type = "similarity" if numeric_intent(question) else search_type
    source_contains = None
    if selected_policy != "(All)":
        source_contains = selected_policy.lower().replace(" ", "-")
    
    # Use filters only if specific selections are made, otherwise search all
    filt = None
    if selected_type != "(All)" or selected_company != "(All)" or selected_policy != "(All)":
        filt = make_filter_dict(selected_type, selected_company, selected_policy)

    try:
        retriever = get_retriever(top_k, effective_search_type, filt, source_contains=source_contains)
        with st.spinner("Retrieving passages…"):
            q_for_search = to_str(question)
            if use_llm_for_query:
                q_for_search = llm_rewrite_query(llm, q_for_search)
            t0 = time.time(); docs = retriever.invoke(q_for_search); elapsed = time.time() - t0

            if not docs:
                # --------- AUTO FALLBACKS WHEN 0 RESULTS ----------
                retriever_relaxed = get_retriever(max(12, top_k), "similarity", None, source_contains=None)
                docs = retriever_relaxed.invoke(q_for_search)

                if not docs:
                    retriever_mmr = get_retriever(16, "mmr", None, source_contains=None)
                    docs = retriever_mmr.invoke(q_for_search)

                if not docs and use_llm_for_query:
                    raw_q = to_str(question)
                    retriever_raw = get_retriever(16, "similarity", None, source_contains=None)
                    docs = retriever_raw.invoke(raw_q)

                if not docs:
                    st.error("No relevant documents found for your question.")
                    return
            else:
                st.success(f"Retrieved {len(docs)} passages in {elapsed:.2f}s · k={top_k} · type={effective_search_type}")

        # Skip heuristic answer - only show LLM answer
        pass

        # Build contexts with IDs for LLM
        contexts = []
        for i, d in enumerate(docs, 1):
            cid = f"C{i}"
            md = d.metadata or {}
            contexts.append({
                "id": cid,
                "text": d.page_content,
                "meta": {"page": md.get("page", "N/A"),
                         "source": md.get("source", "N/A"),
                         "policy_name": md.get("policy_name"),
                         "type": md.get("type"),
                         "company": md.get("company")}
            })

        # Optional LLM re‑rank
        ordered_contexts = contexts
        if use_llm_for_rerank:
            with st.spinner("LLM re‑ranking…"):
                ordered_contexts = llm_rerank(llm, to_str(question), contexts)

        # LLM synthesis (always enabled)
        with st.spinner("Generating answer…"):
            try:
                js = llm_synthesize_answer(llm, to_str(question), ordered_contexts[:12])
                if js and isinstance(js, dict) and js.get("answer"):
                    st.markdown("### ✅ Answer")
                    st.write(js["answer"])
                    if js.get("bullets"):
                        for b in js["bullets"]: st.markdown(f"- {b}")
                    if js.get("citations"):
                        st.caption("Citations:")
                        for c in js["citations"]:
                            st.caption(f"• {c.get('id')} · Page {c.get('page')} · Source: `{c.get('source')}`")
                    if js.get("quotes"):
                        with st.expander("Supporting quotes"):
                            for q in js["quotes"]:
                                st.code(f"{q.get('id')}: {q.get('quote')}")
                    
                    # Display confidence with color coding
                    confidence = js.get("confidence", "medium")
                    if confidence == "high":
                        st.success(f"🎯 Confidence: {confidence.upper()}")
                    elif confidence == "medium":
                        st.info(f"⚖️ Confidence: {confidence.upper()}")
                    else:
                        st.warning(f"⚠️ Confidence: {confidence.upper()}")

                else:
                    st.error("Could not generate answer. Please try again or check your query.")
            except Exception as e:
                st.error(f"Error generating answer: {str(e)[:100]}...")

        # Diagnostics removed since we only show LLM answer now
        pass
        # Retrieved passages section removed for cleaner UI
        pass

    except Exception as e:
        st.error(f"Error during retrieval: {e}")

def section_summarize(
    selected_type, selected_company, selected_policy,
    llm_model, llm_temp
):
    st.markdown('<div style="background: linear-gradient(90deg, #f3e5f5, #e1bee7); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;"><h2 style="color: #7b1fa2; margin: 0;">📄 Policy Summaries</h2></div>', unsafe_allow_html=True)
    # Summary mode toggle — default to Paragraph
    summary_mode = st.radio("Summary mode", ["Paragraph", "Structured JSON"], index=0, horizontal=True, key="sum_mode")

    col_sum1, col_sum2 = st.columns([2, 1])
    with col_sum1:
        st.caption("Select **Insurance Type**, **Company**, and a **Policy** in the sidebar, then click summarize.")
    with col_sum2:
        summarize_one_clicked = st.button("Summarize selected policy", key="sum_selected_policy_btn")

    if summarize_one_clicked:
        if selected_type == "(All)" or selected_company == "(All)" or selected_policy == "(All)":
            st.warning("Please select **Insurance Type**, **Company**, and a specific **Policy** in the sidebar.")
            return

        docs_for_policy = get_docs_for_policy(selected_type, selected_company, selected_policy)
        if not docs_for_policy:
            policy_docs_relaxed = build_policy_catalog_for_scope(selected_type, selected_company)
            if not policy_docs_relaxed:
                st.error("No documents found. Please check your selections.")
                return
            # Choose the largest policy cluster as a proxy to summarize
            selected_policy_relaxed, docs_for_policy = max(policy_docs_relaxed.items(), key=lambda kv: len(kv[1]))
            st.caption(f"Using policy: {selected_policy_relaxed}")

        llm = LLMClient(model=to_str(llm_model), temperature=float(llm_temp))

        if summary_mode == "Paragraph":
            with st.spinner(f"Summarizing **{selected_policy}** (paragraph)…"):
                # 1) Build robust snippet
                one_map = {selected_policy: docs_for_policy}
                snippet_map = make_policy_snippets_for_llm(one_map, per_policy_chars=20000, sents_per_doc=80)
                text_snippet = to_str(snippet_map.get(selected_policy, ""))

                # 2) Raw-text fallback if sentence-level snippet is empty
                if not text_snippet:
                    raw_chunks = []
                    for d in docs_for_policy[:12]:
                        raw = to_str(d.page_content)
                        if raw:
                            raw_chunks.append(raw[:1800])  # 1.8k chars per doc
                    text_snippet = "\n".join(raw_chunks)

            

                # 3) Normal prompt
                paragraph = llm_paragraph_summary(llm, selected_policy, selected_company, selected_type, text_snippet)
                safe_paragraph = to_str(paragraph)

                # 4) Simplified retry if blank
                if not safe_paragraph:
                    st.info("LLM returned empty reply. Retrying with a simplified prompt …")
                    simplified_prompt = (
                        "Write a concise, grounded paragraph (4–6 sentences) summarizing the following excerpts. "
                        "Be factual, keep units (₹/%, caps/co-pay), and avoid inventing details.\n\n"
                        + text_snippet[:8000]
                    )
                    retry_out = llm.chat(system="", user=simplified_prompt, json_mode=False)
                    safe_paragraph = to_str(retry_out)

                # 5) Heuristic fallback if still blank
                if not safe_paragraph:
                    st.info("Using heuristic paragraph fallback (no LLM output) …")
                    lines = [ln.strip() for ln in text_snippet.splitlines() if ln.strip()]
                    heur = " ".join(lines[:6])
                    safe_paragraph = heur[:1800]

            if safe_paragraph:
                st.success(f"Paragraph summary for **{selected_policy}**")
                st.write(safe_paragraph)

                # Show a tiny context preview for transparency
                with st.expander("Context used (sample)"):
                    preview_sents = to_str(text_snippet).split("\n")[:8]
                    st.code("\n".join(preview_sents) if preview_sents else "(no context)")

                # Download TXT
                st.download_button(
                    "⬇️ Download paragraph (.txt)",
                    data=safe_paragraph.encode("utf-8"),
                    file_name=f"summary_{selected_company}_{selected_type}_{selected_policy}.txt",
                    mime="text/plain",
                    key="sum_paragraph_download_btn",
                )
            else:
                st.warning("Could not produce a paragraph even after fallbacks.")

        else:
            with st.spinner(f"Summarizing **{selected_policy}** (structured)…"):
                # Use the same robust snippet builder for structured mode too
                one_map = {selected_policy: docs_for_policy}
                snippet_map = make_policy_snippets_for_llm(one_map, per_policy_chars=20000, sents_per_doc=80)
                text_snippet = to_str(snippet_map.get(selected_policy, ""))

                if not text_snippet:
                    raw_chunks = []
                    for d in docs_for_policy[:12]:
                        raw = to_str(d.page_content)
                        if raw:
                            raw_chunks.append(raw[:1800])
                    text_snippet = "\n".join(raw_chunks)

                js = llm_summarize_policy(llm, selected_policy, selected_company, selected_type, text_snippet)
                row = normalize_summary_row(js)

            # Show summary table (single row)
            st.success(f"Structured summary for **{selected_policy}**")
            st.dataframe([row], width="stretch", hide_index=True)

            # CSV/Excel downloads for the one summary (Excel fallback enabled)
            csv_bytes = to_csv_bytes([row])
            excel_bytes, engine_used = to_excel_bytes([row])

            col_csv, col_xlsx = st.columns(2)
            with col_csv:
                st.download_button(
                    "⬇️ Download CSV (policy summary)",
                    data=csv_bytes,
                    file_name=f"summary_{selected_company}_{selected_type}_{selected_policy}.csv",
                    mime="text/csv",
                    key="sum_policy_csv_btn",
                )
            with col_xlsx:
                if engine_used == "openpyxl":
                    st.download_button(
                        "⬇️ Download Excel (.xlsx)",
                        data=excel_bytes,
                        file_name=f"summary_{selected_company}_{selected_type}_{selected_policy}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="sum_policy_xlsx_btn",
                    )
                else:
                    st.download_button(
                        "⬇️ Download (CSV fallback – Excel not available)",
                        data=excel_bytes,
                        file_name=f"summary_{selected_company}_{selected_type}_{selected_policy}.csv",
                        mime="text/csv",
                        key="sum_policy_csv_fallback_btn",
                    )
                    st.caption("Note: `openpyxl` not found, provided CSV instead.")



def section_recommend(
    selected_type, selected_company,
    llm_model, llm_temp
):
    st.markdown('<div style="background: linear-gradient(90deg, #e8f5e8, #c8e6c9); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;"><h2 style="color: #2e7d32; margin: 0;">🎯 Find Your Perfect Plan</h2></div>', unsafe_allow_html=True)
    st.caption("Tell me your requirements and I’ll rank policies within the selected Insurance Type + Company.")
    req_text = st.text_area(
        "Your requirements (e.g., ₹10 lakh cover, private room rent, preventive check‑up, maternity, diabetes program)",
        height=140,
        key="reco_req_text",
    )
    
    if st.button("⚙️ Get Recommendations", type="primary"):
        if not req_text.strip():
            st.warning("Please enter your requirements to get personalized recommendations.")
            return

        # Group docs by policy_name
        vs = load_vectorstore()
        policy_docs: Dict[str, List] = {}
        for d in getattr(vs.docstore, "_dict", {}).values():
            md = getattr(d, "metadata", {}) or {}
            # Apply filters only if specific selections are made
            if selected_type != "(All)" and md.get("type") != selected_type:
                continue
            if selected_company != "(All)" and md.get("company") != selected_company:
                continue
            pol = md.get("policy_name") or "Unknown"
            policy_docs.setdefault(pol, []).append(d)

        if not policy_docs:
            st.info("No policies found under this Type + Company in the index.")
            return

        st.success(f"Found {len(policy_docs)} policies for {selected_type} · {selected_company}")
        
        # Rank policies based on requirements
        policy_rankings = []
        
        for pol_name, docs in policy_docs.items():
            # Get policy content for matching
            policy_content = ""
            for doc in docs[:3]:  # Use first 3 docs for efficiency
                policy_content += doc.page_content[:1000] + " "
            
            # Simple keyword matching for ranking
            req_lower = req_text.lower()
            content_lower = policy_content.lower()
            
            # Count requirement matches
            req_words = [w.strip() for w in req_lower.replace(',', ' ').split() if len(w.strip()) > 2]
            matches = sum(1 for word in req_words if word in content_lower)
            match_score = matches / max(len(req_words), 1)
            
            # Generate policy link (placeholder - adjust based on actual URL structure)
            policy_link = f"https://insurance-portal.com/policies/{selected_company.lower().replace(' ', '-')}/{pol_name.lower().replace(' ', '-')}"
            
            policy_rankings.append({
                'name': pol_name,
                'score': match_score,
                'matches': matches,
                'content': policy_content[:400],
                'link': policy_link,
                'docs_count': len(docs)
            })
        
        # Sort by match score (best match first)
        policy_rankings.sort(key=lambda x: x['score'], reverse=True)
        
        # Display ranked recommendations
        st.markdown("### 📊 Policy Rankings Based on Your Requirements")
        
        for i, policy in enumerate(policy_rankings, 1):
            match_percentage = int(policy['score'] * 100)
            
            if match_percentage >= 70:
                match_color = "🟢 Best Match"
                bg_color = "#e8f5e8"
            elif match_percentage >= 40:
                match_color = "🟡 Good Match"
                bg_color = "#fff3e0"
            else:
                match_color = "🔴 Lower Match"
                bg_color = "#ffebee"
            
            st.markdown(f'<div style="background: {bg_color}; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">', unsafe_allow_html=True)
            st.markdown(f"### #{i} · **{policy['name']}** · {match_color} ({match_percentage}%)")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Policy Overview:**")
                st.info(policy['content'] + "...")
                st.caption(f"Based on {policy['docs_count']} document(s) | {policy['matches']} requirement matches")
            
            with col2:
                st.markdown("**Policy Link:**")
                st.markdown(f"[🔗 View Policy Details]({policy['link']})")
                st.markdown(f"**Match Score:** {match_percentage}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.info(f"💡 Tip: Higher match percentages indicate better alignment with your requirements. Click the policy links to view full details.")

# Action Selection with Cards
st.markdown("""
<div style="margin: 2rem 0;">
    <h3 style="color: #1e293b; font-size: 1.8rem; font-weight: 600; margin-bottom: 1.5rem; text-align: center;">🚀 Choose Your AI Assistant Mode</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    retrieve_btn = st.button("🔎 Smart Q&A", use_container_width=True, key="retrieve_btn", help="Ask specific questions about insurance policies")
with col2:
    summarize_btn = st.button("📄 Policy Analysis", use_container_width=True, key="summarize_btn", help="Get detailed policy summaries and insights")
with col3:
    recommend_btn = st.button("🎯 AI Recommendations", use_container_width=True, key="recommend_btn", help="Find the best policies for your needs")

# Determine mode based on button clicks
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "Retrieve Answer"

if retrieve_btn:
    st.session_state.current_mode = "Retrieve Answer"
elif summarize_btn:
    st.session_state.current_mode = "Summarize"
elif recommend_btn:
    st.session_state.current_mode = "Recommend Best Plan"

mode = st.session_state.current_mode
st.markdown(f'<div class="mode-indicator">🤖 Active Mode: {mode}</div>', unsafe_allow_html=True)
st.markdown("<hr style='margin: 2rem 0; border: none; height: 1px; background: linear-gradient(90deg, transparent, #e2e8f0, transparent);'>", unsafe_allow_html=True)

if mode == "Retrieve Answer":
    section_retrieve_answer(
        selected_type, selected_company, selected_policy,
        search_type, top_k, diagnostics,
        use_llm_for_query, use_llm_for_rerank, use_llm_for_answer,
        llm_model, llm_temp
    )
elif mode == "Summarize":
    section_summarize(
        selected_type, selected_company, selected_policy,
               llm_model, llm_temp
    )
elif mode == "Recommend Best Plan":
    section_recommend(
        selected_type, selected_company,
        llm_model, llm_temp
    )