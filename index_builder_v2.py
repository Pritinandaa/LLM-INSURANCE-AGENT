
# index_builder.py (category-aware; infers type + policy_name)
from __future__ import annotations
import os, re, json, time, ssl
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import requests, urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ====== CONFIG ======
# ====== CONFIG ======
# 1. Dynamically find the project root (where this script lives)
# This works on any computer (User 91720, pnanda, or anyone else)
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LLM_AGENT_ROOT", CURRENT_DIR))

# 2. Define paths relative to the root
SOURCES_JSON = ROOT / "sources.json"
DOWNLOAD_ROOT = ROOT / "downloads"
INDEX_DIR = ROOT / "indexes" / "policies_faiss"

# 3. Use the Model Name directly (Best Practice)
# By giving the string name, the library will automatically download 
# the model to a standard cache folder. You don't need to manage local paths.
EMBED_MODEL = "all-MiniLM-L6-v2"

# 4. Cache File
CACHE_FILE = DOWNLOAD_ROOT / "download_cache.json"

if not DOWNLOAD_ROOT.exists():
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

# Optional caps (set None to disable)
MAX_PDFS_PER_COMPANY: Optional[int] = None
MAX_PAGES_PER_PDF: Optional[int] = None
MAX_TOTAL_PAGES: Optional[int] = None
SKIP_PDFS_LARGER_MB: Optional[int] = None

# Pattern filters to reduce noise from landing pages (adjust freely)
INCLUDE_PATTERNS = [
    # health examples
    "optima-restore", "energy-combined", "optima-cash", "policy-wording", "policy-wordings",
    # motor examples
    "motor", "two-wheeler", "twowheeler", "bike", "car",
    # travel examples
    "travel", "overseas",
    # business examples
    "business", "suraksha"
]
EXCLUDE_PATTERNS = [
    "brochure", "faq", "claim", "proposal", "network", "hospitals"
]

REQUEST_TIMEOUT = 30
MAX_WORKERS = 5
PDF_EXT_RE = re.compile(r"\.pdf($|\?)", re.IGNORECASE)

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ====== Helpers ======
def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._\-]+", "_", name.strip())

def _matches_patterns(url: str, includes, excludes) -> bool:
    u = (url or "").lower()
    if includes:
        if not any(p.lower() in u for p in includes):
            return False
    if excludes:
        if any(p.lower() in u for p in excludes):
            return False
    return True

def infer_policy_name(pdf_path: Path) -> str:
    stem = pdf_path.stem.lower().replace("_", "-")
    patterns = [
        ("Optima Restore", ["optima-restore", "optima-restore-revision"]),
        ("Optima Cash", ["optima-cash", "optima-cash_policy-wordings_a5"]),
        ("Energy", ["energy", "energy-combined-pw-cis"]),
        ("Business Suraksha Plus", ["business-suraksha-plus"]),
    ]
    for friendly, keys in patterns:
        for k in keys:
            if k in stem:
                return friendly
    return stem.replace("-", " ").title()

# filename-based type inference. Extend as needed.
TYPE_RULES = {
    "Health": ["health", "optima", "restore", "energy", "cash", "diabetes", "hypertension"],
    "Motor": ["motor", "car", "bike", "two-wheeler", "twowheeler", "private car"],
    "Travel": ["travel", "overseas", "trip"],
    "Business": ["business", "suraksha", "commercial", "sme", "asset", "liability"]
}

def infer_type_from_name(name: str) -> str:
    n = (name or "").lower()
    for t, keys in TYPE_RULES.items():
        if any(k in n for k in keys):
            return t
    return "Unknown"

# ====== Cache ======
def load_cache() -> Dict[str, str]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_cache(cache: Dict[str, str]):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache))

# ====== Networking ======
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; PolicyBot/1.0)"})

def fetch(url: str, retries: int = 3) -> Tuple[Optional[bytes], Optional[str]]:
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, timeout=REQUEST_TIMEOUT, verify=False)
            r.raise_for_status()
            return r.content, r.headers.get("content-type", "")
        except Exception as e:
            print(f"[WARN] {url} attempt {attempt}/{retries}: {e}")
            time.sleep(min(2 * attempt, 6))
    return None, None

def head_size_mb(url: str) -> Optional[float]:
    try:
        r = SESSION.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True, verify=False)
        sz = r.headers.get("content-length")
        if sz: return int(sz) / (1024 * 1024)
    except Exception:
        pass
    return None

# ====== HTML parsing ======
def find_pdf_links(html: bytes, base_url: str) -> List[str]:
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []
    links = []
    for a in soup.find_all("a", href=True):
        href = (a["href"] or "").strip()
        if href.lower().startswith(("javascript:", "blob:")): continue
        if PDF_EXT_RE.search(href):
            links.append(requests.compat.urljoin(base_url, href))
    return list(dict.fromkeys(links))

# ====== Download logic ======
def smart_download(url: str, dest_dir: Path, cache: Dict[str, str],
                   include_patterns=None, exclude_patterns=None) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    content, ctype = fetch(url)
    if not content:
        print(f"[FAIL] No content for {url}")
        return []
    is_pdf = (ctype or "").lower().startswith("application/pdf") or PDF_EXT_RE.search(url)
    if is_pdf:
        name = Path(requests.utils.urlparse(url).path).name or "download.pdf"
        out = dest_dir / sanitize_name(name)
        if out.exists():
            print(f"[SKIP] Already exists: {out}")
            return [out]
        out.write_bytes(content)
        print(f"[SAVE] {out} ({len(content)} bytes)")
        return [out]

    pdf_links = find_pdf_links(content, base_url=url)
    # Apply pattern filters
    if include_patterns or exclude_patterns:
        pdf_links = [u for u in pdf_links if _matches_patterns(u, include_patterns, exclude_patterns)]

    if MAX_PDFS_PER_COMPANY:
        pdf_links = pdf_links[:MAX_PDFS_PER_COMPANY]

    if not pdf_links:
        print(f"[INFO] No PDF links passed filters at {url}")
        return []

    out_paths: List[Path] = []
    def download_pdf(purl: str) -> Optional[Path]:
        if SKIP_PDFS_LARGER_MB:
            mb = head_size_mb(purl)
            if mb and mb > SKIP_PDFS_LARGER_MB:
                print(f"[SKIP] {purl} is {mb:.1f}MB > {SKIP_PDFS_LARGER_MB}MB")
                return None
        pdata, ptype = fetch(purl)
        if pdata and ((ptype or "").lower().startswith("application/pdf") or PDF_EXT_RE.search(purl)):
            name = Path(requests.utils.urlparse(purl).path).name or "doc.pdf"
            out = dest_dir / sanitize_name(name)
            if out.exists():
                print(f"[SKIP] Already exists: {out}")
                return out
            out.write_bytes(pdata)
            print(f"[SAVE] {out} ({len(pdata)} bytes) from {purl}")
            return out
        print(f"[WARN] Not a PDF or failed: {purl}")
        return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(download_pdf, u) for u in pdf_links]
        for f in as_completed(futs):
            p = f.result()
            if p: out_paths.append(p)
    return out_paths

# ====== PDF extraction ======
def pdf_to_pages(pdf_path: Path, company: str) -> List[Tuple[str, Dict]]:
    out: List[Tuple[str, Dict]] = []
    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        limit = min(total_pages, MAX_PAGES_PER_PDF) if MAX_PAGES_PER_PDF else total_pages
        policy_name = infer_policy_name(pdf_path)
        inferred_type = infer_type_from_name(pdf_path.name)
        print(f"[PDF] {pdf_path.name}: pages={total_pages} (reading {limit}) · policy={policy_name} · type={inferred_type}")
        for i in range(limit):
            p = reader.pages[i]
            text = (p.extract_text() or "").replace("\u200b", "").strip()
            if text:
                out.append((text, {
                    "company": company,
                    "type": inferred_type,
                    "policy_name": policy_name,
                    "source": str(pdf_path),
                    "page": i + 1
                }))
            if (i + 1) % 10 == 0:
                print(f"[PDF] parsed {i + 1}/{limit} pages...")
    except Exception as e:
        print(f"[PDF] Cannot open {pdf_path}: {e}")
    return out

# ====== Embeddings + FAISS ======
def get_embedding():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})

def build_index(texts: List[str], metas: List[Dict], index_dir: Path):
    if not texts: raise RuntimeError("No texts to index.")
    print(f"[INDEX] Building FAISS on {len(texts)} chunks...")
    embedder = get_embedding()
    vs = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metas)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    print(f"[INDEX] Saved FAISS index to {index_dir}")

# ====== Main ======
def main():
    if not SOURCES_JSON.exists():
        raise FileNotFoundError(f"sources.json not found: {SOURCES_JSON}")
    entries = json.loads(SOURCES_JSON.read_text())
    cache = load_cache()
    all_texts: List[str] = []
    all_meta: List[Dict] = []
    total_pages_added = 0

    for e in entries:
        company = e.get("company") or e.get("name") or "Unknown"
        url = e["url"]
        print(f"\n[FLOW] Company={company} URL={url}")
        company_dir = DOWNLOAD_ROOT / sanitize_name(company)
        pdf_paths = smart_download(url, company_dir, cache, INCLUDE_PATTERNS, EXCLUDE_PATTERNS)
        if not pdf_paths:
            print(f"[FLOW] No PDFs for {company}. Skipping...")
            continue

        for pdf in pdf_paths:
            if MAX_TOTAL_PAGES and total_pages_added >= MAX_TOTAL_PAGES:
                print(f"[CUT] Reached MAX_TOTAL_PAGES={MAX_TOTAL_PAGES}, skipping remaining PDFs.")
                break
            pages = pdf_to_pages(pdf, company)
            if not pages: continue
            if MAX_TOTAL_PAGES:
                remaining = MAX_TOTAL_PAGES - total_pages_added
                pages = pages[:max(0, remaining)]
            for t, md in pages:
                all_texts.append(t); all_meta.append(md)
            total_pages_added += len(pages)
            print(f"[FLOW] Added {len(pages)} pages from {pdf.name}. Total pages: {total_pages_added}")

        print(f"[FLOW] Company {company} complete. PDFs: {len(pdf_paths)}")

    if not all_texts: raise RuntimeError("No text extracted from any PDF.")
    build_index(all_texts, all_meta, INDEX_DIR)
    print(f"\n[DONE] Index built successfully. Total pages indexed: {len(all_texts)}")

if __name__ == "__main__":
    main()
