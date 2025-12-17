
import os
import json
import warnings
import time
import requests

warnings.filterwarnings("ignore", category=UserWarning)

# Silence urllib3 insecure warnings since verify=False is intentional in dev
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass


# =========================
# Configuration / Defaults
# =========================
_DEFAULT_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
_DEFAULT_PROJECT = (
    os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("PROJECT_ID")
    or "my-project-28112025-479604"
)

# 🔥 HARD-CODED SA JSON path (Windows). Use forward slashes or raw string.
_HARDCODED_SA_PATH = r"C:/Users/pnanda/LLM-Insurance-Agent/my-project-28112025-479604-051567097943.json"

# Vertex REST endpoint format
_VERTEX_URL_FMT = (
    "https://{location}-aiplatform.googleapis.com/v1/"
    "projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent"
)


# =========================
# Utilities
# =========================
def safe_json_loads(text: str):
    """Safe JSON parse: returns dict/None, never raises."""
    if not text or not str(text).strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _make_session_for_auth():
    """
    Build a requests.Session for token operations.
    DEV behavior: SSL verify disabled unless REQUESTS_CA_BUNDLE is provided.
    Inherits HTTP(S)_PROXY if present.
    """
    s = requests.Session()
    ca_bundle = os.getenv("REQUESTS_CA_BUNDLE")
    if ca_bundle:
        s.verify = ca_bundle  # Use corporate/root CA in prod
        # print("DEBUG: Using REQUESTS_CA_BUNDLE for auth:", ca_bundle)
    else:
        s.verify = False  # DEV ONLY. Replace with org CA for production.
        # print("DEBUG: Auth session verify=False (DEV)")

    http_proxy = os.getenv("HTTP_PROXY")
    https_proxy = os.getenv("HTTPS_PROXY")
    if http_proxy or https_proxy:
        s.proxies.update({
            "http": http_proxy or "",
            "https": https_proxy or "",
        })
        # print("DEBUG: Auth session proxies:", s.proxies)
    return s


def _get_access_token():
    """
    Get an OAuth2 access token for Vertex AI using the hardcoded SA JSON.
    Strategy:
      1) Try google-auth refresh via a proxy-friendly Session (verify configurable).
      2) If that fails, do a manual JWT bearer exchange to token_uri.
    Returns: token (str) or None. Prints diagnostics on failure paths.
    """
    # Validate path and load SA info (for manual exchange)
    sa_path = _HARDCODED_SA_PATH
    # print("DEBUG: Checking SA path:", sa_path)
    if not os.path.exists(sa_path):
        # print("ERROR: Service account file not found.")
        return None

    try:
        with open(sa_path, "r", encoding="utf-8") as f:
            sa_info = json.load(f)
        token_uri = sa_info.get("token_uri", "https://oauth2.googleapis.com/token")
        client_email = sa_info.get("client_email")
        if not client_email:
            # print("ERROR: client_email missing in SA file.")
            return None
        # print("DEBUG: token_uri:", token_uri)
        # print("DEBUG: client_email:", client_email)
    except Exception as e:
        # print("ERROR: Could not parse SA JSON:", repr(e))
        return None

    # ---- Attempt 1: google-auth refresh via custom session
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request

        # print("DEBUG: Loading creds via google-auth...")
        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        sess = _make_session_for_auth()
        req = Request(session=sess)

        # Force refresh to ensure we have a live token
        # print("DEBUG: Refreshing google-auth credentials...")
        creds.refresh(req)

        if getattr(creds, "token", None):
            # print("DEBUG: google-auth refresh OK, token prefix:", creds.token[:24], "...")
            return creds.token
        else:
            # print("WARN: Token is None after google-auth refresh (no exception).")
            pass
    except Exception as e:
        # print("ERROR: google-auth refresh failed:", repr(e))
        pass

    # ---- Attempt 2: Manual JWT → access_token exchange
    try:
        # print("DEBUG: Trying manual JWT bearer exchange...")
        from google.auth import crypt, jwt

        signer = crypt.RSASigner.from_service_account_file(sa_path)
        now = int(time.time())
        claim = {
            "iss": client_email,
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": token_uri,
            "exp": now + 3600,
            "iat": now,
        }
        assertion = jwt.encode(signer, claim)  # RS256

        sess = _make_session_for_auth()
        resp = sess.post(
            token_uri,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": assertion,
            },
            timeout=45,
        )
        # print("DEBUG: Manual exchange status:", resp.status_code)
        if resp.status_code != 200:
            # print("ERROR: Manual exchange failed. Body:", (resp.text or "")[:600])
            return None

        data = resp.json()
        token = data.get("access_token")
        if token:
            # print("DEBUG: Manual exchange OK, token prefix:", token[:24], "...")
            return token
        else:
            # print("ERROR: Manual exchange returned no access_token. Payload:", data)
            return None

    except Exception as e:
        # print("ERROR: Manual JWT exchange failed:", repr(e))
        return None


def _build_url(project: str, location: str, model: str) -> str:
    return _VERTEX_URL_FMT.format(project=project, location=location, model=model)


class LLMClient:
    """
    Minimal REST Vertex AI Gemini client with robust error handling:
      - requests.post(..., verify=False)  # dev only
      - chat(system, user, json_mode=True/False) returns a string
      - OAuth2 bearer token from hardcoded SA path
      - Never raises to caller; returns helpful error text on failure
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        timeout: int = 60,
        max_output_tokens: int = 2048,
        project: str | None = None,
        location: str | None = None,
    ):
        self.model = model
        self.temperature = float(temperature or 0.2)
        self.timeout = int(timeout or 60)
        self.max_output_tokens = int(max_output_tokens or 2048)

        # Resolve project/location (allow env fallbacks)
        self.project = project or _DEFAULT_PROJECT
        self.location = location or _DEFAULT_LOCATION

        if not self.project:
            warnings.warn(
                "GOOGLE_CLOUD_PROJECT not set. Set env or pass project=... to LLMClient."
            )

    def _call(self, system: str, user: str) -> dict | None:
        if not self.project:
            return {
                "error": {
                    "code": "NO_PROJECT",
                    "message": (
                        "Missing GOOGLE_CLOUD_PROJECT / PROJECT_ID. "
                        "Set it before running (e.g., $env:GOOGLE_CLOUD_PROJECT='my-project-28112025-479604')."
                    ),
                }
            }

        token = _get_access_token()
        if not token:
            return {
                "error": {
                    "code": "NO_ACCESS_TOKEN",
                    "message": (
                        "Could not obtain OAuth2 access token. "
                        "Check the hardcoded service-account path, file existence, network/proxy, and IAM roles."
                    ),
                }
            }

        url = _build_url(self.project, self.location, self.model)

        payload = {
            "systemInstruction": {"role": "system", "parts": [{"text": (system or "").strip()}]},
            "contents": [{"role": "user", "parts": [{"text": (user or "").strip()}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }

        try:
            resp = requests.post(
                url,
                headers={
                    "Content-Type": "application/json; charset=UTF-8",
                    "Authorization": f"Bearer {token}",
                },
                data=json.dumps(payload),
                timeout=self.timeout,
                verify=False,  # ❗ DEV ONLY — enable with org CA in prod
            )

            # Debug snapshot (commented out for Streamlit)
            # print("STATUS:", resp.status_code)
            # print("CONTENT-TYPE:", resp.headers.get("Content-Type"))
            # print("REDIRECT HISTORY:", resp.history)
            # print("RAW RESPONSE:", (resp.text or "")[:300])

            text = resp.text or ""
            if not text.strip():
                return {"error": {"code": resp.status_code, "message": "Empty response body."}}

            # Proxy/firewall HTML check
            if text.strip().startswith("<html") or "<title>" in text[:500]:
                return {
                    "error": {
                        "code": "PROXY_BLOCK",
                        "message": "HTML response detected — likely blocked by proxy/firewall (ZScaler, etc.)",
                    }
                }

            try:
                return resp.json()
            except Exception:
                return {
                    "error": {
                        "code": resp.status_code,
                        "message": f"Non-JSON response: {text[:400]}",
                    }
                }
        except requests.RequestException as e:
            return {"error": {"code": "HTTP_ERROR", "message": str(e)}}

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        data = self._call(system or "", user or "")

        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            return (
                f'{{"error":"{err.get("code","unknown")}","message":"{err.get("message","")}"}}'
                if json_mode
                else f'Error: {err.get("code","unknown")} — {err.get("message","")}'
            )

        # Extract text from candidates; fallback to raw JSON
        txt = ""
        try:
            cands = data.get("candidates") or []
            for c in cands:
                parts = (c.get("content") or {}).get("parts") or []
                for p in parts:
                    if "text" in p:
                        txt += p["text"]
        except Exception:
            pass
        
        return txt or json.dumps(data or {})
