from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, time
import pandas as pd, numpy as np
import tldextract, httpx
import dns.resolver
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

load_dotenv()

app = FastAPI(title="Chyll FastAPI MVP - Railway Free Tier")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Config
SIRENE_MODE = os.getenv("SIRENE_MODE", "demo")  # Default to demo for Railway free tier
SIRENE_TOKEN = os.getenv("SIRENE_TOKEN", "")
SIRENE_BASE  = "https://api.insee.fr/entreprises/sirene/V3/siren"
SIRENE_BULK_PATH = os.getenv("SIRENE_BULK_PATH", "")

HTTP = httpx.Client(timeout=10.0, follow_redirects=True)
DNS  = dns.resolver.Resolver(); DNS.timeout = DNS.lifetime = 1.0

ARTIFACTS: Dict[str, Dict[str, Any]] = {}
CACHE: Dict[str, Dict[str, Any]] = {}

# ---- Schemas
class HistoryRow(BaseModel):
    company_name: str
    deal_status: str  # "won" | "lost"
    website: Optional[str] = None
    siren: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    ape: Optional[str] = None
    legal_form: Optional[str] = None
    created_year: Optional[int] = None
    deal_date: Optional[str] = None

class CandidateRow(BaseModel):
    company_name: str
    website: Optional[str] = None
    siren: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    ape: Optional[str] = None
    legal_form: Optional[str] = None
    created_year: Optional[int] = None
    region: Optional[str] = None
    department: Optional[str] = None
    active: Optional[bool] = None

class TrainRequest(BaseModel):
    tenant_id: str
    rows: List[HistoryRow]

class DiscoverFilters(BaseModel):
    ape_codes: Optional[List[str]] = None
    regions: Optional[List[str]] = None
    departments: Optional[List[str]] = None
    age_buckets: Optional[List[str]] = None
    headcount_bands: Optional[List[str]] = None

class DiscoverRequest(BaseModel):
    tenant_id: str
    filters: DiscoverFilters
    limit: Optional[int] = 50000

class ScoreRequest(BaseModel):
    tenant_id: str
    rows: List[CandidateRow]

# ---- Helper functions (ultra-lightweight)
def cget(k):
    v = CACHE.get(k); 
    return (None if (not v or v["exp"] < time.time()) else v["val"])
def cset(k, v, ttl=3600*48): CACHE[k] = {"val": v, "exp": time.time()+ttl}

def norm(s): return (s or "").strip()
def extract_domain(x):
    s = norm(x).lower().replace("http://","").replace("https://","").split("/")[0]
    if not s: return ""
    e = tldextract.extract(s)
    return f"{e.domain}.{e.suffix}" if e.suffix else e.domain

def check_mx(domain):
    if not domain: return 0
    k=f"mx:{domain}"; v=cget(k)
    if v is not None: return v
    try:
        DNS.resolve(domain,"MX"); cset(k,1); return 1
    except: cset(k,0); return 0

def check_http_tls(domain):
    if not domain: return (0,0)
    k=f"http:{domain}"; v=cget(k)
    if v is not None: return v
    http_ok, tls_ok = 0,0
    try:
        r=HTTP.get(f"https://{domain}"); http_ok = 1 if r.status_code in (200,301,302,403) else 0; tls_ok=1
    except:
        try:
            r=HTTP.get(f"http://{domain}"); http_ok = 1 if r.status_code in (200,301,302,403) else 0
        except: pass
    cset(k,(http_ok,tls_ok)); return (http_ok,tls_ok)

def web_footprint_label(has_domain, mx_ok, http_ok, tls_ok):
    if not has_domain: return "none"
    if not (mx_ok or http_ok or tls_ok): return "weak"
    if http_ok and not (mx_ok and tls_ok): return "basic"
    if mx_ok and http_ok and tls_ok: return "healthy"
    return "weak"

def text_fingerprint(name, ape): return f"{norm(name)} — {norm(ape)}".strip()

# Ultra-simple similarity (no heavy ML)
def simple_similarity(text1: str, text2: str) -> float:
    """Simple text similarity based on common words"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["domain"] = df.get("website","").map(extract_domain)
    df["has_domain"] = (df["domain"]!="").astype(int)
    df["has_siren"]  = df.get("siren","").astype(str).str.replace(r"\D","",regex=True).str.len().ge(9).astype(int)
    mx, httpo, tlso = [], [], []
    for d in df["domain"].fillna(""):
        m = check_mx(d); h,t = check_http_tls(d)
        mx.append(m); httpo.append(h); tlso.append(t)
    df["mx_present"] = mx; df["http_ok"] = httpo; df["tls_ok"] = tlso
    df["web_footprint"] = [web_footprint_label(hd, m, h, t) for hd,m,h,t in zip(df["has_domain"], df["mx_present"], df["http_ok"], df["tls_ok"])]
    df["ape"] = df.get("ape","")
    df["created_year"] = pd.to_numeric(df.get("created_year", np.nan), errors="coerce")
    df["age_years"] = np.where(df["created_year"].notna(), pd.Timestamp.now().year - df["created_year"], np.nan)
    return df

# ---- Routes
@app.get("/health")
def health(): 
    return {
        "ok": True, 
        "service": "chyll-fastapi-railway-free", 
        "sirene_mode": SIRENE_MODE,
        "memory_limit": "0.5GB",
        "plan": "Railway Free Tier"
    }

@app.post("/train")
def train(req: TrainRequest):
    tenant = req.tenant_id
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    if df.empty: return {"ok": False, "error":"no rows"}
    df = featurize(df)
    
    # Ultra-simple feature engineering (minimal memory usage)
    Xtab = pd.DataFrame({
        "has_domain": df["has_domain"], 
        "has_siren": df["has_siren"],
        "mx_present": df["mx_present"], 
        "http_ok": df["http_ok"], 
        "tls_ok": df["tls_ok"],
        "age_years": df["age_years"].fillna(-1),
    }).fillna(0)
    
    y = (df["deal_status"].str.lower()=="won").astype(int).to_numpy()
    
    # Use lightweight RandomForest (no heavy ML libraries)
    base = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)  # Reduced for memory
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtab, y)
    
    # Store minimal artifacts
    ARTIFACTS[tenant] = {
        "clf": clf,
        "hist_labels": y,
        "hist_texts": [text_fingerprint(n,a) for n,a in zip(df["company_name"], df.get("ape",""))]
    }
    
    return {
        "ok": True, 
        "stats": {"rows": int(len(df)), "wins": int(y.sum()), "losses": int((1-y).sum())}, 
        "model_version": f"{tenant}-v1-free-tier"
    }

@app.post("/discover")
def discover(req: DiscoverRequest):
    tenant = req.tenant_id
    if tenant not in ARTIFACTS:
        return {"ok": False, "error":"Model not trained"}
    
    f = req.filters or DiscoverFilters()
    limit = int(req.limit or 100)  # Very small limit for free tier
    
    # Minimal demo data for free tier
    items = [
        {"company_name":"GENERATIVSCHOOL","siren":"938422896","website":"generativschool.com","ape":"8559B","region":"11","department":"75","created_year":2024},
        {"company_name":"EDU LAB FR","siren":"111222333","website":"edulab.fr","ape":"8559B","region":"11","department":"92","created_year":2019},
        {"company_name":"QUIET SCHOOL","siren":"444555666","website":"quietschooledu.fr","ape":"8559A","region":"11","department":"94","created_year":2016},
    ]
    
    df = pd.DataFrame(items[:limit])
    df = featurize(df)
    
    clf = ARTIFACTS[tenant]["clf"]
    hist_texts = ARTIFACTS[tenant]["hist_texts"]
    
    out = []
    for i in range(len(df)):
        # Simple similarity scoring
        current_text = text_fingerprint(df.iloc[i]["company_name"], df.iloc[i].get("ape",""))
        similarities = [simple_similarity(current_text, hist_text) for hist_text in hist_texts]
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # Predict with classifier
        Xtab = pd.DataFrame([{
            "has_domain": int(df.iloc[i]["has_domain"]),
            "has_siren": int(df.iloc[i]["has_siren"]),
            "mx_present": int(df.iloc[i]["mx_present"]),
            "http_ok": int(df.iloc[i]["http_ok"]),
            "tls_ok": int(df.iloc[i]["tls_ok"]),
            "age_years": float(df.iloc[i]["age_years"]) if pd.notna(df.iloc[i]["age_years"]) else -1.0,
        }]).fillna(0)
        
        p = float(clf.predict_proba(Xtab)[0][1])
        # Boost score based on similarity
        p = min(0.95, p + avg_similarity * 0.1)
        
        band = "High" if p>=0.75 else ("Medium" if p>=0.5 else "Low")
        
        badge = "Verified (SIREN)" if int(df.iloc[i]["has_siren"]) else ("Verified (Domain)" if int(df.iloc[i]["has_domain"]) else "High-confidence")
        
        out.append({
            "company_id": f"disc-{i}",
            "name": df.iloc[i]["company_name"],
            "siren": df.iloc[i].get("siren",""),
            "ape": df.iloc[i].get("ape",""),
            "region": df.iloc[i].get("region",""),
            "department": df.iloc[i].get("department",""),
            "win_score": p, 
            "band": band,
            "neighbors": [{"name": f"similar_{j}", "sim": float(similarities[j]), "outcome": "won" if ARTIFACTS[tenant]["hist_labels"][j]==1 else "lost"} for j in range(min(2, len(similarities)))],
            "reasons": [
                ("APE "+(df.iloc[i].get("ape") or "")).strip(),
                ("Web "+df.iloc[i]["web_footprint"]).strip(),
                (df.iloc[i].get("region","") or "Region n/a")
            ],
            "confidence_badge": badge,
            "web_footprint": df.iloc[i]["web_footprint"],
            "source": "discover-free-tier"
        })
    
    out = sorted(out, key=lambda r: r["win_score"], reverse=True)
    return {"ok": True, "batch_id": "demo-batch", "items": out, "total": len(out), "sirene_used": "demo"}

@app.post("/score")
def score(req: ScoreRequest):
    return {"ok": True, "items": []}

@app.post("/export")
def export(payload: Dict[str, Any]):
    return {"ok": True, "signed_url": None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
