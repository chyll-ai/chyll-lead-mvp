from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, time
import pandas as pd, numpy as np
import tldextract, httpx
import dns.resolver
from sentence_transformers import SentenceTransformer
import faiss
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Chyll FastAPI MVP")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ---- Config
SIRENE_MODE = os.getenv("SIRENE_MODE", "api")  # "api" | "bulk" | "demo"
SIRENE_TOKEN = os.getenv("SIRENE_TOKEN", "")
SIRENE_BASE  = "https://api.insee.fr/entreprises/sirene/V3/siren"
SIRENE_BULK_PATH = os.getenv("SIRENE_BULK_PATH", "")  # Parquet/CSV file path for bulk mode

HTTP = httpx.Client(timeout=10.0, follow_redirects=True)
DNS  = dns.resolver.Resolver(); DNS.timeout = DNS.lifetime = 1.0
EMB  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
    age_buckets: Optional[List[str]] = None  # ["0_1","2_5","6_10","10p"]
    headcount_bands: Optional[List[str]] = None  # ["0_9","10_19","20_49","50_99","100p"]

class DiscoverRequest(BaseModel):
    tenant_id: str
    filters: DiscoverFilters
    limit: Optional[int] = 50000

class ScoreRequest(BaseModel):
    tenant_id: str
    rows: List[CandidateRow]

# ---- Small cache helpers
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

def embed_texts(texts: List[str]) -> np.ndarray:
    return np.asarray(EMB.encode(texts, normalize_embeddings=True), dtype="float32")

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

def neighbors_aggregates(cvec: np.ndarray, index, y_hist: np.ndarray, topk=20):
    # FAISS inner product on normalized vectors ~ cosine similarity
    D,I = index.search(cvec.reshape(1,-1), topk)
    sims = D[0]  # already IP (cosine)
    idx  = I[0]
    wins = (y_hist[idx]==1).astype(int)
    mean_win_sim  = float((sims * wins).sum() / max(wins.sum(),1))
    mean_lose_sim = float((sims * (1-wins)).sum() / max((1-wins).sum(),1))
    pct_wins_topk = float(wins.mean())
    return mean_win_sim, mean_lose_sim, pct_wins_topk, idx, sims

# ---- Sirene client (API & BULK)
def sirene_fetch_api(query: str, rows: int = 1000, cap: int = 50000):
    if not SIRENE_TOKEN:
        raise RuntimeError("SIRENE_TOKEN missing for api mode")
    headers = {"Authorization": f"Bearer {SIRENE_TOKEN}"}
    url = f"{SIRENE_BASE}?q={query}&nombre=1000"
    out = []
    fetched = 0
    while url and fetched < cap:
        r = HTTP.get(url, headers=headers)
        r.raise_for_status()
        j = r.json()
        for u in j.get("unitesLegales", []):
            out.append(u)
        fetched += len(j.get("unitesLegales", []))
        # Paginate by "links" if provided; if not, break (INSEE v3 sometimes no next)
        next_url = None
        for l in j.get("links", []):
            if l.get("rel")=="next": next_url = l.get("href")
        url = next_url
        if not url: break
    return out[:cap]

def sirene_query_from_filters(f: DiscoverFilters):
    clauses = ["etatAdministratifUniteLegale:A"]
    if f.ape_codes:
        ape_or = " OR ".join([f"activitePrincipaleUniteLegale:{a}" for a in f.ape_codes])
        clauses.append(f"({ape_or})")
    if f.regions:
        reg_or = " OR ".join([f"codeRegion:{r}" for r in f.regions])
        clauses.append(f"({reg_or})")
    if f.departments:
        dep_or = " OR ".join([f"codeDepartement:{d}" for d in f.departments])
        clauses.append(f"({dep_or})")
    # NOTE: age_buckets & headcount_bands can be approximated post-fetch (filter locally) if API filter is limited.
    return " AND ".join(clauses)

def map_unites_to_candidates(unites: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    items = []
    for u in unites:
        siren = u.get("siren","")
        periods = u.get("periodesUniteLegale", [])
        p = periods[0] if periods else {}
        name = p.get("denominationUniteLegale") or p.get("nomUniteLegale") or ""
        ape  = p.get("activitePrincipaleUniteLegale","")
        created = p.get("dateCreationUniteLegale","")
        created_year = int(created[:4]) if created[:4].isdigit() else None
        # Note: region/department sometimes at establishment level; we keep empty if unknown at UL level
        items.append({
            "company_name": name, "siren": siren, "ape": ape,
            "created_year": created_year, "region": "", "department": "", "active": True
        })
    return items

def sirene_fetch_bulk(limit: int, f: DiscoverFilters):
    if not SIRENE_BULK_PATH:
        raise RuntimeError("SIRENE_BULK_PATH missing for bulk mode")
    df = pd.read_parquet(SIRENE_BULK_PATH) if SIRENE_BULK_PATH.endswith(".parquet") else pd.read_csv(SIRENE_BULK_PATH)
    df = df.rename(columns=str.lower)
    # Expect columns like: siren, denominationunitelegale, activiteprincipaleunitelegale, datecreationunitelegale, etatadministratifunitelegale, coderegion, codedepartement
    df = df[df.get("etatadministratifunitelegale","A")=="A"]
    if f.ape_codes:
        df = df[df["activiteprincipaleunitelegale"].isin(f.ape_codes)]
    if f.regions:
        if "coderegion" in df.columns:
            df = df[df["coderegion"].isin(f.regions)]
    if f.departments:
        if "codedepartement" in df.columns:
            df = df[df["codedepartement"].isin(f.departments)]
    # age buckets post-filter:
    if f.age_buckets:
        year_now = pd.Timestamp.now().year
        df["created_year"] = pd.to_numeric(df.get("datecreationunitelegale","").astype(str).str[:4], errors="coerce")
        df["age_years"] = year_now - df["created_year"]
        mask = False
        for b in f.age_buckets:
            if b=="0_1": mask |= (df["age_years"]<=1)
            if b=="2_5": mask |= (df["age_years"].between(2,5, inclusive="both"))
            if b=="6_10": mask |= (df["age_years"].between(6,10, inclusive="both"))
            if b=="10p": mask |= (df["age_years"]>=11)
        df = df[mask]
    out = []
    for _,r in df.head(limit).iteritems() if False else df.head(limit).iterrows():  # keep simple
        out.append({
            "company_name": r.get("denominationunitelegale","") or r.get("nomunitelegale",""),
            "siren": str(r.get("siren","")),
            "ape": r.get("activiteprincipaleunitelegale",""),
            "created_year": int(r["created_year"]) if "created_year" in r and pd.notna(r["created_year"]) else None,
            "region": str(r.get("coderegion","") or ""),
            "department": str(r.get("codedepartement","") or ""),
            "active": True
        })
    return out

# ---- Routes
@app.get("/health")
def health(): 
    return {"ok": True, "service": "chyll-fastapi", "sirene_mode": SIRENE_MODE}

@app.post("/train")
def train(req: TrainRequest):
    tenant = req.tenant_id
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    if df.empty: return {"ok": False, "error":"no rows"}
    df = featurize(df)
    texts = [text_fingerprint(n,a) for n,a in zip(df["company_name"], df.get("ape",""))]
    Xvec = embed_texts(texts)
    y = (df["deal_status"].str.lower()=="won").astype(int).to_numpy()

    # FAISS (inner product) on normalized embeddings
    index = faiss.IndexFlatIP(Xvec.shape[1]); index.add(Xvec)

    # Build training tabular with similarity aggregates
    agg_win, agg_lose, agg_pct = [],[],[]
    for i in range(len(Xvec)):
        mw, ml, pw, _, _ = neighbors_aggregates(Xvec[i], index, y, topk=20)
        agg_win.append(mw); agg_lose.append(ml); agg_pct.append(pw)

    Xtab = pd.DataFrame({
        "has_domain": df["has_domain"], "has_siren": df["has_siren"],
        "mx_present": df["mx_present"], "http_ok": df["http_ok"], "tls_ok": df["tls_ok"],
        "age_years": df["age_years"].fillna(-1),
        "sim_mean_to_wins": agg_win, "sim_mean_to_losses": agg_lose, "pct_wins_topk": agg_pct
    }).fillna(0)

    X_tr, X_va, y_tr, y_va = train_test_split(Xtab, y, test_size=0.2, stratify=y, random_state=42)
    base = LGBMClassifier(n_estimators=200, learning_rate=0.05)
    clf  = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_tr, y_tr)

    ARTIFACTS[tenant] = {"index": index, "hist_labels": y, "clf": clf}
    return {"ok": True, "stats": {"rows": int(len(df)), "wins": int(y.sum()), "losses": int((1-y).sum())}, "model_version": f"{tenant}-v1"}

@app.post("/discover")
def discover(req: DiscoverRequest):
    tenant = req.tenant_id
    if tenant not in ARTIFACTS:
        return {"ok": False, "error":"Model not trained"}
    f = req.filters or DiscoverFilters()
    limit = int(req.limit or 50000)

    # Fetch universe from Sirene
    if SIRENE_MODE == "bulk":
        items = sirene_fetch_bulk(limit, f)
        sirene_used = "bulk"
    elif SIRENE_MODE == "api":
        q = sirene_query_from_filters(f)
        raw = sirene_fetch_api(q, rows=1000, cap=limit)
        items = map_unites_to_candidates(raw)
        sirene_used = "api"
    else:
        # Dev fallback demo (very small)
        items = [
            {"company_name":"GENERATIVSCHOOL","siren":"938422896","website":"generativschool.com","ape":"8559B","region":"11","department":"75","created_year":2024},
            {"company_name":"EDU LAB FR","siren":"111222333","website":"edulab.fr","ape":"8559B","region":"11","department":"92","created_year":2019},
            {"company_name":"QUIET SCHOOL","siren":"444555666","website":"quietschooledu.fr","ape":"8559A","region":"11","department":"94","created_year":2016},
        ]
        sirene_used = "demo"

    df = pd.DataFrame(items)
    if df.empty: return {"ok": True, "batch_id":"empty", "items": [], "total": 0, "sirene_used": sirene_used}
    df = featurize(df)
    texts = [text_fingerprint(n,a) for n,a in zip(df["company_name"], df.get("ape",""))]
    Cvec = embed_texts(texts)

    index = ARTIFACTS[tenant]["index"]
    y_hist = ARTIFACTS[tenant]["hist_labels"]
    clf   = ARTIFACTS[tenant]["clf"]

    out = []
    for i in range(len(df)):
        mw, ml, pw, idx, sims = neighbors_aggregates(Cvec[i], index, y_hist, topk=20)
        Xtab = pd.DataFrame([{
            "has_domain": int(df.iloc[i]["has_domain"]),
            "has_siren": int(df.iloc[i]["has_siren"]),
            "mx_present": int(df.iloc[i]["mx_present"]),
            "http_ok": int(df.iloc[i]["http_ok"]),
            "tls_ok": int(df.iloc[i]["tls_ok"]),
            "age_years": float(df.iloc[i]["age_years"]) if pd.notna(df.iloc[i]["age_years"]) else -1.0,
            "sim_mean_to_wins": float(mw),
            "sim_mean_to_losses": float(ml),
            "pct_wins_topk": float(pw)
        }]).fillna(0)
        p = float(clf.predict_proba(Xtab)[0][1])
        band = "High" if p>=0.75 else ("Medium" if p>=0.5 else "Low")

        # neighbors preview (top 2)
        preview = []
        for j in range(min(2, len(idx))):
            preview.append({"name": f"past_{idx[j]}", "siren": "", "sim": float(sims[j]), "outcome": "won" if y_hist[idx[j]]==1 else "lost"})

        badge = "Verified (SIREN)" if int(df.iloc[i]["has_siren"]) else ("Verified (Domain)" if int(df.iloc[i]["has_domain"]) else "High-confidence")

        out.append({
            "company_id": f"disc-{i}",
            "name": df.iloc[i]["company_name"],
            "siren": df.iloc[i].get("siren",""),
            "ape": df.iloc[i].get("ape",""),
            "region": df.iloc[i].get("region",""),
            "department": df.iloc[i].get("department",""),
            "win_score": p, "band": band,
            "neighbors": preview,
            "reasons": [
                ("APE "+(df.iloc[i].get("ape") or "")).strip(),
                ("Web "+df.iloc[i]["web_footprint"]).strip(),
                (df.iloc[i].get("region","") or "Region n/a")
            ],
            "confidence_badge": badge,
            "web_footprint": df.iloc[i]["web_footprint"],
            "source": "discover"
        })

    out = sorted(out, key=lambda r: r["win_score"], reverse=True)
    return {"ok": True, "batch_id": "sirene-batch", "items": out, "total": len(out), "sirene_used": sirene_used}

@app.post("/score")
def score(req: ScoreRequest):
    # Optional: same scoring path for user-supplied candidates
    return {"ok": True, "items": []}

@app.post("/export")
def export(payload: Dict[str, Any]):
    # MVP: front-end will export visible table to CSV client-side
    return {"ok": True, "signed_url": None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
