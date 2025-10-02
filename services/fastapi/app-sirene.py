from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import httpx
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Try to import sklearn, fallback gracefully if not available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: sklearn not available: {e}")
    SKLEARN_AVAILABLE = False

load_dotenv()

app = FastAPI(title="Chyll FastAPI MVP - SIRENE Integration")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Config
SIRENE_MODE = os.getenv("SIRENE_MODE", "api")
SIRENE_TOKEN = os.getenv("SIRENE_TOKEN", "")
SIRENE_BASE = "https://api.insee.fr/entreprises/sirene/V3/siren"

# Initialize HTTP client
HTTP = httpx.Client(timeout=30.0, follow_redirects=True)

ARTIFACTS: Dict[str, Dict[str, Any]] = {}

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
    limit: Optional[int] = 100

class ScoreRequest(BaseModel):
    tenant_id: str
    rows: List[CandidateRow]

# ---- SIRENE API Functions
def sirene_fetch_api(query: str, rows: int = 1000, cap: int = 1000):
    """Fetch companies from SIRENE API"""
    if not SIRENE_TOKEN:
        print("Warning: SIRENE_TOKEN not set, using demo data")
        return []
    
    try:
        headers = {"Authorization": f"Bearer {SIRENE_TOKEN}"}
        url = f"{SIRENE_BASE}?q={query}&nombre={rows}"
        
        response = HTTP.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        companies = []
        for unit in data.get("unitesLegales", []):
            siren = unit.get("siren", "")
            periods = unit.get("periodesUniteLegale", [])
            if periods:
                period = periods[0]
                name = period.get("denominationUniteLegale") or period.get("nomUniteLegale", "")
                ape = period.get("activitePrincipaleUniteLegale", "")
                created = period.get("dateCreationUniteLegale", "")
                created_year = int(created[:4]) if created[:4].isdigit() else None
                
                companies.append({
                    "company_name": name,
                    "siren": siren,
                    "ape": ape,
                    "created_year": created_year,
                    "region": "",
                    "department": "",
                    "active": True
                })
        
        return companies[:cap]
    
    except Exception as e:
        print(f"SIRENE API error: {e}")
        return []

def sirene_query_from_filters(f: DiscoverFilters):
    """Build SIRENE query from filters"""
    clauses = ["etatAdministratifUniteLegale:A"]  # Active companies only
    
    if f.ape_codes:
        ape_or = " OR ".join([f"activitePrincipaleUniteLegale:{a}" for a in f.ape_codes])
        clauses.append(f"({ape_or})")
    
    if f.regions:
        reg_or = " OR ".join([f"codeRegion:{r}" for r in f.regions])
        clauses.append(f"({reg_or})")
    
    if f.departments:
        dep_or = " OR ".join([f"codeDepartement:{d}" for d in f.departments])
        clauses.append(f"({dep_or})")
    
    return " AND ".join(clauses)

def simple_similarity(text1: str, text2: str) -> float:
    """Simple text similarity based on common words"""
    try:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0

def featurize_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature engineering"""
    df = df.copy()
    df["has_siren"] = df.get("siren","").astype(str).str.replace(r"\D","",regex=True).str.len().ge(9).astype(int)
    df["created_year"] = pd.to_numeric(df.get("created_year", np.nan), errors="coerce")
    df["age_years"] = np.where(df["created_year"].notna(), pd.Timestamp.now().year - df["created_year"], np.nan)
    return df

# ---- Routes
@app.get("/health")
def health(): 
    return {
        "ok": True, 
        "service": "chyll-fastapi-sirene", 
        "sirene_mode": SIRENE_MODE,
        "sirene_token_set": bool(SIRENE_TOKEN),
        "sklearn_available": SKLEARN_AVAILABLE
    }

@app.post("/train")
def train(req: TrainRequest):
    try:
        tenant = req.tenant_id
        df = pd.DataFrame([r.model_dump() for r in req.rows])
        
        if df.empty: 
            return {"ok": False, "error": "no rows"}
        
        # Simple feature engineering
        df = featurize_simple(df)
        
        Xtab = pd.DataFrame({
            "has_siren": df["has_siren"],
            "age_years": df["age_years"].fillna(-1),
        }).fillna(0)
        
        y = (df["deal_status"].str.lower()=="won").astype(int).to_numpy()
        
        if SKLEARN_AVAILABLE and len(y) > 1:
            base = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            clf = CalibratedClassifierCV(base, method="isotonic", cv=min(3, len(y)))
            clf.fit(Xtab, y)
        else:
            clf = None
        
        # Store artifacts
        ARTIFACTS[tenant] = {
            "clf": clf,
            "hist_labels": y,
            "hist_texts": [f"{row.company_name} {row.ape or ''}" for row in req.rows],
            "sklearn_available": SKLEARN_AVAILABLE
        }
        
        return {
            "ok": True, 
            "stats": {"rows": int(len(df)), "wins": int(y.sum()), "losses": int((1-y).sum())}, 
            "model_version": f"{tenant}-v1-sirene"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/discover")
def discover(req: DiscoverRequest):
    try:
        tenant = req.tenant_id
        if tenant not in ARTIFACTS:
            return {"ok": False, "error": "Model not trained"}
        
        f = req.filters or DiscoverFilters()
        limit = int(req.limit or 100)
        
        # Try to fetch from SIRENE API
        if SIRENE_MODE == "api" and SIRENE_TOKEN:
            query = sirene_query_from_filters(f)
            if query:
                companies = sirene_fetch_api(query, rows=1000, cap=limit)
                sirene_used = "api"
            else:
                companies = []
                sirene_used = "no_query"
        else:
            # Fallback demo data
            companies = [
                {"company_name":"GENERATIVSCHOOL","siren":"938422896","ape":"8559B","region":"11","department":"75","created_year":2024},
                {"company_name":"EDU LAB FR","siren":"111222333","ape":"8559B","region":"11","department":"92","created_year":2019},
                {"company_name":"QUIET SCHOOL","siren":"444555666","ape":"8559A","region":"11","department":"94","created_year":2016},
                {"company_name":"TECH STARTUP","siren":"777888999","ape":"6201Z","region":"11","department":"75","created_year":2023},
                {"company_name":"INNOVATION CORP","siren":"123456789","ape":"6202A","region":"11","department":"92","created_year":2020},
            ]
            sirene_used = "demo"
        
        if not companies:
            return {"ok": True, "items": [], "total": 0, "sirene_used": sirene_used}
        
        df = pd.DataFrame(companies[:limit])
        df = featurize_simple(df)
        
        artifacts = ARTIFACTS[tenant]
        clf = artifacts["clf"]
        hist_texts = artifacts["hist_texts"]
        hist_labels = artifacts["hist_labels"]
        
        out = []
        for i in range(len(df)):
            # Simple similarity scoring
            current_text = f"{df.iloc[i]['company_name']} {df.iloc[i].get('ape', '')}"
            similarities = [simple_similarity(current_text, hist_text) for hist_text in hist_texts]
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            if clf and SKLEARN_AVAILABLE:
                # Predict with classifier
                Xtab = pd.DataFrame([{
                    "has_siren": int(df.iloc[i]["has_siren"]),
                    "age_years": float(df.iloc[i]["age_years"]) if pd.notna(df.iloc[i]["age_years"]) else -1.0,
                }]).fillna(0)
                
                p = float(clf.predict_proba(Xtab)[0][1])
            else:
                # Fallback scoring
                p = 0.5 + avg_similarity * 0.3
            
            # Boost score based on similarity
            p = min(0.95, p + avg_similarity * 0.1)
            
            band = "High" if p>=0.75 else ("Medium" if p>=0.5 else "Low")
            
            badge = "Verified (SIREN)" if int(df.iloc[i]["has_siren"]) else "High-confidence"
            
            out.append({
                "company_id": f"disc-{i}",
                "name": df.iloc[i]["company_name"],
                "siren": df.iloc[i].get("siren",""),
                "ape": df.iloc[i].get("ape",""),
                "region": df.iloc[i].get("region",""),
                "department": df.iloc[i].get("department",""),
                "win_score": round(p, 3), 
                "band": band,
                "neighbors": [{"name": f"similar_{j}", "sim": round(float(similarities[j]), 3), "outcome": "won" if hist_labels[j]==1 else "lost"} for j in range(min(2, len(similarities)))],
                "reasons": [
                    f"APE {df.iloc[i].get('ape', '')}".strip(),
                    f"Age {int(df.iloc[i]['age_years']) if pd.notna(df.iloc[i]['age_years']) else 'Unknown'} years",
                    f"Region {df.iloc[i].get('region', '') or 'Unknown'}"
                ],
                "confidence_badge": badge,
                "source": "sirene" if sirene_used == "api" else "demo"
            })
        
        out = sorted(out, key=lambda r: r["win_score"], reverse=True)
        return {
            "ok": True, 
            "items": out, 
            "total": len(out), 
            "sirene_used": sirene_used,
            "message": f"Found {len(out)} companies from {sirene_used}"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/score")
def score(req: ScoreRequest):
    return {"ok": True, "items": []}

@app.post("/export")
def export(payload: Dict[str, Any]):
    return {"ok": True, "signed_url": None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
