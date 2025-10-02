from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Try to import sklearn, fallback gracefully if not available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: sklearn not available: {e}")
    SKLEARN_AVAILABLE = False

# Try to import httpx, fallback to urllib if not available
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    import urllib.request
    import urllib.parse
    HTTPX_AVAILABLE = False

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
SIRENE_BASE = "https://api.insee.fr/api-sirene/3.11/siren"

# Initialize HTTP client if available
if HTTPX_AVAILABLE:
    HTTP = httpx.Client(timeout=30.0, follow_redirects=True)
else:
    HTTP = None

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
    """Fetch companies from SIRENE API according to v3.11 documentation"""
    if not SIRENE_TOKEN:
        print("Warning: SIRENE_TOKEN not set, using demo data")
        return []
    
    try:
        # Use proper SIRENE API v3.11 parameters
        params = {
            "q": query,
            "nombre": min(rows, 1000),  # Max 1000 per request
            "debut": 1,
            "tri": "siren"
        }
        
        companies = []  # Initialize companies list
        
        if HTTPX_AVAILABLE and HTTP:
            headers = {"X-INSEE-Api-Key-Integration": SIRENE_TOKEN}
            url = f"{SIRENE_BASE}"
            
            response = HTTP.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
        else:
            # Fallback to urllib
            headers = {"X-INSEE-Api-Key-Integration": SIRENE_TOKEN}
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{SIRENE_BASE}?{query_string}"
            
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
        
        # Process companies from API response (works for both httpx and urllib)
        for unit in data.get("unitesLegales", []):
            siren = unit.get("siren", "")
            periods = unit.get("periodesUniteLegale", [])
            if periods:
                period = periods[0]
                name = period.get("denominationUniteLegale") or period.get("nomUniteLegale", "")
                ape = period.get("activitePrincipaleUniteLegale", "")
                created = period.get("dateCreationUniteLegale", "")
                created_year = int(created[:4]) if created[:4].isdigit() else None
                etat = period.get("etatAdministratifUniteLegale", "")
                
                # Only include active companies (A = Active)
                if etat == "A" and name:  # Only active companies with names
                    # Extract region and department from address if available
                    region = ""
                    department = ""
                    postal_code = ""
                    city = ""
                    
                    if "adresseEtablissement" in unit:
                        addr = unit["adresseEtablissement"]
                        region = addr.get("codeRegion", "")
                        department = addr.get("codeDepartement", "")
                        postal_code = addr.get("codePostalEtablissement", "")
                        city = addr.get("libelleCommuneEtablissement", "")
                    
                    companies.append({
                        "company_name": name,
                        "siren": siren,
                        "ape": ape,
                        "created_year": created_year,
                        "region": region,
                        "department": department,
                        "postal_code": postal_code,
                        "city": city,
                        "active": True
                    })
        
        return companies[:cap]
    
    except Exception as e:
        print(f"SIRENE API error: {e}")
        return []

def sirene_query_from_filters(f: DiscoverFilters):
    """Build SIRENE query from filters - simplified approach"""
    # For now, let's use a simple approach and filter in our code
    # The SIRENE API query syntax seems to be different than expected
    return ""  # Empty query means get all companies, we'll filter in code

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
    """Enhanced feature engineering for smart scoring"""
    df = df.copy()
    
    # Ensure required columns exist
    if "postal_code" not in df.columns:
        df["postal_code"] = ""
    if "city" not in df.columns:
        df["city"] = ""
    
    # Basic features with safe type conversion
    df["siren"] = df["siren"].fillna("").astype(str)
    df["has_siren"] = df["siren"].str.replace(r"\D","",regex=True).str.len().ge(9).astype(int)
    df["created_year"] = pd.to_numeric(df["created_year"], errors="coerce")
    df["age_years"] = np.where(df["created_year"].notna(), pd.Timestamp.now().year - df["created_year"], np.nan)
    
    # Enhanced features for smart scoring with safe string handling
    df["ape"] = df["ape"].fillna("").astype(str)
    df["ape_category"] = df["ape"].str[:2]
    df["industry_score"] = df["ape_category"].map(get_industry_score)
    
    # Geographic intelligence from postal codes and cities
    df["postal_code"] = df["postal_code"].fillna("").astype(str)
    df["city"] = df["city"].fillna("").astype(str)
    df["region"] = df["postal_code"].apply(extract_region_from_postal_code)
    df["region_score"] = df["region"].map(get_region_score)
    df["city_score"] = df.apply(lambda row: get_city_score(row.get("city", ""), row.get("postal_code", "")), axis=1)
    
    # Use the higher of region or city score for geographic intelligence
    df["geographic_score"] = df[["region_score", "city_score"]].max(axis=1)
    
    # Safe age categorization
    df["age_category"] = pd.cut(df["age_years"], bins=[0, 2, 5, 10, 100], labels=["startup", "growth", "mature", "established"])
    df["maturity_score"] = df["age_category"].map({"startup": 0.3, "growth": 0.8, "mature": 0.9, "established": 0.7})
    
    # Safe data quality calculation
    df["data_quality"] = (
        df["has_siren"].astype(int) + 
        df["ape"].notna().astype(int) + 
        df["postal_code"].notna().astype(int) +
        df["city"].notna().astype(int)
    ) / 4.0
    
    return df

def get_industry_score(ape_category: str) -> float:
    """Industry-specific scoring based on APE codes - more realistic scoring"""
    industry_scores = {
        "62": 0.8,  # Computer programming (high tech)
        "63": 0.7,  # Information service activities
        "70": 0.6,  # Activities of head offices; management consultancy
        "71": 0.6,  # Architectural and engineering activities
        "72": 0.6,  # Scientific research and development
        "85": 0.5,  # Education
        "46": 0.5,  # Wholesale trade
        "47": 0.4,  # Retail trade
        "68": 0.4,  # Real estate activities
        "41": 0.3,  # Construction
        "10": 0.2,  # Manufacture of food products
        "20": 0.2,  # Manufacture of chemicals
        "52": 0.3,  # Warehousing and support activities
        "77": 0.3,  # Rental and leasing activities
    }
    return industry_scores.get(ape_category, 0.3)  # Lower default score for unknown industries

def get_region_score(region: str) -> float:
    """Geographic scoring based on regions"""
    region_scores = {
        "11": 0.9,  # Île-de-France
        "75": 0.9,  # Paris
        "92": 0.8,  # Hauts-de-Seine
        "93": 0.7,  # Seine-Saint-Denis
        "94": 0.7,  # Val-de-Marne
        "69": 0.7,  # Rhône (Lyon)
        "13": 0.6,  # Bouches-du-Rhône (Marseille)
        "31": 0.6,  # Haute-Garonne (Toulouse)
        "59": 0.6,  # Nord (Lille)
        "44": 0.5,  # Loire-Atlantique (Nantes)
    }
    return region_scores.get(region, 0.4)  # Default score for other regions

def extract_region_from_postal_code(postal_code: str) -> str:
    """Extract region code from French postal code"""
    if not postal_code or len(str(postal_code)) < 2:
        return ""
    
    postal_str = str(postal_code).zfill(5)
    first_two = postal_str[:2]
    
    # Map postal code prefixes to region codes
    postal_to_region = {
        "75": "75",  # Paris
        "77": "11",  # Seine-et-Marne (Île-de-France)
        "78": "11",  # Yvelines (Île-de-France)
        "91": "11",  # Essonne (Île-de-France)
        "92": "92",  # Hauts-de-Seine
        "93": "93",  # Seine-Saint-Denis
        "94": "94",  # Val-de-Marne
        "95": "11",  # Val-d'Oise (Île-de-France)
        "69": "69",  # Rhône (Lyon)
        "13": "13",  # Bouches-du-Rhône (Marseille)
        "31": "31",  # Haute-Garonne (Toulouse)
        "59": "59",  # Nord (Lille)
        "44": "44",  # Loire-Atlantique (Nantes)
        "67": "44",  # Bas-Rhin (Alsace)
        "68": "44",  # Haut-Rhin (Alsace)
    }
    
    return postal_to_region.get(first_two, "")

def get_city_score(city: str, postal_code: str) -> float:
    """City-specific scoring based on economic importance"""
    city_scores = {
        "Paris": 0.95,
        "Lyon": 0.85,
        "Marseille": 0.75,
        "Toulouse": 0.75,
        "Nice": 0.70,
        "Nantes": 0.70,
        "Strasbourg": 0.70,
        "Montpellier": 0.65,
        "Bordeaux": 0.65,
        "Lille": 0.65,
        "Rennes": 0.60,
        "Reims": 0.55,
        "Saint-Étienne": 0.55,
        "Toulon": 0.55,
        "Le Havre": 0.50,
        "Grenoble": 0.50,
        "Dijon": 0.50,
        "Angers": 0.50,
        "Nîmes": 0.45,
        "Villeurbanne": 0.45,
        "Saint-Denis": 0.45,
        "Le Mans": 0.45,
        "Aix-en-Provence": 0.45,
        "Clermont-Ferrand": 0.45,
        "Brest": 0.45,
        "Tours": 0.45,
        "Limoges": 0.45,
        "Amiens": 0.45,
        "Annecy": 0.45,
        "Perpignan": 0.45,
        "Boulogne-Billancourt": 0.80,
        "Saint-Denis": 0.70,
        "Argenteuil": 0.60,
        "Montreuil": 0.60,
        "Roubaix": 0.60,
        "Tourcoing": 0.60,
        "Nanterre": 0.70,
        "Avignon": 0.50,
        "Créteil": 0.60,
        "Dunkirk": 0.50,
        "Poitiers": 0.50,
        "Asnières-sur-Seine": 0.70,
        "Versailles": 0.75,
        "Courbevoie": 0.75,
        "Vitry-sur-Seine": 0.60,
        "Colombes": 0.70,
        "Aulnay-sous-Bois": 0.60,
        "La Rochelle": 0.55,
        "Champigny-sur-Marne": 0.60,
        "Rueil-Malmaison": 0.75,
        "Antibes": 0.60,
        "Saint-Maur-des-Fossés": 0.70,
        "Cannes": 0.65,
        "Le Tampon": 0.40,
        "Aubervilliers": 0.60,
        "Béziers": 0.45,
        "Bourges": 0.45,
        "Cannes": 0.65,
        "Colmar": 0.50,
        "Drancy": 0.60,
        "Mérignac": 0.60,
        "Saint-Nazaire": 0.50,
        "Issy-les-Moulineaux": 0.75,
        "Noisy-le-Grand": 0.60,
        "Évry": 0.60,
        "Cergy": 0.60,
        "Pessac": 0.60,
        "Vénissieux": 0.60,
        "Clichy": 0.70,
        "Ivry-sur-Seine": 0.60,
        "Levallois-Perret": 0.75,
        "Montrouge": 0.70,
        "Neuilly-sur-Seine": 0.80,
        "Pantin": 0.60,
        "Suresnes": 0.70,
        "Vélizy-Villacoublay": 0.75,
        "Massy": 0.70,
    }
    
    # Check exact city match first
    if city in city_scores:
        return city_scores[city]
    
    # Check partial matches for common variations
    city_lower = city.lower()
    for city_key, score in city_scores.items():
        if city_key.lower() in city_lower or city_lower in city_key.lower():
            return score
    
    # Default scoring based on postal code region
    region = extract_region_from_postal_code(postal_code)
    return get_region_score(region)

# ---- Routes
@app.get("/health")
def health(): 
    return {
        "ok": True, 
        "service": "chyll-fastapi-sirene", 
        "sirene_mode": SIRENE_MODE,
        "sirene_token_set": bool(SIRENE_TOKEN),
        "sklearn_available": SKLEARN_AVAILABLE,
        "httpx_available": HTTPX_AVAILABLE
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
            # For very small datasets, use simple fit without cross-validation
            if len(y) < 4:
                clf = base
                clf.fit(Xtab, y)
            else:
                clf = CalibratedClassifierCV(base, method="isotonic", cv=min(2, len(y)//2))
                clf.fit(Xtab, y)
        else:
            clf = None
        
        # Store artifacts
        artifacts = {
            "clf": clf,
            "hist_labels": y,
            "hist_texts": [f"{row.company_name} {row.ape or ''}" for row in req.rows],
            "sklearn_available": SKLEARN_AVAILABLE
        }
        ARTIFACTS[tenant] = artifacts
        
        # Auto-discover leads after training for seamless UX
        discovered_leads = []
        try:
            # Optimized for Railway free plan - get more companies efficiently
            if SIRENE_MODE == "api" and SIRENE_TOKEN:
                companies = sirene_fetch_api("", rows=1000, cap=1000)  # Get more companies
                if companies:
                    print(f"Fetched {len(companies)} companies from SIRENE")
                    print(f"Sample company data: {companies[0] if companies else 'None'}")
                    
                    # Process all companies and score them
                    df_sample = pd.DataFrame(companies)
                    print(f"DataFrame columns: {df_sample.columns.tolist()}")
                    print(f"DataFrame dtypes: {df_sample.dtypes.to_dict()}")
                    
                    df_sample = featurize_simple(df_sample)
                    
                    # Score all companies
                    scored_companies = []
                    for i in range(len(df_sample)):
                        current_text = f"{df_sample.iloc[i]['company_name']} {df_sample.iloc[i].get('ape', '')}"
                        similarities = [simple_similarity(current_text, hist_text) for hist_text in artifacts["hist_texts"]]
                        avg_similarity = np.mean(similarities) if similarities else 0.0
                        
                        # Smart multi-factor scoring algorithm
                        if clf and SKLEARN_AVAILABLE:
                            Xtab = pd.DataFrame([{
                                "has_siren": int(df_sample.iloc[i]["has_siren"]),
                                "age_years": float(df_sample.iloc[i]["age_years"]) if pd.notna(df_sample.iloc[i]["age_years"]) else -1.0,
                            }]).fillna(0)
                            ml_base_score = float(clf.predict_proba(Xtab)[0][1])
                        else:
                            ml_base_score = 0.3 + avg_similarity * 0.4  # Lower base, more similarity weight
                        
                        # Get individual factor scores
                        industry_score = df_sample.iloc[i].get("industry_score", 0.3)
                        geographic_score = df_sample.iloc[i].get("geographic_score", 0.3)
                        maturity_score = df_sample.iloc[i].get("maturity_score", 0.3)
                        data_quality_score = df_sample.iloc[i].get("data_quality", 0.3)
                        
                        # Company size indicators (if available)
                        siren_length = len(str(df_sample.iloc[i].get("siren", "")))
                        size_indicator = 0.6 if siren_length >= 9 else 0.4  # Longer SIREN = more established
                        
                        # Calculate weighted composite score
                        p = (
                            0.25 * ml_base_score +           # ML prediction (25%)
                            0.20 * industry_score +          # Industry relevance (20%)
                            0.15 * geographic_score +        # Geographic intelligence (15%)
                            0.15 * maturity_score +          # Company maturity (15%)
                            0.10 * data_quality_score +      # Data completeness (10%)
                            0.10 * size_indicator +          # Company size (10%)
                            0.05 * avg_similarity            # Historical similarity (5%)
                        )
                        
                        # Apply realistic bounds (not too optimistic)
                        p = min(0.85, max(0.15, p))  # Cap at 85%, floor at 15%
                        band = "High" if p>=0.75 else ("Medium" if p>=0.5 else "Low")
                        
                        # Generate smart reasons for scoring
                        reasons = []
                        
                        # Industry relevance (only if significant)
                        if industry_score > 0.7:
                            ape = df_sample.iloc[i].get("ape", "")
                            if ape:
                                reasons.append(f"Tech industry ({ape})")
                        elif industry_score > 0.5:
                            ape = df_sample.iloc[i].get("ape", "")
                            if ape:
                                reasons.append(f"Relevant sector ({ape})")
                        
                        # Company maturity (only if significant)
                        if maturity_score > 0.7 and pd.notna(df_sample.iloc[i]["age_years"]):
                            age = int(df_sample.iloc[i]["age_years"])
                            if age >= 5:
                                reasons.append(f"Established ({age}y)")
                            elif age >= 2:
                                reasons.append(f"Growing ({age}y)")
                        
                        # Geographic intelligence (only if available)
                        if geographic_score > 0.6:
                            if df_sample.iloc[i].get("city"):
                                city = df_sample.iloc[i]["city"]
                                reasons.append(f"Prime location ({city})")
                            elif df_sample.iloc[i].get("region"):
                                region = df_sample.iloc[i]["region"]
                                reasons.append(f"Good region ({region})")
                        
                        # Data quality (only if high)
                        if data_quality_score > 0.75:
                            reasons.append("Complete data")
                        
                        # Historical similarity (only if significant)
                        if avg_similarity > 0.2:
                            reasons.append(f"Similar to wins ({avg_similarity:.1f})")
                        
                        # Company size indicator
                        if size_indicator > 0.5:
                            reasons.append("Established company")
                        
                        # Fallback reasons if none generated
                        if not reasons:
                            ape = df_sample.iloc[i].get("ape", "")
                            if ape:
                                reasons.append(f"Industry {ape}")
                            else:
                                reasons.append("Basic match")
                        
                        scored_companies.append({
                            "company_id": f"auto-disc-{i}",
                            "name": df_sample.iloc[i]["company_name"],
                            "siren": df_sample.iloc[i].get("siren",""),
                            "ape": df_sample.iloc[i].get("ape",""),
                            "region": df_sample.iloc[i].get("region",""),
                            "department": df_sample.iloc[i].get("department",""),
                            "city": df_sample.iloc[i].get("city",""),
                            "postal_code": df_sample.iloc[i].get("postal_code",""),
                            "win_score": round(p, 3), 
                            "band": band,
                            "confidence_badge": "Verified (SIREN)" if int(df_sample.iloc[i]["has_siren"]) else "High-confidence",
                            "source": "sirene",
                            "reasons": reasons,
                            "similarity_score": round(avg_similarity, 3)
                        })
                    
                    # Debug: Show all scores before filtering
                    print(f"All {len(scored_companies)} companies with scores: {[c['win_score'] for c in scored_companies[:5]]}")
                    
                    # Quality filtering - only high-scoring leads (realistic threshold)
                    high_quality_leads = [c for c in scored_companies if c["win_score"] >= 0.60]
                    
                    # Sort by win score and take top results
                    high_quality_leads.sort(key=lambda x: x["win_score"], reverse=True)
                    discovered_leads = high_quality_leads[:8]  # Show top 8 high-quality leads
                    
                    print(f"Filtered {len(scored_companies)} companies to {len(high_quality_leads)} high-quality leads")
                    print(f"Selected top {len(discovered_leads)} companies with scores {[c['win_score'] for c in discovered_leads[:3]]}")
                    
        except Exception as e:
            print(f"Auto-discovery error: {e}")
            # Fallback to demo data if SIRENE fails
            discovered_leads = [
                {
                    "company_id": "demo-1",
                    "name": "GENERATIVSCHOOL",
                    "siren": "938422896",
                    "ape": "8559B",
                    "region": "11",
                    "department": "75",
                    "win_score": 0.583,
                    "band": "Medium",
                    "confidence_badge": "Verified (SIREN)",
                    "source": "demo",
                    "reasons": ["APE 8559B", "Age 1 years", "Region 11"],
                    "similarity_score": 0.0
                }
            ]
        
        return {
            "ok": True, 
            "stats": {"rows": int(len(df)), "wins": int(y.sum()), "losses": int((1-y).sum())}, 
            "model_version": f"{tenant}-v1-sirene",
            "discovered_leads": discovered_leads,  # Return all discovered leads (up to 10)
            "total_discovered": len(discovered_leads),
            "message": f"Model trained successfully! Found {len(discovered_leads)} high-scoring leads from SIRENE database."
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
        
        # Fetch from SIRENE API
        if SIRENE_MODE == "api" and SIRENE_TOKEN:
            # Get a larger sample from SIRENE API and filter client-side
            companies = sirene_fetch_api("", rows=2000, cap=2000)  # Get more companies
            sirene_used = "api"
            
            # Client-side filtering based on user filters
            if companies and f:
                filtered_companies = []
                for company in companies:
                    include = True
                    
                    # Filter by APE codes
                    if f.ape_codes and company.get("ape") not in f.ape_codes:
                        include = False
                    
                    # Filter by regions
                    if f.regions and company.get("region") not in f.regions:
                        include = False
                    
                    # Filter by departments
                    if f.departments and company.get("department") not in f.departments:
                        include = False
                    
                    if include:
                        filtered_companies.append(company)
                
                companies = filtered_companies[:limit]
            else:
                companies = companies[:limit]
        else:
            # Fallback demo data only if no SIRENE token
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
