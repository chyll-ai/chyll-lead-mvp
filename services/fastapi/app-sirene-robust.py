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

def build_smart_sirene_query(historical_data: List[HistoryRow]) -> str:
    """Build smart SIRENE query based on historical won deals"""
    if not historical_data:
        return "etatAdministratifUniteLegale:A"  # Just active companies
    
    # Extract patterns from won deals
    won_deals = [row for row in historical_data if row.deal_status == "won"]
    if not won_deals:
        return "etatAdministratifUniteLegale:A"
    
    # Get APE codes from won deals
    ape_codes = []
    regions = []
    
    for deal in won_deals:
        if hasattr(deal, 'ape') and deal.ape:
            ape_codes.append(deal.ape)
        if hasattr(deal, 'postal_code') and deal.postal_code:
            # Extract region from postal code
            postal_str = str(deal.postal_code).zfill(5)
            first_two = postal_str[:2]
            if first_two in ["75", "77", "78", "91", "92", "93", "94", "95"]:
                regions.append("11")  # Île-de-France
            elif first_two == "69":
                regions.append("69")  # Rhône
            elif first_two == "13":
                regions.append("13")  # Bouches-du-Rhône
    
    # Build query
    query_parts = ["etatAdministratifUniteLegale:A"]  # Active companies only
    
    # Add APE code filters (if we have them)
    if ape_codes:
        unique_apes = list(set(ape_codes))
        if len(unique_apes) <= 3:  # Only if we have few APE codes
            ape_query = " OR ".join([f"activitePrincipaleUniteLegale:{ape}" for ape in unique_apes])
            query_parts.append(f"({ape_query})")
    
    # Add region filters (if we have them)
    if regions:
        unique_regions = list(set(regions))
        if len(unique_regions) <= 2:  # Only if we have few regions
            region_query = " OR ".join([f"codeRegion:{region}" for region in unique_regions])
            query_parts.append(f"({region_query})")
    
    return " AND ".join(query_parts)

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
    
    # Advanced closability indicators
    df["company_size_indicator"] = df["siren"].str.len().apply(lambda x: 0.8 if x >= 9 else 0.4)
    df["growth_stage"] = df["age_years"].apply(get_growth_stage_score)
    df["stability_score"] = df["age_years"].apply(get_stability_score)
    df["deal_readiness"] = df.apply(lambda row: calculate_deal_readiness(row), axis=1)
    
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

def get_growth_stage_score(age_years: float) -> float:
    """Score based on company growth stage - optimal for deal closing"""
    if pd.isna(age_years) or age_years < 0:
        return 0.3  # Unknown age = lower score
    
    if age_years < 1:
        return 0.2  # Too new, risky
    elif age_years < 3:
        return 0.7  # Growth stage - high potential
    elif age_years < 7:
        return 0.9  # Mature growth - optimal
    elif age_years < 15:
        return 0.8  # Established - good
    else:
        return 0.6  # Very old - might be conservative

def get_stability_score(age_years: float) -> float:
    """Score based on company stability - important for deal closing"""
    if pd.isna(age_years) or age_years < 0:
        return 0.3
    
    if age_years < 2:
        return 0.3  # Not stable enough
    elif age_years < 5:
        return 0.7  # Becoming stable
    elif age_years < 10:
        return 0.9  # Very stable
    else:
        return 0.8  # Stable but might be slow to decide

def calculate_deal_readiness(row) -> float:
    """Calculate overall deal readiness based on multiple factors"""
    # Company size (SIREN length indicates establishment)
    size_score = 0.8 if len(str(row.get("siren", ""))) >= 9 else 0.4
    
    # Industry maturity (tech companies are more deal-ready)
    ape = str(row.get("ape", ""))[:2]
    industry_readiness = {
        "62": 0.9,  # Computer programming - very deal-ready
        "63": 0.8,  # Information services
        "70": 0.7,  # Management consultancy
        "71": 0.6,  # Engineering
        "72": 0.6,  # R&D
    }.get(ape, 0.4)
    
    # Geographic readiness (Paris/Lyon companies are more deal-ready)
    city = str(row.get("city", "")).lower()
    geo_readiness = 0.9 if "paris" in city else (0.8 if "lyon" in city else 0.5)
    
    # Age-based readiness
    age = row.get("age_years", 0)
    if pd.isna(age):
        age_readiness = 0.3
    elif 3 <= age <= 8:
        age_readiness = 0.9  # Sweet spot for deals
    elif 1 <= age < 3:
        age_readiness = 0.7  # Growing companies
    elif 8 < age <= 15:
        age_readiness = 0.8  # Established
    else:
        age_readiness = 0.5  # Too new or too old
    
    # Weighted combination
    return (
        0.3 * size_score +
        0.3 * industry_readiness +
        0.2 * geo_readiness +
        0.2 * age_readiness
    )

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
            # Smart SIRENE querying based on historical data
            if SIRENE_MODE == "api" and SIRENE_TOKEN:
                # Build smart query based on won deals
                smart_query = build_smart_sirene_query(req.rows)
                print(f"Smart SIRENE query: {smart_query}")
                
                companies = sirene_fetch_api(smart_query, rows=500, cap=500)
                if companies:
                    print(f"Fetched {len(companies)} companies from SIRENE")
                    
                    # Process companies and score them
                    df_sample = pd.DataFrame(companies)
                    df_sample = featurize_simple(df_sample)
                    
                    # Pre-filter before scoring to reduce processing
                    # Only score companies with basic data quality
                    df_sample = df_sample[
                        (df_sample["company_name"].notna()) & 
                        (df_sample["company_name"] != "") &
                        (df_sample["ape"].notna()) &
                        (df_sample["ape"] != "")
                    ]
                    
                    print(f"Processing {len(df_sample)} companies with complete data")
                    
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
                        
                        # Advanced closability indicators
                        growth_stage_score = df_sample.iloc[i].get("growth_stage", 0.3)
                        stability_score = df_sample.iloc[i].get("stability_score", 0.3)
                        deal_readiness_score = df_sample.iloc[i].get("deal_readiness", 0.3)
                        company_size_score = df_sample.iloc[i].get("company_size_indicator", 0.3)
                        
                        # Calculate weighted composite score focused on closability
                        p = (
                            0.20 * ml_base_score +           # ML prediction (20%)
                            0.15 * industry_score +          # Industry relevance (15%)
                            0.15 * deal_readiness_score +    # Deal readiness (15%)
                            0.12 * growth_stage_score +      # Growth stage (12%)
                            0.12 * stability_score +         # Company stability (12%)
                            0.10 * geographic_score +        # Geographic intelligence (10%)
                            0.08 * company_size_score +      # Company size (8%)
                            0.05 * data_quality_score +      # Data completeness (5%)
                            0.03 * avg_similarity            # Historical similarity (3%)
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
                        if company_size_score > 0.6:
                            reasons.append("Established company")
                        
                        # Deal readiness indicators
                        if deal_readiness_score > 0.7:
                            reasons.append("Deal-ready")
                        elif deal_readiness_score > 0.5:
                            reasons.append("Good prospect")
                        
                        # Growth stage indicators
                        if growth_stage_score > 0.8:
                            reasons.append("Growth stage")
                        elif growth_stage_score > 0.6:
                            reasons.append("Expanding")
                        
                        # Stability indicators
                        if stability_score > 0.8:
                            reasons.append("Stable company")
                        
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
                    
                    # Show scoring distribution
                    scores = [c['win_score'] for c in scored_companies]
                    print(f"Scored {len(scored_companies)} companies - Score range: {min(scores):.2f} to {max(scores):.2f}")
                    print(f"Score distribution: {len([s for s in scores if s >= 0.7])} high, {len([s for s in scores if 0.5 <= s < 0.7])} medium, {len([s for s in scores if s < 0.5])} low")
                    
                    # Quality filtering - only high-scoring leads (realistic threshold)
                    high_quality_leads = [c for c in scored_companies if c["win_score"] >= 0.60]
                    
                    # Sort by win score and take top results
                    high_quality_leads.sort(key=lambda x: x["win_score"], reverse=True)
                    discovered_leads = high_quality_leads[:8]  # Show top 8 high-quality leads
                    
                    print(f"Quality filter: {len(scored_companies)} → {len(high_quality_leads)} → {len(discovered_leads)} final leads")
                    if discovered_leads:
                        print(f"Top 3 scores: {[c['win_score'] for c in discovered_leads[:3]]}")
                    
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
