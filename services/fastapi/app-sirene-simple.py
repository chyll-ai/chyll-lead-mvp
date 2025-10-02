#!/usr/bin/env python3
"""
Simple SIRENE-based lead scoring using pattern learning from user data
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
from datetime import datetime

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="SIRENE Simple Lead Scoring", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SIRENE_TOKEN = os.getenv("SIRENE_TOKEN")
SIRENE_BASE_URL = "https://api.insee.fr/entreprises/sirene/V3.11"
SKLEARN_AVAILABLE = True
HTTPX_AVAILABLE = True

# Global storage for learned patterns
LEARNED_PATTERNS = {}

class TrainRow(BaseModel):
    company_name: str
    deal_status: str
    siren: Optional[str] = ""
    ape: Optional[str] = ""
    postal_code: Optional[str] = ""
    city: Optional[str] = ""
    legal_form: Optional[str] = ""
    employee_range: Optional[str] = ""
    created_year: Optional[int] = None
    website: Optional[str] = ""
    email: Optional[str] = ""

class TrainRequest(BaseModel):
    tenant_id: str
    rows: List[TrainRow]

class DiscoverRequest(BaseModel):
    tenant_id: str
    filters: Dict[str, Any]
    limit: int = 100

class DiscoverResponse(BaseModel):
    companies: List[Dict[str, Any]]
    total_found: int
    patterns_used: Dict[str, Any]

def get_age_range(created_year: int) -> str:
    """Convert creation year to age range"""
    if not created_year or created_year == 0:
        return "unknown"
    
    current_year = datetime.now().year
    age = current_year - created_year
    
    if age < 2:
        return "startup"
    elif age < 5:
        return "growth"
    elif age < 10:
        return "mature"
    else:
        return "established"

def analyze_user_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract distribution patterns from user's successful deals"""
    print(f"[DEBUG] Analyzing patterns from {len(df)} total deals")
    
    won_deals = df[df['deal_status'].str.lower() == 'won']
    print(f"[DEBUG] Found {len(won_deals)} won deals")
    
    if len(won_deals) == 0:
        return {
            'ape_distribution': {},
            'region_distribution': {},
            'size_distribution': {},
            'legal_form_distribution': {},
            'age_distribution': {},
            'total_won': 0
        }
    
    # Clean and prepare data
    won_deals = won_deals.copy()
    won_deals['ape'] = won_deals['ape'].fillna('').astype(str)
    won_deals['postal_code'] = won_deals['postal_code'].fillna('').astype(str)
    won_deals['employee_range'] = won_deals['employee_range'].fillna('').astype(str)
    won_deals['legal_form'] = won_deals['legal_form'].fillna('').astype(str)
    won_deals['created_year'] = pd.to_numeric(won_deals['created_year'], errors='coerce')
    
    patterns = {
        'ape_distribution': won_deals['ape'].value_counts(normalize=True).to_dict(),
        'region_distribution': won_deals['postal_code'].str[:2].value_counts(normalize=True).to_dict(),
        'size_distribution': won_deals['employee_range'].value_counts(normalize=True).to_dict(),
        'legal_form_distribution': won_deals['legal_form'].value_counts(normalize=True).to_dict(),
        'age_distribution': won_deals['created_year'].apply(get_age_range).value_counts(normalize=True).to_dict(),
        'total_won': len(won_deals)
    }
    
    print(f"[DEBUG] Patterns extracted:")
    print(f"  - APE codes: {len(patterns['ape_distribution'])}")
    print(f"  - Regions: {len(patterns['region_distribution'])}")
    print(f"  - Sizes: {len(patterns['size_distribution'])}")
    print(f"  - Legal forms: {len(patterns['legal_form_distribution'])}")
    print(f"  - Age ranges: {len(patterns['age_distribution'])}")
    
    return patterns

async def fetch_sirene_companies(filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch companies from SIRENE API based on filters"""
    if not SIRENE_TOKEN:
        print("[WARNING] No SIRENE token available, returning empty results")
        return []
    
    try:
        # Build SIRENE query
        query_parts = []
        
        if 'ape' in filters and filters['ape']:
            ape_codes = filters['ape']
            if isinstance(ape_codes, list):
                ape_condition = " OR ".join([f"activitePrincipaleUniteLegale:{ape}" for ape in ape_codes])
                query_parts.append(f"({ape_condition})")
            else:
                query_parts.append(f"activitePrincipaleUniteLegale:{ape_codes}")
        
        if 'postal_code' in filters and filters['postal_code']:
            regions = filters['postal_code']
            if isinstance(regions, list):
                region_condition = " OR ".join([f"codePostalEtablissement:{region}*" for region in regions])
                query_parts.append(f"({region_condition})")
            else:
                query_parts.append(f"codePostalEtablissement:{regions}*")
        
        if 'etatAdministratifUniteLegale' in filters:
            query_parts.append(f"etatAdministratifUniteLegale:{filters['etatAdministratifUniteLegale']}")
        
        query = " AND ".join(query_parts) if query_parts else "*"
        
        print(f"[DEBUG] SIRENE query: {query}")
        
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {SIRENE_TOKEN}",
                "Accept": "application/json"
            }
            
            params = {
                "q": query,
                "nombre": min(limit, 1000),
                "debut": 1
            }
            
            response = await client.get(f"{SIRENE_BASE_URL}/siret", headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            companies = []
            
            for etablissement in data.get('etablissements', []):
                unite_legale = etablissement.get('uniteLegale', {})
                adresse = etablissement.get('adresseEtablissement', {})
                
                company = {
                    'siren': unite_legale.get('siren', ''),
                    'siret': etablissement.get('siret', ''),
                    'company_name': unite_legale.get('denominationUniteLegale', ''),
                    'ape': unite_legale.get('activitePrincipaleUniteLegale', ''),
                    'legal_form': unite_legale.get('categorieJuridiqueUniteLegale', ''),
                    'postal_code': adresse.get('codePostalEtablissement', ''),
                    'city': adresse.get('libelleCommuneEtablissement', ''),
                    'employee_range': unite_legale.get('trancheEffectifsUniteLegale', ''),
                    'created_year': int(unite_legale.get('dateCreationUniteLegale', '0')[:4]) if unite_legale.get('dateCreationUniteLegale') else None,
                    'is_active': unite_legale.get('etatAdministratifUniteLegale') == 'A'
                }
                companies.append(company)
            
            print(f"[DEBUG] Fetched {len(companies)} companies from SIRENE")
            return companies
            
    except Exception as e:
        print(f"[ERROR] SIRENE API error: {e}")
        return []

def score_company(company: Dict[str, Any], patterns: Dict[str, Any]) -> float:
    """Score company based on how well it matches the user's success distribution"""
    if patterns['total_won'] == 0:
        return 0.5  # Default score if no patterns learned
    
    score = 0.0
    
    # APE code score (40% weight) - based on actual frequency in user's successful deals
    ape = company.get('ape', '')
    if ape in patterns['ape_distribution']:
        score += 0.4 * patterns['ape_distribution'][ape]
    
    # Region score (30% weight) - based on actual frequency in user's successful deals
    postal_code = company.get('postal_code', '')
    if len(postal_code) >= 2:
        region = postal_code[:2]
        if region in patterns['region_distribution']:
            score += 0.3 * patterns['region_distribution'][region]
    
    # Company size score (20% weight) - based on actual frequency in user's successful deals
    employee_range = company.get('employee_range', '')
    if employee_range in patterns['size_distribution']:
        score += 0.2 * patterns['size_distribution'][employee_range]
    
    # Legal form score (10% weight) - based on actual frequency in user's successful deals
    legal_form = company.get('legal_form', '')
    if legal_form in patterns['legal_form_distribution']:
        score += 0.1 * patterns['legal_form_distribution'][legal_form]
    
    return min(score, 1.0)  # Cap at 1.0

def build_sirene_filters(patterns: Dict[str, Any], min_frequency: float = 0.01) -> Dict[str, Any]:
    """Build SIRENE filters using patterns that appear in at least min_frequency of successful deals"""
    filters = {
        'etatAdministratifUniteLegale': 'A'  # Active companies only
    }
    
    # Get APE codes that appear in at least min_frequency of successful deals
    significant_ape_codes = [
        ape for ape, freq in patterns['ape_distribution'].items() 
        if freq >= min_frequency and ape and ape != ''
    ]
    if significant_ape_codes:
        filters['ape'] = significant_ape_codes
    
    # Get regions that appear in at least min_frequency of successful deals
    significant_regions = [
        region for region, freq in patterns['region_distribution'].items() 
        if freq >= min_frequency and region and region != ''
    ]
    if significant_regions:
        filters['postal_code'] = significant_regions
    
    print(f"[DEBUG] Built SIRENE filters: {filters}")
    return filters

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "sirene-simple",
        "sirene_token_set": bool(SIRENE_TOKEN),
        "sklearn_available": SKLEARN_AVAILABLE,
        "httpx_available": HTTPX_AVAILABLE
    }

@app.post("/train")
def train(req: TrainRequest):
    """Learn patterns from user's training data"""
    try:
        print(f"[DEBUG] Training with {len(req.rows)} rows")
        
        # Convert to DataFrame
        df = pd.DataFrame([r.model_dump() for r in req.rows])
        
        if df.empty:
            return {"ok": False, "error": "no rows"}
        
        # Analyze patterns from user data
        patterns = analyze_user_patterns(df)
        
        # Store patterns for this tenant
        LEARNED_PATTERNS[req.tenant_id] = patterns
        
        return {
            "ok": True,
            "stats": {
                "total_deals": len(df),
                "won_deals": patterns['total_won'],
                "patterns_learned": {
                    "ape_codes": len(patterns['ape_distribution']),
                    "regions": len(patterns['region_distribution']),
                    "sizes": len(patterns['size_distribution']),
                    "legal_forms": len(patterns['legal_form_distribution']),
                    "age_ranges": len(patterns['age_distribution'])
                }
            },
            "model_version": f"{req.tenant_id}-v1-simple"
        }
        
    except Exception as e:
        print(f"[ERROR] Training error: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/discover")
async def discover(req: DiscoverRequest):
    """Discover companies using learned patterns"""
    try:
        print(f"[DEBUG] Discovery request for tenant: {req.tenant_id}")
        
        # Get learned patterns
        patterns = LEARNED_PATTERNS.get(req.tenant_id)
        if not patterns:
            return {"ok": False, "error": "No patterns learned. Please train first."}
        
        # Build SIRENE filters based on patterns
        sirene_filters = build_sirene_filters(patterns, min_frequency=0.01)
        
        # Fetch companies from SIRENE
        companies = await fetch_sirene_companies(sirene_filters, limit=req.limit)
        
        # Score companies based on patterns
        scored_companies = []
        for company in companies:
            score = score_company(company, patterns)
            company['lead_score'] = score
            scored_companies.append(company)
        
        # Sort by score (highest first)
        scored_companies.sort(key=lambda x: x['lead_score'], reverse=True)
        
        return {
            "ok": True,
            "companies": scored_companies[:req.limit],
            "total_found": len(scored_companies),
            "patterns_used": {
                "ape_codes": list(patterns['ape_distribution'].keys())[:10],
                "regions": list(patterns['region_distribution'].keys())[:10],
                "total_patterns": sum(len(p) for p in patterns.values() if isinstance(p, dict))
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Discovery error: {e}")
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
