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
SIRENE_TOKEN = os.getenv("SIRENE_TOKEN", "your-sirene-token-here")
SIRENE_BASE_URL = "https://api.insee.fr/api-sirene/3.11"
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
    """Extract positive and negative patterns from user's deals"""
    print(f"[DEBUG] Analyzing patterns from {len(df)} total deals")
    
    # Clean and prepare data
    df_clean = df.copy()
    df_clean['ape'] = df_clean['ape'].fillna('').astype(str)
    df_clean['postal_code'] = df_clean['postal_code'].fillna('').astype(str)
    df_clean['employee_range'] = df_clean['employee_range'].fillna('').astype(str)
    df_clean['legal_form'] = df_clean['legal_form'].fillna('').astype(str)
    df_clean['created_year'] = pd.to_numeric(df_clean['created_year'], errors='coerce')
    
    # Separate won and lost deals
    won_deals = df_clean[df_clean['deal_status'].str.lower() == 'won']
    lost_deals = df_clean[df_clean['deal_status'].str.lower() == 'lost']
    
    print(f"[DEBUG] Found {len(won_deals)} won deals and {len(lost_deals)} lost deals")
    
    if len(won_deals) == 0 and len(lost_deals) == 0:
        return {
            'positive_patterns': {},
            'negative_patterns': {},
            'total_won': 0,
            'total_lost': 0
        }
    
    # Analyze positive patterns (what's common in won deals)
    positive_patterns = {}
    if len(won_deals) > 0:
        positive_patterns = {
            'ape_distribution': won_deals['ape'].value_counts(normalize=True).to_dict(),
            'region_distribution': won_deals['postal_code'].str[:2].value_counts(normalize=True).to_dict(),
            'size_distribution': won_deals['employee_range'].value_counts(normalize=True).to_dict(),
            'legal_form_distribution': won_deals['legal_form'].value_counts(normalize=True).to_dict(),
            'age_distribution': won_deals['created_year'].apply(get_age_range).value_counts(normalize=True).to_dict()
        }
    
    # Analyze negative patterns (what's common in lost deals)
    negative_patterns = {}
    if len(lost_deals) > 0:
        negative_patterns = {
            'ape_distribution': lost_deals['ape'].value_counts(normalize=True).to_dict(),
            'region_distribution': lost_deals['postal_code'].str[:2].value_counts(normalize=True).to_dict(),
            'size_distribution': lost_deals['employee_range'].value_counts(normalize=True).to_dict(),
            'legal_form_distribution': lost_deals['legal_form'].value_counts(normalize=True).to_dict(),
            'age_distribution': lost_deals['created_year'].apply(get_age_range).value_counts(normalize=True).to_dict()
        }
    
    patterns = {
        'positive_patterns': positive_patterns,
        'negative_patterns': negative_patterns,
        'total_won': len(won_deals),
        'total_lost': len(lost_deals)
    }
    
    print(f"[DEBUG] Positive patterns extracted:")
    for key, value in positive_patterns.items():
        print(f"  - {key}: {len(value)}")
    
    print(f"[DEBUG] Negative patterns extracted:")
    for key, value in negative_patterns.items():
        print(f"  - {key}: {len(value)}")
    
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
        
        params = {
            "q": query,
            "nombre": min(limit, 1000),
            "debut": 1
        }
        
        print(f"[DEBUG] SIRENE query: {query}")
        print(f"[DEBUG] SIRENE URL: {SIRENE_BASE_URL}/siret")
        print(f"[DEBUG] SIRENE params: {params}")
        
        async with httpx.AsyncClient() as client:
            headers = {
                "X-INSEE-Api-Key-Integration": SIRENE_TOKEN,
                "Accept": "application/json"
            }
            
            response = await client.get(f"{SIRENE_BASE_URL}/siret", headers=headers, params=params)
            print(f"[DEBUG] SIRENE response status: {response.status_code}")
            response.raise_for_status()
            
            data = response.json()
            print(f"[DEBUG] SIRENE response data keys: {list(data.keys())}")
            print(f"[DEBUG] SIRENE etablissements count: {len(data.get('etablissements', []))}")
            companies = []
            
            for i, etablissement in enumerate(data.get('etablissements', [])):
                unite_legale = etablissement.get('uniteLegale', {})
                adresse = etablissement.get('adresseEtablissement', {})
                
                # Debug first company structure
                if i == 0:
                    print(f"[DEBUG] First etablissement keys: {list(etablissement.keys())}")
                    print(f"[DEBUG] uniteLegale keys: {list(unite_legale.keys())}")
                    print(f"[DEBUG] adresseEtablissement keys: {list(adresse.keys())}")
                    print(f"[DEBUG] Sample siren: {unite_legale.get('siren', 'MISSING')}")
                    print(f"[DEBUG] Sample company_name: {unite_legale.get('denominationUniteLegale', 'MISSING')}")
                    print(f"[DEBUG] Sample postal_code: {adresse.get('codePostalEtablissement', 'MISSING')}")
                    print(f"[DEBUG] Sample city: {adresse.get('libelleCommuneEtablissement', 'MISSING')}")
                
                # Extract all available data from SIRENE response
                company = {
                    'siren': etablissement.get('siren', ''),  # siren is at establishment level
                    'siret': etablissement.get('siret', ''),
                    'company_name': unite_legale.get('denominationUniteLegale') or 
                                   unite_legale.get('denominationUsuelle1UniteLegale') or 
                                   unite_legale.get('nomUniteLegale') or 
                                   unite_legale.get('sigleUniteLegale') or '',
                    'ape': unite_legale.get('activitePrincipaleUniteLegale', ''),
                    'legal_form': unite_legale.get('categorieJuridiqueUniteLegale', ''),
                    'postal_code': adresse.get('codePostalEtablissement', ''),
                    'city': adresse.get('libelleCommuneEtablissement', ''),
                    'employee_range': unite_legale.get('trancheEffectifsUniteLegale', ''),
                    'created_year': int(unite_legale.get('dateCreationUniteLegale', '0')[:4]) if unite_legale.get('dateCreationUniteLegale') else None,
                    'is_active': unite_legale.get('etatAdministratifUniteLegale') == 'A',
                    # Additional fields from SIRENE
                    'address': f"{adresse.get('numeroVoieEtablissement', '')} {adresse.get('typeVoieEtablissement', '')} {adresse.get('libelleVoieEtablissement', '')}".strip(),
                    'complement_address': adresse.get('complementAdresseEtablissement', ''),
                    'cedex': adresse.get('codeCedexEtablissement', ''),
                    'cedex_label': adresse.get('libelleCedexEtablissement', ''),
                    'commune_code': adresse.get('codeCommuneEtablissement', ''),
                    'is_headquarters': etablissement.get('etablissementSiege', False),
                    'establishment_activity': etablissement.get('activitePrincipaleRegistreMetiersEtablissement', ''),
                    'establishment_employees': etablissement.get('trancheEffectifsEtablissement', ''),
                    'establishment_creation_date': etablissement.get('dateCreationEtablissement', ''),
                    'nic': etablissement.get('nic', ''),
                    'diffusion_status': etablissement.get('statutDiffusionEtablissement', ''),
                    'legal_unit_diffusion_status': unite_legale.get('statutDiffusionUniteLegale', ''),
                    'legal_unit_creation_date': unite_legale.get('dateCreationUniteLegale', ''),
                    'legal_unit_employees': unite_legale.get('trancheEffectifsUniteLegale', ''),
                    'legal_unit_employees_year': unite_legale.get('anneeEffectifsUniteLegale', ''),
                    'is_employer': unite_legale.get('caractereEmployeurUniteLegale', ''),
                    'company_category': unite_legale.get('categorieEntreprise', ''),
                    'company_category_year': unite_legale.get('anneeCategorieEntreprise', ''),
                    'social_economy': unite_legale.get('economieSocialeSolidaireUniteLegale', ''),
                    'mission_company': unite_legale.get('societeMissionUniteLegale', ''),
                    'association_id': unite_legale.get('identifiantAssociationUniteLegale', ''),
                    'nomenclature_activity': unite_legale.get('nomenclatureActivitePrincipaleUniteLegale', ''),
                    'headquarters_nic': unite_legale.get('nicSiegeUniteLegale', ''),
                    'last_processing_date': unite_legale.get('dateDernierTraitementUniteLegale', ''),
                    'establishment_last_processing_date': etablissement.get('dateDernierTraitementEtablissement', ''),
                    'periods_count': etablissement.get('nombrePeriodesEtablissement', 0)
                }
                companies.append(company)
            
            print(f"[DEBUG] Fetched {len(companies)} companies from SIRENE")
            return companies
            
    except Exception as e:
        print(f"[ERROR] SIRENE API error: {e}")
        return []

def score_company(company: Dict[str, Any], patterns: Dict[str, Any]) -> float:
    """Score company based on positive and negative patterns with weighted scoring"""
    if patterns['total_won'] == 0 and patterns['total_lost'] == 0:
        return 0.5  # Default score if no patterns learned
    
    positive_score = 0.0
    negative_score = 0.0
    total_weight = 0.0
    
    # Get positive and negative patterns
    pos_patterns = patterns.get('positive_patterns', {})
    neg_patterns = patterns.get('negative_patterns', {})
    
    # APE code scoring (50% weight - most important)
    ape = company.get('ape', '')
    if ape in pos_patterns.get('ape_distribution', {}):
        ape_freq = pos_patterns['ape_distribution'][ape]
        positive_score += 0.5 * ape_freq
        total_weight += 0.5
    if ape in neg_patterns.get('ape_distribution', {}):
        ape_freq = neg_patterns['ape_distribution'][ape]
        negative_score += 0.5 * ape_freq
    
    # Region scoring (30% weight)
    postal_code = company.get('postal_code', '')
    if len(postal_code) >= 2:
        region = postal_code[:2]
        if region in pos_patterns.get('region_distribution', {}):
            region_freq = pos_patterns['region_distribution'][region]
            positive_score += 0.3 * region_freq
            total_weight += 0.3
        if region in neg_patterns.get('region_distribution', {}):
            region_freq = neg_patterns['region_distribution'][region]
            negative_score += 0.3 * region_freq
    
    # Company size scoring (15% weight)
    employee_range = company.get('employee_range', '')
    if employee_range in pos_patterns.get('size_distribution', {}):
        size_freq = pos_patterns['size_distribution'][employee_range]
        positive_score += 0.15 * size_freq
        total_weight += 0.15
    if employee_range in neg_patterns.get('size_distribution', {}):
        size_freq = neg_patterns['size_distribution'][employee_range]
        negative_score += 0.15 * size_freq
    
    # Legal form scoring (5% weight)
    legal_form = company.get('legal_form', '')
    if legal_form in pos_patterns.get('legal_form_distribution', {}):
        legal_freq = pos_patterns['legal_form_distribution'][legal_form]
        positive_score += 0.05 * legal_freq
        total_weight += 0.05
    if legal_form in neg_patterns.get('legal_form_distribution', {}):
        legal_freq = neg_patterns['legal_form_distribution'][legal_form]
        negative_score += 0.05 * legal_freq
    
    # Calculate final score: positive patterns boost, negative patterns reduce
    if total_weight > 0:
        # Normalize by total weight applied
        final_score = (positive_score - (negative_score * 0.3)) / total_weight
    else:
        final_score = 0.5  # Default if no patterns match
    
    # Ensure score is between 0 and 1
    final_score = max(0.0, min(1.0, final_score))
    
    return final_score

def build_sirene_filters(patterns: Dict[str, Any], min_frequency: float = 0.1) -> Dict[str, Any]:
    """Build SIRENE filters using only the strongest positive patterns (high win probability)"""
    filters = {
        'etatAdministratifUniteLegale': 'A'  # Active companies only
    }
    
    # Use only the strongest positive patterns (what's most common in won deals)
    pos_patterns = patterns.get('positive_patterns', {})
    
    # Get APE codes that appear in at least min_frequency of won deals (default 10%)
    significant_ape_codes = [
        ape for ape, freq in pos_patterns.get('ape_distribution', {}).items() 
        if freq >= min_frequency and ape and ape != ''
    ]
    if significant_ape_codes:
        filters['ape'] = significant_ape_codes
        print(f"[DEBUG] Filtering by winning APE codes: {significant_ape_codes}")
    
    # Get regions that appear in at least min_frequency of won deals (default 10%)
    significant_regions = [
        region for region, freq in pos_patterns.get('region_distribution', {}).items() 
        if freq >= min_frequency and region and region != ''
    ]
    if significant_regions:
        filters['postal_code'] = significant_regions
        print(f"[DEBUG] Filtering by winning regions: {significant_regions}")
    
    # If we have both APE and region filters, we're being very specific
    if 'ape' in filters and 'postal_code' in filters:
        print(f"[DEBUG] Using precise filters: APE {filters['ape']} + Region {filters['postal_code']}")
    
    print(f"[DEBUG] Built SIRENE filters from positive patterns: {filters}")
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

async def enrich_user_data_with_sirene(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich user data with SIRENE information"""
    print(f"[DEBUG] Enriching {len(df)} companies with SIRENE data...")
    
    enriched_rows = []
    for _, row in df.iterrows():
        company_name = row.get('company_name', '')
        siren = row.get('siren', '')
        
        # Try to find company in SIRENE
        sirene_data = None
        if siren:
            # Search by SIREN
            sirene_data = await search_sirene_by_siren(siren)
        elif company_name:
            # Search by company name
            sirene_data = await search_sirene_by_name(company_name)
        
        # Merge user data with SIRENE data
        enriched_row = row.to_dict()
        if sirene_data:
            enriched_row.update(sirene_data)
            print(f"[DEBUG] Enriched {company_name} with SIRENE data")
        else:
            print(f"[DEBUG] No SIRENE data found for {company_name}")
        
        enriched_rows.append(enriched_row)
    
    enriched_df = pd.DataFrame(enriched_rows)
    print(f"[DEBUG] Enriched dataset shape: {enriched_df.shape}")
    return enriched_df

async def search_sirene_by_siren(siren: str) -> dict:
    """Search SIRENE by SIREN number"""
    if not SIRENE_TOKEN or not siren:
        return {}
    
    try:
        query = f"siren:{siren}"
        params = {
            "q": query,
            "nombre": 1,
            "debut": 1
        }
        
        async with httpx.AsyncClient() as client:
            headers = {
                "X-INSEE-Api-Key-Integration": SIRENE_TOKEN,
                "Accept": "application/json"
            }
            
            response = await client.get(f"{SIRENE_BASE_URL}/siret", headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                etablissements = data.get('etablissements', [])
                if etablissements:
                    return extract_sirene_data(etablissements[0])
    except Exception as e:
        print(f"[ERROR] SIRENE search by SIREN failed: {e}")
    
    return {}

async def search_sirene_by_name(company_name: str) -> dict:
    """Search SIRENE by company name"""
    if not SIRENE_TOKEN or not company_name:
        return {}
    
    try:
        # Clean company name for search
        clean_name = company_name.replace("'", " ").replace("-", " ").strip()
        query = f'denominationUniteLegale:"{clean_name}"'
        params = {
            "q": query,
            "nombre": 1,
            "debut": 1
        }
        
        async with httpx.AsyncClient() as client:
            headers = {
                "X-INSEE-Api-Key-Integration": SIRENE_TOKEN,
                "Accept": "application/json"
            }
            
            response = await client.get(f"{SIRENE_BASE_URL}/siret", headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                etablissements = data.get('etablissements', [])
                if etablissements:
                    return extract_sirene_data(etablissements[0])
    except Exception as e:
        print(f"[ERROR] SIRENE search by name failed: {e}")
    
    return {}

def extract_sirene_data(etablissement: dict) -> dict:
    """Extract SIRENE data from etablissement response"""
    unite_legale = etablissement.get('uniteLegale', {})
    adresse = etablissement.get('adresseEtablissement', {})
    
    return {
        'siren': etablissement.get('siren', ''),
        'siret': etablissement.get('siret', ''),
        'company_name': unite_legale.get('denominationUniteLegale') or 
                       unite_legale.get('denominationUsuelle1UniteLegale') or 
                       unite_legale.get('nomUniteLegale') or 
                       unite_legale.get('sigleUniteLegale') or '',
        'ape': unite_legale.get('activitePrincipaleUniteLegale', ''),
        'legal_form': unite_legale.get('categorieJuridiqueUniteLegale', ''),
        'postal_code': adresse.get('codePostalEtablissement', ''),
        'city': adresse.get('libelleCommuneEtablissement', ''),
        'employee_range': unite_legale.get('trancheEffectifsUniteLegale', ''),
        'created_year': int(unite_legale.get('dateCreationUniteLegale', '0')[:4]) if unite_legale.get('dateCreationUniteLegale') else None,
        'is_active': unite_legale.get('etatAdministratifUniteLegale') == 'A',
        'company_category': unite_legale.get('categorieEntreprise', ''),
        'social_economy': unite_legale.get('economieSocialeSolidaireUniteLegale', ''),
        'mission_company': unite_legale.get('societeMissionUniteLegale', ''),
        'is_employer': unite_legale.get('caractereEmployeurUniteLegale', ''),
        'nomenclature_activity': unite_legale.get('nomenclatureActivitePrincipaleUniteLegale', ''),
        'headquarters_nic': unite_legale.get('nicSiegeUniteLegale', ''),
        'legal_unit_creation_date': unite_legale.get('dateCreationUniteLegale', ''),
        'legal_unit_employees_year': unite_legale.get('anneeEffectifsUniteLegale', ''),
        'company_category_year': unite_legale.get('anneeCategorieEntreprise', ''),
        'legal_unit_diffusion_status': unite_legale.get('statutDiffusionUniteLegale', ''),
        'last_processing_date': unite_legale.get('dateDernierTraitementUniteLegale', ''),
        'establishment_creation_date': etablissement.get('dateCreationEtablissement', ''),
        'establishment_employees': etablissement.get('trancheEffectifsEtablissement', ''),
        'establishment_activity': etablissement.get('activitePrincipaleRegistreMetiersEtablissement', ''),
        'is_headquarters': etablissement.get('etablissementSiege', False),
        'nic': etablissement.get('nic', ''),
        'diffusion_status': etablissement.get('statutDiffusionEtablissement', ''),
        'establishment_last_processing_date': etablissement.get('dateDernierTraitementEtablissement', ''),
        'periods_count': etablissement.get('nombrePeriodesEtablissement', 0)
    }

async def find_similar_companies(patterns: dict, limit: int = 50) -> pd.DataFrame:
    """Find similar companies in SIRENE based on learned patterns"""
    print(f"[DEBUG] Finding similar companies based on patterns...")
    
    # Build filters for similar companies
    pos_patterns = patterns.get('positive_patterns', {})
    neg_patterns = patterns.get('negative_patterns', {})
    
    # Get top positive patterns
    top_ape_codes = sorted(pos_patterns.get('ape_distribution', {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_regions = sorted(pos_patterns.get('region_distribution', {}).items(), key=lambda x: x[1], reverse=True)[:3]
    
    similar_companies = []
    
    # Search for companies with similar APE codes
    for ape_code, freq in top_ape_codes:
        if freq > 0.1:  # Only if significant
            filters = {
                'etatAdministratifUniteLegale': 'A',
                'ape': [ape_code]
            }
            companies = await fetch_sirene_companies(filters, limit=limit//len(top_ape_codes))
            for company in companies:
                company['similarity_reason'] = f"Similar APE code {ape_code}"
                similar_companies.append(company)
    
    # Search for companies in similar regions
    for region, freq in top_regions:
        if freq > 0.1:  # Only if significant
            filters = {
                'etatAdministratifUniteLegale': 'A',
                'postal_code': [region]
            }
            companies = await fetch_sirene_companies(filters, limit=limit//len(top_regions))
            for company in companies:
                company['similarity_reason'] = f"Similar region {region}"
                similar_companies.append(company)
    
    # Remove duplicates and convert to DataFrame
    unique_companies = []
    seen_sirens = set()
    for company in similar_companies:
        siren = company.get('siren', '')
        if siren and siren not in seen_sirens:
            seen_sirens.add(siren)
            unique_companies.append(company)
    
    similar_df = pd.DataFrame(unique_companies)
    print(f"[DEBUG] Found {len(similar_df)} similar companies")
    return similar_df

@app.post("/train")
async def train(req: TrainRequest):
    """Enhanced training with SIRENE enrichment and similarity expansion"""
    try:
        print(f"[DEBUG] Enhanced training with {len(req.rows)} rows")
        
        # Convert to DataFrame
        df = pd.DataFrame([r.model_dump() for r in req.rows])
        print(f"[DEBUG] Original DataFrame shape: {df.shape}")
        
        if df.empty:
            return {"ok": False, "error": "no rows"}
        
        # Step 1: Enrich user data with SIRENE information
        enriched_df = await enrich_user_data_with_sirene(df)
        
        # Step 2: Analyze patterns from enriched data
        patterns = analyze_user_patterns(enriched_df)
        print(f"[DEBUG] Patterns extracted from enriched data")
        
        # Step 3: Find similar companies based on patterns
        similar_df = await find_similar_companies(patterns, limit=100)
        
        # Step 4: Combine original + enriched + similar data
        # Add similarity labels to similar companies
        if len(similar_df) > 0:
            # For similar companies, we'll use the pattern analysis to determine if they should be positive or negative
            # This is a simplified approach - in practice, you might want more sophisticated similarity scoring
            similar_df['deal_status'] = 'similar'  # Mark as similar for now
        
        # Step 5: Re-analyze patterns on the expanded dataset
        if len(similar_df) > 0:
            # Combine enriched data with similar companies
            combined_df = pd.concat([enriched_df, similar_df], ignore_index=True)
            print(f"[DEBUG] Combined dataset shape: {combined_df.shape}")
            
            # Re-analyze patterns on the expanded dataset
            final_patterns = analyze_user_patterns(combined_df)
        else:
            final_patterns = patterns
        
        # Store patterns for this tenant
        LEARNED_PATTERNS[req.tenant_id] = final_patterns
        
        # Automatically discover companies using the learned patterns
        print(f"[DEBUG] Auto-discovering companies after training...")
        try:
            # Build SIRENE filters from positive patterns (only strong patterns)
            sirene_filters = build_sirene_filters(patterns, min_frequency=0.1)
            print(f"[DEBUG] Built SIRENE filters: {sirene_filters}")
            
            # Fetch companies from SIRENE (already filtered by winning patterns)
            companies = await fetch_sirene_companies(sirene_filters, limit=20)
            print(f"[DEBUG] Fetched {len(companies)} companies from SIRENE (pre-filtered by winning patterns)")
            
            # Score companies (they should already be high probability due to server-side filtering)
            scored_companies = []
            for company in companies:
                score = score_company(company, patterns)
                
                # Build reasons based on pattern matches
                reasons = []
                ape = company.get("ape", "")
                region = company.get("postal_code", "")[:2] if company.get("postal_code") else ""
                
                # Build detailed reasons based on pattern matches
                if ape in patterns['positive_patterns'].get('ape_distribution', {}):
                    ape_freq = patterns['positive_patterns']['ape_distribution'][ape]
                    reasons.append(f"Winning APE code {ape} ({ape_freq:.1%} of wins)")
                
                if region in patterns['positive_patterns'].get('region_distribution', {}):
                    region_freq = patterns['positive_patterns']['region_distribution'][region]
                    reasons.append(f"Winning region {region} ({region_freq:.1%} of wins)")
                
                # Add company size if available
                employee_range = company.get("employee_range", "")
                if employee_range in patterns['positive_patterns'].get('size_distribution', {}):
                    size_freq = patterns['positive_patterns']['size_distribution'][employee_range]
                    reasons.append(f"Winning size {employee_range} ({size_freq:.1%} of wins)")
                
                # Add legal form if available
                legal_form = company.get("legal_form", "")
                if legal_form in patterns['positive_patterns'].get('legal_form_distribution', {}):
                    legal_freq = patterns['positive_patterns']['legal_form_distribution'][legal_form]
                    reasons.append(f"Winning legal form {legal_form} ({legal_freq:.1%} of wins)")
                
                # Get location info using SIRENE postal code
                city = company.get("city", "")
                postal_code = company.get("postal_code", "")
                
                # Use postal code as primary location identifier
                if postal_code:
                    location = f"{city}, {postal_code}" if city else postal_code
                elif city:
                    location = city
                else:
                    location = "N/A"
                
                # Calculate derived fields
                siren = company.get("siren", "")
                nic_siege = company.get("headquarters_nic", "")
                seat_siret = f"{siren}{nic_siege}" if siren and nic_siege else ""
                
                # Company size labels
                headcount_codes = {
                    "NN": "Non employeur / inconnu",
                    "00": "0 salarié (N)",
                    "01": "1–2",
                    "02": "3–5", 
                    "03": "6–9",
                    "11": "10–19",
                    "12": "20–49",
                    "21": "50–99",
                    "22": "100–199",
                    "31": "200–249",
                    "32": "250–499",
                    "41": "500–999",
                    "42": "1 000–1 999",
                    "51": "2 000–4 999",
                    "52": "5 000–9 999",
                    "53": "10 000+"
                }
                
                employee_range = company.get("employee_range", "")
                headcount_label = headcount_codes.get(employee_range, employee_range)
                
                # Company category labels
                company_category = company.get("company_category", "")
                company_size_label = "PME" if company_category == "PME" else "ETI" if company_category == "ETI" else "GE" if company_category == "GE" else company_category
                
                scored_companies.append({
                    # Core identification
                    "name": company.get("company_name", "N/A"),
                    "siren": siren,
                    "siret": company.get("siret", "N/A"),
                    "ape": ape,
                    "region": region,
                    "location": location,
                    
                    # Legal unit data
                    "denominationUniteLegale": company.get("company_name", "N/A"),
                    "denominationUsuelle1UniteLegale": company.get("denominationUsuelle1UniteLegale", ""),
                    "denominationUsuelle2UniteLegale": company.get("denominationUsuelle2UniteLegale", ""),
                    "denominationUsuelle3UniteLegale": company.get("denominationUsuelle3UniteLegale", ""),
                    "sigleUniteLegale": company.get("sigleUniteLegale", ""),
                    "categorieJuridiqueUniteLegale": company.get("legal_form", "N/A"),
                    "activitePrincipaleUniteLegale": ape,
                    "nomenclatureActivitePrincipaleUniteLegale": company.get("nomenclature_activity", ""),
                    "categorieEntreprise": company_category,
                    "anneeCategorieEntreprise": company.get("company_category_year", ""),
                    "trancheEffectifsUniteLegale": employee_range,
                    "anneeEffectifsUniteLegale": company.get("legal_unit_employees_year", ""),
                    "economieSocialeSolidaireUniteLegale": company.get("social_economy", ""),
                    "societeMissionUniteLegale": company.get("mission_company", ""),
                    "etatAdministratifUniteLegale": "A" if company.get("is_active", False) else "C",
                    "dateCreationUniteLegale": company.get("legal_unit_creation_date", ""),
                    "dateDernierTraitementUniteLegale": company.get("last_processing_date", ""),
                    "nicSiegeUniteLegale": nic_siege,
                    "statutDiffusionUniteLegale": company.get("legal_unit_diffusion_status", ""),
                    "unitePurgeeUniteLegale": company.get("unitePurgeeUniteLegale", ""),
                    
                    # Derived fields
                    "seatSiret": seat_siret,
                    "isActive": company.get("is_active", False),
                    "isESS": company.get("social_economy") == "O",
                    "isMissionCompany": company.get("mission_company") == "O",
                    "isDiffusionPartialUL": company.get("legal_unit_diffusion_status") == "P",
                    "companySizeLabel": company_size_label,
                    "headcountLabelUL": headcount_label,
                    
                    # Scoring
                    "win_score": score,
                    "band": "High" if score > 0.8 else "Medium" if score > 0.6 else "Low",
                    "confidence_badge": f"{score:.1%}",
                    "reasons": reasons if reasons else [f"Matches winning patterns (score: {score:.1%})"],
                    "source": "SIRENE"
                })
            
            # Sort by score (highest first)
            scored_companies.sort(key=lambda x: x["win_score"], reverse=True)
            
            print(f"[DEBUG] Returned {len(scored_companies)} companies (pre-filtered by SIRENE)")
            
            return {
                "ok": True,
                "stats": {
                    "rows": len(df),
                    "wins": patterns['total_won'],
                    "losses": patterns['total_lost'],
                    "total_deals": len(df),
                    "won_deals": patterns['total_won'],
                    "lost_deals": patterns['total_lost'],
                    "positive_patterns": {
                        "ape_codes": len(patterns['positive_patterns'].get('ape_distribution', {})),
                        "regions": len(patterns['positive_patterns'].get('region_distribution', {})),
                        "sizes": len(patterns['positive_patterns'].get('size_distribution', {})),
                        "legal_forms": len(patterns['positive_patterns'].get('legal_form_distribution', {})),
                        "age_ranges": len(patterns['positive_patterns'].get('age_distribution', {}))
                    },
                    "negative_patterns": {
                        "ape_codes": len(patterns['negative_patterns'].get('ape_distribution', {})),
                        "regions": len(patterns['negative_patterns'].get('region_distribution', {})),
                        "sizes": len(patterns['negative_patterns'].get('size_distribution', {})),
                        "legal_forms": len(patterns['negative_patterns'].get('legal_form_distribution', {})),
                        "age_ranges": len(patterns['negative_patterns'].get('age_distribution', {}))
                    }
                },
                "model_version": f"{req.tenant_id}-v1-simple",
                "discovered_leads": scored_companies,
                "message": f"Model trained successfully! Found {len(scored_companies)} matching companies from SIRENE."
            }
            
        except Exception as e:
            print(f"[ERROR] Auto-discovery failed: {e}")
            return {
                "ok": True,
                "stats": {
                    "rows": len(df),
                    "wins": patterns['total_won'],
                    "losses": patterns['total_lost'],
                    "total_deals": len(df),
                    "won_deals": patterns['total_won'],
                    "lost_deals": patterns['total_lost'],
                    "positive_patterns": {
                        "ape_codes": len(patterns['positive_patterns'].get('ape_distribution', {})),
                        "regions": len(patterns['positive_patterns'].get('region_distribution', {})),
                        "sizes": len(patterns['positive_patterns'].get('size_distribution', {})),
                        "legal_forms": len(patterns['positive_patterns'].get('legal_form_distribution', {})),
                        "age_ranges": len(patterns['positive_patterns'].get('age_distribution', {}))
                    },
                    "negative_patterns": {
                        "ape_codes": len(patterns['negative_patterns'].get('ape_distribution', {})),
                        "regions": len(patterns['negative_patterns'].get('region_distribution', {})),
                        "sizes": len(patterns['negative_patterns'].get('size_distribution', {})),
                        "legal_forms": len(patterns['negative_patterns'].get('legal_form_distribution', {})),
                        "age_ranges": len(patterns['negative_patterns'].get('age_distribution', {}))
                    }
                },
                "model_version": f"{req.tenant_id}-v1-simple",
                "discovered_leads": [],
                "message": f"Model trained successfully, but auto-discovery failed: {str(e)}"
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
                "positive_ape_codes": list(patterns['positive_patterns'].get('ape_distribution', {}).keys())[:10],
                "positive_regions": list(patterns['positive_patterns'].get('region_distribution', {}).keys())[:10],
                "negative_ape_codes": list(patterns['negative_patterns'].get('ape_distribution', {}).keys())[:10],
                "negative_regions": list(patterns['negative_patterns'].get('region_distribution', {}).keys())[:10],
                "total_positive_patterns": sum(len(p) for p in patterns['positive_patterns'].values() if isinstance(p, dict)),
                "total_negative_patterns": sum(len(p) for p in patterns['negative_patterns'].values() if isinstance(p, dict))
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Discovery error: {e}")
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
