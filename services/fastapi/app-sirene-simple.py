#!/usr/bin/env python3
"""
Enhanced SIRENE-based lead scoring with data enrichment and age analysis
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
import re

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

# Legal Form Mapping (INSEE categorization)
LEGAL_FORM_MAPPING = {
    "5499": "Autres formes juridiques",
    "5710": "SAS, société par actions simplifiée",
    "5599": "Autres sociétés par actions",
    "9220": "Association déclarée",
    "7410": "SASU, société par actions simplifiée unipersonnelle",
    "5499": "Autres formes juridiques",
    "5710": "SAS, société par actions simplifiée",
    "5599": "Autres sociétés par actions",
    "9220": "Association déclarée",
    "7410": "SASU, société par actions simplifiée unipersonnelle",
    "5499": "Autres formes juridiques",
    "5710": "SAS, société par actions simplifiée",
    "5599": "Autres sociétés par actions",
    "9220": "Association déclarée",
    "7410": "SASU, société par actions simplifiée unipersonnelle"
}

# APE Code Mapping (Key business activities)
APE_CODE_MAPPING = {
    "62.01Z": "Programmation informatique",
    "62.02A": "Conseil en systèmes et logiciels informatiques",
    "63.11Z": "Traitement de données, hébergement et activités connexes",
    "70.10Z": "Activités des sièges sociaux",
    "55.10Z": "Hôtels et hébergement similaire",
    "58.29C": "Édition de jeux électroniques",
    "72.19Z": "Recherche-développement en autres sciences physiques et naturelles",
    "71.12Z": "Activités d'architecture",
    "71.11Z": "Activités d'architecture et d'ingénierie",
    "62.03Z": "Gestion d'installations informatiques",
    "63.12Z": "Portails Internet",
    "63.91Z": "Activités des agences de presse",
    "63.99Z": "Autres services d'information"
}

# Employee Range Mapping
EMPLOYEE_RANGE_MAPPING = {
    "NN": "Unité non-employeuse ou présumée non-employeuse",
    "00": "0 salarié (ayant employé au cours de l'année)",
    "01": "1 ou 2 salariés",
    "02": "3 à 5 salariés", 
    "03": "6 à 9 salariés",
    "11": "10 à 19 salariés",
    "12": "20 à 49 salariés",
    "21": "50 à 99 salariés",
    "22": "100 à 199 salariés",
    "31": "200 à 249 salariés",
    "32": "250 à 499 salariés",
    "41": "500 à 999 salariés",
    "42": "1 000 à 1 999 salariés",
    "51": "2 000 à 4 999 salariés",
    "52": "5 000 à 9 999 salariés",
    "53": "10 000 salariés et plus"
}

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
    filters: Dict[str, Any] = {}
    limit: int = 100

class DiscoverResponse(BaseModel):
    companies: List[Dict[str, Any]]
    total_found: int
    patterns_used: Dict[str, Any]

def calculate_age_and_category(creation_date: str) -> tuple[int, str]:
    """Calculate company age and category from creation date"""
    if not creation_date:
        return None, "unknown"
    
    try:
        # Parse date (format: YYYY-MM-DD)
        creation_year = int(creation_date[:4])
        current_year = datetime.now().year
        age = current_year - creation_year
        
        if age < 2:
            category = "startup"
        elif age < 5:
            category = "growth"
        elif age < 10:
            category = "mature"
        else:
            category = "established"
            
        return age, category
    except:
        return None, "unknown"

def get_legal_form_description(code: str) -> str:
    """Get human-readable legal form description"""
    return LEGAL_FORM_MAPPING.get(code, f"Forme juridique {code}")

def get_ape_description(code: str) -> str:
    """Get human-readable APE activity description"""
    return APE_CODE_MAPPING.get(code, f"Activité {code}")

def get_employee_description(code: str) -> str:
    """Get human-readable employee range description"""
    return EMPLOYEE_RANGE_MAPPING.get(code, f"Effectif {code}")

def clean_company_name(name: str) -> str:
    """Clean company name for Sirene search"""
    if not name:
        return ""
    
    # Remove common suffixes and clean up
    cleaned = name.upper().strip()
    cleaned = re.sub(r'\s+(SAS|SARL|SA|SASU|EURL|SNC|SCI|ASSOCIATION|ASSOC|LTD|INC|CORP)$', '', cleaned)
    cleaned = re.sub(r'[^\w\s\-&]', ' ', cleaned)  # Remove special chars except & and -
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Normalize spaces
    
    return cleaned

def analyze_user_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract positive and negative patterns from user's deals with enhanced analysis"""
    print(f"[DEBUG] Analyzing patterns from {len(df)} total deals")
    
    # Clean and prepare data
    df_clean = df.copy()
    df_clean['ape'] = df_clean['ape'].fillna('').astype(str)
    df_clean['postal_code'] = df_clean['postal_code'].fillna('').astype(str)
    df_clean['employee_range'] = df_clean['employee_range'].fillna('').astype(str)
    df_clean['legal_form'] = df_clean['legal_form'].fillna('').astype(str)
    df_clean['company_category'] = df_clean['company_category'].fillna('').astype(str)
    df_clean['age_category'] = df_clean['age_category'].fillna('').astype(str)
    
    # Separate won and lost deals
    won_deals = df_clean[df_clean['deal_status'].str.lower() == 'won']
    lost_deals = df_clean[df_clean['deal_status'].str.lower() == 'lost']
    
    print(f"[DEBUG] Found {len(won_deals)} won deals and {len(lost_deals)} lost deals")
    
    if len(won_deals) == 0 and len(lost_deals) == 0:
        return {
            'positive_patterns': {},
            'negative_patterns': {},
            'total_won': 0,
            'total_lost': 0,
            'age_analysis': {},
            'company_category_analysis': {}
        }
    
    # Analyze positive patterns (what's common in won deals)
    positive_patterns = {}
    if len(won_deals) > 0:
        positive_patterns = {
            'ape_distribution': won_deals['ape'].value_counts(normalize=True).to_dict(),
            'region_distribution': won_deals['postal_code'].str[:2].value_counts(normalize=True).to_dict(),
            'size_distribution': won_deals['employee_range'].value_counts(normalize=True).to_dict(),
            'legal_form_distribution': won_deals['legal_form'].value_counts(normalize=True).to_dict(),
            'age_distribution': won_deals['age_category'].value_counts(normalize=True).to_dict(),
            'company_category_distribution': won_deals['company_category'].value_counts(normalize=True).to_dict()
        }
    
    # Analyze negative patterns (what's common in lost deals)
    negative_patterns = {}
    if len(lost_deals) > 0:
        negative_patterns = {
            'ape_distribution': lost_deals['ape'].value_counts(normalize=True).to_dict(),
            'region_distribution': lost_deals['postal_code'].str[:2].value_counts(normalize=True).to_dict(),
            'size_distribution': lost_deals['employee_range'].value_counts(normalize=True).to_dict(),
            'legal_form_distribution': lost_deals['legal_form'].value_counts(normalize=True).to_dict(),
            'age_distribution': lost_deals['age_category'].value_counts(normalize=True).to_dict(),
            'company_category_distribution': lost_deals['company_category'].value_counts(normalize=True).to_dict()
        }
    
    # Age analysis
    age_analysis = {}
    if len(won_deals) > 0 and len(lost_deals) > 0:
        won_age_dist = won_deals['age_category'].value_counts(normalize=True)
        lost_age_dist = lost_deals['age_category'].value_counts(normalize=True)
        
        age_analysis = {
            'won_age_distribution': won_age_dist.to_dict(),
            'lost_age_distribution': lost_age_dist.to_dict(),
            'age_significance': {}
        }
        
        # Calculate significance for each age category
        for age_cat in set(won_age_dist.index) | set(lost_age_dist.index):
            won_pct = won_age_dist.get(age_cat, 0)
            lost_pct = lost_age_dist.get(age_cat, 0)
            total_pct = won_pct + lost_pct
            
            if total_pct > 0:
                significance = abs(won_pct - lost_pct) / total_pct
                age_analysis['age_significance'][age_cat] = {
                    'significance': significance,
                    'won_percentage': won_pct,
                    'lost_percentage': lost_pct,
                    'is_significant': significance > 0.2  # 20% difference threshold
                }
    
    # Company category analysis
    company_category_analysis = {}
    if len(won_deals) > 0 and len(lost_deals) > 0:
        won_cat_dist = won_deals['company_category'].value_counts(normalize=True)
        lost_cat_dist = lost_deals['company_category'].value_counts(normalize=True)
        
        company_category_analysis = {
            'won_category_distribution': won_cat_dist.to_dict(),
            'lost_category_distribution': lost_cat_dist.to_dict(),
            'category_significance': {}
        }
        
        # Calculate significance for each company category
        for cat in set(won_cat_dist.index) | set(lost_cat_dist.index):
            won_pct = won_cat_dist.get(cat, 0)
            lost_pct = lost_cat_dist.get(cat, 0)
            total_pct = won_pct + lost_pct
            
            if total_pct > 0:
                significance = abs(won_pct - lost_pct) / total_pct
                company_category_analysis['category_significance'][cat] = {
                    'significance': significance,
                    'won_percentage': won_pct,
                    'lost_percentage': lost_pct,
                    'is_significant': significance > 0.2
                }
    
    patterns = {
        'positive_patterns': positive_patterns,
        'negative_patterns': negative_patterns,
        'total_won': len(won_deals),
        'total_lost': len(lost_deals),
        'age_analysis': age_analysis,
        'company_category_analysis': company_category_analysis
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
                
                # Extract and enrich data from SIRENE response
                ape_code = unite_legale.get('activitePrincipaleUniteLegale', '')
                legal_form_code = unite_legale.get('categorieJuridiqueUniteLegale', '')
                employee_range = unite_legale.get('trancheEffectifsUniteLegale', '')
                creation_date = unite_legale.get('dateCreationUniteLegale', '')
                age_years, age_category = calculate_age_and_category(creation_date)
                
                company = {
                    # Core identification
                    'siren': etablissement.get('siren', ''),
                    'siret': etablissement.get('siret', ''),
                    'company_name': unite_legale.get('denominationUniteLegale') or 
                                   unite_legale.get('denominationUsuelle1UniteLegale') or 
                                   unite_legale.get('nomUniteLegale') or 
                                   unite_legale.get('sigleUniteLegale') or '',
                    
                    # Activity and legal information
                    'ape': ape_code,
                    'ape_description': get_ape_description(ape_code),
                    'legal_form': legal_form_code,
                    'legal_form_description': get_legal_form_description(legal_form_code),
                    
                    # Location information
                    'postal_code': adresse.get('codePostalEtablissement', ''),
                    'city': adresse.get('libelleCommuneEtablissement', ''),
                    'region': adresse.get('codePostalEtablissement', '')[:2] if adresse.get('codePostalEtablissement') else '',
                    'location': f"{adresse.get('libelleCommuneEtablissement', '')}, {adresse.get('codePostalEtablissement', '')}".strip(', '),
                    
                    # Company characteristics
                    'employee_range': employee_range,
                    'employee_description': get_employee_description(employee_range),
                    'age_years': age_years,
                    'age_category': age_category,
                    'creation_date': creation_date,
                    'is_active': unite_legale.get('etatAdministratifUniteLegale') == 'A',
                    'is_ess': unite_legale.get('economieSocialeSolidaireUniteLegale') == 'O',
                    'is_mission_company': unite_legale.get('societeMissionUniteLegale') == 'O',
                    
                    # Company category and size
                    'company_category': unite_legale.get('categorieEntreprise', ''),
                    'company_category_year': unite_legale.get('anneeCategorieEntreprise', ''),
                    
                    # Additional useful fields
                    'is_headquarters': etablissement.get('etablissementSiege', False),
                    'association_id': unite_legale.get('identifiantAssociationUniteLegale', ''),
                    'last_processing_date': unite_legale.get('dateDernierTraitementUniteLegale', '')
                }
                companies.append(company)
            
            print(f"[DEBUG] Fetched {len(companies)} companies from SIRENE")
            return companies
            
    except Exception as e:
        print(f"[ERROR] SIRENE API error: {e}")
        return []

def generate_hypotheses(patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Generate hypotheses based on pattern analysis"""
    hypotheses = {
        'strong_patterns': [],
        'moderate_patterns': [],
        'weak_patterns': []
    }
    
    # Analyze APE patterns
    ape_pos = patterns.get('positive_patterns', {}).get('ape_distribution', {})
    ape_neg = patterns.get('negative_patterns', {}).get('ape_distribution', {})
    
    for ape_code, freq in ape_pos.items():
        if ape_code and freq > 0.3:  # 30% threshold for strong pattern
            neg_freq = ape_neg.get(ape_code, 0)
            if neg_freq < freq * 0.5:  # Much more common in won deals
                confidence = "high" if freq > 0.5 else "medium"
                hypotheses[f"{confidence}_patterns"].append({
                    'pattern': f"Companies with APE code {ape_code} ({get_ape_description(ape_code)}) tend to win more deals",
                    'confidence': confidence,
                    'evidence': f"{freq:.1%} of won deals vs {neg_freq:.1%} of lost deals",
                    'category': 'activity'
                })
    
    # Analyze company category patterns
    cat_pos = patterns.get('positive_patterns', {}).get('company_category_distribution', {})
    cat_neg = patterns.get('negative_patterns', {}).get('company_category_distribution', {})
    
    for category, freq in cat_pos.items():
        if category and freq > 0.3:
            neg_freq = cat_neg.get(category, 0)
            if neg_freq < freq * 0.5:
                confidence = "high" if freq > 0.5 else "medium"
                hypotheses[f"{confidence}_patterns"].append({
                    'pattern': f"{category} companies tend to win more deals",
                    'confidence': confidence,
                    'evidence': f"{freq:.1%} of won deals vs {neg_freq:.1%} of lost deals",
                    'category': 'company_size'
                })
    
    # Analyze age patterns
    age_analysis = patterns.get('age_analysis', {})
    age_significance = age_analysis.get('age_significance', {})
    
    for age_cat, data in age_significance.items():
        if data.get('is_significant', False):
            won_pct = data.get('won_percentage', 0)
            lost_pct = data.get('lost_percentage', 0)
            significance = data.get('significance', 0)
            
            confidence = "high" if significance > 0.4 else "medium"
            hypotheses[f"{confidence}_patterns"].append({
                'pattern': f"{age_cat.title()} companies show different win rates",
                'confidence': confidence,
                'evidence': f"{won_pct:.1%} won vs {lost_pct:.1%} lost (significance: {significance:.1%})",
                'category': 'age'
            })
    
    # Analyze region patterns
    region_pos = patterns.get('positive_patterns', {}).get('region_distribution', {})
    region_neg = patterns.get('negative_patterns', {}).get('region_distribution', {})
    
    for region, freq in region_pos.items():
        if region and freq > 0.2:  # Lower threshold for regions
            neg_freq = region_neg.get(region, 0)
            if neg_freq < freq * 0.7:
                confidence = "medium" if freq > 0.4 else "weak"
                hypotheses[f"{confidence}_patterns"].append({
                    'pattern': f"Companies in region {region} show positive trend",
                    'confidence': confidence,
                    'evidence': f"{freq:.1%} of won deals vs {neg_freq:.1%} of lost deals",
                    'category': 'location'
                })
    
    return hypotheses

def build_discovery_criteria(patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Build discovery criteria based on learned patterns"""
    criteria = {
        'primary_filters': {
            'etat_administratif': 'A'  # Active companies only
        },
        'secondary_filters': {},
        'scoring_weights': {}
    }
    
    # Get significant patterns
    pos_patterns = patterns.get('positive_patterns', {})
    
    # Company category filters (highest priority)
    cat_dist = pos_patterns.get('company_category_distribution', {})
    significant_categories = [cat for cat, freq in cat_dist.items() if freq >= 0.2 and cat]
    if significant_categories:
        criteria['primary_filters']['company_category'] = significant_categories
        criteria['scoring_weights']['company_category'] = 0.4
    
    # Age category filters
    age_dist = pos_patterns.get('age_distribution', {})
    significant_ages = [age for age, freq in age_dist.items() if freq >= 0.2 and age]
    if significant_ages:
        criteria['primary_filters']['age_category'] = significant_ages
        criteria['scoring_weights']['age_category'] = 0.3
    
    # APE code filters
    ape_dist = pos_patterns.get('ape_distribution', {})
    significant_apes = [ape for ape, freq in ape_dist.items() if freq >= 0.15 and ape]
    if significant_apes:
        criteria['secondary_filters']['ape_codes'] = significant_apes
        criteria['scoring_weights']['ape_code'] = 0.2
    
    # Region filters
    region_dist = pos_patterns.get('region_distribution', {})
    significant_regions = [region for region, freq in region_dist.items() if freq >= 0.15 and region]
    if significant_regions:
        criteria['secondary_filters']['regions'] = significant_regions
        criteria['scoring_weights']['region'] = 0.1
    
    return criteria

def score_company(company: Dict[str, Any], patterns: Dict[str, Any]) -> float:
    """Score company based on positive and negative patterns with enhanced scoring"""
    if patterns['total_won'] == 0 and patterns['total_lost'] == 0:
        return 0.5  # Default score if no patterns learned
    
    positive_score = 0.0
    negative_score = 0.0
    total_weight = 0.0
    
    # Get positive and negative patterns
    pos_patterns = patterns.get('positive_patterns', {})
    neg_patterns = patterns.get('negative_patterns', {})
    
    # Company category scoring (40% weight - most important)
    company_category = company.get('company_category', '')
    if company_category in pos_patterns.get('company_category_distribution', {}):
        cat_freq = pos_patterns['company_category_distribution'][company_category]
        positive_score += 0.4 * cat_freq
        total_weight += 0.4
    if company_category in neg_patterns.get('company_category_distribution', {}):
        cat_freq = neg_patterns['company_category_distribution'][company_category]
        negative_score += 0.4 * cat_freq
    
    # Age category scoring (30% weight)
    age_category = company.get('age_category', '')
    if age_category in pos_patterns.get('age_distribution', {}):
        age_freq = pos_patterns['age_distribution'][age_category]
        positive_score += 0.3 * age_freq
        total_weight += 0.3
    if age_category in neg_patterns.get('age_distribution', {}):
        age_freq = neg_patterns['age_distribution'][age_category]
        negative_score += 0.3 * age_freq
    
    # APE code scoring (20% weight)
    ape = company.get('ape', '')
    if ape in pos_patterns.get('ape_distribution', {}):
        ape_freq = pos_patterns['ape_distribution'][ape]
        positive_score += 0.2 * ape_freq
        total_weight += 0.2
    if ape in neg_patterns.get('ape_distribution', {}):
        ape_freq = neg_patterns['ape_distribution'][ape]
        negative_score += 0.2 * ape_freq
    
    # Region scoring (10% weight)
    postal_code = company.get('postal_code', '')
    if len(postal_code) >= 2:
        region = postal_code[:2]
        if region in pos_patterns.get('region_distribution', {}):
            region_freq = pos_patterns['region_distribution'][region]
            positive_score += 0.1 * region_freq
            total_weight += 0.1
        if region in neg_patterns.get('region_distribution', {}):
            region_freq = neg_patterns['region_distribution'][region]
            negative_score += 0.1 * region_freq
    
    # Calculate final score: positive patterns boost, negative patterns reduce
    if total_weight > 0:
        # Normalize by total weight applied
        final_score = (positive_score - (negative_score * 0.3)) / total_weight
    else:
        final_score = 0.5  # Default if no patterns match
    
    # Ensure score is between 0 and 1
    final_score = max(0.0, min(1.0, final_score))
    
    return final_score

def build_sirene_filters(patterns: Dict[str, Any], min_frequency: float = 0.01) -> Dict[str, Any]:
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

async def search_sirene_by_name_and_location(company_name: str, postal_code: str, city: str) -> dict:
    """Search Sirene by company name and location"""
    if not SIRENE_TOKEN or not company_name:
        return {}
    
    try:
        # Clean company name for search
        clean_name = clean_company_name(company_name)
        if not clean_name:
            return {}
        
        # Build search query
        query_parts = [f'denominationUniteLegale:"{clean_name}"']
        
        # Add location filters if available
        if postal_code:
            query_parts.append(f'codePostalEtablissement:{postal_code}')
        elif city:
            # Try to match city name
            clean_city = city.upper().strip()
            query_parts.append(f'libelleCommuneEtablissement:"{clean_city}"')
        
        query = " AND ".join(query_parts)
        
        params = {
            "q": query,
            "nombre": 5,  # Get top 5 matches
            "debut": 1
        }
        
        print(f"[DEBUG] Searching Sirene for: {company_name} in {city}, {postal_code}")
        print(f"[DEBUG] Query: {query}")
        
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
                    # Return the best match (first result)
                    return extract_enriched_sirene_data(etablissements[0])
                    
    except Exception as e:
        print(f"[ERROR] Sirene search failed for {company_name}: {e}")
    
    return {}

def extract_enriched_sirene_data(etablissement: dict) -> dict:
    """Extract enriched Sirene data from etablissement response"""
    unite_legale = etablissement.get('uniteLegale', {})
    adresse = etablissement.get('adresseEtablissement', {})
    
    ape_code = unite_legale.get('activitePrincipaleUniteLegale', '')
    legal_form_code = unite_legale.get('categorieJuridiqueUniteLegale', '')
    employee_range = unite_legale.get('trancheEffectifsUniteLegale', '')
    creation_date = unite_legale.get('dateCreationUniteLegale', '')
    age_years, age_category = calculate_age_and_category(creation_date)
    
    return {
        # Core identification
        'siren': etablissement.get('siren', ''),
        'siret': etablissement.get('siret', ''),
        'company_name': unite_legale.get('denominationUniteLegale') or 
                       unite_legale.get('denominationUsuelle1UniteLegale') or 
                       unite_legale.get('nomUniteLegale') or 
                       unite_legale.get('sigleUniteLegale') or '',
        
        # Activity and legal information
        'ape': ape_code,
        'ape_description': get_ape_description(ape_code),
        'legal_form': legal_form_code,
        'legal_form_description': get_legal_form_description(legal_form_code),
        
        # Location information
        'postal_code': adresse.get('codePostalEtablissement', ''),
        'city': adresse.get('libelleCommuneEtablissement', ''),
        'region': adresse.get('codePostalEtablissement', '')[:2] if adresse.get('codePostalEtablissement') else '',
        'location': f"{adresse.get('libelleCommuneEtablissement', '')}, {adresse.get('codePostalEtablissement', '')}".strip(', '),
        
        # Company characteristics
        'employee_range': employee_range,
        'employee_description': get_employee_description(employee_range),
        'age_years': age_years,
        'age_category': age_category,
        'creation_date': creation_date,
        'is_active': unite_legale.get('etatAdministratifUniteLegale') == 'A',
        'is_ess': unite_legale.get('economieSocialeSolidaireUniteLegale') == 'O',
        'is_mission_company': unite_legale.get('societeMissionUniteLegale') == 'O',
        
        # Company category and size
        'company_category': unite_legale.get('categorieEntreprise', ''),
        'company_category_year': unite_legale.get('anneeCategorieEntreprise', ''),
        
        # Additional useful fields
        'is_headquarters': etablissement.get('etablissementSiege', False),
        'association_id': unite_legale.get('identifiantAssociationUniteLegale', ''),
        'last_processing_date': unite_legale.get('dateDernierTraitementUniteLegale', '')
    }

async def enrich_user_data_with_sirene(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich user data with SIRENE information using name and location"""
    print(f"[DEBUG] Enriching {len(df)} companies with SIRENE data...")
    
    enriched_rows = []
    success_count = 0
    
    for _, row in df.iterrows():
        company_name = row.get('company_name', '')
        postal_code = row.get('postal_code', '')
        city = row.get('city', '')
        
        # Try to find company in Sirene
        sirene_data = await search_sirene_by_name_and_location(company_name, postal_code, city)
        
        # Merge user data with Sirene data
        enriched_row = row.to_dict()
        if sirene_data:
            enriched_row.update(sirene_data)
            success_count += 1
            print(f"[DEBUG] Enriched {company_name} with Sirene data")
        else:
            print(f"[DEBUG] No Sirene data found for {company_name}")
            # Keep original data but mark as not enriched
            enriched_row['enrichment_status'] = 'not_found'
        
        enriched_rows.append(enriched_row)
    
    enriched_df = pd.DataFrame(enriched_rows)
    print(f"[DEBUG] Enriched dataset shape: {enriched_df.shape}")
    print(f"[DEBUG] Successfully enriched {success_count}/{len(df)} companies ({success_count/len(df)*100:.1f}%)")
    
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
    """Enhanced training with Sirene enrichment and pattern analysis"""
    try:
        print(f"[DEBUG] Enhanced training with {len(req.rows)} rows")
        
        # Convert to DataFrame
        df = pd.DataFrame([r.model_dump() for r in req.rows])
        print(f"[DEBUG] Original DataFrame shape: {df.shape}")
        
        if df.empty:
            return {"ok": False, "error": "no rows"}
        
        # Step 1: Enrich user data with Sirene information
        enriched_df = await enrich_user_data_with_sirene(df)
        
        # Step 2: Analyze patterns from enriched data
        patterns = analyze_user_patterns(enriched_df)
        print(f"[DEBUG] Patterns extracted from enriched data")
        
        # Step 3: Generate hypotheses
        hypotheses = generate_hypotheses(patterns)
        
        # Step 4: Build discovery criteria
        discovery_criteria = build_discovery_criteria(patterns)
        
        # Step 5: Prepare enriched data for frontend
        enriched_data = []
        for _, row in enriched_df.iterrows():
            enriched_data.append({
                # Core identification
                "company_name": row.get("company_name", ""),
                "deal_status": row.get("deal_status", ""),
                "siren": row.get("siren", ""),
                "siret": row.get("siret", ""),
                
                # Activity and legal information
                "ape": row.get("ape", ""),
                "ape_description": row.get("ape_description", ""),
                "legal_form": row.get("legal_form", ""),
                "legal_form_description": row.get("legal_form_description", ""),
                
                # Location information
                "postal_code": row.get("postal_code", ""),
                "city": row.get("city", ""),
                "region": row.get("region", ""),
                "location": row.get("location", ""),
                
                # Company characteristics
                "employee_range": row.get("employee_range", ""),
                "employee_description": row.get("employee_description", ""),
                "age_years": row.get("age_years"),
                "age_category": row.get("age_category", ""),
                "creation_date": row.get("creation_date", ""),
                "is_active": row.get("is_active", False),
                "is_ess": row.get("is_ess", False),
                "is_mission_company": row.get("is_mission_company", False),
                
                # Company category and size
                "company_category": row.get("company_category", ""),
                "company_category_year": row.get("company_category_year", ""),
                
                # Additional fields
                "is_headquarters": row.get("is_headquarters", False),
                "association_id": row.get("association_id", ""),
                "last_processing_date": row.get("last_processing_date", ""),
                "enrichment_status": row.get("enrichment_status", "enriched")
            })
        
        # Store patterns for this tenant
        LEARNED_PATTERNS[req.tenant_id] = patterns
        
        # Calculate enrichment success rate
        enriched_count = len([d for d in enriched_data if d.get("siren")])
        enrichment_rate = enriched_count / len(enriched_data) if enriched_data else 0
        
        return {
            "ok": True,
            "enriched_data": enriched_data,
            "analysis": {
                "total_companies": len(df),
                "won_companies": patterns['total_won'],
                "lost_companies": patterns['total_lost'],
                "enrichment_success_rate": f"{enrichment_rate:.1%}",
                "patterns_analysis": {
                    "ape_distribution": {
                        "won": patterns['positive_patterns'].get('ape_distribution', {}),
                        "lost": patterns['negative_patterns'].get('ape_distribution', {})
                    },
                    "region_distribution": {
                        "won": patterns['positive_patterns'].get('region_distribution', {}),
                        "lost": patterns['negative_patterns'].get('region_distribution', {})
                    },
                    "age_distribution": {
                        "won": patterns['positive_patterns'].get('age_distribution', {}),
                        "lost": patterns['negative_patterns'].get('age_distribution', {})
                    },
                    "company_category_distribution": {
                        "won": patterns['positive_patterns'].get('company_category_distribution', {}),
                        "lost": patterns['negative_patterns'].get('company_category_distribution', {})
                    },
                    "employee_range_distribution": {
                        "won": patterns['positive_patterns'].get('size_distribution', {}),
                        "lost": patterns['negative_patterns'].get('size_distribution', {})
                    }
                }
            },
            "hypotheses": hypotheses,
            "discovery_criteria": discovery_criteria,
            "model_version": f"{req.tenant_id}-v2-enhanced",
            "message": f"Model trained successfully! Enriched {enriched_count}/{len(df)} companies with Sirene data."
        }
        
    except Exception as e:
        print(f"[ERROR] Training error: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/discover")
async def discover(req: DiscoverRequest):
    """Discover companies using learned patterns with enhanced scoring"""
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
            
            # Build reasons for scoring
            reasons = []
            ape = company.get("ape", "")
            region = company.get("region", "")
            company_category = company.get("company_category", "")
            age_category = company.get("age_category", "")
            
            # Build detailed reasons based on pattern matches
            if ape in patterns['positive_patterns'].get('ape_distribution', {}):
                ape_freq = patterns['positive_patterns']['ape_distribution'][ape]
                reasons.append(f"Winning APE {ape} ({ape_freq:.1%} of wins)")
            
            if region in patterns['positive_patterns'].get('region_distribution', {}):
                region_freq = patterns['positive_patterns']['region_distribution'][region]
                reasons.append(f"Winning region {region} ({region_freq:.1%} of wins)")
            
            if company_category in patterns['positive_patterns'].get('company_category_distribution', {}):
                cat_freq = patterns['positive_patterns']['company_category_distribution'][company_category]
                reasons.append(f"Winning company category {company_category} ({cat_freq:.1%} of wins)")
            
            if age_category in patterns['positive_patterns'].get('age_distribution', {}):
                age_freq = patterns['positive_patterns']['age_distribution'][age_category]
                reasons.append(f"Winning age category {age_category} ({age_freq:.1%} of wins)")
            
            # Determine band
            if score >= 0.7:
                band = "High"
            elif score >= 0.4:
                band = "Medium"
            else:
                band = "Low"
            
            scored_companies.append({
                # Core identification
                "name": company.get("company_name", "N/A"),
                "siren": company.get("siren", "N/A"),
                "siret": company.get("siret", "N/A"),
                "ape": ape,
                "ape_description": company.get("ape_description", ""),
                "legal_form": company.get("legal_form", ""),
                "legal_form_description": company.get("legal_form_description", ""),
                "region": region,
                "location": company.get("location", "N/A"),
                
                # Company characteristics
                "employee_range": company.get("employee_range", ""),
                "employee_description": company.get("employee_description", ""),
                "age_years": company.get("age_years"),
                "age_category": age_category,
                "creation_date": company.get("creation_date", ""),
                "is_active": company.get("is_active", False),
                "is_ess": company.get("is_ess", False),
                "is_mission_company": company.get("is_mission_company", False),
                
                # Company category and size
                "company_category": company_category,
                "company_category_year": company.get("company_category_year", ""),
                
                # Scoring
                "win_score": score,
                "band": band,
                "confidence_badge": f"{score:.1%}",
                "reasons": reasons if reasons else [f"Matches winning patterns (score: {score:.1%})"],
                "source": "SIRENE"
            })
        
        # Sort by score (highest first)
        scored_companies.sort(key=lambda x: x['win_score'], reverse=True)
        
        return {
            "ok": True,
            "companies": scored_companies[:req.limit],
            "total_found": len(scored_companies),
            "patterns_used": {
                "company_categories": list(patterns['positive_patterns'].get('company_category_distribution', {}).keys())[:10],
                "age_categories": list(patterns['positive_patterns'].get('age_distribution', {}).keys())[:10],
                "ape_codes": list(patterns['positive_patterns'].get('ape_distribution', {}).keys())[:10],
                "regions": list(patterns['positive_patterns'].get('region_distribution', {}).keys())[:10]
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Discovery error: {e}")
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
