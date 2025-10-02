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
        # Sanitize query to prevent API errors
        safe_query = sanitize_sirene_query(query)
        print(f"Original query: {query}")
        print(f"Safe query: {safe_query}")
        
        # Use proper SIRENE API v3.11 parameters
        params = {
            "q": safe_query,
            "nombre": min(rows, 1000),  # Max 1000 per request
            "debut": 1,
            "tri": "siren"
        }
        
        companies = []  # Initialize companies list
        
        if HTTPX_AVAILABLE and HTTP:
            headers = {"X-INSEE-Api-Key-Integration": SIRENE_TOKEN}
            url = f"{SIRENE_BASE}"
            
            response = HTTP.get(url, headers=headers, params=params)
            
            # Check for API errors
            if response.status_code == 400:
                print(f"SIRENE API 400 error: {response.text}")
                return []
            elif response.status_code == 401:
                print("SIRENE API 401 error: Invalid token")
                return []
            elif response.status_code == 429:
                print("SIRENE API 429 error: Rate limit exceeded")
                return []
            elif response.status_code != 200:
                print(f"SIRENE API error {response.status_code}: {response.text}")
                return []
            
            response.raise_for_status()
            data = response.json()
        else:
            # Fallback to urllib
            headers = {"X-INSEE-Api-Key-Integration": SIRENE_TOKEN}
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{SIRENE_BASE}?{query_string}"
            
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    print(f"SIRENE API error {response.status}")
                    return []
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
                    
                    # Extract additional SIRENE data points
                    legal_category = period.get("categorieJuridiqueUniteLegale", "")
                    employee_range = period.get("trancheEffectifsUniteLegale", "")
                    is_employer = period.get("caractereEmployeurUniteLegale", "")
                    company_category = period.get("categorieEntreprise", "")
                    company_size_year = period.get("anneeCategorieEntreprise", "")
                    company_size = period.get("categorieEntreprise", "")
                    
                    # Extract establishment data if available
                    establishment_count = 0
                    if "etablissements" in unit and unit["etablissements"]:
                        establishment_count = len(unit["etablissements"])
                    
                    companies.append({
                        "company_name": name,
                        "siren": siren,
                        "ape": ape,
                        "created_year": created_year,
                        "region": region,
                        "department": department,
                        "postal_code": postal_code,
                        "city": city,
                        "legal_category": legal_category,
                        "employee_range": employee_range,
                        "is_employer": is_employer,
                        "company_category": company_category,
                        "company_size_year": company_size_year,
                        "company_size": company_size,
                        "establishment_count": establishment_count,
                        "active": True
                    })
        
        return companies[:cap]
    
    except Exception as e:
        print(f"SIRENE API error: {e}")
        return []

def sanitize_sirene_query(query: str) -> str:
    """Sanitize SIRENE query to prevent API errors"""
    if not query or not query.strip():
        return ""
    
    # Remove potentially problematic characters
    safe_query = query.strip()
    
    # Limit query length
    if len(safe_query) > 200:
        safe_query = safe_query[:200]
    
    # Remove special characters that might break the API
    import re
    safe_query = re.sub(r'[^\w\s:()\-]', '', safe_query)
    
    # Ensure proper formatting
    safe_query = re.sub(r'\s+', ' ', safe_query)  # Multiple spaces to single space
    
    return safe_query

def sirene_query_from_filters(f: DiscoverFilters):
    """Build SIRENE query from filters - simplified approach"""
    # For now, let's use a simple approach and filter in our code
    # The SIRENE API query syntax seems to be different than expected
    return ""  # Empty query means get all companies, we'll filter in code

def analyze_deal_patterns(historical_data: List[HistoryRow]) -> dict:
    """Advanced analysis of won vs lost deal patterns"""
    if not historical_data:
        return {}
    
    won_deals = [row for row in historical_data if row.deal_status == "won"]
    lost_deals = [row for row in historical_data if row.deal_status == "lost"]
    
    analysis = {
        "won_count": len(won_deals),
        "lost_count": len(lost_deals),
        "win_rate": len(won_deals) / len(historical_data) if historical_data else 0,
        "patterns": {}
    }
    
    # Analyze company naming patterns
    won_names = [deal.company_name.lower() for deal in won_deals]
    lost_names = [deal.company_name.lower() for deal in lost_deals]
    
    # Extract naming patterns
    won_keywords = extract_company_keywords(won_names)
    lost_keywords = extract_company_keywords(lost_names)
    
    analysis["patterns"]["naming"] = {
        "won_keywords": won_keywords,
        "lost_keywords": lost_keywords,
        "successful_patterns": [kw for kw in won_keywords if kw not in lost_keywords]
    }
    
    # Analyze geographic patterns
    won_regions = extract_regions_from_deals(won_deals)
    lost_regions = extract_regions_from_deals(lost_deals)
    
    analysis["patterns"]["geographic"] = {
        "won_regions": won_regions,
        "lost_regions": lost_regions,
        "successful_regions": [r for r in won_regions if r not in lost_regions]
    }
    
    # Analyze industry patterns
    won_apes = extract_ape_codes_from_deals(won_deals)
    lost_apes = extract_ape_codes_from_deals(lost_deals)
    
    analysis["patterns"]["industry"] = {
        "won_apes": won_apes,
        "lost_apes": lost_apes,
        "successful_industries": [ape for ape in won_apes if ape not in lost_apes]
    }
    
    # Analyze company maturity patterns
    won_ages = extract_company_ages_from_deals(won_deals)
    lost_ages = extract_company_ages_from_deals(lost_deals)
    
    analysis["patterns"]["maturity"] = {
        "won_age_range": analyze_age_patterns(won_ages),
        "lost_age_range": analyze_age_patterns(lost_ages),
        "optimal_age_range": find_optimal_age_range(won_ages, lost_ages)
    }
    
    return analysis

def extract_company_keywords(names: List[str]) -> List[str]:
    """Extract meaningful keywords from company names"""
    keywords = []
    tech_keywords = ["tech", "digital", "data", "soft", "system", "solution", "innovation", "intelligence", "cloud", "ai", "ml"]
    business_keywords = ["consulting", "services", "group", "corp", "sas", "sarl", "sa", "ltd"]
    
    for name in names:
        words = name.split()
        for word in words:
            clean_word = word.lower().strip(".,-()")
            if len(clean_word) > 3 and clean_word in tech_keywords + business_keywords:
                keywords.append(clean_word)
    
    # Return most common keywords
    from collections import Counter
    return [kw for kw, count in Counter(keywords).most_common(5)]

def extract_regions_from_deals(deals: List[HistoryRow]) -> List[str]:
    """Extract regions from deals"""
    regions = []
    for deal in deals:
        if hasattr(deal, 'postal_code') and deal.postal_code:
            postal_str = str(deal.postal_code).zfill(5)
            first_two = postal_str[:2]
            if first_two in ["75", "77", "78", "91", "92", "93", "94", "95"]:
                regions.append("11")  # Île-de-France
            elif first_two == "69":
                regions.append("69")  # Rhône
            elif first_two == "13":
                regions.append("13")  # Bouches-du-Rhône
    return list(set(regions))

def extract_ape_codes_from_deals(deals: List[HistoryRow]) -> List[str]:
    """Extract APE codes from deals"""
    apes = []
    for deal in deals:
        if hasattr(deal, 'ape') and deal.ape:
            apes.append(deal.ape)
    return list(set(apes))

def extract_company_ages_from_deals(deals: List[HistoryRow]) -> List[int]:
    """Extract company ages from deals"""
    ages = []
    current_year = pd.Timestamp.now().year
    for deal in deals:
        if hasattr(deal, 'created_year') and deal.created_year:
            age = current_year - int(deal.created_year)
            if age > 0:
                ages.append(age)
    return ages

def analyze_age_patterns(ages: List[int]) -> dict:
    """Analyze age patterns"""
    if not ages:
        return {"avg": 0, "min": 0, "max": 0, "range": "unknown"}
    
    return {
        "avg": sum(ages) / len(ages),
        "min": min(ages),
        "max": max(ages),
        "range": f"{min(ages)}-{max(ages)}"
    }

def find_optimal_age_range(won_ages: List[int], lost_ages: List[int]) -> str:
    """Find optimal age range based on won vs lost patterns"""
    if not won_ages:
        return "unknown"
    
    # Find age ranges where won deals are more common
    won_avg = sum(won_ages) / len(won_ages) if won_ages else 0
    lost_avg = sum(lost_ages) / len(lost_ages) if lost_ages else 0
    
    if won_avg < lost_avg:
        return "younger companies"
    elif won_avg > lost_avg:
        return "older companies"
    else:
        return "mixed age range"

def build_smart_sirene_query_from_patterns(won_deals_df: pd.DataFrame) -> str:
    """Build smart SIRENE query based on comprehensive won deal patterns"""
    if won_deals_df.empty:
        return "etatAdministratifUniteLegale:A"  # Fallback to active companies only
    
    print("Building smart SIRENE query from won deal patterns...")
    
    # Extract patterns from won deals using 110 data points
    query_parts = ["etatAdministratifUniteLegale:A"]  # Always active companies
    
    # 1. APE Code Patterns (Industry Intelligence)
    ape_patterns = won_deals_df["ape"].value_counts()
    if len(ape_patterns) > 0:
        # Get top 2 most common APE codes from won deals
        top_apes = ape_patterns.head(2).index.tolist()
        if len(top_apes) == 1:
            query_parts.append(f"activitePrincipaleUniteLegale:{top_apes[0]}")
        elif len(top_apes) == 2:
            ape_query = f"activitePrincipaleUniteLegale:{top_apes[0]} OR activitePrincipaleUniteLegale:{top_apes[1]}"
            query_parts.append(f"({ape_query})")
        print(f"APE patterns: {dict(ape_patterns.head(3))}")
    
    # 2. Employee Range Patterns (Size Intelligence)
    employee_patterns = won_deals_df["employee_range"].value_counts()
    if len(employee_patterns) > 0:
        # Get most common employee range
        top_employee_range = employee_patterns.index[0]
        if top_employee_range and top_employee_range != "":
            query_parts.append(f"trancheEffectifsUniteLegale:{top_employee_range}")
        print(f"Employee range patterns: {dict(employee_patterns.head(3))}")
    
    # 3. Legal Category Patterns (Structure Intelligence)
    legal_patterns = won_deals_df["legal_category"].value_counts()
    if len(legal_patterns) > 0:
        # Get most common legal category
        top_legal = legal_patterns.index[0]
        if top_legal and top_legal != "":
            query_parts.append(f"categorieJuridiqueUniteLegale:{top_legal}")
        print(f"Legal category patterns: {dict(legal_patterns.head(3))}")
    
    # 4. Geographic Patterns (Location Intelligence)
    region_patterns = won_deals_df["region"].value_counts()
    city_patterns = won_deals_df["city"].value_counts()
    
    if len(region_patterns) > 0:
        # Get top region
        top_region = region_patterns.index[0]
        if top_region and top_region != "":
            query_parts.append(f"codeRegion:{top_region}")
        print(f"Region patterns: {dict(region_patterns.head(3))}")
    
    # 5. Company Size Patterns
    size_patterns = won_deals_df["company_size"].value_counts()
    if len(size_patterns) > 0:
        top_size = size_patterns.index[0]
        if top_size and top_size != "":
            query_parts.append(f"categorieEntreprise:{top_size}")
        print(f"Company size patterns: {dict(size_patterns.head(3))}")
    
    # 6. Age Patterns (Maturity Intelligence)
    age_stats = won_deals_df["age_years"].describe()
    if not age_stats.empty and pd.notna(age_stats["mean"]):
        mean_age = int(age_stats["mean"])
        # Create age range around mean
        age_min = max(0, mean_age - 2)
        age_max = mean_age + 2
        print(f"Age patterns: mean={mean_age}, range={age_min}-{age_max}")
    
    # Build final query
    final_query = " AND ".join(query_parts)
    
    # Safety check: limit query length
    if len(final_query) > 200:
        # Fallback to simpler query
        final_query = " AND ".join(query_parts[:3])  # Keep only first 3 parts
    
    print(f"Final SIRENE query: {final_query}")
    return final_query

def build_safe_sirene_query(historical_data: List[HistoryRow]) -> str:
    """Build safe SIRENE query that won't break the API"""
    if not historical_data:
        return "etatAdministratifUniteLegale:A"  # Fallback to active companies
    
    # Convert to DataFrame for pattern analysis
    df = pd.DataFrame([row.model_dump() for row in historical_data])
    if df.empty:
        return "etatAdministratifUniteLegale:A"
    
    # Apply feature engineering to get 110 data points
    df = featurize_simple(df)
    
    # Filter won deals
    won_deals = df[df["deal_status"].str.lower() == "won"]
    
    # Use smart pattern-based query building
    return build_smart_sirene_query_from_patterns(won_deals)

def build_smart_sirene_query(historical_data: List[HistoryRow]) -> str:
    """Build smart SIRENE query based on advanced deal pattern analysis"""
    return build_safe_sirene_query(historical_data)  # Use safe version

def build_smart_sirene_query_from_filters(filters: DiscoverFilters, tenant: str) -> str:
    """Build smart SIRENE query from user filters and training patterns"""
    query_parts = ["etatAdministratifUniteLegale:A"]  # Always active companies
    
    # Get training patterns if available
    training_patterns = None
    if tenant in ARTIFACTS:
        training_patterns = ARTIFACTS[tenant].get("won_deals_patterns", {})
    
    # 1. APE Code Filtering (Industry Intelligence)
    if filters.ape_codes and len(filters.ape_codes) > 0:
        if len(filters.ape_codes) == 1:
            query_parts.append(f"activitePrincipaleUniteLegale:{filters.ape_codes[0]}")
        else:
            ape_query = " OR ".join([f"activitePrincipaleUniteLegale:{ape}" for ape in filters.ape_codes[:3]])  # Limit to 3
            query_parts.append(f"({ape_query})")
    elif training_patterns and training_patterns.get("ape_codes"):
        # Use training patterns if no user filters
        top_apes = list(training_patterns["ape_codes"].keys())[:2]
        if len(top_apes) == 1:
            query_parts.append(f"activitePrincipaleUniteLegale:{top_apes[0]}")
        elif len(top_apes) == 2:
            ape_query = f"activitePrincipaleUniteLegale:{top_apes[0]} OR activitePrincipaleUniteLegale:{top_apes[1]}"
            query_parts.append(f"({ape_query})")
    
    # 2. Region Filtering (Geographic Intelligence)
    if filters.regions and len(filters.regions) > 0:
        if len(filters.regions) == 1:
            query_parts.append(f"codeRegion:{filters.regions[0]}")
        else:
            region_query = " OR ".join([f"codeRegion:{region}" for region in filters.regions[:2]])  # Limit to 2
            query_parts.append(f"({region_query})")
    elif training_patterns and training_patterns.get("regions"):
        # Use training patterns if no user filters
        top_regions = list(training_patterns["regions"].keys())[:1]
        if top_regions:
            query_parts.append(f"codeRegion:{top_regions[0]}")
    
    # 3. Department Filtering
    if filters.departments and len(filters.departments) > 0:
        if len(filters.departments) == 1:
            query_parts.append(f"codeDepartement:{filters.departments[0]}")
        else:
            dept_query = " OR ".join([f"codeDepartement:{dept}" for dept in filters.departments[:2]])  # Limit to 2
            query_parts.append(f"({dept_query})")
    
    # 4. Employee Range Filtering (from training patterns)
    if training_patterns and training_patterns.get("employee_ranges"):
        top_employee_range = list(training_patterns["employee_ranges"].keys())[0]
        if top_employee_range and top_employee_range != "":
            query_parts.append(f"trancheEffectifsUniteLegale:{top_employee_range}")
    
    # 5. Legal Category Filtering (from training patterns)
    if training_patterns and training_patterns.get("legal_categories"):
        top_legal = list(training_patterns["legal_categories"].keys())[0]
        if top_legal and top_legal != "":
            query_parts.append(f"categorieJuridiqueUniteLegale:{top_legal}")
    
    # Build final query
    final_query = " AND ".join(query_parts)
    
    # Safety check: limit query length
    if len(final_query) > 200:
        # Fallback to simpler query
        final_query = " AND ".join(query_parts[:3])  # Keep only first 3 parts
    
    print(f"Built smart query from filters: {final_query}")
    return final_query

def analyze_company_name(name: str) -> float:
    """Analyze company name for business intelligence"""
    if not name or pd.isna(name):
        return 0.3
    
    name_lower = name.lower()
    score = 0.5  # Base score
    
    # Tech indicators
    tech_keywords = ["tech", "digital", "data", "soft", "system", "solution", "innovation", "intelligence", "cloud", "ai", "ml", "cyber", "smart"]
    tech_count = sum(1 for keyword in tech_keywords if keyword in name_lower)
    score += tech_count * 0.1
    
    # Business maturity indicators
    business_suffixes = ["sas", "sarl", "sa", "ltd", "corp", "group", "holding", "international"]
    if any(suffix in name_lower for suffix in business_suffixes):
        score += 0.2
    
    # International presence
    if any(word in name_lower for word in ["international", "global", "world", "europe", "france"]):
        score += 0.15
    
    # Brand strength (name length and complexity)
    if len(name) > 15:
        score += 0.1  # Longer names often indicate established brands
    
    return min(1.0, score)

def analyze_website(website: str) -> float:
    """Analyze website for business intelligence"""
    if not website or pd.isna(website):
        return 0.3
    
    website_lower = website.lower()
    score = 0.5  # Base score
    
    # Domain quality
    if website_lower.endswith('.com'):
        score += 0.2  # International presence
    elif website_lower.endswith('.fr'):
        score += 0.1  # French presence
    
    # Professional indicators
    if 'www.' in website_lower:
        score += 0.1  # Professional setup
    
    # Tech company indicators
    tech_domains = ["tech", "digital", "data", "soft", "system", "solution", "innovation", "intelligence", "cloud", "ai", "ml"]
    if any(domain in website_lower for domain in tech_domains):
        score += 0.15
    
    # Subdomain analysis
    if len(website_lower.split('.')) > 2:
        score += 0.1  # Complex domain structure
    
    return min(1.0, score)

def analyze_email(email: str) -> float:
    """Analyze email for business intelligence"""
    if not email or pd.isna(email):
        return 0.3
    
    email_lower = email.lower()
    score = 0.5  # Base score
    
    # Professional email patterns
    professional_patterns = ["contact", "info", "hello", "bonjour", "ceo", "director", "manager"]
    if any(pattern in email_lower for pattern in professional_patterns):
        score += 0.2
    
    # Generic vs specific
    if email_lower.startswith('contact@') or email_lower.startswith('info@'):
        score += 0.1  # Professional contact
    
    # Domain matching (if website available)
    if '@' in email_lower:
        domain = email_lower.split('@')[1]
        if domain.endswith('.com') or domain.endswith('.fr'):
            score += 0.1
    
    return min(1.0, score)

def analyze_geographic_intelligence(row) -> float:
    """Analyze geographic intelligence from multiple data points"""
    score = 0.5  # Base score
    
    # City analysis
    city = str(row.get("city", "")).lower()
    if "paris" in city:
        score += 0.3  # Paris premium
    elif "lyon" in city:
        score += 0.2  # Lyon premium
    elif any(city_name in city for city_name in ["marseille", "toulouse", "nice", "nantes"]):
        score += 0.1  # Major cities
    
    # Postal code analysis
    postal_code = str(row.get("postal_code", ""))
    if postal_code:
        first_two = postal_code[:2]
        if first_two in ["75", "92", "93", "94"]:  # Paris and inner suburbs
            score += 0.2
        elif first_two in ["77", "78", "91", "95"]:  # Outer Paris region
            score += 0.1
    
    return min(1.0, score)

def get_business_district_score(row) -> float:
    """Score based on business district location"""
    city = str(row.get("city", "")).lower()
    postal_code = str(row.get("postal_code", ""))
    
    # Paris business districts
    if "paris" in city and postal_code:
        arrondissement = postal_code[2:] if len(postal_code) >= 4 else ""
        
        # Premium business districts
        premium_districts = ["75001", "75002", "75008", "75009", "75016", "75017"]
        if postal_code in premium_districts:
            return 0.9
        
        # Good business districts
        good_districts = ["75003", "75004", "75011", "75012", "75015"]
        if postal_code in good_districts:
            return 0.7
        
        # Other Paris districts
        if postal_code.startswith("75"):
            return 0.6
    
    # Lyon business districts
    if "lyon" in city:
        return 0.7
    
    # Other major cities
    if any(city_name in city for city_name in ["marseille", "toulouse", "nice", "nantes"]):
        return 0.5
    
    return 0.3  # Default score

def add_comprehensive_data_points(df: pd.DataFrame) -> pd.DataFrame:
    """Add 100 comprehensive data points for advanced lead scoring"""
    
    # Ensure all required columns exist with safe defaults
    safe_columns = {
        "company_name": "",
        "siren": "",
        "ape": "",
        "postal_code": "",
        "city": "",
        "website": "",
        "email": "",
        "created_year": None,
        "active": True,
        "legal_category": "",
        "employee_range": "",
        "is_employer": "",
        "company_category": "",
        "company_size_year": "",
        "company_size": "",
        "establishment_count": 0
    }
    
    for col, default_val in safe_columns.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            # Safe type conversion
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(default_val if isinstance(default_val, str) else "")
            else:
                df[col] = df[col].fillna(default_val)
    
    # 1-10: Company Name Analysis (10 points) - Safe string operations
    df["name_length"] = df["company_name"].astype(str).str.len()
    df["name_word_count"] = df["company_name"].astype(str).str.split().str.len()
    df["name_has_tech"] = df["company_name"].astype(str).str.lower().str.contains("tech|digital|data|soft|system|solution|innovation|intelligence|cloud|ai|ml|cyber|smart", na=False).astype(int)
    df["name_has_business"] = df["company_name"].astype(str).str.lower().str.contains("sas|sarl|sa|ltd|corp|group|holding|international", na=False).astype(int)
    df["name_has_international"] = df["company_name"].astype(str).str.lower().str.contains("international|global|world|europe|france", na=False).astype(int)
    df["name_complexity"] = df["company_name"].astype(str).str.count("[A-Z]") / df["name_length"].replace(0, 1)  # Avoid division by zero
    df["name_has_numbers"] = df["company_name"].astype(str).str.contains(r"\d", na=False).astype(int)
    df["name_has_special_chars"] = df["company_name"].astype(str).str.contains(r"[^a-zA-Z0-9\s]", na=False).astype(int)
    df["name_starts_capital"] = df["company_name"].astype(str).str[0].str.isupper().fillna(False).astype(int)
    df["name_tech_score"] = df["company_name"].astype(str).str.lower().str.count("tech|digital|data|soft|system|solution|innovation|intelligence|cloud|ai|ml|cyber|smart")
    
    # 11-20: SIREN Analysis (10 points) - Safe string operations
    df["siren_length"] = df["siren"].astype(str).str.len()
    df["siren_is_valid"] = df["siren"].astype(str).str.len().ge(9).astype(int)
    df["siren_has_leading_zeros"] = df["siren"].astype(str).str.startswith("0").astype(int)
    df["siren_checksum_valid"] = df["siren"].astype(str).apply(validate_siren_checksum)
    df["siren_age_indicator"] = pd.to_numeric(df["siren"].astype(str).str[:2], errors="coerce").fillna(0)
    df["siren_region_code"] = df["siren"].astype(str).str[2:4]
    df["siren_department_code"] = df["siren"].astype(str).str[4:6]
    df["siren_sequence"] = df["siren"].astype(str).str[6:9]
    df["siren_parity"] = pd.to_numeric(df["siren"].astype(str).str[-1], errors="coerce").fillna(0) % 2
    df["siren_establishment_count"] = df["siren"].astype(str).apply(estimate_establishment_count)
    
    # 21-30: APE Code Analysis (10 points) - Safe string operations
    df["ape_length"] = df["ape"].astype(str).str.len()
    df["ape_is_valid"] = df["ape"].astype(str).str.len().ge(5).astype(int)
    df["ape_category"] = df["ape"].astype(str).str[:2]
    df["ape_subcategory"] = df["ape"].astype(str).str[2:4]
    df["ape_activity"] = df["ape"].astype(str).str[4:5]
    df["ape_is_tech"] = df["ape"].astype(str).str.startswith("62|63|70|71|72").astype(int)
    df["ape_is_services"] = df["ape"].astype(str).str.startswith("70|71|72|73|74|75|76|77|78|79|80|81|82").astype(int)
    df["ape_is_manufacturing"] = df["ape"].astype(str).str.startswith("10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33").astype(int)
    df["ape_complexity"] = df["ape"].astype(str).str.count("[A-Z]")
    df["ape_numeric_part"] = pd.to_numeric(df["ape"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(0)
    
    # 31-40: Geographic Analysis (10 points) - Safe string operations
    df["postal_code_length"] = df["postal_code"].astype(str).str.len()
    df["postal_code_is_valid"] = df["postal_code"].astype(str).str.len().eq(5).astype(int)
    df["postal_code_department"] = df["postal_code"].astype(str).str[:2]
    df["postal_code_arrondissement"] = df["postal_code"].astype(str).str[2:4]
    df["postal_code_is_paris"] = df["postal_code"].astype(str).str.startswith("75").astype(int)
    df["postal_code_is_lyon"] = df["postal_code"].astype(str).str.startswith("69").astype(int)
    df["postal_code_is_major_city"] = df["postal_code"].astype(str).str.startswith("75|69|13|31|59|44|67|68").astype(int)
    df["city_length"] = df["city"].astype(str).str.len()
    df["city_has_arrondissement"] = df["city"].astype(str).str.contains(r"\d+e", na=False).astype(int)
    df["city_is_capital"] = df["city"].astype(str).str.lower().str.contains("paris|lyon|marseille|toulouse|nice|nantes|strasbourg|montpellier|bordeaux|lille", na=False).astype(int)
    
    # 41-50: Age and Maturity Analysis (10 points)
    df["age_years"] = pd.Timestamp.now().year - df["created_year"]
    df["age_category"] = pd.cut(df["age_years"], bins=[0, 1, 3, 7, 15, 100], labels=["startup", "early", "growth", "mature", "established"])
    df["age_is_startup"] = (df["age_years"] < 2).astype(int)
    df["age_is_growth"] = ((df["age_years"] >= 2) & (df["age_years"] < 7)).astype(int)
    df["age_is_mature"] = ((df["age_years"] >= 7) & (df["age_years"] < 15)).astype(int)
    df["age_is_established"] = (df["age_years"] >= 15).astype(int)
    df["age_sweet_spot"] = ((df["age_years"] >= 3) & (df["age_years"] <= 8)).astype(int)
    df["age_risk_factor"] = ((df["age_years"] < 1) | (df["age_years"] > 20)).astype(int)
    df["age_stability"] = ((df["age_years"] >= 5) & (df["age_years"] <= 10)).astype(int)
    df["age_innovation"] = ((df["age_years"] >= 1) & (df["age_years"] <= 5)).astype(int)
    
    # 51-60: SIREN Pattern Analysis (10 points) - Based on actual SIREN data
    df["siren_sequence_pattern"] = df["siren"].astype(str).str[6:9].astype(int, errors="coerce").fillna(0)
    df["siren_region_pattern"] = df["siren"].astype(str).str[2:4]
    df["siren_department_pattern"] = df["siren"].astype(str).str[4:6]
    df["siren_creation_pattern"] = df["siren"].astype(str).str[:2].astype(int, errors="coerce").fillna(0)
    df["siren_parity_pattern"] = df["siren"].astype(str).str[-1].astype(int, errors="coerce").fillna(0) % 2
    df["siren_leading_digits"] = df["siren"].astype(str).str[:3].astype(int, errors="coerce").fillna(0)
    df["siren_middle_digits"] = df["siren"].astype(str).str[3:6].astype(int, errors="coerce").fillna(0)
    df["siren_trailing_digits"] = df["siren"].astype(str).str[6:9].astype(int, errors="coerce").fillna(0)
    df["siren_checksum_digit"] = df["siren"].astype(str).str[-1].astype(int, errors="coerce").fillna(0)
    df["siren_validation_score"] = df["siren"].astype(str).apply(validate_siren_checksum)
    
    # 61-70: APE Code Deep Analysis (10 points) - Based on actual APE data
    df["ape_tech_intensity"] = df["ape"].astype(str).str.startswith("62|63").astype(int)
    df["ape_services_intensity"] = df["ape"].astype(str).str.startswith("70|71|72|73|74|75|76|77|78|79|80|81|82").astype(int)
    df["ape_manufacturing_intensity"] = df["ape"].astype(str).str.startswith("10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33").astype(int)
    df["ape_retail_intensity"] = df["ape"].astype(str).str.startswith("47|56").astype(int)
    df["ape_construction_intensity"] = df["ape"].astype(str).str.startswith("41|42|43").astype(int)
    df["ape_education_intensity"] = df["ape"].astype(str).str.startswith("85").astype(int)
    df["ape_health_intensity"] = df["ape"].astype(str).str.startswith("86|87").astype(int)
    df["ape_finance_intensity"] = df["ape"].astype(str).str.startswith("64|65|66").astype(int)
    df["ape_transport_intensity"] = df["ape"].astype(str).str.startswith("49|50|51|52").astype(int)
    df["ape_agriculture_intensity"] = df["ape"].astype(str).str.startswith("01|02|03").astype(int)
    
    # 71-80: Geographic Intelligence (10 points) - Based on actual SIRENE data
    df["is_paris_region"] = df["postal_code"].astype(str).str.startswith("75|77|78|91|92|93|94|95").astype(int)
    df["is_lyon_region"] = df["postal_code"].astype(str).str.startswith("69").astype(int)
    df["is_marseille_region"] = df["postal_code"].astype(str).str.startswith("13").astype(int)
    df["is_toulouse_region"] = df["postal_code"].astype(str).str.startswith("31").astype(int)
    df["is_nice_region"] = df["postal_code"].astype(str).str.startswith("06").astype(int)
    df["is_nantes_region"] = df["postal_code"].astype(str).str.startswith("44").astype(int)
    df["is_strasbourg_region"] = df["postal_code"].astype(str).str.startswith("67|68").astype(int)
    df["is_montpellier_region"] = df["postal_code"].astype(str).str.startswith("34").astype(int)
    df["is_bordeaux_region"] = df["postal_code"].astype(str).str.startswith("33").astype(int)
    df["is_lille_region"] = df["postal_code"].astype(str).str.startswith("59").astype(int)
    
    # 81-90: Company Name Intelligence (10 points) - Based on actual SIRENE data
    df["name_has_sas"] = df["company_name"].astype(str).str.lower().str.contains("sas", na=False).astype(int)
    df["name_has_sarl"] = df["company_name"].astype(str).str.lower().str.contains("sarl", na=False).astype(int)
    df["name_has_sa"] = df["company_name"].astype(str).str.lower().str.contains(" sa ", na=False).astype(int)
    df["name_has_snc"] = df["company_name"].astype(str).str.lower().str.contains("snc", na=False).astype(int)
    df["name_has_sci"] = df["company_name"].astype(str).str.lower().str.contains("sci", na=False).astype(int)
    df["name_has_eurl"] = df["company_name"].astype(str).str.lower().str.contains("eurl", na=False).astype(int)
    df["name_has_sarlu"] = df["company_name"].astype(str).str.lower().str.contains("sarlu", na=False).astype(int)
    df["name_has_sasu"] = df["company_name"].astype(str).str.lower().str.contains("sasu", na=False).astype(int)
    df["name_has_snc"] = df["company_name"].astype(str).str.lower().str.contains("snc", na=False).astype(int)
    df["name_has_auto_entrepreneur"] = df["company_name"].astype(str).str.lower().str.contains("auto-entrepreneur|auto entrepreneur", na=False).astype(int)
    
    # 91-100: Employee & Company Size Analysis (10 points) - Based on SIRENE data
    df["employee_range_score"] = df["employee_range"].apply(get_employee_range_score)
    df["is_employer_score"] = (df["is_employer"] == "O").astype(int)
    df["company_category_score"] = df["company_category"].apply(get_company_category_score)
    df["legal_category_score"] = df["legal_category"].apply(get_legal_category_score)
    df["establishment_count_score"] = df["establishment_count"].apply(get_establishment_count_score)
    df["company_size_indicator"] = df["company_size"].apply(get_company_size_indicator)
    df["employer_characteristic"] = (df["is_employer"] == "O").astype(int)
    df["multi_establishment"] = (df["establishment_count"] > 1).astype(int)
    df["large_company_indicator"] = (df["employee_range"].isin(["53", "54", "55", "56", "57", "58", "59", "60", "61", "62"])).astype(int)
    df["small_company_indicator"] = (df["employee_range"].isin(["00", "01", "02", "03", "11", "12"])).astype(int)
    
    # Add missing columns that are referenced later
    df["has_complete_data"] = (
        df["siren"].notna().astype(int) + 
        df["ape"].notna().astype(int) + 
        df["postal_code"].notna().astype(int) + 
        df["city"].notna().astype(int)
    ) / 4.0
    df["is_active"] = df["active"].fillna(True).astype(int)
    
    # 101-110: Composite Scores (10 points) - Based on actual SIRENE data
    df["composite_tech_score"] = (df["ape_tech_intensity"] * 0.6 + df["name_has_tech"] * 0.4)
    df["composite_geographic_score"] = (df["is_paris_region"] * 0.4 + df["is_lyon_region"] * 0.3 + df["is_marseille_region"] * 0.2 + df["is_toulouse_region"] * 0.1)
    df["composite_maturity_score"] = (df["age_sweet_spot"] * 0.4 + df["age_stability"] * 0.3 + df["age_innovation"] * 0.3)
    df["composite_data_quality"] = (df["siren_is_valid"] * 0.3 + df["ape_is_valid"] * 0.3 + df["postal_code_is_valid"] * 0.2 + df["has_complete_data"] * 0.2)
    df["composite_business_score"] = (df["ape_services_intensity"] * 0.4 + df["ape_tech_intensity"] * 0.3 + df["name_has_sas"] * 0.3)
    df["composite_establishment_score"] = (df["siren_is_valid"] * 0.3 + df["age_is_mature"] * 0.3 + df["name_has_sas"] * 0.4)
    df["composite_innovation_score"] = (df["age_innovation"] * 0.4 + df["name_has_tech"] * 0.3 + df["ape_tech_intensity"] * 0.3)
    df["composite_stability_score"] = (df["age_stability"] * 0.4 + df["siren_is_valid"] * 0.3 + df["is_active"] * 0.3)
    df["composite_growth_score"] = (df["age_is_growth"] * 0.4 + df["ape_tech_intensity"] * 0.3 + df["age_sweet_spot"] * 0.3)
    df["composite_lead_score"] = (df["composite_tech_score"] * 0.2 + df["composite_geographic_score"] * 0.2 + df["composite_maturity_score"] * 0.2 + df["composite_business_score"] * 0.2 + df["composite_innovation_score"] * 0.2)
    
    return df

def validate_siren_checksum(siren: str) -> int:
    """Validate SIREN checksum using Luhn algorithm"""
    if not siren or len(siren) != 9:
        return 0
    try:
        digits = [int(d) for d in siren]
        checksum = 0
        for i, digit in enumerate(digits[:-1]):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit = digit // 10 + digit % 10
            checksum += digit
        return int((10 - (checksum % 10)) % 10 == digits[-1])
    except:
        return 0

def estimate_establishment_count(siren: str) -> int:
    """Estimate number of establishments based on SIREN patterns"""
    if not siren or len(siren) != 9:
        return 0
    try:
        # Simple heuristic based on SIREN patterns
        sequence = int(siren[6:9])
        if sequence < 100:
            return 1  # Single establishment
        elif sequence < 500:
            return 2  # Small multi-establishment
        else:
            return 3  # Large multi-establishment
    except:
        return 0

def get_employee_range_score(employee_range: str) -> float:
    """Score based on employee range - optimal for deal closing"""
    if not employee_range:
        return 0.3
    
    # SIRENE employee range codes (trancheEffectifsUniteLegale)
    range_scores = {
        "00": 0.2,  # 0 salarié
        "01": 0.3,  # 1 ou 2 salariés
        "02": 0.4,  # 3 à 5 salariés
        "03": 0.5,  # 6 à 9 salariés
        "11": 0.6,  # 10 à 19 salariés
        "12": 0.7,  # 20 à 49 salariés
        "21": 0.8,  # 50 à 99 salariés
        "22": 0.9,  # 100 à 199 salariés
        "31": 0.8,  # 200 à 249 salariés
        "32": 0.7,  # 250 à 499 salariés
        "41": 0.6,  # 500 à 999 salariés
        "42": 0.5,  # 1000 à 1999 salariés
        "51": 0.4,  # 2000 à 4999 salariés
        "52": 0.3,  # 5000 à 9999 salariés
        "53": 0.2,  # 10000 salariés et plus
    }
    return range_scores.get(employee_range, 0.3)

def get_company_category_score(company_category: str) -> float:
    """Score based on company category"""
    if not company_category:
        return 0.3
    
    # SIRENE company categories
    category_scores = {
        "PME": 0.8,  # Petites et moyennes entreprises
        "ETI": 0.9,  # Entreprises de taille intermédiaire
        "GE": 0.6,   # Grandes entreprises
        "TPE": 0.7,  # Très petites entreprises
    }
    return category_scores.get(company_category, 0.3)

def get_legal_category_score(legal_category: str) -> float:
    """Score based on legal category"""
    if not legal_category:
        return 0.3
    
    # SIRENE legal categories - higher scores for more professional structures
    legal_scores = {
        "5710": 0.9,  # SAS
        "5499": 0.8,  # SARL
        "5710": 0.9,  # SASU
        "5499": 0.8,  # EURL
        "5499": 0.8,  # SARLU
        "5710": 0.9,  # SAS
        "5499": 0.8,  # SARL
        "5499": 0.8,  # SNC
        "5499": 0.8,  # SCI
        "5499": 0.8,  # Auto-entrepreneur
    }
    return legal_scores.get(legal_category, 0.3)

def get_establishment_count_score(establishment_count: int) -> float:
    """Score based on number of establishments"""
    if establishment_count == 0:
        return 0.3
    elif establishment_count == 1:
        return 0.7  # Single establishment - good for targeting
    elif establishment_count <= 3:
        return 0.8  # Small multi-establishment - good growth potential
    elif establishment_count <= 10:
        return 0.9  # Medium multi-establishment - excellent
    else:
        return 0.6  # Large multi-establishment - might be harder to reach decision makers

def get_company_size_indicator(company_size: str) -> float:
    """Score based on company size indicator"""
    if not company_size:
        return 0.3
    
    # SIRENE company size indicators
    size_scores = {
        "00": 0.2,  # 0 salarié
        "01": 0.3,  # 1 ou 2 salariés
        "02": 0.4,  # 3 à 5 salariés
        "03": 0.5,  # 6 à 9 salariés
        "11": 0.6,  # 10 à 19 salariés
        "12": 0.7,  # 20 à 49 salariés
        "21": 0.8,  # 50 à 99 salariés
        "22": 0.9,  # 100 à 199 salariés
        "31": 0.8,  # 200 à 249 salariés
        "32": 0.7,  # 250 à 499 salariés
        "41": 0.6,  # 500 à 999 salariés
        "42": 0.5,  # 1000 à 1999 salariés
        "51": 0.4,  # 2000 à 4999 salariés
        "52": 0.3,  # 5000 à 9999 salariés
        "53": 0.2,  # 10000 salariés et plus
    }
    return size_scores.get(company_size, 0.3)

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
    
    # Ensure required columns exist with safe defaults
    required_columns = {
        "postal_code": "",
        "city": "",
        "website": "",
        "email": "",
        "siren": "",
        "ape": "",
        "created_year": None,
        "active": True
    }
    
    for col, default_val in required_columns.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            # Handle NaN values safely
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(default_val if isinstance(default_val, str) else "")
            else:
                df[col] = df[col].fillna(default_val)
    
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
    
    # Enhanced data points
    df["company_name_intelligence"] = df["company_name"].apply(analyze_company_name)
    df["website_intelligence"] = df.get("website", "").apply(analyze_website)
    df["email_intelligence"] = df.get("email", "").apply(analyze_email)
    df["geographic_intelligence"] = df.apply(lambda row: analyze_geographic_intelligence(row), axis=1)
    df["business_district_score"] = df.apply(lambda row: get_business_district_score(row), axis=1)
    
    # 100 Data Points System - Comprehensive Analysis
    df = add_comprehensive_data_points(df)
    
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
        
        # Comprehensive feature engineering with 110 data points
        df = featurize_simple(df)
        
        # Extract patterns from won deals using 110 data points
        won_deals = df[df["deal_status"].str.lower() == "won"].copy()
        lost_deals = df[df["deal_status"].str.lower() == "lost"].copy()
        
        print(f"Analyzing {len(won_deals)} won deals vs {len(lost_deals)} lost deals")
        
        # Analyze patterns in won deals
        if len(won_deals) > 0:
            # Employee range patterns
            won_employee_ranges = won_deals["employee_range"].value_counts()
            print(f"Won deals employee ranges: {dict(won_employee_ranges)}")
            
            # Legal category patterns
            won_legal_categories = won_deals["legal_category"].value_counts()
            print(f"Won deals legal categories: {dict(won_legal_categories)}")
            
            # Company size patterns
            won_company_sizes = won_deals["company_size"].value_counts()
            print(f"Won deals company sizes: {dict(won_company_sizes)}")
            
            # Geographic patterns
            won_regions = won_deals["region"].value_counts()
            won_cities = won_deals["city"].value_counts()
            print(f"Won deals regions: {dict(won_regions)}")
            print(f"Won deals cities: {dict(won_cities)}")
            
            # Industry patterns (APE codes)
            won_ape_codes = won_deals["ape"].value_counts()
            print(f"Won deals APE codes: {dict(won_ape_codes)}")
            
            # Age patterns
            won_ages = won_deals["age_years"].describe()
            print(f"Won deals age stats: {dict(won_ages)}")
            
            # Establishment count patterns
            won_establishments = won_deals["establishment_count"].value_counts()
            print(f"Won deals establishment counts: {dict(won_establishments)}")
        
        # Build enhanced training features using 110 data points
        # Ensure all required columns exist with safe defaults
        required_features = {
            "has_siren": 0,
            "age_years": -1,
            "employee_range_score": 0.3,
            "is_employer_score": 0,
            "company_category_score": 0.3,
            "legal_category_score": 0.3,
            "establishment_count_score": 0.3,
            "company_size_indicator": 0.3,
            "composite_tech_score": 0.3,
            "composite_geographic_score": 0.3,
            "composite_maturity_score": 0.3,
            "composite_business_score": 0.3,
            "composite_establishment_score": 0.3,
            "composite_innovation_score": 0.3,
            "composite_stability_score": 0.3,
            "composite_growth_score": 0.3,
            "composite_lead_score": 0.3
        }
        
        # Create feature dictionary with safe defaults for missing columns
        feature_dict = {}
        for feature, default_val in required_features.items():
            if feature in df.columns:
                feature_dict[feature] = df[feature]
            else:
                feature_dict[feature] = pd.Series([default_val] * len(df), index=df.index)
        
        Xtab = pd.DataFrame(feature_dict).fillna(0)
        
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
        
        # Store artifacts with comprehensive pattern analysis
        artifacts = {
            "clf": clf,
            "hist_labels": y,
            "hist_texts": [f"{row.company_name} {row.ape or ''}" for row in req.rows],
            "sklearn_available": SKLEARN_AVAILABLE,
            "won_deals_patterns": {
                "employee_ranges": dict(won_deals["employee_range"].value_counts()) if len(won_deals) > 0 else {},
                "legal_categories": dict(won_deals["legal_category"].value_counts()) if len(won_deals) > 0 else {},
                "company_sizes": dict(won_deals["company_size"].value_counts()) if len(won_deals) > 0 else {},
                "regions": dict(won_deals["region"].value_counts()) if len(won_deals) > 0 else {},
                "cities": dict(won_deals["city"].value_counts()) if len(won_deals) > 0 else {},
                "ape_codes": dict(won_deals["ape"].value_counts()) if len(won_deals) > 0 else {},
                "age_stats": dict(won_deals["age_years"].describe()) if len(won_deals) > 0 else {},
                "establishment_counts": dict(won_deals["establishment_count"].value_counts()) if len(won_deals) > 0 else {}
            },
            "training_features": list(Xtab.columns),
            "feature_importance": None
        }
        
        # Calculate feature importance if model is available
        if clf and hasattr(clf, 'feature_importances_'):
            feature_importance = dict(zip(Xtab.columns, clf.feature_importances_))
            artifacts["feature_importance"] = feature_importance
            print(f"Top 5 most important features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
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
                        
                        # Enhanced data points
                        company_name_score = df_sample.iloc[i].get("company_name_intelligence", 0.3)
                        website_score = df_sample.iloc[i].get("website_intelligence", 0.3)
                        email_score = df_sample.iloc[i].get("email_intelligence", 0.3)
                        geographic_intelligence_score = df_sample.iloc[i].get("geographic_intelligence", 0.3)
                        business_district_score = df_sample.iloc[i].get("business_district_score", 0.3)
                        
                        # 110 Data Points Scoring System - Based on comprehensive SIRENE data
                        # Core ML and Business Intelligence (25%)
                        core_score = (
                            0.12 * ml_base_score +                    # ML prediction (12%)
                            0.07 * industry_score +                   # Industry relevance (7%)
                            0.06 * deal_readiness_score               # Deal readiness (6%)
                        )
                        
                        # Company Characteristics (25%) - Based on SIRENE data
                        company_score = (
                            0.05 * df_sample.iloc[i].get("composite_tech_score", 0.3) +
                            0.04 * df_sample.iloc[i].get("composite_maturity_score", 0.3) +
                            0.04 * df_sample.iloc[i].get("composite_establishment_score", 0.3) +
                            0.04 * df_sample.iloc[i].get("composite_stability_score", 0.3) +
                            0.04 * df_sample.iloc[i].get("composite_growth_score", 0.3) +
                            0.04 * df_sample.iloc[i].get("composite_innovation_score", 0.3)
                        )
                        
                        # Employee & Company Size Intelligence (20%) - NEW SIRENE data
                        size_score = (
                            0.05 * df_sample.iloc[i].get("employee_range_score", 0.3) +
                            0.04 * df_sample.iloc[i].get("is_employer_score", 0) +
                            0.04 * df_sample.iloc[i].get("company_category_score", 0.3) +
                            0.03 * df_sample.iloc[i].get("legal_category_score", 0.3) +
                            0.02 * df_sample.iloc[i].get("establishment_count_score", 0.3) +
                            0.02 * df_sample.iloc[i].get("company_size_indicator", 0.3)
                        )
                        
                        # Geographic Intelligence (15%) - Based on SIRENE data
                        geographic_score = (
                            0.04 * df_sample.iloc[i].get("composite_geographic_score", 0.3) +
                            0.03 * df_sample.iloc[i].get("is_paris_region", 0) +
                            0.02 * df_sample.iloc[i].get("is_lyon_region", 0) +
                            0.02 * df_sample.iloc[i].get("is_marseille_region", 0) +
                            0.02 * df_sample.iloc[i].get("is_toulouse_region", 0) +
                            0.01 * df_sample.iloc[i].get("is_nice_region", 0) +
                            0.01 * df_sample.iloc[i].get("is_nantes_region", 0)
                        )
                        
                        # Business Intelligence (10%) - Based on SIRENE data
                        business_score = (
                            0.03 * df_sample.iloc[i].get("composite_business_score", 0.3) +
                            0.02 * df_sample.iloc[i].get("ape_tech_intensity", 0) +
                            0.02 * df_sample.iloc[i].get("ape_services_intensity", 0) +
                            0.02 * df_sample.iloc[i].get("name_has_sas", 0) +
                            0.01 * df_sample.iloc[i].get("name_has_sarl", 0)
                        )
                        
                        # Data Quality and Completeness (5%) - Based on SIRENE data
                        quality_score = (
                            0.02 * df_sample.iloc[i].get("composite_data_quality", 0.3) +
                            0.02 * df_sample.iloc[i].get("siren_is_valid", 0) +
                            0.01 * df_sample.iloc[i].get("ape_is_valid", 0)
                        )
                        
                        # Final composite score
                        p = core_score + company_score + size_score + geographic_score + business_score + quality_score
                        
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
                        
                        # Enhanced data point indicators
                        if company_name_score > 0.7:
                            reasons.append("Strong brand")
                        elif company_name_score > 0.5:
                            reasons.append("Professional name")
                        
                        if business_district_score > 0.8:
                            reasons.append("Prime location")
                        elif business_district_score > 0.6:
                            reasons.append("Good district")
                        
                        if website_score > 0.7:
                            reasons.append("Professional website")
                        
                        if email_score > 0.7:
                            reasons.append("Professional contact")
                        
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
            "message": f"Model trained successfully! Found {len(discovered_leads)} high-scoring leads from SIRENE database.",
            "training_insights": {
                "won_deals_patterns": artifacts["won_deals_patterns"],
                "feature_importance": artifacts["feature_importance"],
                "training_features": artifacts["training_features"],
                "smart_query_criteria": "Based on won deal patterns: " + ", ".join([
                    f"APE codes: {list(artifacts['won_deals_patterns']['ape_codes'].keys())[:2]}",
                    f"Employee ranges: {list(artifacts['won_deals_patterns']['employee_ranges'].keys())[:2]}",
                    f"Legal categories: {list(artifacts['won_deals_patterns']['legal_categories'].keys())[:2]}",
                    f"Regions: {list(artifacts['won_deals_patterns']['regions'].keys())[:2]}"
                ]) if len(won_deals) > 0 else "No patterns detected"
            }
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
        
        # Fetch from SIRENE API with smart querying
        if SIRENE_MODE == "api" and SIRENE_TOKEN:
            # Build smart query based on user filters and training patterns
            smart_query = build_smart_sirene_query_from_filters(f, tenant)
            print(f"Smart SIRENE query for discover: {smart_query}")
            
            # Fetch only companies that match our criteria (no client-side filtering needed)
            companies = sirene_fetch_api(smart_query, rows=limit*2, cap=limit*2)  # Get 2x limit for quality filtering
            sirene_used = "api"
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
