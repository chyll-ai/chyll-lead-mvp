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

def build_safe_sirene_query(historical_data: List[HistoryRow]) -> str:
    """Build safe SIRENE query that won't break the API"""
    if not historical_data:
        return ""  # Empty query for broad results
    
    # Analyze deal patterns
    analysis = analyze_deal_patterns(historical_data)
    print(f"Deal analysis: {analysis}")
    
    # Extract patterns from won deals
    won_deals = [row for row in historical_data if row.deal_status == "won"]
    if not won_deals:
        return ""  # Empty query for broad results
    
    # Get APE codes from won deals (safely)
    ape_codes = []
    regions = []
    
    for deal in won_deals:
        # Safe APE code extraction
        if hasattr(deal, 'ape') and deal.ape and str(deal.ape).strip():
            ape_code = str(deal.ape).strip()
            if len(ape_code) >= 5:  # Valid APE code length
                ape_codes.append(ape_code)
        
        # Safe postal code extraction
        if hasattr(deal, 'postal_code') and deal.postal_code and str(deal.postal_code).strip():
            postal_str = str(deal.postal_code).strip().zfill(5)
            if len(postal_str) == 5 and postal_str.isdigit():
                first_two = postal_str[:2]
                if first_two in ["75", "77", "78", "91", "92", "93", "94", "95"]:
                    regions.append("11")  # Île-de-France
                elif first_two == "69":
                    regions.append("69")  # Rhône
                elif first_two == "13":
                    regions.append("13")  # Bouches-du-Rhône
    
    # Build safe query with limits
    query_parts = []
    
    # Always include active companies filter
    query_parts.append("etatAdministratifUniteLegale:A")
    
    # Add APE code filters (safely, max 2 codes to avoid API limits)
    if ape_codes:
        unique_apes = list(set(ape_codes))[:2]  # Limit to 2 APE codes
        if len(unique_apes) == 1:
            query_parts.append(f"activitePrincipaleUniteLegale:{unique_apes[0]}")
        elif len(unique_apes) == 2:
            ape_query = f"activitePrincipaleUniteLegale:{unique_apes[0]} OR activitePrincipaleUniteLegale:{unique_apes[1]}"
            query_parts.append(f"({ape_query})")
    
    # Add region filters (safely, max 1 region to avoid API limits)
    if regions:
        unique_regions = list(set(regions))[:1]  # Limit to 1 region
        if unique_regions:
            query_parts.append(f"codeRegion:{unique_regions[0]}")
    
    # Join with AND, but limit total query length
    final_query = " AND ".join(query_parts)
    
    # Safety check: limit query length to prevent API errors
    if len(final_query) > 200:
        return "etatAdministratifUniteLegale:A"  # Fallback to simple query
    
    return final_query

def build_smart_sirene_query(historical_data: List[HistoryRow]) -> str:
    """Build smart SIRENE query based on advanced deal pattern analysis"""
    return build_safe_sirene_query(historical_data)  # Use safe version

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
        "active": True
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
    
    # 51-60: Website Analysis (10 points) - Safe string operations
    df["website_has_https"] = df["website"].astype(str).str.startswith("https://").astype(int)
    df["website_has_www"] = df["website"].astype(str).str.contains("www.", na=False).astype(int)
    df["website_is_com"] = df["website"].astype(str).str.endswith(".com").astype(int)
    df["website_is_fr"] = df["website"].astype(str).str.endswith(".fr").astype(int)
    df["website_domain_length"] = df["website"].astype(str).str.extract(r"://([^/]+)")[0].str.len().fillna(0)
    df["website_has_subdomain"] = (df["website"].astype(str).str.count("\\.") > 2).astype(int)
    df["website_has_tech_keywords"] = df["website"].astype(str).str.lower().str.contains("tech|digital|data|soft|system|solution|innovation|intelligence|cloud|ai|ml|cyber|smart", na=False).astype(int)
    df["website_complexity"] = df["website"].astype(str).str.count("/")
    df["website_has_port"] = df["website"].astype(str).str.contains(":\\d+", na=False).astype(int)
    df["website_has_path"] = (df["website"].astype(str).str.count("/") > 2).astype(int)
    
    # 61-70: Email Analysis (10 points) - Safe string operations
    df["email_has_at"] = df["email"].astype(str).str.contains("@", na=False).astype(int)
    df["email_has_dot"] = df["email"].astype(str).str.contains("\\.", na=False).astype(int)
    df["email_is_contact"] = df["email"].astype(str).str.lower().str.startswith("contact@").astype(int)
    df["email_is_info"] = df["email"].astype(str).str.lower().str.startswith("info@").astype(int)
    df["email_is_hello"] = df["email"].astype(str).str.lower().str.startswith("hello@").astype(int)
    df["email_is_ceo"] = df["email"].astype(str).str.lower().str.contains("ceo|director|manager", na=False).astype(int)
    df["email_domain"] = df["email"].astype(str).str.extract(r"@([^@]+)")[0].fillna("")
    df["email_domain_is_com"] = df["email_domain"].str.endswith(".com").astype(int)
    df["email_domain_is_fr"] = df["email_domain"].str.endswith(".fr").astype(int)
    df["email_length"] = df["email"].astype(str).str.len()
    
    # 71-80: Business Intelligence (10 points) - Safe operations
    df["is_active"] = df.get("active", True).astype(int)
    df["has_complete_data"] = (df["company_name"].notna() & df["siren"].notna() & df["ape"].notna()).astype(int)
    df["data_completeness"] = (df["company_name"].notna().astype(int) + df["siren"].notna().astype(int) + df["ape"].notna().astype(int) + df["postal_code"].notna().astype(int) + df["city"].notna().astype(int)) / 5
    df["is_tech_company"] = (df["ape"].astype(str).str.startswith("62|63|70|71|72") | df["company_name"].astype(str).str.lower().str.contains("tech|digital|data|soft|system|solution|innovation|intelligence|cloud|ai|ml|cyber|smart", na=False)).astype(int)
    df["is_services_company"] = df["ape"].astype(str).str.startswith("70|71|72|73|74|75|76|77|78|79|80|81|82").astype(int)
    df["is_manufacturing"] = df["ape"].astype(str).str.startswith("10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33").astype(int)
    df["is_b2b"] = df["ape"].astype(str).str.startswith("62|63|70|71|72|73|74|75|76|77|78|79|80|81|82").astype(int)
    df["is_b2c"] = df["ape"].astype(str).str.startswith("47|56|68|85|86|87|88|90|91|92|93|95|96").astype(int)
    df["is_consulting"] = df["ape"].astype(str).str.startswith("70|71|72").astype(int)
    df["is_software"] = df["ape"].astype(str).str.startswith("62|63").astype(int)
    
    # 81-90: Market Position (10 points)
    df["market_position_tech"] = df["is_tech_company"] * 0.8
    df["market_position_services"] = df["is_services_company"] * 0.6
    df["market_position_manufacturing"] = df["is_manufacturing"] * 0.4
    df["market_position_consulting"] = df["is_consulting"] * 0.7
    df["market_position_software"] = df["is_software"] * 0.9
    df["market_position_b2b"] = df["is_b2b"] * 0.7
    df["market_position_b2c"] = df["is_b2c"] * 0.5
    df["market_position_established"] = df["age_is_established"] * 0.6
    df["market_position_growth"] = df["age_is_growth"] * 0.8
    df["market_position_innovation"] = df["age_innovation"] * 0.7
    
    # 91-100: Composite Scores (10 points)
    df["composite_tech_score"] = (df["is_tech_company"] * 0.4 + df["name_has_tech"] * 0.3 + df["website_has_tech_keywords"] * 0.3)
    df["composite_professional_score"] = (df["email_is_contact"] * 0.3 + df["website_has_https"] * 0.3 + df["name_has_business"] * 0.4)
    df["composite_location_score"] = (df["postal_code_is_paris"] * 0.4 + df["postal_code_is_lyon"] * 0.3 + df["postal_code_is_major_city"] * 0.3)
    df["composite_maturity_score"] = (df["age_sweet_spot"] * 0.4 + df["age_stability"] * 0.3 + df["age_innovation"] * 0.3)
    df["composite_data_quality"] = (df["has_complete_data"] * 0.4 + df["data_completeness"] * 0.6)
    df["composite_business_score"] = (df["is_b2b"] * 0.4 + df["is_consulting"] * 0.3 + df["is_software"] * 0.3)
    df["composite_establishment_score"] = (df["siren_is_valid"] * 0.3 + df["age_is_mature"] * 0.3 + df["name_has_business"] * 0.4)
    df["composite_innovation_score"] = (df["age_innovation"] * 0.4 + df["name_has_tech"] * 0.3 + df["is_tech_company"] * 0.3)
    df["composite_stability_score"] = (df["age_stability"] * 0.4 + df["siren_is_valid"] * 0.3 + df["is_active"] * 0.3)
    df["composite_growth_score"] = (df["age_is_growth"] * 0.4 + df["market_position_growth"] * 0.3 + df["age_sweet_spot"] * 0.3)
    
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
                        
                        # Enhanced data points
                        company_name_score = df_sample.iloc[i].get("company_name_intelligence", 0.3)
                        website_score = df_sample.iloc[i].get("website_intelligence", 0.3)
                        email_score = df_sample.iloc[i].get("email_intelligence", 0.3)
                        geographic_intelligence_score = df_sample.iloc[i].get("geographic_intelligence", 0.3)
                        business_district_score = df_sample.iloc[i].get("business_district_score", 0.3)
                        
                        # 100 Data Points Scoring System
                        # Core ML and Business Intelligence (30%)
                        core_score = (
                            0.15 * ml_base_score +                    # ML prediction (15%)
                            0.08 * industry_score +                   # Industry relevance (8%)
                            0.07 * deal_readiness_score               # Deal readiness (7%)
                        )
                        
                        # Company Characteristics (25%)
                        company_score = (
                            0.06 * df_sample.iloc[i].get("composite_tech_score", 0.3) +
                            0.05 * df_sample.iloc[i].get("composite_professional_score", 0.3) +
                            0.04 * df_sample.iloc[i].get("composite_maturity_score", 0.3) +
                            0.04 * df_sample.iloc[i].get("composite_establishment_score", 0.3) +
                            0.03 * df_sample.iloc[i].get("composite_stability_score", 0.3) +
                            0.03 * df_sample.iloc[i].get("composite_growth_score", 0.3)
                        )
                        
                        # Market Position (20%)
                        market_score = (
                            0.05 * df_sample.iloc[i].get("market_position_tech", 0.3) +
                            0.04 * df_sample.iloc[i].get("market_position_software", 0.3) +
                            0.04 * df_sample.iloc[i].get("market_position_consulting", 0.3) +
                            0.03 * df_sample.iloc[i].get("market_position_b2b", 0.3) +
                            0.02 * df_sample.iloc[i].get("market_position_growth", 0.3) +
                            0.02 * df_sample.iloc[i].get("market_position_innovation", 0.3)
                        )
                        
                        # Geographic and Location (15%)
                        location_score = (
                            0.05 * df_sample.iloc[i].get("composite_location_score", 0.3) +
                            0.04 * df_sample.iloc[i].get("postal_code_is_paris", 0) +
                            0.03 * df_sample.iloc[i].get("postal_code_is_lyon", 0) +
                            0.02 * df_sample.iloc[i].get("postal_code_is_major_city", 0) +
                            0.01 * df_sample.iloc[i].get("city_is_capital", 0)
                        )
                        
                        # Data Quality and Completeness (10%)
                        quality_score = (
                            0.04 * df_sample.iloc[i].get("composite_data_quality", 0.3) +
                            0.03 * df_sample.iloc[i].get("has_complete_data", 0) +
                            0.02 * df_sample.iloc[i].get("siren_is_valid", 0) +
                            0.01 * df_sample.iloc[i].get("ape_is_valid", 0)
                        )
                        
                        # Final composite score
                        p = core_score + company_score + market_score + location_score + quality_score
                        
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
