from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import time
import httpx
import tldextract
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Chyll FastAPI MVP - Minimal")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Config
SIRENE_MODE = os.getenv("SIRENE_MODE", "demo")
SIRENE_TOKEN = os.getenv("SIRENE_TOKEN", "")
SIRENE_BASE = "https://api.insee.fr/entreprises/sirene/V3/siren"
SIRENE_BULK_PATH = os.getenv("SIRENE_BULK_PATH", "")

# ---- Pydantic Schemas
class TrainRequest(BaseModel):
    tenant_id: str
    rows: List[Dict[str, Any]]

class DiscoverRequest(BaseModel):
    tenant_id: str
    filters: Dict[str, Any]

class ScoreRequest(BaseModel):
    tenant_id: str
    companies: List[Dict[str, Any]]

class ExportRequest(BaseModel):
    tenant_id: str
    companies: List[Dict[str, Any]]

# ---- Utility Functions
def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    if not url:
        return ""
    try:
        extracted = tldextract.extract(url)
        return f"{extracted.domain}.{extracted.suffix}" if extracted.domain else ""
    except:
        return ""

def compute_web_footprint(domain: str) -> str:
    """Compute web footprint label (simplified for demo)"""
    if not domain:
        return "None"
    
    # Simplified version for demo
    if domain.endswith('.com') or domain.endswith('.fr'):
        return "Strong"
    elif domain.endswith('.org') or domain.endswith('.net'):
        return "Medium"
    else:
        return "Weak"

def get_company_age(created_year: str) -> str:
    """Get company age bucket"""
    if not created_year:
        return "Unknown"
    
    try:
        year = int(created_year)
        current_year = 2024
        age = current_year - year
        
        if age <= 5:
            return "0-5"
        elif age <= 10:
            return "6-10"
        elif age <= 20:
            return "11-20"
        else:
            return "20+"
    except:
        return "Unknown"

# ---- API Endpoints
@app.get("/health")
def health():
    return {
        "ok": True, 
        "service": "chyll-fastapi-minimal", 
        "sirene_mode": SIRENE_MODE,
        "python_version": "3.11.9"
    }

@app.post("/train")
def train(req: TrainRequest):
    """Train ML model (simplified version)"""
    try:
        print(f"Training request for tenant: {req.tenant_id}")
        print(f"Number of rows: {len(req.rows)}")
        
        # Simulate training process
        time.sleep(1)
        
        return {
            "ok": True,
            "message": "Model training completed (minimal version)",
            "tenant_id": req.tenant_id,
            "rows_processed": len(req.rows),
            "features": ["company_name", "website", "ape", "created_year"],
            "model_type": "simplified"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/discover")
def discover(req: DiscoverRequest):
    """Discover companies (simplified version)"""
    try:
        print(f"Discovery request for tenant: {req.tenant_id}")
        print(f"Filters: {req.filters}")
        
        # Simulate discovery process
        time.sleep(1)
        
        # Return demo companies
        demo_companies = [
            {
                "company_name": "TechCorp France",
                "website": "techcorp.fr",
                "ape": "6201Z",
                "region": "Île-de-France",
                "created_year": "2020",
                "headcount": "5",
                "score": 0.85,
                "band": "High",
                "confidence": "High",
                "similar_past_wins": ["TechCorp France", "Innovate SAS"],
                "why": ["Similar APE code", "Same region"],
                "web_footprint": "Strong"
            },
            {
                "company_name": "Innovate SAS",
                "website": "innovate-sas.com",
                "ape": "6201Z",
                "region": "Île-de-France",
                "created_year": "2019",
                "headcount": "8",
                "score": 0.78,
                "band": "High",
                "confidence": "Medium",
                "similar_past_wins": ["TechCorp France"],
                "why": ["Similar APE code", "Same region"],
                "web_footprint": "Strong"
            }
        ]
        
        return {
            "ok": True,
            "companies": demo_companies,
            "total_found": len(demo_companies),
            "filters_applied": req.filters,
            "sirene_mode": SIRENE_MODE
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/score")
def score(req: ScoreRequest):
    """Score companies (simplified version)"""
    try:
        print(f"Scoring request for tenant: {req.tenant_id}")
        print(f"Number of companies: {len(req.companies)}")
        
        # Simulate scoring process
        time.sleep(0.5)
        
        scored_companies = []
        for company in req.companies:
            # Simple scoring based on domain
            domain = extract_domain(company.get("website", ""))
            web_footprint = compute_web_footprint(domain)
            
            # Simple score calculation
            base_score = 0.5
            if web_footprint == "Strong":
                base_score += 0.3
            elif web_footprint == "Medium":
                base_score += 0.2
            
            scored_companies.append({
                **company,
                "score": base_score,
                "web_footprint": web_footprint,
                "band": "High" if base_score > 0.7 else "Medium" if base_score > 0.5 else "Low"
            })
        
        return {
            "ok": True,
            "companies": scored_companies,
            "scoring_method": "simplified"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/export")
def export(req: ExportRequest):
    """Export companies to CSV (simplified version)"""
    try:
        print(f"Export request for tenant: {req.tenant_id}")
        print(f"Number of companies: {len(req.companies)}")
        
        # Simulate export process
        time.sleep(0.5)
        
        return {
            "ok": True,
            "message": "Export completed (simplified version)",
            "companies_exported": len(req.companies),
            "format": "CSV",
            "download_url": "https://example.com/export.csv"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
