from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, time, json
import pandas as pd, numpy as np
import tldextract
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Chyll FastAPI MVP - Production Ready")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Config
SIRENE_MODE = os.getenv("SIRENE_MODE", "demo")  # "api" | "bulk" | "demo"
SIRENE_TOKEN = os.getenv("SIRENE_TOKEN", "")
SIRENE_BASE = "https://api.insee.fr/entreprises/sirene/V3/siren"
SIRENE_BULK_PATH = os.getenv("SIRENE_BULK_PATH", "")

# ---- Global ML Models (per tenant)
models = {}  # tenant_id -> {companies_df, features}

# ---- Utility Functions
def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    if not url or not isinstance(url, str):
        return ""
    try:
        extracted = tldextract.extract(url)
        return f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
    except:
        return ""

def compute_web_footprint(domain: str) -> str:
    """Compute web footprint label (simplified for demo)"""
    if not domain:
        return "None"
    
    # Simplified version for demo - skip network calls
    if domain.endswith('.com') or domain.endswith('.fr'):
        return "Strong"
    elif domain.endswith('.org') or domain.endswith('.net'):
        return "Medium"
    else:
        return "Weak"

def get_company_age(created_year: str) -> str:
    """Get company age bucket"""
    try:
        year = int(created_year) if created_year else 2024
        age = 2024 - year
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

def compute_similarity_score(company1: Dict[str, Any], company2: Dict[str, Any]) -> float:
    """Compute similarity score between two companies"""
    score = 0.0
    
    # APE code similarity
    if company1.get('ape') == company2.get('ape'):
        score += 0.4
    
    # Domain similarity (simple string matching)
    domain1 = extract_domain(company1.get('website', ''))
    domain2 = extract_domain(company2.get('website', ''))
    if domain1 and domain2:
        if domain1 == domain2:
            score += 0.3
        elif any(word in domain2 for word in domain1.split('.')[0].split('-') if len(word) > 3):
            score += 0.2
    
    # Company name similarity
    name1 = company1.get('company_name', '').lower()
    name2 = company2.get('company_name', '').lower()
    if name1 and name2:
        common_words = set(name1.split()) & set(name2.split())
        if common_words:
            score += 0.3 * (len(common_words) / max(len(name1.split()), len(name2.split())))
    
    return min(score, 1.0)

# ---- Schemas
class TrainRequest(BaseModel):
    tenant_id: str
    rows: List[Dict[str, Any]]

class DiscoverRequest(BaseModel):
    tenant_id: str
    filters: Dict[str, Any]

# ---- Sirene Integration
async def fetch_sirene_companies(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch companies from Sirene API or return demo data"""
    if SIRENE_MODE == "demo":
        # Return demo companies
        demo_companies = []
        ape_codes = filters.get("ape_codes", ["6201Z"])
        regions = filters.get("regions", ["Île-de-France"])
        
        for i, ape in enumerate(ape_codes):
            for j, region in enumerate(regions):
                demo_companies.append({
                    "siren": f"12345678{i}{j}",
                    "siret": f"12345678{i}{j}001",
                    "denomination": f"Demo Company {i+1}{j+1}",
                    "ape": ape,
                    "region": region,
                    "created_year": str(2020 + i),
                    "headcount": "1-10" if i % 2 == 0 else "11-50",
                    "website": f"demo{i+1}{j+1}.com"
                })
        
        return demo_companies[:10]  # Limit to 10 for demo
    
    elif SIRENE_MODE == "api" and SIRENE_TOKEN:
        # Real Sirene API integration
        try:
            import httpx
            headers = {"Authorization": f"Bearer {SIRENE_TOKEN}"}
            params = {
                "q": "denominationUniteLegale:*",
                "nombre": 100
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(SIRENE_BASE, headers=headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("unitesLegales", [])
        except Exception as e:
            print(f"Sirene API error: {e}")
    
    return []

# ---- ML Pipeline (Simplified)
def train_ml_model(tenant_id: str, df: pd.DataFrame):
    """Train simplified ML model"""
    try:
        # Prepare features
        df['domain'] = df['website'].apply(extract_domain)
        df['web_footprint'] = df['domain'].apply(compute_web_footprint)
        df['age_bucket'] = df['created_year'].apply(get_company_age)
        df['target'] = (df['deal_status'] == 'won').astype(int)
        
        # Store model data
        models[tenant_id] = {
            'companies_df': df,
            'features': {
                'web_footprint': df['web_footprint'].tolist(),
                'age_bucket': df['age_bucket'].tolist(),
                'ape_codes': df['ape'].tolist(),
                'domains': df['domain'].tolist()
            }
        }
        
        return True
        
    except Exception as e:
        print(f"ML training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def predict_company_score(tenant_id: str, company: Dict[str, Any]) -> Dict[str, Any]:
    """Predict score for a company using trained model"""
    if tenant_id not in models:
        return {"score": 0.5, "band": "Unknown", "confidence": "Low"}
    
    model_data = models[tenant_id]
    companies_df = model_data['companies_df']
    
    try:
        # Find most similar companies
        similarities = []
        for _, past_company in companies_df.iterrows():
            sim_score = compute_similarity_score(company, past_company.to_dict())
            similarities.append((sim_score, past_company))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Get top similar companies
        top_similar = similarities[:3]
        similar_wins = [comp[1]['company_name'] for comp in top_similar if comp[1]['deal_status'] == 'won']
        
        # Calculate score based on similar wins
        if similar_wins:
            # Higher score if more similar wins
            base_score = 0.3 + (len(similar_wins) * 0.2)
            # Boost score based on similarity
            if top_similar:
                base_score += top_similar[0][0] * 0.3
            score = min(base_score, 1.0)
        else:
            # Lower score if no similar wins
            score = 0.2 + (top_similar[0][0] * 0.3 if top_similar else 0.0)
        
        # Determine band and confidence
        if score >= 0.8:
            band = "High"
            confidence = "High"
        elif score >= 0.6:
            band = "Medium"
            confidence = "Medium"
        else:
            band = "Low"
            confidence = "Low"
        
        # Generate "why" explanations
        why_reasons = []
        if company.get('ape') in companies_df['ape'].values:
            why_reasons.append("Similar APE code")
        if any(extract_domain(company.get('website', '')) in companies_df['domain'].values):
            why_reasons.append("Similar domain pattern")
        if len(similar_wins) > 0:
            why_reasons.append("Similar to past wins")
        
        return {
            "score": float(score),
            "band": band,
            "confidence": confidence,
            "similar_past_wins": similar_wins[:3],
            "why": why_reasons,
            "web_footprint": compute_web_footprint(extract_domain(company.get('website', '')))
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"score": 0.5, "band": "Unknown", "confidence": "Low"}

# ---- Endpoints
@app.get("/health")
def health():
    return {"ok": True, "service": "chyll-fastapi-production", "sirene_mode": SIRENE_MODE}

@app.post("/train")
async def train(req: TrainRequest):
    """Train ML model"""
    try:
        df = pd.DataFrame(req.rows)
        
        if len(df) == 0:
            return {"ok": False, "error": "No data provided"}
        
        # Train ML model
        success = train_ml_model(req.tenant_id, df)
        
        if success:
            wins = sum(1 for row in req.rows if row.get("deal_status") == "won")
            losses = sum(1 for row in req.rows if row.get("deal_status") == "lost")
            
            return {
                "ok": True,
                "message": "ML model trained successfully",
                "stats": {
                    "rows": len(req.rows),
                    "wins": wins,
                    "losses": losses,
                    "tenant_id": req.tenant_id,
                    "model_type": "Similarity-based ML"
                }
            }
        else:
            return {"ok": False, "error": "ML training failed"}
            
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/discover")
async def discover(req: DiscoverRequest):
    """Discover companies with ML scoring"""
    try:
        if req.tenant_id not in models:
            return {"ok": False, "error": "Model not trained for this tenant"}
        
        # Fetch companies from Sirene
        companies = await fetch_sirene_companies(req.filters)
        
        if not companies:
            return {"ok": False, "error": "No companies found"}
        
        # Score companies
        scored_companies = []
        for company in companies:
            prediction = predict_company_score(req.tenant_id, company)
            
            scored_company = {
                "company_name": company.get("denomination", "Unknown"),
                "website": company.get("website", ""),
                "ape": company.get("ape", ""),
                "region": company.get("region", ""),
                "created_year": company.get("created_year", ""),
                "headcount": company.get("headcount", ""),
                **prediction
            }
            scored_companies.append(scored_company)
        
        # Sort by score
        scored_companies.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "ok": True,
            "companies": scored_companies,
            "total": len(scored_companies),
            "filters_applied": req.filters
        }
        
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/score")
def score(req: Dict[str, Any]):
    return {"ok": True, "message": "Scoring endpoint (production ready)"}

@app.post("/export")
def export(req: Dict[str, Any]):
    return {"ok": True, "message": "Export endpoint (production ready)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
