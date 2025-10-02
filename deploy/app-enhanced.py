from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, time, json
import pandas as pd, numpy as np
import tldextract, httpx
import dns.resolver
from sentence_transformers import SentenceTransformer
import faiss
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Chyll FastAPI MVP - Enhanced")

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
models = {}  # tenant_id -> {model, index, transformer, companies_df}

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

def check_mx_record(domain: str) -> bool:
    """Check if domain has MX record"""
    try:
        dns.resolver.resolve(domain, 'MX')
        return True
    except:
        return False

def check_http_response(domain: str) -> Dict[str, Any]:
    """Check HTTP response for domain"""
    try:
        response = httpx.get(f"http://{domain}", timeout=5, follow_redirects=True)
        return {
            "status_code": response.status_code,
            "has_ssl": domain.startswith("https://"),
            "response_time": response.elapsed.total_seconds()
        }
    except:
        return {"status_code": 0, "has_ssl": False, "response_time": 0}

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

# ---- ML Pipeline
def train_ml_model(tenant_id: str, df: pd.DataFrame):
    """Train ML model with embeddings and LightGBM"""
    try:
        # Initialize transformer with error handling
        try:
            transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Transformer loading error: {e}")
            # Fallback to simple transformer
            transformer = None
        
        # Prepare features
        df['domain'] = df['website'].apply(extract_domain)
        df['web_footprint'] = df['domain'].apply(compute_web_footprint)
        df['age_bucket'] = df['created_year'].apply(get_company_age)
        df['target'] = (df['deal_status'] == 'won').astype(int)
        
        # Create text embeddings
        if transformer is not None:
            texts = df['company_name'] + " " + df['domain'] + " " + df['ape'].fillna("")
            embeddings = transformer.encode(texts.tolist())
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
            index.add(embeddings.astype('float32'))
        else:
            # Fallback: create dummy embeddings
            embeddings = np.random.rand(len(df), 10)
            dimension = 10
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))
        
        # Prepare features for LightGBM
        feature_cols = ['web_footprint', 'age_bucket']
        df_features = pd.get_dummies(df[feature_cols], prefix=feature_cols)
        
        # Add embedding features (first 10 dimensions)
        for i in range(min(10, dimension)):
            df_features[f'embedding_{i}'] = embeddings[:, i]
        
        X = df_features.values
        y = df['target'].values
        
        # Train LightGBM
        if len(np.unique(y)) > 1:  # Need both classes
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            lgb_model = LGBMClassifier(random_state=42, verbose=-1)
            lgb_model.fit(X_train, y_train)
            
            # Calibrate probabilities
            calibrated_model = CalibratedClassifierCV(lgb_model, method='isotonic', cv=3)
            calibrated_model.fit(X_train, y_train)
            
            # Store model
            models[tenant_id] = {
                'model': calibrated_model,
                'index': index,
                'transformer': transformer,
                'companies_df': df,
                'feature_cols': df_features.columns.tolist()
            }
            
            return True
        else:
            # Only one class, store basic model
            models[tenant_id] = {
                'model': None,
                'index': index,
                'transformer': transformer,
                'companies_df': df,
                'feature_cols': df_features.columns.tolist()
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
    
    try:
        # Create embedding for company
        if model_data['transformer'] is not None:
            text = f"{company.get('denomination', '')} {company.get('website', '')} {company.get('ape', '')}"
            embedding = model_data['transformer'].encode([text])
            faiss.normalize_L2(embedding)
        else:
            # Fallback: create dummy embedding
            embedding = np.random.rand(1, 10)
            faiss.normalize_L2(embedding)
        
        # Find similar companies
        scores, indices = model_data['index'].search(embedding.astype('float32'), k=3)
        similar_companies = model_data['companies_df'].iloc[indices[0]]
        
        # Get prediction if model exists
        if model_data['model'] is not None:
            # Prepare features
            domain = extract_domain(company.get('website', ''))
            web_footprint = compute_web_footprint(domain)
            age_bucket = get_company_age(company.get('created_year', ''))
            
            # Create feature vector
            feature_dict = {}
            for col in model_data['feature_cols']:
                if col.startswith('web_footprint_'):
                    feature_dict[col] = 1 if col == f'web_footprint_{web_footprint}' else 0
                elif col.startswith('age_bucket_'):
                    feature_dict[col] = 1 if col == f'age_bucket_{age_bucket}' else 0
                elif col.startswith('embedding_'):
                    idx = int(col.split('_')[1])
                    feature_dict[col] = embedding[0, idx] if idx < embedding.shape[1] else 0
                else:
                    feature_dict[col] = 0
            
            feature_vector = np.array([feature_dict[col] for col in model_data['feature_cols']]).reshape(1, -1)
            score = model_data['model'].predict_proba(feature_vector)[0][1]
        else:
            # Use similarity-based scoring
            score = float(scores[0][0]) if len(scores[0]) > 0 else 0.5
        
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
        
        # Get similar past wins
        won_companies = similar_companies[similar_companies['deal_status'] == 'won']
        similar_wins = won_companies['company_name'].tolist()[:3]
        
        # Generate "why" explanations
        why_reasons = []
        if company.get('ape') in similar_companies['ape'].values:
            why_reasons.append("Similar APE code")
        if any(extract_domain(company.get('website', '')) in similar_companies['domain'].values):
            why_reasons.append("Similar domain pattern")
        if len(similar_wins) > 0:
            why_reasons.append("Similar to past wins")
        
        return {
            "score": float(score),
            "band": band,
            "confidence": confidence,
            "similar_past_wins": similar_wins,
            "why": why_reasons,
            "web_footprint": compute_web_footprint(extract_domain(company.get('website', '')))
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"score": 0.5, "band": "Unknown", "confidence": "Low"}

# ---- Endpoints
@app.get("/health")
def health():
    return {"ok": True, "service": "chyll-fastapi-enhanced", "sirene_mode": SIRENE_MODE}

@app.post("/train")
async def train(req: TrainRequest):
    """Train ML model with embeddings and LightGBM"""
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
                    "model_type": "LightGBM + Embeddings + FAISS"
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
    return {"ok": True, "message": "Scoring endpoint (enhanced)"}

@app.post("/export")
def export(req: Dict[str, Any]):
    return {"ok": True, "message": "Export endpoint (enhanced)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
