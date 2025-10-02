from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, time

app = FastAPI(title="Chyll FastAPI MVP - Simple")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ---- Config
SIRENE_MODE = os.getenv("SIRENE_MODE", "demo")  # "api" | "bulk" | "demo"

# ---- Schemas
class TrainRequest(BaseModel):
    tenant_id: str
    rows: List[Dict[str, Any]]

class DiscoverRequest(BaseModel):
    tenant_id: str
    filters: Dict[str, Any]

# ---- Endpoints
@app.get("/health")
def health():
    return {"ok": True, "service": "chyll-fastapi-simple", "sirene_mode": SIRENE_MODE}

@app.post("/train")
def train(req: TrainRequest):
    """Simple training endpoint for testing"""
    wins = sum(1 for row in req.rows if row.get("deal_status") == "won")
    losses = sum(1 for row in req.rows if row.get("deal_status") == "lost")
    
    return {
        "ok": True,
        "message": "Model trained successfully (simplified version)",
        "stats": {
            "rows": len(req.rows),
            "wins": wins,
            "losses": losses,
            "tenant_id": req.tenant_id
        }
    }

@app.post("/discover")
def discover(req: DiscoverRequest):
    """Simple discovery endpoint for testing"""
    # Return demo companies
    demo_companies = [
        {
            "company_name": "Demo Tech Corp",
            "website": "demo-tech.com",
            "ape": "6201Z",
            "score": 0.85,
            "band": "High",
            "confidence": "High",
            "similar_past_wins": ["TechCorp France"],
            "why": ["Similar APE code", "Tech industry"],
            "web_footprint": "Strong"
        },
        {
            "company_name": "Sample Innovation SAS",
            "website": "sample-innovation.com", 
            "ape": "6201Z",
            "score": 0.72,
            "band": "Medium",
            "confidence": "Medium",
            "similar_past_wins": ["Innovate SAS"],
            "why": ["Similar naming pattern", "Same APE code"],
            "web_footprint": "Medium"
        }
    ]
    
    return {
        "ok": True,
        "companies": demo_companies,
        "total": len(demo_companies),
        "filters_applied": req.filters
    }

@app.post("/score")
def score(req: Dict[str, Any]):
    return {"ok": True, "message": "Scoring endpoint (simplified)"}

@app.post("/export")
def export(req: Dict[str, Any]):
    return {"ok": True, "message": "Export endpoint (simplified)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
