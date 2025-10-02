from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os

app = FastAPI(title="Chyll FastAPI MVP - Simple Test")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class TrainRequest(BaseModel):
    tenant_id: str
    rows: List[HistoryRow]

# ---- Routes
@app.get("/health")
def health(): 
    return {
        "ok": True, 
        "service": "chyll-fastapi-simple-test", 
        "message": "Simple test version working"
    }

@app.post("/train")
def train(req: TrainRequest):
    try:
        tenant = req.tenant_id
        rows = req.rows
        
        if not rows:
            return {"ok": False, "error": "no rows"}
        
        # Simple response without complex processing
        wins = sum(1 for row in rows if row.deal_status.lower() == "won")
        losses = len(rows) - wins
        
        return {
            "ok": True, 
            "stats": {
                "rows": len(rows), 
                "wins": wins, 
                "losses": losses
            }, 
            "model_version": f"{tenant}-v1-simple-test",
            "message": "Training completed successfully"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/discover")
def discover():
    return {
        "ok": True, 
        "items": [
            {
                "company_id": "test-1",
                "name": "Test Company",
                "win_score": 0.75,
                "band": "High",
                "confidence_badge": "Test Badge"
            }
        ], 
        "total": 1,
        "message": "Discovery completed successfully"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
