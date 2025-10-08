#!/usr/bin/env python3
"""
Tabula Virtutis - ESS Companies Map API
Simplified API serving only ess_companies_filtered_table
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tabula Virtutis - ESS Companies API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
import os

DB_CONFIG = {
    'host': os.getenv('PGHOST'),
    'port': os.getenv('PGPORT'),
    'database': os.getenv('PGDATABASE'),
    'user': os.getenv('PGUSER'),
    'password': os.getenv('PGPASSWORD')
}

class Company(BaseModel):
    siren: str
    siret: str
    denomination_unite_legale: Optional[str] = None
    libelle_commune: Optional[str] = None
    code_postal: Optional[str] = None
    latitude: float
    longitude: float
    tags: List[str]  # ['ESS', 'QPV', 'ZRR']
    qpv_label: Optional[str] = None
    zrr_classification: Optional[str] = None
    activite_principale_unite_legale: Optional[str] = None

class CompanyStats(BaseModel):
    total_companies: int
    qpv_companies: int
    zrr_companies: int
    unique_communes: int

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG)

@app.get("/")
async def root():
    return {"message": "Tabula Virtutis - ESS Companies API", "version": "2.0.0"}

@app.get("/test")
async def test():
    return {"status": "ok", "message": "Tabula Virtutis API is working"}

@app.post("/fix-saint-denis-zrr")
async def fix_saint_denis_zrr():
    """Fix ZRR misclassification for companies in Seine-Saint-Denis (93xxx)"""
    try:
        cursor = conn.cursor()
        
        # Check current state
        cursor.execute("""
            SELECT COUNT(*) as total_companies,
                   COUNT(CASE WHEN is_zrr = true THEN 1 END) as zrr_classified
            FROM ess_companies_filtered_table 
            WHERE code_postal LIKE '93%'
        """)
        
        before_result = cursor.fetchone()
        
        # Update ZRR classification for Seine-Saint-Denis companies
        cursor.execute("""
            UPDATE ess_companies_filtered_table 
            SET is_zrr = false, 
                zrr_classification = NULL
            WHERE code_postal LIKE '93%' 
            AND is_zrr = true
        """)
        
        updated_count = cursor.rowcount
        conn.commit()
        
        # Check after state
        cursor.execute("""
            SELECT COUNT(*) as remaining_zrr_in_93
            FROM ess_companies_filtered_table 
            WHERE code_postal LIKE '93%' AND is_zrr = true
        """)
        
        after_result = cursor.fetchone()
        
        cursor.close()
        
        return {
            "message": "Saint-Denis ZRR classification fixed",
            "before": {
                "total_companies": before_result[0],
                "zrr_classified": before_result[1]
            },
            "updated_count": updated_count,
            "after": {
                "remaining_zrr": after_result[0]
            },
            "success": True
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

@app.get("/companies", response_model=List[Company])
async def get_companies(
    ess: bool = Query(False, description="Filter by ESS companies only (excludes ZRR)"),
    qpv: bool = Query(False, description="Filter by QPV companies only"),
    zrr: bool = Query(False, description="Filter by ZRR companies only"),
    commune: Optional[str] = Query(None, description="Filter by commune name"),
    department: Optional[str] = Query(None, description="Filter by department code"),
    code_postal: Optional[str] = Query(None, description="Filter by postal code prefix"),
    activity_code: Optional[str] = Query(None, description="Filter by activity code"),
    search: Optional[str] = Query(None, description="Search in company names"),
    limit: int = Query(1000, description="Maximum number of companies to return"),
    offset: int = Query(0, description="Number of companies to skip")
):
    """Get ESS companies with comprehensive filtering"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Simple base query - no complex joins or unions
        query = """
            SELECT 
                siren, siret, denomination_unite_legale, libelle_commune, code_postal,
                latitude, longitude, in_qpv, is_zrr, qpv_label, zrr_classification, 
                activite_principale_unite_legale
            FROM ess_companies_filtered_table
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """
        
        # Add filters
        filters = []
        params = []
        
        if ess:
            filters.append("is_zrr = FALSE")  # ESS companies exclude ZRR
        
        if qpv:
            filters.append("in_qpv = TRUE")
        
        if zrr:
            filters.append("is_zrr = TRUE")
            
        if commune:
            filters.append("LOWER(libelle_commune) LIKE LOWER(%s)")
            params.append(f"%{commune}%")
            
        if department:
            filters.append("code_postal LIKE %s")
            params.append(f"{department}%")
            
        if code_postal:
            filters.append("code_postal LIKE %s")
            params.append(f"{code_postal}%")
            
        if activity_code:
            filters.append("activite_principale_unite_legale LIKE %s")
            params.append(f"{activity_code}%")
            
        if search:
            filters.append("LOWER(denomination_unite_legale) LIKE LOWER(%s)")
            params.append(f"%{search}%")
        
        # Combine query
        if filters:
            query += " AND " + " AND ".join(filters)
        
        query += f" ORDER BY denomination_unite_legale, siren LIMIT {limit} OFFSET {offset}"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        companies = []
        for row in results:
            tags = ['ESS']
            
            # Add QPV and ZRR tags if applicable
            if row[7]:  # in_qpv
                tags.append('QPV')
            if row[8]:  # is_zrr
                tags.append('ZRR')
            
            companies.append(Company(
                siren=row[0],
                siret=row[1],
                denomination_unite_legale=row[2],
                libelle_commune=row[3],
                code_postal=row[4],
                latitude=float(row[5]),
                longitude=float(row[6]),
                tags=tags,
                qpv_label=row[9],
                zrr_classification=row[10],
                activite_principale_unite_legale=row[11]
            ))
        
        cursor.close()
        conn.close()
        
        logger.info(f"Returned {len(companies)} ESS companies")
        return companies
        
    except Exception as e:
        logger.error(f"Error fetching companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/companies/region")
async def get_companies_by_region(
    region: str = Query("mayotte", description="Region to load (mayotte, paris, france)"),
    limit: int = Query(200, description="Maximum number of companies to return")
):
    """Get ESS companies by region for smart initial loading"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Define region-specific queries
        if region.lower() == "mayotte":
            # Mayotte - focus on Mamoudzou and surrounding areas
            query = """
                SELECT 
                    siren, siret, denomination_unite_legale, libelle_commune, code_postal,
                    latitude, longitude, in_qpv, is_zrr, qpv_label, zrr_classification, 
                    activite_principale_unite_legale
                FROM ess_companies_filtered_table
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                AND (code_postal LIKE '976%' OR libelle_commune ILIKE '%MAMOUDZOU%' OR libelle_commune ILIKE '%MAYOTTE%')
                ORDER BY denomination_unite_legale
                LIMIT %s
            """
        elif region.lower() == "paris":
            # Paris region (75 + surrounding areas)
            query = """
                SELECT 
                    siren, siret, denomination_unite_legale, libelle_commune, code_postal,
                    latitude, longitude, in_qpv, is_zrr, qpv_label, zrr_classification, 
                    activite_principale_unite_legale
                FROM ess_companies_filtered_table
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                AND (code_postal LIKE '75%' OR libelle_commune ILIKE '%paris%')
                ORDER BY denomination_unite_legale
                LIMIT %s
            """
        else:
            # Default to a sample from France
            query = """
                SELECT 
                    siren, siret, denomination_unite_legale, libelle_commune, code_postal,
                    latitude, longitude, in_qpv, is_zrr, qpv_label, zrr_classification, 
                    activite_principale_unite_legale
                FROM ess_companies_filtered_table
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                ORDER BY denomination_unite_legale
                LIMIT %s
            """
        
        cursor.execute(query, [limit])
        results = cursor.fetchall()
        
        companies = []
        for row in results:
            tags = ['ESS']
            
            # Add QPV and ZRR tags if applicable
            if row[7]:  # in_qpv
                tags.append('QPV')
            if row[8]:  # is_zrr
                tags.append('ZRR')
            
            companies.append(Company(
                siren=row[0],
                siret=row[1],
                denomination_unite_legale=row[2],
                libelle_commune=row[3],
                code_postal=row[4],
                latitude=float(row[5]),
                longitude=float(row[6]),
                tags=tags,
                qpv_label=row[9],
                zrr_classification=row[10],
                activite_principale_unite_legale=row[11]
            ))
        
        cursor.close()
        conn.close()
        
        logger.info(f"Returned {len(companies)} ESS companies from {region}")
        return {
            "companies": companies,
            "region": region,
            "total": len(companies)
        }
        
    except Exception as e:
        logger.error(f"Error fetching companies by region: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/filter")
async def filter_companies(
    ess: bool = Query(False, description="Filter by ESS companies only (excludes ZRR)"),
    qpv: bool = Query(False, description="Filter by QPV companies only"),
    zrr: bool = Query(False, description="Filter by ZRR companies only"),
    commune: Optional[str] = Query(None, description="Filter by commune name"),
    department: Optional[str] = Query(None, description="Filter by department code"),
    code_postal: Optional[str] = Query(None, description="Filter by postal code prefix"),
    activity_code: Optional[str] = Query(None, description="Filter by activity code"),
    search: Optional[str] = Query(None, description="Search in company names"),
    limit: int = Query(1000, description="Maximum number of companies to return"),
    offset: int = Query(0, description="Number of companies to skip")
):
    """Filter ESS companies - frontend-compatible endpoint"""
    try:
        companies = await get_companies(
            ess=ess, qpv=qpv, zrr=zrr, commune=commune, department=department,
            code_postal=code_postal, activity_code=activity_code, search=search, limit=limit, offset=offset
        )
        
        return {
            "companies": companies,
            "total": len(companies),
            "filters_applied": {
                "ess": ess,
                "qpv": qpv,
                "zrr": zrr,
                "commune": commune,
                "department": department,
                "activity_code": activity_code,
                "search": search
            }
        }
        
    except Exception as e:
        logger.error(f"Error in filter endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=CompanyStats)
async def get_stats():
    """Get simplified ESS company statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total ESS companies count
        cursor.execute("SELECT COUNT(*) FROM ess_companies_filtered_table WHERE latitude IS NOT NULL")
        total_count = cursor.fetchone()[0]
        
        # QPV companies count
        cursor.execute("SELECT COUNT(*) FROM ess_companies_filtered_table WHERE in_qpv = TRUE AND latitude IS NOT NULL")
        qpv_count = cursor.fetchone()[0]
        
        # ZRR companies count
        cursor.execute("SELECT COUNT(*) FROM ess_companies_filtered_table WHERE is_zrr = TRUE AND latitude IS NOT NULL")
        zrr_count = cursor.fetchone()[0]
        
        # Unique communes count
        cursor.execute("""
            SELECT COUNT(DISTINCT libelle_commune) 
            FROM ess_companies_filtered_table 
            WHERE libelle_commune IS NOT NULL AND latitude IS NOT NULL
        """)
        unique_communes = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return CompanyStats(
            total_companies=total_count,
            qpv_companies=qpv_count,
            zrr_companies=zrr_count,
            unique_communes=unique_communes
        )
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)