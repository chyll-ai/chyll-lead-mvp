#!/usr/bin/env python3
"""
Tabula Virtutis - ESS Companies Map API
Serves data from ess_companies_filtered_table with comprehensive filtering
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
    ess_companies: int
    qpv_companies: int
    zrr_companies: int
    companies_with_multiple_tags: int
    unique_communes: int
    unique_activities: int

class FilterOptions(BaseModel):
    communes: List[str]
    activities: List[str]
    departments: List[str]

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG)

@app.get("/")
async def root():
    return {"message": "Tabula Virtutis - ESS Companies API", "version": "2.0.0"}

@app.get("/test")
async def test():
    return {"status": "ok", "message": "Tabula Virtutis API is working"}

@app.get("/companies", response_model=List[Company])
async def get_companies(
    qpv: bool = Query(False, description="Filter by QPV companies only"),
    zrr: bool = Query(False, description="Filter by ZRR companies only"),
    commune: Optional[str] = Query(None, description="Filter by commune name"),
    department: Optional[str] = Query(None, description="Filter by department code"),
    activity_code: Optional[str] = Query(None, description="Filter by activity code"),
    search: Optional[str] = Query(None, description="Search in company names"),
    limit: int = Query(1000, description="Maximum number of companies to return"),
    offset: int = Query(0, description="Number of companies to skip")
):
    """Get ESS companies with comprehensive filtering"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build the base query
        base_query = """
            SELECT 
                siren, siret, denomination_unite_legale, libelle_commune, code_postal,
                latitude, longitude, 
                CASE WHEN in_qpv THEN TRUE ELSE FALSE END as in_qpv,
                CASE WHEN is_zrr THEN TRUE ELSE FALSE END as is_zrr,
                qpv_label, zrr_classification, 
                activite_principale_unite_legale
            FROM ess_companies_filtered_table
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """
        
        # Add filters
        filters = []
        params = []
        
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
            
        if activity_code:
            filters.append("activite_principale_unite_legale LIKE %s")
            params.append(f"{activity_code}%")
            
        if search:
            filters.append("LOWER(denomination_unite_legale) LIKE LOWER(%s)")
            params.append(f"%{search}%")
        
        # Combine query
        if filters:
            base_query += " AND " + " AND ".join(filters)
        
        base_query += f" ORDER BY denomination_unite_legale LIMIT {limit} OFFSET {offset}"
        
        cursor.execute(base_query, params)
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


@app.get("/filter")
async def filter_companies(
    qpv: bool = Query(False, description="Filter by QPV companies only"),
    zrr: bool = Query(False, description="Filter by ZRR companies only"),
    commune: Optional[str] = Query(None, description="Filter by commune name"),
    department: Optional[str] = Query(None, description="Filter by department code"),
    activity_code: Optional[str] = Query(None, description="Filter by activity code"),
    search: Optional[str] = Query(None, description="Search in company names"),
    limit: int = Query(1000, description="Maximum number of companies to return"),
    offset: int = Query(0, description="Number of companies to skip")
):
    """Filter ESS companies - frontend-compatible endpoint"""
    try:
        companies = await get_companies(
            qpv=qpv, zrr=zrr, commune=commune, department=department,
            activity_code=activity_code, search=search, limit=limit, offset=offset
        )
        
        return {
            "companies": companies,
            "total": len(companies),
            "filters_applied": {
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

@app.get("/filter-options", response_model=FilterOptions)
async def get_filter_options():
    """Get available filter options for the frontend"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get unique communes
        cursor.execute("""
            SELECT DISTINCT libelle_commune 
            FROM ess_companies_filtered_table 
            WHERE libelle_commune IS NOT NULL 
            ORDER BY libelle_commune 
            LIMIT 100
        """)
        communes = [row[0] for row in cursor.fetchall()]
        
        # Get unique activity codes
        cursor.execute("""
            SELECT DISTINCT activite_principale_unite_legale
            FROM ess_companies_filtered_table 
            WHERE activite_principale_unite_legale IS NOT NULL 
            ORDER BY activite_principale_unite_legale 
            LIMIT 50
        """)
        activities = [row[0] for row in cursor.fetchall()]
        
        # Get unique departments (first 2 digits of postal code)
        cursor.execute("""
            SELECT DISTINCT LEFT(code_postal, 2) as dept
            FROM ess_companies_filtered_table 
            WHERE code_postal IS NOT NULL AND LENGTH(code_postal) >= 2
            ORDER BY dept
        """)
        departments = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return FilterOptions(
            communes=communes,
            activities=activities,
            departments=departments
        )
        
    except Exception as e:
        logger.error(f"Error fetching filter options: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=CompanyStats)
async def get_stats():
    """Get ESS company statistics"""
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
        
        # Companies with multiple tags (QPV + ZRR)
        cursor.execute("""
            SELECT COUNT(*) FROM ess_companies_filtered_table 
            WHERE latitude IS NOT NULL 
            AND (in_qpv = TRUE AND is_zrr = TRUE)
        """)
        multi_tag_count = cursor.fetchone()[0]
        
        # Unique communes count
        cursor.execute("""
            SELECT COUNT(DISTINCT libelle_commune) 
            FROM ess_companies_filtered_table 
            WHERE libelle_commune IS NOT NULL AND latitude IS NOT NULL
        """)
        unique_communes = cursor.fetchone()[0]
        
        # Unique activities count
        cursor.execute("""
            SELECT COUNT(DISTINCT activite_principale_unite_legale) 
            FROM ess_companies_filtered_table 
            WHERE activite_principale_unite_legale IS NOT NULL AND latitude IS NOT NULL
        """)
        unique_activities = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return CompanyStats(
            total_companies=total_count,
            ess_companies=total_count,  # All companies are ESS
            qpv_companies=qpv_count,
            zrr_companies=zrr_count,
            companies_with_multiple_tags=multi_tag_count,
            unique_communes=unique_communes,
            unique_activities=unique_activities
        )
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
