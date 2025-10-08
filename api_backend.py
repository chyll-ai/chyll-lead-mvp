#!/usr/bin/env python3
"""
FastAPI Backend for ESS and Mission Companies Map
Serves data from both ess_companies_filtered_table and companies_societe_mission
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

app = FastAPI(title="ESS Mission Companies API", version="1.0.0")

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
    tags: List[str]  # ['ESS', 'Mission', 'QPV', 'ZRR']
    qpv_label: Optional[str] = None
    zrr_classification: Optional[str] = None
    activite_principale_unite_legale: Optional[str] = None

class CompanyStats(BaseModel):
    total_companies: int
    ess_companies: int
    mission_companies: int
    qpv_companies: int
    zrr_companies: int
    companies_with_multiple_tags: int

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG)

@app.get("/")
async def root():
    return {"message": "ESS Mission Companies API", "version": "1.0.0"}

@app.get("/test")
async def test():
    return {"status": "ok", "message": "API is working"}

@app.get("/companies", response_model=List[Company])
async def get_companies(
    ess: bool = Query(True, description="Include ESS companies"),
    mission: bool = Query(True, description="Include Mission companies"),
    qpv: bool = Query(False, description="Filter by QPV companies"),
    zrr: bool = Query(False, description="Filter by ZRR companies"),
    limit: int = Query(1000, description="Maximum number of companies to return"),
    offset: int = Query(0, description="Number of companies to skip")
):
    """Get companies with optional filters"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build the main query using UNION to avoid duplicates
        query_parts = []
        
        # ESS companies query
        if ess:
            ess_query = """
                SELECT 
                    siren, siret, denomination_unite_legale, libelle_commune, code_postal,
                    latitude, longitude, 
                    CASE WHEN in_qpv THEN TRUE ELSE FALSE END as in_qpv,
                    CASE WHEN is_zrr THEN TRUE ELSE FALSE END as is_zrr,
                    qpv_label, zrr_classification, activite_principale_unite_legale,
                    'ESS' as source_table
                FROM ess_companies_filtered_table
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            """
            
            if qpv:
                ess_query += " AND in_qpv = TRUE"
            if zrr:
                ess_query += " AND is_zrr = TRUE"
                
            query_parts.append(ess_query)
        
        # Mission companies query
        if mission:
            mission_query = """
                SELECT 
                    siren, siret, denomination_unite_legale, libelle_commune, code_postal,
                    latitude, longitude,
                    CASE WHEN in_qpv THEN TRUE ELSE FALSE END as in_qpv,
                    CASE WHEN is_zrr THEN TRUE ELSE FALSE END as is_zrr,
                    qpv_label, zrr_classification, activite_principale_unite_legale,
                    'Mission' as source_table
                FROM companies_societe_mission
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            """
            
            if qpv:
                mission_query += " AND in_qpv = TRUE"
            if zrr:
                mission_query += " AND is_zrr = TRUE"
                
            query_parts.append(mission_query)
        
        if not query_parts:
            return []
        
        # Combine queries with UNION and deduplicate by siret
        main_query = f"""
            SELECT DISTINCT ON (siret)
                siren, siret, denomination_unite_legale, libelle_commune, code_postal,
                latitude, longitude, in_qpv, is_zrr, qpv_label, zrr_classification, 
                activite_principale_unite_legale, source_table
            FROM (
                {' UNION ALL '.join(query_parts)}
            ) combined
            ORDER BY siret, source_table
            LIMIT {limit} OFFSET {offset}
        """
        
        cursor.execute(main_query)
        results = cursor.fetchall()
        
        companies = []
        for row in results:
            tags = []
            
            # Determine tags based on source table and flags
            if row[12] == 'ESS':  # source_table
                tags.append('ESS')
            elif row[12] == 'Mission':
                tags.append('Mission')
            
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
        
        logger.info(f"Returned {len(companies)} companies")
        return companies
        
    except Exception as e:
        logger.error(f"Error fetching companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/companies/batch")
async def get_companies_batch(
    batch_size: int = Query(500, description="Number of companies per batch"),
    batch_number: int = Query(0, description="Batch number (0-based)"),
    ess: bool = Query(True, description="Include ESS companies"),
    mission: bool = Query(True, description="Include Mission companies"),
    qpv: bool = Query(False, description="Filter by QPV companies"),
    zrr: bool = Query(False, description="Filter by ZRR companies")
):
    """Get companies in batches for efficient map loading"""
    offset = batch_number * batch_size
    return await get_companies(
        ess=ess, mission=mission, qpv=qpv, zrr=zrr,
        limit=batch_size, offset=offset
    )

@app.get("/filter")
async def filter_companies(
    ess: bool = Query(True, description="Include ESS companies"),
    mission: bool = Query(True, description="Include Mission companies"),
    qpv: bool = Query(False, description="Filter by QPV companies"),
    zrr: bool = Query(False, description="Filter by ZRR companies"),
    limit: int = Query(1000, description="Maximum number of companies to return"),
    offset: int = Query(0, description="Number of companies to skip")
):
    """Filter companies - frontend-compatible endpoint"""
    try:
        companies = await get_companies(
            ess=ess, mission=mission, qpv=qpv, zrr=zrr,
            limit=limit, offset=offset
        )
        
        return {
            "companies": companies,
            "total": len(companies),
            "filters_applied": {
                "ess": ess,
                "mission": mission,
                "qpv": qpv,
                "zrr": zrr
            }
        }
        
    except Exception as e:
        logger.error(f"Error in filter endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=CompanyStats)
async def get_stats():
    """Get company statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # ESS companies count
        cursor.execute("SELECT COUNT(*) FROM ess_companies_filtered_table WHERE latitude IS NOT NULL")
        ess_count = cursor.fetchone()[0]
        
        # Mission companies count
        cursor.execute("SELECT COUNT(*) FROM companies_societe_mission WHERE latitude IS NOT NULL")
        mission_count = cursor.fetchone()[0]
        
        # QPV companies count
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT siren FROM ess_companies_filtered_table WHERE in_qpv = TRUE AND latitude IS NOT NULL
                UNION
                SELECT siren FROM companies_societe_mission WHERE in_qpv = TRUE AND latitude IS NOT NULL
            ) as qpv_companies
        """)
        qpv_count = cursor.fetchone()[0]
        
        # ZRR companies count
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT siren FROM ess_companies_filtered_table WHERE is_zrr = TRUE AND latitude IS NOT NULL
                UNION
                SELECT siren FROM companies_societe_mission WHERE is_zrr = TRUE AND latitude IS NOT NULL
            ) as zrr_companies
        """)
        zrr_count = cursor.fetchone()[0]
        
        # Companies with multiple tags
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT siren FROM ess_companies_filtered_table 
                WHERE latitude IS NOT NULL 
                AND (in_qpv = TRUE OR is_zrr = TRUE)
                UNION
                SELECT siren FROM companies_societe_mission 
                WHERE latitude IS NOT NULL 
                AND (in_qpv = TRUE OR is_zrr = TRUE)
            ) as multi_tag_companies
        """)
        multi_tag_count = cursor.fetchone()[0]
        
        total_count = ess_count + mission_count
        
        cursor.close()
        conn.close()
        
        return CompanyStats(
            total_companies=total_count,
            ess_companies=ess_count,
            mission_companies=mission_count,
            qpv_companies=qpv_count,
            zrr_companies=zrr_count,
            companies_with_multiple_tags=multi_tag_count
        )
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
