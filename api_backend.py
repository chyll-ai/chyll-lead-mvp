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
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
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
    limit: int = Query(1000, description="Maximum number of companies to return")
):
    """Get companies with optional filters"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        companies = []
        
        # Get ESS companies
        if ess:
            ess_query = """
                SELECT 
                    siren, siret, denomination_unite_legale, libelle_commune, code_postal,
                    latitude, longitude, in_qpv, is_zrr, qpv_label, zrr_classification,
                    activite_principale_unite_legale
                FROM ess_companies_filtered_table
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            """
            
            if qpv:
                ess_query += " AND in_qpv = TRUE"
            if zrr:
                ess_query += " AND is_zrr = TRUE"
                
            ess_query += f" LIMIT {limit}"
            
            cursor.execute(ess_query)
            ess_results = cursor.fetchall()
            
            for row in ess_results:
                tags = ['ESS']
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
        
        # Get Mission companies
        if mission:
            mission_query = """
                SELECT 
                    siren, siret, denomination_unite_legale, libelle_commune, code_postal,
                    latitude, longitude, societe_mission_unite_legale, economie_sociale_solidaire_unite_legale,
                    in_qpv, is_zrr, qpv_label, zrr_classification, activite_principale_unite_legale
                FROM companies_societe_mission
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            """
            
            if qpv:
                mission_query += " AND in_qpv = TRUE"
            if zrr:
                mission_query += " AND is_zrr = TRUE"
                
            mission_query += f" LIMIT {limit}"
            
            cursor.execute(mission_query)
            mission_results = cursor.fetchall()
            
            for row in mission_results:
                tags = []
                if row[7] == 'T':  # societe_mission_unite_legale
                    tags.append('Mission')
                if row[8] == 'T':  # economie_sociale_solidaire_unite_legale
                    tags.append('ESS')
                if row[9]:  # in_qpv
                    tags.append('QPV')
                if row[10]:  # is_zrr
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
                    qpv_label=row[11],
                    zrr_classification=row[12],
                    activite_principale_unite_legale=row[13]
                ))
        
        cursor.close()
        conn.close()
        
        logger.info(f"Returned {len(companies)} companies")
        return companies
        
    except Exception as e:
        logger.error(f"Error fetching companies: {e}")
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
                AND (societe_mission_unite_legale = 'T' OR economie_sociale_solidaire_unite_legale = 'T' OR in_qpv = TRUE OR is_zrr = TRUE)
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
