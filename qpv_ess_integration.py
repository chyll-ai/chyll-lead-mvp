#!/usr/bin/env python3
"""
QPV Integration for ESS Companies Filtered Table
Adds QPV (Quartiers Prioritaires de la Ville) data to ess_companies_filtered_table
"""

import os
import sys
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.wkt import loads
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qpv_ess_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QPVESSIntegrator:
    def __init__(self):
        self.db_connection = None
        self.qpv_data = None
        
    def connect_database(self):
        """Connect to the database"""
        try:
            # Use Railway database connection (working configuration)
            self.db_connection = psycopg2.connect(
                host='ballast.proxy.rlwy.net',
                port='30865',
                database='railway',
                user='postgres',
                password='mlnbusCQpeWkMTFpMasYdILtfnHEyfyG'
            )
            logger.info("✅ Connected to Railway database")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def load_qpv_data(self):
        """Load QPV GeoJSON data"""
        try:
            logger.info("📂 Loading QPV GeoJSON data...")
            
            # Load main QPV data (France hexagonale)
            qpv_file = "GEOdata/QP2024_France_hexagonale_LB93.geojson"
            if os.path.exists(qpv_file):
                self.qpv_data = gpd.read_file(qpv_file)
                logger.info(f"   ✅ Loaded {len(self.qpv_data)} QPV zones from {qpv_file}")
            else:
                logger.error(f"❌ QPV file not found: {qpv_file}")
                return False
            
            # Convert to WGS84 for coordinate matching
            self.qpv_data = self.qpv_data.to_crs('EPSG:4326')
            logger.info("   ✅ Converted QPV data to WGS84")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading QPV data: {e}")
            return False
    
    def check_qpv_columns_exist(self):
        """Check if QPV columns already exist in ess_companies_filtered_table"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'ess_companies_filtered_table' 
                AND column_name = 'in_qpv'
            """)
            
            exists = cursor.fetchone() is not None
            cursor.close()
            
            if exists:
                logger.info("✅ QPV columns already exist in ess_companies_filtered_table")
            else:
                logger.info("⚠️ QPV columns do not exist, will add them")
            
            return exists
            
        except Exception as e:
            logger.error(f"❌ Error checking QPV columns: {e}")
            return False
    
    def add_qpv_columns_to_ess_table(self):
        """Add QPV columns to ess_companies_filtered_table"""
        try:
            cursor = self.db_connection.cursor()
            
            logger.info("🔧 Adding QPV columns to ess_companies_filtered_table...")
            
            # Add QPV columns
            qpv_columns = [
                "ALTER TABLE ess_companies_filtered_table ADD COLUMN IF NOT EXISTS in_qpv BOOLEAN DEFAULT FALSE",
                "ALTER TABLE ess_companies_filtered_table ADD COLUMN IF NOT EXISTS qpv_code TEXT",
                "ALTER TABLE ess_companies_filtered_table ADD COLUMN IF NOT EXISTS qpv_label TEXT",
                "ALTER TABLE ess_companies_filtered_table ADD COLUMN IF NOT EXISTS qpv_region TEXT",
                "ALTER TABLE ess_companies_filtered_table ADD COLUMN IF NOT EXISTS qpv_department TEXT",
                "ALTER TABLE ess_companies_filtered_table ADD COLUMN IF NOT EXISTS qpv_commune TEXT",
                "ALTER TABLE ess_companies_filtered_table ADD COLUMN IF NOT EXISTS qpv_updated_at TIMESTAMP WITH TIME ZONE"
            ]
            
            for column_sql in qpv_columns:
                cursor.execute(column_sql)
                logger.info(f"   ✅ Added QPV column: {column_sql.split('ADD COLUMN IF NOT EXISTS')[1].split()[0]}")
            
            self.db_connection.commit()
            logger.info("✅ Successfully added QPV columns to ess_companies_filtered_table")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding QPV columns: {e}")
            return False
    
    def get_ess_companies_for_qpv_matching(self):
        """Get ESS companies with coordinates for QPV matching"""
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            
            logger.info("📊 Fetching ESS companies with coordinates...")
            
            # Get ESS companies with valid coordinates
            query = """
            SELECT 
                siren,
                siret,
                denomination_unite_legale,
                libelle_commune,
                code_postal,
                latitude,
                longitude,
                activite_principale_unite_legale
            FROM ess_companies_filtered_table 
            WHERE latitude IS NOT NULL 
            AND longitude IS NOT NULL
            AND latitude != 0 
            AND longitude != 0
            AND latitude BETWEEN -90 AND 90
            AND longitude BETWEEN -180 AND 180
            """
            
            cursor.execute(query)
            companies_data = cursor.fetchall()
            
            if not companies_data:
                logger.warning("⚠️ No ESS companies found with valid coordinates")
                return pd.DataFrame()
            
            # Convert to DataFrame
            companies_df = pd.DataFrame(companies_data)
            
            logger.info(f"   ✅ Found {len(companies_df):,} ESS companies with valid coordinates")
            
            cursor.close()
            return companies_df
            
        except Exception as e:
            logger.error(f"❌ Error fetching ESS companies: {e}")
            return pd.DataFrame()
    
    def match_companies_to_qpv(self, companies_df: pd.DataFrame):
        """Match companies to QPV zones using spatial join"""
        try:
            logger.info("🗺️ Matching ESS companies to QPV zones...")
            
            # Create GeoDataFrame for companies
            companies_gdf = gpd.GeoDataFrame(
                companies_df,
                geometry=[Point(xy) for xy in zip(companies_df.longitude, companies_df.latitude)],
                crs='EPSG:4326'
            )
            
            # Perform spatial join
            result_gdf = gpd.sjoin(companies_gdf, self.qpv_data, how='left', predicate='within')
            
            # Process results
            result_df = result_gdf.copy()
            
            # Add QPV information
            result_df['in_qpv'] = result_df['index_right'].notna()
            
            # Fill QPV details for companies in QPV zones
            result_df['qpv_code'] = result_df['code_qp']
            result_df['qpv_label'] = result_df['lib_qp']
            result_df['qpv_region'] = result_df['lib_reg']
            result_df['qpv_department'] = result_df['lib_dep']
            result_df['qpv_commune'] = result_df['lib_com']
            
            # Clean up
            result_df = result_df.drop(columns=['geometry', 'index_right'])
            
            # Statistics
            total_companies = len(result_df)
            companies_in_qpv = len(result_df[result_df['in_qpv'] == True])
            qpv_coverage = (companies_in_qpv / total_companies) * 100 if total_companies > 0 else 0
            
            logger.info(f"📊 QPV Matching Results:")
            logger.info(f"   Total ESS companies processed: {total_companies:,}")
            logger.info(f"   ESS companies in QPV zones: {companies_in_qpv:,}")
            logger.info(f"   QPV coverage: {qpv_coverage:.2f}%")
            
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Error matching companies to QPV: {e}")
            return pd.DataFrame()
    
    def update_ess_table_with_qpv_data(self, qpv_df: pd.DataFrame):
        """Update ess_companies_filtered_table with QPV data"""
        try:
            cursor = self.db_connection.cursor()
            
            logger.info("💾 Updating ess_companies_filtered_table with QPV data...")
            
            # Prepare update statements
            update_sql = """
            UPDATE ess_companies_filtered_table 
            SET 
                in_qpv = %s,
                qpv_code = %s,
                qpv_label = %s,
                qpv_region = %s,
                qpv_department = %s,
                qpv_commune = %s,
                qpv_updated_at = NOW()
            WHERE siren = %s AND siret = %s
            """
            
            # Process updates in batches
            batch_size = 100
            total_updates = 0
            
            for i in range(0, len(qpv_df), batch_size):
                batch = qpv_df.iloc[i:i+batch_size]
                
                batch_data = []
                for _, row in batch.iterrows():
                    batch_data.append((
                        row['in_qpv'],
                        row['qpv_code'] if pd.notna(row['qpv_code']) else None,
                        row['qpv_label'] if pd.notna(row['qpv_label']) else None,
                        row['qpv_region'] if pd.notna(row['qpv_region']) else None,
                        row['qpv_department'] if pd.notna(row['qpv_department']) else None,
                        row['qpv_commune'] if pd.notna(row['qpv_commune']) else None,
                        row['siren'],
                        row['siret']
                    ))
                
                cursor.executemany(update_sql, batch_data)
                total_updates += len(batch_data)
                
                if i % 500 == 0:
                    logger.info(f"   Updated {total_updates:,} ESS companies...")
            
            # Commit all changes
            self.db_connection.commit()
            logger.info(f"✅ Successfully updated {total_updates:,} ESS companies with QPV data")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating ESS table with QPV data: {e}")
            return False
    
    def verify_qpv_integration(self):
        """Verify QPV integration results"""
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            
            logger.info("🔍 Verifying QPV integration results...")
            
            # Check total ESS companies
            cursor.execute("SELECT COUNT(*) as total FROM ess_companies_filtered_table")
            total_ess = cursor.fetchone()['total']
            
            # Check ESS companies in QPV
            cursor.execute("SELECT COUNT(*) as qpv_count FROM ess_companies_filtered_table WHERE in_qpv = TRUE")
            qpv_count = cursor.fetchone()['qpv_count']
            
            # Check QPV coverage
            qpv_coverage = (qpv_count / total_ess) * 100 if total_ess > 0 else 0
            
            logger.info(f"📊 QPV Integration Verification:")
            logger.info(f"   Total ESS companies: {total_ess:,}")
            logger.info(f"   ESS companies in QPV: {qpv_count:,}")
            logger.info(f"   QPV coverage: {qpv_coverage:.2f}%")
            
            # Sample QPV companies
            cursor.execute("""
                SELECT 
                    denomination_unite_legale,
                    libelle_commune,
                    qpv_label,
                    qpv_region
                FROM ess_companies_filtered_table 
                WHERE in_qpv = TRUE 
                LIMIT 5
            """)
            
            sample_companies = cursor.fetchall()
            
            if sample_companies:
                logger.info("   📋 Sample ESS companies in QPV zones:")
                for company in sample_companies:
                    logger.info(f"      • {company['denomination_unite_legale']} ({company['libelle_commune']}) - {company['qpv_label']} ({company['qpv_region']})")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying QPV integration: {e}")
            return False
    
    def run_qpv_integration(self):
        """Run complete QPV integration process for ESS companies"""
        try:
            logger.info("🚀 Starting QPV integration for ESS companies...")
            
            # Connect to database
            if not self.connect_database():
                return False
            
            # Load QPV data
            if not self.load_qpv_data():
                return False
            
            # Check if QPV columns already exist (skip if they do)
            if not self.check_qpv_columns_exist():
                if not self.add_qpv_columns_to_ess_table():
                    return False
            
            # Get ESS companies for matching
            companies_df = self.get_ess_companies_for_qpv_matching()
            if companies_df.empty:
                logger.error("❌ No ESS companies found for QPV matching")
                return False
            
            # Match companies to QPV zones
            qpv_df = self.match_companies_to_qpv(companies_df)
            if qpv_df.empty:
                logger.error("❌ QPV matching failed")
                return False
            
            # Update ESS table
            if not self.update_ess_table_with_qpv_data(qpv_df):
                return False
            
            # Verify results
            if not self.verify_qpv_integration():
                return False
            
            logger.info("✅ QPV integration for ESS companies completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in QPV integration: {e}")
            return False
        finally:
            if self.db_connection:
                self.db_connection.close()

def main():
    """Main function"""
    logger.info("🎯 QPV Integration for ESS Companies Filtered Table")
    logger.info("=" * 60)
    
    integrator = QPVESSIntegrator()
    
    if integrator.run_qpv_integration():
        logger.info("🎉 QPV integration completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 QPV integration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
