#!/usr/bin/env python3
"""
ZRR Integration for ESS Companies Filtered Table
Adds ZRR (Zone de Revitalisation Rurale) data to ess_companies_filtered_table
"""

import os
import sys
import logging
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zrr_ess_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ZRESSIntegrator:
    def __init__(self):
        self.db_connection = None
        self.zrr_data = None
        
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
    
    def load_zrr_data(self):
        """Load ZRR Excel data"""
        try:
            logger.info("📂 Loading ZRR Excel data...")
            
            # Load ZRR data from Excel file
            zrr_file = "GEOdata/diffusion-zonages-zrr-cog2021 (1).xls"
            if os.path.exists(zrr_file):
                self.zrr_data = pd.read_excel(zrr_file, skiprows=5)
                logger.info(f"   ✅ Loaded {len(self.zrr_data)} ZRR commune records from {zrr_file}")
            else:
                logger.error(f"❌ ZRR file not found: {zrr_file}")
                return False
            
            # Clean up the data
            self.zrr_data = self.zrr_data.dropna(subset=['CODGEO'])
            self.zrr_data['CODGEO'] = self.zrr_data['CODGEO'].astype(str).str.strip()
            
            logger.info(f"   ✅ Processed {len(self.zrr_data)} valid ZRR commune records")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading ZRR data: {e}")
            return False
    
    def check_zrr_columns_exist(self):
        """Check if ZRR columns already exist in ess_companies_filtered_table"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'ess_companies_filtered_table' 
                AND column_name = 'is_zrr'
            """)
            
            exists = cursor.fetchone() is not None
            cursor.close()
            
            if exists:
                logger.info("✅ ZRR columns already exist in ess_companies_filtered_table")
            else:
                logger.info("⚠️ ZRR columns do not exist, will add them")
            
            return exists
            
        except Exception as e:
            logger.error(f"❌ Error checking ZRR columns: {e}")
            return False
    
    def add_zrr_columns_to_ess_table(self):
        """Add ZRR columns to ess_companies_filtered_table"""
        try:
            cursor = self.db_connection.cursor()
            
            logger.info("🔧 Adding ZRR columns to ess_companies_filtered_table...")
            
            # Add ZRR columns
            zrr_columns = [
                "ALTER TABLE ess_companies_filtered_table ADD COLUMN IF NOT EXISTS is_zrr BOOLEAN DEFAULT FALSE",
                "ALTER TABLE ess_companies_filtered_table ADD COLUMN IF NOT EXISTS zrr_classification TEXT",
                "ALTER TABLE ess_companies_filtered_table ADD COLUMN IF NOT EXISTS zrr_zonage TEXT",
                "ALTER TABLE ess_companies_filtered_table ADD COLUMN IF NOT EXISTS zrr_commune TEXT",
                "ALTER TABLE ess_companies_filtered_table ADD COLUMN IF NOT EXISTS zrr_updated_at TIMESTAMP WITH TIME ZONE"
            ]
            
            for column_sql in zrr_columns:
                cursor.execute(column_sql)
                logger.info(f"   ✅ Added ZRR column: {column_sql.split('ADD COLUMN IF NOT EXISTS')[1].split()[0]}")
            
            self.db_connection.commit()
            logger.info("✅ Successfully added ZRR columns to ess_companies_filtered_table")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding ZRR columns: {e}")
            return False
    
    def get_ess_companies_for_zrr_matching(self):
        """Get ESS companies with commune codes for ZRR matching"""
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            
            logger.info("📊 Fetching ESS companies with commune codes...")
            
            # Get ESS companies with commune codes
            query = """
            SELECT 
                siren,
                siret,
                denomination_unite_legale,
                libelle_commune,
                code_postal,
                activite_principale_unite_legale
            FROM ess_companies_filtered_table 
            WHERE code_postal IS NOT NULL
            AND code_postal != ''
            ORDER BY siren
            """
            
            cursor.execute(query)
            companies_data = cursor.fetchall()
            
            if not companies_data:
                logger.warning("⚠️ No ESS companies found with commune codes")
                return pd.DataFrame()
            
            # Convert to DataFrame
            companies_df = pd.DataFrame(companies_data)
            
            # Extract commune code from postal code (first 3 digits)
            companies_df['code_commune'] = companies_df['code_postal'].str[:3]
            
            logger.info(f"   ✅ Found {len(companies_df):,} ESS companies with commune codes")
            
            cursor.close()
            return companies_df
            
        except Exception as e:
            logger.error(f"❌ Error fetching ESS companies: {e}")
            return pd.DataFrame()
    
    def match_companies_to_zrr(self, companies_df: pd.DataFrame):
        """Match companies to ZRR zones using commune names"""
        try:
            logger.info("🔍 Matching ESS companies to ZRR zones using commune names...")
            
            # Create a mapping from commune name to ZRR data
            zrr_mapping = {}
            for _, row in self.zrr_data.iterrows():
                commune_name = str(row['LIBGEO']).strip().upper()
                zrr_mapping[commune_name] = {
                    'zrr_classification': row['ZRR_SIMP'],
                    'zrr_zonage': row['ZONAGE_ZRR'],
                    'zrr_commune': row['LIBGEO']
                }
            
            logger.info(f"   Created ZRR mapping for {len(zrr_mapping)} communes")
            
            # Match companies to ZRR using commune names
            companies_df['is_zrr'] = False
            companies_df['zrr_classification'] = None
            companies_df['zrr_zonage'] = None
            companies_df['zrr_commune'] = None
            
            matched_count = 0
            zrr_count = 0
            
            for idx, row in companies_df.iterrows():
                commune_name = str(row['libelle_commune']).strip().upper()
                
                if commune_name in zrr_mapping:
                    zrr_info = zrr_mapping[commune_name]
                    companies_df.at[idx, 'zrr_classification'] = zrr_info['zrr_classification']
                    companies_df.at[idx, 'zrr_zonage'] = zrr_info['zrr_zonage']
                    companies_df.at[idx, 'zrr_commune'] = zrr_info['zrr_commune']
                    
                    # Check if it's actually a ZRR commune
                    if zrr_info['zrr_classification'] == 'C - Classée en ZRR':
                        companies_df.at[idx, 'is_zrr'] = True
                        zrr_count += 1
                    
                    matched_count += 1
            
            # Statistics
            total_companies = len(companies_df)
            zrr_coverage = (zrr_count / total_companies) * 100 if total_companies > 0 else 0
            
            logger.info(f"📊 ZRR Matching Results:")
            logger.info(f"   Total ESS companies processed: {total_companies:,}")
            logger.info(f"   ESS companies in ZRR zones: {zrr_count:,}")
            logger.info(f"   ZRR coverage: {zrr_coverage:.2f}%")
            
            return companies_df
            
        except Exception as e:
            logger.error(f"❌ Error matching companies to ZRR: {e}")
            return pd.DataFrame()
    
    def update_ess_table_with_zrr_data(self, zrr_df: pd.DataFrame):
        """Update ess_companies_filtered_table with ZRR data"""
        try:
            cursor = self.db_connection.cursor()
            
            logger.info("💾 Updating ess_companies_filtered_table with ZRR data...")
            
            # Prepare update statements
            update_sql = """
            UPDATE ess_companies_filtered_table 
            SET 
                is_zrr = %s,
                zrr_classification = %s,
                zrr_zonage = %s,
                zrr_commune = %s,
                zrr_updated_at = NOW()
            WHERE siren = %s AND siret = %s
            """
            
            # Process updates in larger batches for speed
            batch_size = 1000
            total_updates = 0
            zrr_updates = 0
            
            logger.info(f"   Processing {len(zrr_df):,} companies in batches of {batch_size:,}...")
            
            for i in range(0, len(zrr_df), batch_size):
                batch = zrr_df.iloc[i:i+batch_size]
                
                batch_data = []
                batch_zrr_count = 0
                
                for _, row in batch.iterrows():
                    batch_data.append((
                        row['is_zrr'],
                        row['zrr_classification'] if pd.notna(row['zrr_classification']) else None,
                        row['zrr_zonage'] if pd.notna(row['zrr_zonage']) else None,
                        row['zrr_commune'] if pd.notna(row['zrr_commune']) else None,
                        row['siren'],
                        row['siret']
                    ))
                    
                    if row['is_zrr']:
                        batch_zrr_count += 1
                
                cursor.executemany(update_sql, batch_data)
                total_updates += len(batch_data)
                zrr_updates += batch_zrr_count
                
                # Show progress every batch
                logger.info(f"   ✅ Batch {i//batch_size + 1}: Updated {total_updates:,}/{len(zrr_df):,} companies ({batch_zrr_count} ZRR in this batch)")
                
                # Show sample ZRR companies from this batch
                if batch_zrr_count > 0 and i < 5000:  # Show samples for first few batches
                    sample_zrr = batch[batch['is_zrr'] == True].head(2)
                    for _, row in sample_zrr.iterrows():
                        logger.info(f"      🏢 ZRR: {row['denomination_unite_legale'][:40]}... ({row['libelle_commune']})")
            
            # Commit all changes
            self.db_connection.commit()
            logger.info(f"✅ Successfully updated {total_updates:,} ESS companies with ZRR data")
            logger.info(f"🎯 Total ZRR companies identified: {zrr_updates:,}")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating ESS table with ZRR data: {e}")
            return False
    
    def verify_zrr_integration(self):
        """Verify ZRR integration results"""
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            
            logger.info("🔍 Verifying ZRR integration results...")
            
            # Check total ESS companies
            cursor.execute("SELECT COUNT(*) as total FROM ess_companies_filtered_table")
            total_ess = cursor.fetchone()['total']
            
            # Check ESS companies in ZRR
            cursor.execute("SELECT COUNT(*) as zrr_count FROM ess_companies_filtered_table WHERE is_zrr = TRUE")
            zrr_count = cursor.fetchone()['zrr_count']
            
            # Check ZRR coverage
            zrr_coverage = (zrr_count / total_ess) * 100 if total_ess > 0 else 0
            
            logger.info(f"📊 ZRR Integration Verification:")
            logger.info(f"   Total ESS companies: {total_ess:,}")
            logger.info(f"   ESS companies in ZRR: {zrr_count:,}")
            logger.info(f"   ZRR coverage: {zrr_coverage:.2f}%")
            
            # Sample ZRR companies
            cursor.execute("""
                SELECT 
                    denomination_unite_legale,
                    libelle_commune,
                    zrr_commune,
                    zrr_classification
                FROM ess_companies_filtered_table 
                WHERE is_zrr = TRUE 
                LIMIT 5
            """)
            
            sample_companies = cursor.fetchall()
            
            if sample_companies:
                logger.info("   📋 Sample ESS companies in ZRR zones:")
                for company in sample_companies:
                    logger.info(f"      • {company['denomination_unite_legale']} ({company['libelle_commune']}) - {company['zrr_commune']} ({company['zrr_classification']})")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying ZRR integration: {e}")
            return False
    
    def run_zrr_integration(self):
        """Run complete ZRR integration process for ESS companies"""
        try:
            logger.info("🚀 Starting ZRR integration for ESS companies...")
            
            # Connect to database
            if not self.connect_database():
                return False
            
            # Load ZRR data
            if not self.load_zrr_data():
                return False
            
            # Check if ZRR columns already exist (skip if they do)
            if not self.check_zrr_columns_exist():
                if not self.add_zrr_columns_to_ess_table():
                    return False
            
            # Get ESS companies for matching
            companies_df = self.get_ess_companies_for_zrr_matching()
            if companies_df.empty:
                logger.error("❌ No ESS companies found for ZRR matching")
                return False
            
            # Match companies to ZRR zones
            zrr_df = self.match_companies_to_zrr(companies_df)
            if zrr_df.empty:
                logger.error("❌ ZRR matching failed")
                return False
            
            # Update ESS table
            if not self.update_ess_table_with_zrr_data(zrr_df):
                return False
            
            # Verify results
            if not self.verify_zrr_integration():
                return False
            
            logger.info("✅ ZRR integration for ESS companies completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in ZRR integration: {e}")
            return False
        finally:
            if self.db_connection:
                self.db_connection.close()

def main():
    """Main function"""
    logger.info("🌾 ZRR Integration for ESS Companies Filtered Table")
    logger.info("=" * 60)
    
    integrator = ZRESSIntegrator()
    
    if integrator.run_zrr_integration():
        logger.info("🎉 ZRR integration completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 ZRR integration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
