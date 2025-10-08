#!/usr/bin/env python3
"""
Fix Saint-Denis ZRR misclassification
Companies in Seine-Saint-Denis (93xxx postal codes) are incorrectly classified as ZRR
because there's confusion with Saint-Denis in La Réunion (which is indeed ZRR).
"""

import psycopg2
import os
import sys

def fix_saint_denis_zrr_classification():
    """Fix ZRR classification for companies in Seine-Saint-Denis (93xxx)"""
    
    # Database connection
    try:
        # Try to get DATABASE_URL from environment
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            print("DATABASE_URL not found in environment variables")
            print("Please set DATABASE_URL or provide database connection details")
            return False
            
        print("Connecting to database...")
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # First, let's check how many companies in Seine-Saint-Denis are currently classified as ZRR
        print("\n=== Checking current ZRR classification in Seine-Saint-Denis ===")
        cursor.execute("""
            SELECT COUNT(*) as total_companies,
                   COUNT(CASE WHEN is_zrr = true THEN 1 END) as zrr_classified,
                   COUNT(CASE WHEN in_qpv = true THEN 1 END) as qpv_classified
            FROM ess_companies_filtered_table 
            WHERE code_postal LIKE '93%'
        """)
        
        result = cursor.fetchone()
        print(f"Total companies in Seine-Saint-Denis (93xxx): {result[0]}")
        print(f"Currently classified as ZRR: {result[1]}")
        print(f"Currently classified as QPV: {result[2]}")
        
        if result[1] == 0:
            print("No companies in Seine-Saint-Denis are currently classified as ZRR.")
            print("The issue might be elsewhere or already fixed.")
            return True
        
        # Show some examples of misclassified companies
        print("\n=== Examples of misclassified companies ===")
        cursor.execute("""
            SELECT denomination_unite_legale, libelle_commune, code_postal, 
                   is_zrr, zrr_classification, in_qpv, qpv_label
            FROM ess_companies_filtered_table 
            WHERE code_postal LIKE '93%' AND is_zrr = true
            LIMIT 10
        """)
        
        companies = cursor.fetchall()
        for company in companies:
            print(f"Company: {company[0]}")
            print(f"  Commune: {company[1]}")
            print(f"  Postal: {company[2]}")
            print(f"  ZRR: {company[3]} ({company[4]})")
            print(f"  QPV: {company[5]} ({company[6]})")
            print()
        
        # Ask for confirmation before making changes
        print(f"\nFound {result[1]} companies in Seine-Saint-Denis (93xxx) classified as ZRR.")
        print("These should NOT be ZRR because Seine-Saint-Denis is not a ZRR zone.")
        print("Only Saint-Denis in La Réunion (97400) should be ZRR.")
        
        response = input("\nDo you want to fix this classification? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return False
        
        # Update ZRR classification for Seine-Saint-Denis companies
        print("\n=== Updating ZRR classification ===")
        cursor.execute("""
            UPDATE ess_companies_filtered_table 
            SET is_zrr = false, 
                zrr_classification = NULL
            WHERE code_postal LIKE '93%' 
            AND is_zrr = true
        """)
        
        updated_count = cursor.rowcount
        print(f"Updated {updated_count} companies in Seine-Saint-Denis")
        
        # Commit the changes
        conn.commit()
        print("Changes committed to database.")
        
        # Verify the fix
        print("\n=== Verification ===")
        cursor.execute("""
            SELECT COUNT(*) as remaining_zrr_in_93
            FROM ess_companies_filtered_table 
            WHERE code_postal LIKE '93%' AND is_zrr = true
        """)
        
        remaining = cursor.fetchone()[0]
        print(f"Remaining ZRR companies in Seine-Saint-Denis: {remaining}")
        
        if remaining == 0:
            print("✅ Fix successful! No more ZRR misclassification in Seine-Saint-Denis.")
        else:
            print("⚠️  Some companies still classified as ZRR. Manual review needed.")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Saint-Denis ZRR Classification Fix")
    print("=" * 40)
    
    success = fix_saint_denis_zrr_classification()
    
    if success:
        print("\n✅ Operation completed successfully!")
    else:
        print("\n❌ Operation failed!")
        sys.exit(1)
