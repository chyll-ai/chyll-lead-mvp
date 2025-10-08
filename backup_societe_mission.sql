-- Backup script for companies_societe_mission table
-- This will export the data from the view to a CSV file

-- First, let's check what columns the view has
\d companies_societe_mission;

-- Export the data to CSV
\copy companies_societe_mission TO 'companies_societe_mission_backup.csv' WITH CSV HEADER;

-- Also create a SQL dump of the view definition
SELECT 'CREATE VIEW companies_societe_mission AS ' || definition || ';' as view_definition
FROM pg_views 
WHERE viewname = 'companies_societe_mission';
