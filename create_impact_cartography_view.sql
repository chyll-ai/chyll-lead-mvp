-- Impact Cartography Materialized View
-- This creates a unified view of companies with their impact data
-- Ready for integration with existing QPV/ZRR detection system

-- Step 1: Create the base materialized view
CREATE MATERIALIZED VIEW mv_company_impact AS
SELECT 
    -- Company identification
    ul.siren,
    ul.denomination_unite_legale as company_name,
    ul.activite_principale_unite_legale as activity_code,
    ul.nature_juridique_unite_legale as legal_form_code,
    
    -- Business impact indicators
    ul.economie_sociale_solidaire_unite_legale as ess_flag,
    ul.societe_mission_unite_legale as mission_flag,
    
    -- Address information
    CONCAT(
        COALESCE(e.numero_voie::text, ''), ' ',
        COALESCE(e.libelle_voie, ''), ', ',
        COALESCE(e.code_postal, ''), ' ',
        COALESCE(e.libelle_commune, '')
    ) as full_address,
    e.code_postal,
    e.libelle_commune,
    e.code_commune,
    
    -- Geographic coordinates (for QPV detection)
    e.latitude,
    e.longitude,
    
    -- Plain text translations
    CASE 
        WHEN ul.activite_principale_unite_legale LIKE '62%' THEN 'Services informatiques'
        WHEN ul.activite_principale_unite_legale LIKE '56%' THEN 'Restauration'
        WHEN ul.activite_principale_unite_legale LIKE '47%' THEN 'Commerce de détail'
        WHEN ul.activite_principale_unite_legale LIKE '68%' THEN 'Activités immobilières'
        WHEN ul.activite_principale_unite_legale LIKE '70%' THEN 'Activités de conseil'
        WHEN ul.activite_principale_unite_legale LIKE '85%' THEN 'Enseignement'
        WHEN ul.activite_principale_unite_legale LIKE '86%' THEN 'Activités de santé'
        WHEN ul.activite_principale_unite_legale LIKE '46%' THEN 'Commerce de gros'
        WHEN ul.activite_principale_unite_legale LIKE '41%' THEN 'Construction'
        WHEN ul.activite_principale_unite_legale LIKE '10%' THEN 'Industrie alimentaire'
        ELSE 'Autre activité'
    END as activity_plain_text,
    
    CASE 
        WHEN ul.nature_juridique_unite_legale = '1000' THEN 'Entreprise individuelle'
        WHEN ul.nature_juridique_unite_legale = '2000' THEN 'SARL'
        WHEN ul.nature_juridique_unite_legale = '3000' THEN 'SA'
        WHEN ul.nature_juridique_unite_legale = '9200' THEN 'Association'
        WHEN ul.nature_juridique_unite_legale = '9210' THEN 'Association déclarée'
        WHEN ul.nature_juridique_unite_legale = '9220' THEN 'Association non déclarée'
        WHEN ul.nature_juridique_unite_legale = '9300' THEN 'Fondation'
        WHEN ul.nature_juridique_unite_legale = '9400' THEN 'Fondation d''entreprise'
        ELSE 'Autre forme juridique'
    END as legal_form_plain_text,
    
    -- Impact indicators (to be populated by QPV/ZRR detection)
    'Unknown' as geographic_impact,
    'Unknown' as impact_level,
    
    -- Zone details (to be populated by QPV/ZRR detection)
    NULL as qpv_name,
    NULL as qpv_code,
    NULL as zrr_name,
    NULL as zrr_commune,
    
    -- Metadata
    ul.date_creation_unite_legale as creation_date,
    ul.statut_diffusion as diffusion_status,
    e.statut_diffusion as establishment_status

FROM sirene_unites_legales ul
JOIN sirene_etablissements e ON ul.siren = e.siren
WHERE ul.statut_diffusion = 'O' 
  AND e.statut_diffusion = 'O'
  AND ul.siren IS NOT NULL
  AND e.siren IS NOT NULL;

-- Step 2: Create indexes for performance
CREATE INDEX idx_mv_company_impact_siren ON mv_company_impact(siren);
CREATE INDEX idx_mv_company_impact_postal ON mv_company_impact(code_postal);
CREATE INDEX idx_mv_company_impact_commune ON mv_company_impact(libelle_commune);
CREATE INDEX idx_mv_company_impact_geographic ON mv_company_impact(geographic_impact);
CREATE INDEX idx_mv_company_impact_ess ON mv_company_impact(ess_flag);
CREATE INDEX idx_mv_company_impact_mission ON mv_company_impact(mission_flag);
CREATE INDEX idx_mv_company_impact_coords ON mv_company_impact(latitude, longitude) WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Step 3: Create a function to refresh the view
CREATE OR REPLACE FUNCTION refresh_company_impact_view()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW mv_company_impact;
    RAISE NOTICE 'Company impact view refreshed successfully';
END;
$$ LANGUAGE plpgsql;

-- Step 4: Create a function to get impact statistics
CREATE OR REPLACE FUNCTION get_impact_statistics()
RETURNS TABLE(
    total_companies bigint,
    ess_companies bigint,
    mission_companies bigint,
    qpv_companies bigint,
    zrr_companies bigint,
    high_impact_companies bigint,
    medium_impact_companies bigint,
    low_impact_companies bigint
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_companies,
        COUNT(CASE WHEN ess_flag = 'O' THEN 1 END) as ess_companies,
        COUNT(CASE WHEN mission_flag = 'O' THEN 1 END) as mission_companies,
        COUNT(CASE WHEN geographic_impact = 'QPV' THEN 1 END) as qpv_companies,
        COUNT(CASE WHEN geographic_impact = 'ZRR' THEN 1 END) as zrr_companies,
        COUNT(CASE WHEN impact_level = 'High Impact' THEN 1 END) as high_impact_companies,
        COUNT(CASE WHEN impact_level = 'Medium Impact' THEN 1 END) as medium_impact_companies,
        COUNT(CASE WHEN impact_level = 'Low Impact' THEN 1 END) as low_impact_companies
    FROM mv_company_impact;
END;
$$ LANGUAGE plpgsql;

-- Step 5: Create a function to update QPV/ZRR data for a company
CREATE OR REPLACE FUNCTION update_company_impact(
    p_siren text,
    p_geographic_impact text,
    p_qpv_name text DEFAULT NULL,
    p_qpv_code text DEFAULT NULL,
    p_zrr_name text DEFAULT NULL,
    p_zrr_commune text DEFAULT NULL
)
RETURNS void AS $$
DECLARE
    v_impact_level text;
BEGIN
    -- Calculate impact level based on geographic and business impact
    SELECT CASE 
        WHEN p_geographic_impact = 'QPV' AND ess_flag = 'O' THEN 'High Impact'
        WHEN p_geographic_impact = 'QPV' AND mission_flag = 'O' THEN 'High Impact'
        WHEN p_geographic_impact = 'ZRR' AND ess_flag = 'O' THEN 'Medium Impact'
        WHEN p_geographic_impact = 'ZRR' AND mission_flag = 'O' THEN 'Medium Impact'
        WHEN p_geographic_impact = 'QPV' THEN 'Low Impact'
        WHEN p_geographic_impact = 'ZRR' THEN 'Low Impact'
        WHEN ess_flag = 'O' OR mission_flag = 'O' THEN 'Low Impact'
        ELSE 'No Impact'
    END INTO v_impact_level
    FROM mv_company_impact
    WHERE siren = p_siren;
    
    -- Update the company record
    UPDATE mv_company_impact SET
        geographic_impact = p_geographic_impact,
        impact_level = v_impact_level,
        qpv_name = p_qpv_name,
        qpv_code = p_qpv_code,
        zrr_name = p_zrr_name,
        zrr_commune = p_zrr_commune
    WHERE siren = p_siren;
END;
$$ LANGUAGE plpgsql;

-- Step 6: Create a view for API endpoints
CREATE VIEW v_company_map_data AS
SELECT 
    siren,
    company_name,
    full_address,
    activity_plain_text,
    legal_form_plain_text,
    geographic_impact,
    impact_level,
    qpv_name,
    zrr_name,
    latitude,
    longitude,
    creation_date
FROM mv_company_impact
WHERE latitude IS NOT NULL 
  AND longitude IS NOT NULL
  AND geographic_impact != 'Unknown';

-- Step 7: Create indexes on the view
CREATE INDEX idx_v_company_map_data_geographic ON mv_company_impact(geographic_impact) WHERE geographic_impact != 'Unknown';
CREATE INDEX idx_v_company_map_data_impact ON mv_company_impact(impact_level) WHERE impact_level != 'Unknown';
CREATE INDEX idx_v_company_map_data_coords ON mv_company_impact(latitude, longitude) WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Step 8: Grant permissions (adjust as needed)
-- GRANT SELECT ON mv_company_impact TO your_api_user;
-- GRANT SELECT ON v_company_map_data TO your_api_user;
-- GRANT EXECUTE ON FUNCTION refresh_company_impact_view() TO your_api_user;
-- GRANT EXECUTE ON FUNCTION get_impact_statistics() TO your_api_user;
-- GRANT EXECUTE ON FUNCTION update_company_impact(text, text, text, text, text, text) TO your_api_user;

-- Step 9: Add comments
COMMENT ON MATERIALIZED VIEW mv_company_impact IS 'Unified view of companies with impact data for cartography';
COMMENT ON VIEW v_company_map_data IS 'Filtered view for map display with coordinates';
COMMENT ON FUNCTION refresh_company_impact_view() IS 'Refreshes the company impact materialized view';
COMMENT ON FUNCTION get_impact_statistics() IS 'Returns impact statistics for dashboard';
COMMENT ON FUNCTION update_company_impact(text, text, text, text, text, text) IS 'Updates QPV/ZRR data for a specific company';
