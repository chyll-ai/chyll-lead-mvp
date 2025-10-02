-- Create score_source enum
CREATE TYPE score_source AS ENUM ('ml_model', 'manual', 'rule_based');

-- Create tenants table
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create companies table
CREATE TABLE companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    siren TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    address TEXT,
    city TEXT,
    postal_code TEXT,
    country TEXT DEFAULT 'FR',
    industry TEXT,
    employee_count INTEGER,
    revenue DECIMAL(15,2),
    website TEXT,
    phone TEXT,
    email TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create historical_deals table
CREATE TABLE historical_deals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    company_name TEXT NOT NULL,
    company_siren TEXT,
    deal_value DECIMAL(15,2),
    deal_status TEXT NOT NULL CHECK (deal_status IN ('won', 'lost')),
    deal_date DATE NOT NULL,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create features_company table
CREATE TABLE features_company (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    company_id UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    feature_value TEXT NOT NULL,
    feature_type TEXT NOT NULL CHECK (feature_type IN ('text', 'numeric', 'boolean', 'categorical')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(company_id, feature_name)
);

-- Create scores table
CREATE TABLE scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    company_id UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    score_value DECIMAL(5,4) NOT NULL CHECK (score_value >= 0 AND score_value <= 1),
    score_source score_source NOT NULL,
    model_version TEXT,
    confidence DECIMAL(5,4),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(company_id, score_source, model_version)
);

-- Enable Row Level Security on all tables
ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE companies ENABLE ROW LEVEL SECURITY;
ALTER TABLE historical_deals ENABLE ROW LEVEL SECURITY;
ALTER TABLE features_company ENABLE ROW LEVEL SECURITY;
ALTER TABLE scores ENABLE ROW LEVEL SECURITY;

-- Create basic RLS policies (allow all for now, will be restricted later)
CREATE POLICY "Allow all operations on tenants" ON tenants FOR ALL USING (true);
CREATE POLICY "Allow all operations on companies" ON companies FOR ALL USING (true);
CREATE POLICY "Allow all operations on historical_deals" ON historical_deals FOR ALL USING (true);
CREATE POLICY "Allow all operations on features_company" ON features_company FOR ALL USING (true);
CREATE POLICY "Allow all operations on scores" ON scores FOR ALL USING (true);

-- Create indexes for better performance
CREATE INDEX idx_companies_tenant_id ON companies(tenant_id);
CREATE INDEX idx_companies_siren ON companies(siren);
CREATE INDEX idx_historical_deals_tenant_id ON historical_deals(tenant_id);
CREATE INDEX idx_historical_deals_company_id ON historical_deals(company_id);
CREATE INDEX idx_historical_deals_deal_status ON historical_deals(deal_status);
CREATE INDEX idx_features_company_tenant_id ON features_company(tenant_id);
CREATE INDEX idx_features_company_company_id ON features_company(company_id);
CREATE INDEX idx_scores_tenant_id ON scores(tenant_id);
CREATE INDEX idx_scores_company_id ON scores(company_id);
CREATE INDEX idx_scores_score_source ON scores(score_source);
