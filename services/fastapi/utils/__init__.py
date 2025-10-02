"""
Utility functions for the Chyll Lead MVP ML service
"""

import logging
from typing import Dict, Any, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_tenant_id(tenant_id: str) -> bool:
    """Validate tenant ID format"""
    if not tenant_id or not isinstance(tenant_id, str):
        return False
    # Basic UUID format check
    return len(tenant_id) == 36 and tenant_id.count('-') == 4

def log_request(route: str, tenant_id: str, data: Dict[str, Any] = None):
    """Log incoming requests"""
    logger.info(f"Request to {route} for tenant {tenant_id}")
    if data:
        logger.debug(f"Request data: {json.dumps(data, indent=2)}")

def format_score(score: float) -> float:
    """Format score to 4 decimal places"""
    return round(score, 4)

def validate_company_data(company_data: Dict[str, Any]) -> bool:
    """Validate company data structure"""
    required_fields = ['name', 'siren']
    return all(field in company_data for field in required_fields)

def calculate_confidence_score(features: List[float]) -> float:
    """Calculate confidence score based on feature completeness"""
    if not features:
        return 0.0
    
    # Simple confidence based on non-zero features
    non_zero_features = sum(1 for f in features if f != 0)
    return non_zero_features / len(features)
