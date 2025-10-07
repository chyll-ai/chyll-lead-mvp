import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './App.css';

// Fix for default markers
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

interface Company {
  siren: string;
  siret: string;
  denomination_unite_legale: string | null;
  libelle_commune: string | null;
  code_postal: string | null;
  latitude: number;
  longitude: number;
  tags: string[]; // ['ESS', 'Mission', 'QPV', 'ZRR']
  qpv_label: string | null;
  zrr_classification: string | null;
  activite_principale_unite_legale: string | null;
}

function App() {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    mission: true,
    ess: true,
    qpv: false,
    zrr: false
  });

  useEffect(() => {
    const fetchCompanies = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || 'https://chyll-lead-mvp-production.up.railway.app';
        const response = await fetch(`${apiUrl}/companies?limit=50000`);
        if (!response.ok) {
          throw new Error('Failed to fetch companies');
        }
        const data = await response.json();
        setCompanies(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching companies:', error);
        setLoading(false);
      }
    };

    fetchCompanies();
  }, []);

  const filteredCompanies = companies.filter(company => {
    if (filters.mission && company.tags.includes('Mission')) return true;
    if (filters.ess && company.tags.includes('ESS')) return true;
    if (filters.qpv && company.tags.includes('QPV')) return true;
    if (filters.zrr && company.tags.includes('ZRR')) return true;
    return false;
  });

  const createCustomIcon = (company: Company) => {
    let color = '#6B7280';
    let icon = '🏢';
    
    if (company.tags.includes('Mission')) {
      color = '#8B5CF6';
      icon = '🎯';
    } else if (company.tags.includes('ESS')) {
      color = '#10B981';
      icon = '🤝';
    }
    
    // Special colors for multiple tags
    if (company.tags.length > 1) {
      if (company.tags.includes('Mission') && company.tags.includes('ESS')) {
        color = '#7C3AED'; // Purple for Mission + ESS
        icon = '🌟';
      } else if (company.tags.includes('QPV') || company.tags.includes('ZRR')) {
        color = '#F59E0B'; // Orange for QPV/ZRR
        icon = '🏘️';
      }
    }
    
    return L.divIcon({
      className: 'custom-marker',
      html: `
        <div style="
          background: ${color};
          width: 24px;
          height: 24px;
          border-radius: 50%;
          border: 3px solid white;
          box-shadow: 0 4px 12px rgba(0,0,0,0.3);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          color: white;
          font-weight: bold;
        ">
          ${icon}
        </div>
      `,
      iconSize: [30, 30],
      iconAnchor: [15, 15]
    });
  };

  const MapUpdater = ({ companies }: { companies: Company[] }) => {
    const map = useMap();
    
    useEffect(() => {
      if (companies.length > 0) {
        const bounds = L.latLngBounds(
          companies.map(company => [company.latitude, company.longitude])
        );
        map.fitBounds(bounds, { padding: [50, 50] });
      }
    }, [companies, map]);

    return null;
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner"></div>
        <div className="loading-text">Chargement de la carte...</div>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <h1>Impact Map</h1>
        <div className="stats">
          <span className="stat">{filteredCompanies.length} entreprises</span>
        </div>
      </div>

      {/* Filters */}
      <div className="filters">
        <button 
          className={`filter-btn ${filters.mission ? 'active' : ''}`}
          onClick={() => setFilters(prev => ({ ...prev, mission: !prev.mission }))}
        >
          🎯 Mission
        </button>
        <button 
          className={`filter-btn ${filters.ess ? 'active' : ''}`}
          onClick={() => setFilters(prev => ({ ...prev, ess: !prev.ess }))}
        >
          🤝 ESS
        </button>
        <button 
          className={`filter-btn ${filters.qpv ? 'active' : ''}`}
          onClick={() => setFilters(prev => ({ ...prev, qpv: !prev.qpv }))}
        >
          🏙️ QPV
        </button>
        <button 
          className={`filter-btn ${filters.zrr ? 'active' : ''}`}
          onClick={() => setFilters(prev => ({ ...prev, zrr: !prev.zrr }))}
        >
          🌾 ZRR
        </button>
      </div>

      {/* Map */}
      <div className="map-container">
        <MapContainer
          center={[46.2276, 2.2137]}
          zoom={6}
          style={{ height: '100%', width: '100%' }}
          className="map"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          <MapUpdater companies={filteredCompanies} />
          
          {filteredCompanies.map((company) => (
            <Marker
              key={company.siren}
              position={[company.latitude, company.longitude]}
              icon={createCustomIcon(company)}
            >
              <Popup>
                <div className="popup-content">
                  <h3>{company.denomination_unite_legale || 'Nom non disponible'}</h3>
                  <p>📍 {company.libelle_commune || 'Commune non disponible'} {company.code_postal && `(${company.code_postal})`}</p>
                  {company.activite_principale_unite_legale && (
                    <p>🏢 {company.activite_principale_unite_legale}</p>
                  )}
                  <div className="badges">
                    {company.tags.map(tag => (
                      <span key={tag} className={`badge ${tag.toLowerCase()}`}>
                        {tag === 'Mission' && '🎯 Mission'}
                        {tag === 'ESS' && '🤝 ESS'}
                        {tag === 'QPV' && '🏙️ QPV'}
                        {tag === 'ZRR' && '🌾 ZRR'}
                      </span>
                    ))}
                  </div>
                  {company.qpv_label && (
                    <p className="qpv-info">🏙️ QPV: {company.qpv_label}</p>
                  )}
                  {company.zrr_classification && (
                    <p className="zrr-info">🌾 ZRR: {company.zrr_classification}</p>
                  )}
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>

      {/* Legend */}
      <div className="legend">
        <div className="legend-item">
          <div className="legend-marker mission"></div>
          <span>Mission</span>
        </div>
        <div className="legend-item">
          <div className="legend-marker ess"></div>
          <span>ESS</span>
        </div>
        <div className="legend-item">
          <div className="legend-marker qpv"></div>
          <span>QPV</span>
        </div>
        <div className="legend-item">
          <div className="legend-marker zrr"></div>
          <span>ZRR</span>
        </div>
      </div>
    </div>
  );
}

export default App;