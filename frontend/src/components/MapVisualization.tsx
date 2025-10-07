import React, { useState, useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default markers in React Leaflet
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

// Fix for default markers
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

interface Company {
  siren: string;
  siret: string;
  denomination_unite_legale: string;
  libelle_commune: string;
  code_postal: string;
  latitude: number;
  longitude: number;
  activite_principale_unite_legale: string;
  societe_mission_unite_legale: string;
  economie_sociale_solidaire_unite_legale: string;
  in_qpv: boolean;
  qpv_label: string;
  is_zrr: boolean;
  zrr_commune: string;
}

interface MapVisualizationProps {
  companies: Company[];
  showQpvZones: boolean;
  showZrrZones: boolean;
}

const CompanyMarker: React.FC<{
  company: Company;
}> = ({ company }) => {
  const getMarkerColor = () => {
    if (company.societe_mission_unite_legale === 'T') return '#8B5CF6'; // Purple for Mission
    if (company.economie_sociale_solidaire_unite_legale === 'T') return '#10B981'; // Green for ESS
    return '#6B7280'; // Gray for others
  };

  const createCustomIcon = () => {
    const color = getMarkerColor();
    return L.divIcon({
      className: 'custom-marker',
      html: `<div style="background-color: ${color}; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>`,
      iconSize: [16, 16],
      iconAnchor: [8, 8]
    });
  };

  return (
    <Marker
      position={[company.latitude, company.longitude]}
      icon={createCustomIcon()}
    >
      <Popup>
        <div className="p-2 min-w-[200px]">
          <h3 className="font-semibold text-sm mb-2">{company.denomination_unite_legale}</h3>
          <div className="text-xs text-gray-600 mb-2">
            <div>{company.libelle_commune} ({company.code_postal})</div>
            <div>Activité: {company.activite_principale_unite_legale}</div>
          </div>
          <div className="flex flex-wrap gap-1 mb-2">
            {company.societe_mission_unite_legale === 'T' && (
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">
                Mission
              </span>
            )}
            {company.economie_sociale_solidaire_unite_legale === 'T' && (
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">
                ESS
              </span>
            )}
            {company.in_qpv && (
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                QPV: {company.qpv_label}
              </span>
            )}
            {company.is_zrr && (
              <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded text-xs">
                ZRR: {company.zrr_commune}
              </span>
            )}
          </div>
        </div>
      </Popup>
    </Marker>
  );
};

const MapUpdater: React.FC<{ companies: Company[] }> = ({ companies }) => {
  const map = useMap();
  
  useEffect(() => {
    if (companies.length > 0) {
      const bounds = L.latLngBounds(
        companies.map(company => [company.latitude, company.longitude])
      );
      map.fitBounds(bounds, { padding: [20, 20] });
    }
  }, [companies, map]);

  return null;
};

const MapVisualization: React.FC<MapVisualizationProps> = ({
  companies,
  showQpvZones,
  showZrrZones
}) => {
  const [mapCenter] = useState<[number, number]>([46.2276, 2.2137]); // France center
  const [mapError, setMapError] = useState<string | null>(null);

  const markers = useMemo(() => {
    return companies.map((company, index) => (
      <CompanyMarker
        key={`${company.siren}-${index}`}
        company={company}
      />
    ));
  }, [companies]);

  // Handle map loading errors
  useEffect(() => {
    const handleError = () => {
      setMapError('Erreur de chargement de la carte');
    };
    
    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, []);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-900">
          Carte Interactive ({companies.length} entreprises)
        </h2>
        <div className="flex gap-4 text-sm">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-purple-500 rounded-full mr-2"></div>
            <span>Mission</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
            <span>ESS</span>
          </div>
        </div>
      </div>

      <div className="h-96 rounded-lg overflow-hidden border border-gray-200">
        {mapError ? (
          <div className="h-full flex items-center justify-center bg-gray-100">
            <div className="text-center">
              <div className="text-red-600 mb-2">⚠️ {mapError}</div>
              <div className="text-sm text-gray-600">
                <p>Nombre d'entreprises: {companies.length}</p>
                <p>Mission: {companies.filter(c => c.societe_mission_unite_legale === 'T').length}</p>
                <p>ESS: {companies.filter(c => c.economie_sociale_solidaire_unite_legale === 'T').length}</p>
              </div>
            </div>
          </div>
        ) : (
          <MapContainer
            center={mapCenter}
            zoom={6}
            style={{ height: '100%', width: '100%' }}
            className="z-0"
            zoomControl={true}
            scrollWheelZoom={true}
            doubleClickZoom={true}
            dragging={true}
            keyboard={true}
            attributionControl={true}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            
            <MapUpdater companies={companies} />
            {markers}
          </MapContainer>
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <h3 className="text-sm font-semibold text-gray-900 mb-2">Légende</h3>
        <div className="flex flex-wrap gap-4 text-xs">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-purple-500 rounded-full mr-2"></div>
            <span>Entreprises Mission</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
            <span>ESS</span>
          </div>
          {showQpvZones && (
            <div className="flex items-center">
              <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
              <span>Zones QPV</span>
            </div>
          )}
          {showZrrZones && (
            <div className="flex items-center">
              <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
              <span>Zones ZRR</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MapVisualization;
