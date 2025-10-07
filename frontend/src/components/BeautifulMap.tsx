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

interface BeautifulMapProps {
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
    const isMission = company.societe_mission_unite_legale === 'T';
    const isESS = company.economie_sociale_solidaire_unite_legale === 'T';
    
    return L.divIcon({
      className: 'custom-marker',
      html: `
        <div style="
          background: linear-gradient(135deg, ${color} 0%, ${color}dd 100%);
          width: 20px;
          height: 20px;
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
          ${isMission ? '🎯' : isESS ? '🤝' : '🏢'}
        </div>
      `,
      iconSize: [26, 26],
      iconAnchor: [13, 13]
    });
  };

  return (
    <Marker
      position={[company.latitude, company.longitude]}
      icon={createCustomIcon()}
    >
      <Popup className="custom-popup">
        <div className="p-4 min-w-[280px]">
          {/* Company Header */}
          <div className="flex items-start justify-between mb-3">
            <div className="flex-1">
              <h3 className="font-bold text-lg text-gray-900 mb-1">
                {company.denomination_unite_legale}
              </h3>
              <div className="text-sm text-gray-600">
                📍 {company.libelle_commune} ({company.code_postal})
              </div>
            </div>
            <div className="text-2xl">
              {company.societe_mission_unite_legale === 'T' ? '🎯' : 
               company.economie_sociale_solidaire_unite_legale === 'T' ? '🤝' : '🏢'}
            </div>
          </div>

          {/* Activity */}
          <div className="mb-4">
            <div className="text-sm text-gray-700">
              <span className="font-medium">Activité:</span> {company.activite_principale_unite_legale}
            </div>
          </div>

          {/* Impact Badges */}
          <div className="flex flex-wrap gap-2 mb-4">
            {company.societe_mission_unite_legale === 'T' && (
              <span className="px-3 py-1 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-full text-xs font-semibold shadow-md">
                🎯 Mission
              </span>
            )}
            {company.economie_sociale_solidaire_unite_legale === 'T' && (
              <span className="px-3 py-1 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-full text-xs font-semibold shadow-md">
                🤝 ESS
              </span>
            )}
            {company.in_qpv && (
              <span className="px-3 py-1 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-full text-xs font-semibold shadow-md">
                🏙️ QPV: {company.qpv_label}
              </span>
            )}
            {company.is_zrr && (
              <span className="px-3 py-1 bg-gradient-to-r from-yellow-500 to-orange-500 text-white rounded-full text-xs font-semibold shadow-md">
                🌾 ZRR: {company.zrr_commune}
              </span>
            )}
          </div>

          {/* Impact Message */}
          <div className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg p-3">
            <div className="text-sm text-gray-700">
              <span className="font-semibold">Impact:</span> Cette entreprise contribue au{' '}
              {company.societe_mission_unite_legale === 'T' ? 'développement durable' : 'développement social'}{' '}
              {company.in_qpv ? 'dans un quartier prioritaire' : ''}
              {company.is_zrr ? 'en zone rurale' : ''}.
            </div>
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
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [companies, map]);

  return null;
};

const BeautifulMap: React.FC<BeautifulMapProps> = ({
  companies,
  showQpvZones,
  showZrrZones
}) => {
  const [mapCenter] = useState<[number, number]>([46.2276, 2.2137]); // France center
  const [mapLoaded, setMapLoaded] = useState(false);

  const markers = useMemo(() => {
    return companies.map((company, index) => (
      <CompanyMarker
        key={`${company.siren}-${index}`}
        company={company}
      />
    ));
  }, [companies]);

  useEffect(() => {
    const timer = setTimeout(() => setMapLoaded(true), 1000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="bg-white bg-opacity-90 backdrop-blur-xl border border-gray-200 border-opacity-50 rounded-3xl p-8 shadow-xl">
      {/* Map Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            Carte d'Impact Social
          </h2>
          <p className="text-gray-600">
            {companies.length} entreprises qui transforment la France
          </p>
        </div>
        <div className="flex gap-4 text-sm">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full mr-2"></div>
            <span className="font-medium">Mission</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full mr-2"></div>
            <span className="font-medium">ESS</span>
          </div>
        </div>
      </div>

      {/* Map Container */}
      <div className="h-[500px] rounded-2xl overflow-hidden border border-gray-200 shadow-lg">
        {!mapLoaded ? (
          <div className="h-full flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div>
              <p className="text-gray-600 font-medium">Chargement de la carte...</p>
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

      {/* Map Legend */}
      <div className="mt-6 p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-2xl">
        <h3 className="text-sm font-semibold text-gray-900 mb-3">Légende</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full mr-2"></div>
            <span>Entreprises Mission</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full mr-2"></div>
            <span>ESS</span>
          </div>
          {showQpvZones && (
            <div className="flex items-center">
              <div className="w-4 h-4 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full mr-2"></div>
              <span>Zones QPV</span>
            </div>
          )}
          {showZrrZones && (
            <div className="flex items-center">
              <div className="w-4 h-4 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full mr-2"></div>
              <span>Zones ZRR</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BeautifulMap;
