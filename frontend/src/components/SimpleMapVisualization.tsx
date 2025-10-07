import React from 'react';

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

interface SimpleMapProps {
  companies: Company[];
  showQpvZones: boolean;
  showZrrZones: boolean;
}

const SimpleMapVisualization: React.FC<SimpleMapProps> = ({
  companies,
  showQpvZones,
  showZrrZones
}) => {
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

      {/* Simple map placeholder */}
      <div className="h-96 rounded-lg overflow-hidden border border-gray-200 bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">🗺️</div>
          <h3 className="text-lg font-semibold text-gray-700 mb-2">Carte des Entreprises</h3>
          <p className="text-gray-600 mb-4">
            {companies.length} entreprises trouvées
          </p>
          
          {/* Company list */}
          <div className="max-h-48 overflow-y-auto space-y-2">
            {companies.map((company) => (
              <div key={company.siren} className="bg-white p-3 rounded border text-left">
                <div className="font-semibold text-sm">{company.denomination_unite_legale}</div>
                <div className="text-xs text-gray-600">{company.libelle_commune} ({company.code_postal})</div>
                <div className="flex gap-1 mt-1">
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
                      QPV
                    </span>
                  )}
                  {company.is_zrr && (
                    <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded text-xs">
                      ZRR
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
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

export default SimpleMapVisualization;
