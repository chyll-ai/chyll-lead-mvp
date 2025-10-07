import React from 'react';

interface FilterControlsProps {
  filters: {
    mission: boolean;
    ess: boolean;
    qpv: boolean;
    zrr: boolean;
  };
  onFilterChange: (filter: keyof FilterControlsProps['filters']) => void;
  counts: {
    mission: number;
    ess: number;
    qpv: number;
    zrr: number;
  };
}

const FilterControls: React.FC<FilterControlsProps> = ({
  filters,
  onFilterChange,
  counts
}) => {
  const filterOptions = [
    {
      key: 'mission' as const,
      label: 'Entreprises Mission',
      icon: '🎯',
      gradient: 'mission-gradient',
      count: counts.mission,
      description: 'Entreprises avec une raison d\'être'
    },
    {
      key: 'ess' as const,
      label: 'ESS',
      icon: '🤝',
      gradient: 'ess-gradient',
      count: counts.ess,
      description: 'Économie Sociale et Solidaire'
    },
    {
      key: 'qpv' as const,
      label: 'Zones QPV',
      icon: '🏙️',
      gradient: 'social-gradient',
      count: counts.qpv,
      description: 'Quartiers Prioritaires de la Ville'
    },
    {
      key: 'zrr' as const,
      label: 'Zones ZRR',
      icon: '🌾',
      gradient: 'impact-gradient',
      count: counts.zrr,
      description: 'Zones de Revitalisation Rurale'
    }
  ];

  return (
    <div className="bg-white bg-opacity-90 backdrop-blur-xl border border-gray-200 border-opacity-50 rounded-3xl p-8 shadow-xl">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Filtres d'Impact</h2>
        <p className="text-gray-600">Sélectionnez les types d'entreprises à visualiser</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {filterOptions.map((option, index) => (
          <div
            key={option.key}
            className={`relative group cursor-pointer transition-all duration-300 transform hover:scale-105 ${
              filters[option.key] ? 'scale-105' : 'hover:scale-105'
            }`}
            onClick={() => onFilterChange(option.key)}
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <div className={`rounded-2xl p-6 border-2 transition-all duration-300 ${
              filters[option.key] 
                ? 'border-transparent shadow-xl' 
                : 'border-gray-200 hover:border-gray-300 hover:shadow-lg'
            }`}>
              {/* Background Gradient */}
              {filters[option.key] && (
                <div className={`absolute inset-0 ${option.gradient} rounded-2xl opacity-10`}></div>
              )}
              
              {/* Content */}
              <div className="relative z-10">
                {/* Icon */}
                <div className="text-4xl mb-4 text-center">{option.icon}</div>
                
                {/* Label */}
                <h3 className="text-lg font-semibold text-gray-900 mb-2 text-center">
                  {option.label}
                </h3>
                
                {/* Count */}
                <div className="text-center mb-3">
                  <span className={`text-3xl font-bold ${
                    filters[option.key] 
                      ? option.key === 'mission' ? 'text-purple-600' :
                        option.key === 'ess' ? 'text-green-600' :
                        option.key === 'qpv' ? 'text-blue-600' : 'text-yellow-600'
                      : 'text-gray-400'
                  }`}>
                    {option.count.toLocaleString()}
                  </span>
                </div>
                
                {/* Description */}
                <p className="text-sm text-gray-600 text-center mb-4">
                  {option.description}
                </p>
                
                {/* Toggle Indicator */}
                <div className="flex justify-center">
                  <div className={`w-12 h-6 rounded-full transition-all duration-300 ${
                    filters[option.key] 
                      ? option.key === 'mission' ? 'bg-purple-500' :
                        option.key === 'ess' ? 'bg-green-500' :
                        option.key === 'qpv' ? 'bg-blue-500' : 'bg-yellow-500'
                      : 'bg-gray-300'
                  }`}>
                    <div className={`w-5 h-5 bg-white rounded-full shadow-md transform transition-all duration-300 ${
                      filters[option.key] ? 'translate-x-6' : 'translate-x-0.5'
                    } mt-0.5`}></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Active Filters Summary */}
      <div className="mt-8 pt-6 border-t border-gray-200">
        <div className="flex flex-wrap gap-3 justify-center">
          {filterOptions
            .filter(option => filters[option.key])
            .map(option => (
              <div
                key={option.key}
                className={`px-4 py-2 rounded-full text-sm font-medium text-white ${option.gradient} shadow-lg`}
              >
                {option.icon} {option.label} ({option.count})
              </div>
            ))
          }
          {filterOptions.every(option => !filters[option.key]) && (
            <div className="px-4 py-2 rounded-full text-sm font-medium text-gray-500 bg-gray-100">
              Aucun filtre actif
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FilterControls;
