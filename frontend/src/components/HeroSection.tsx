import React from 'react';

interface HeroSectionProps {
  totalCompanies: number;
  missionCompanies: number;
  essCompanies: number;
  qpvCompanies: number;
  zrrCompanies: number;
}

const HeroSection: React.FC<HeroSectionProps> = ({
  totalCompanies,
  missionCompanies,
  essCompanies,
  qpvCompanies,
  zrrCompanies
}) => {
  return (
    <div className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-20">
        <div className="w-full h-full" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          backgroundRepeat: 'repeat'
        }}></div>
      </div>
      
      {/* Floating Elements */}
      <div className="absolute top-20 left-10 w-20 h-20 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full opacity-20 floating-animation"></div>
      <div className="absolute top-40 right-20 w-16 h-16 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full opacity-20 floating-animation" style={{animationDelay: '2s'}}></div>
      <div className="absolute bottom-20 left-1/4 w-12 h-12 bg-gradient-to-r from-green-400 to-emerald-400 rounded-full opacity-20 floating-animation" style={{animationDelay: '4s'}}></div>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
        <div className="text-center">
          {/* Main Title */}
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 slide-up">
            <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
              Impact
            </span>
            <br />
            <span className="text-white">Map</span>
          </h1>
          
          {/* Subtitle */}
          <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-3xl mx-auto fade-in" style={{animationDelay: '0.2s'}}>
            Découvrez les entreprises à mission et ESS qui transforment la France. 
            <span className="text-purple-400 font-semibold"> Visualisez l'impact social</span> sur votre territoire.
          </p>

          {/* Impact Stats */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-6 mb-16">
            <div className="bg-white bg-opacity-80 backdrop-blur-xl border border-white border-opacity-20 shadow-xl rounded-2xl p-6 text-center slide-up" style={{animationDelay: '0.4s'}}>
              <div className="text-3xl font-bold text-gray-900 mb-2">{totalCompanies.toLocaleString()}</div>
              <div className="text-sm text-gray-600 font-medium">Entreprises</div>
            </div>
            
            <div className="bg-white bg-opacity-80 backdrop-blur-xl border border-white border-opacity-20 shadow-xl rounded-2xl p-6 text-center slide-up" style={{animationDelay: '0.5s'}}>
              <div className="text-3xl font-bold text-purple-600 mb-2">{missionCompanies.toLocaleString()}</div>
              <div className="text-sm text-gray-600 font-medium">Mission</div>
            </div>
            
            <div className="bg-white bg-opacity-80 backdrop-blur-xl border border-white border-opacity-20 shadow-xl rounded-2xl p-6 text-center slide-up" style={{animationDelay: '0.6s'}}>
              <div className="text-3xl font-bold text-green-600 mb-2">{essCompanies.toLocaleString()}</div>
              <div className="text-sm text-gray-600 font-medium">ESS</div>
            </div>
            
            <div className="bg-white bg-opacity-80 backdrop-blur-xl border border-white border-opacity-20 shadow-xl rounded-2xl p-6 text-center slide-up" style={{animationDelay: '0.7s'}}>
              <div className="text-3xl font-bold text-blue-600 mb-2">{qpvCompanies.toLocaleString()}</div>
              <div className="text-sm text-gray-600 font-medium">QPV</div>
            </div>
            
            <div className="bg-white bg-opacity-80 backdrop-blur-xl border border-white border-opacity-20 shadow-xl rounded-2xl p-6 text-center slide-up" style={{animationDelay: '0.8s'}}>
              <div className="text-3xl font-bold text-yellow-600 mb-2">{zrrCompanies.toLocaleString()}</div>
              <div className="text-sm text-gray-600 font-medium">ZRR</div>
            </div>
          </div>

          {/* Call to Action */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center fade-in" style={{animationDelay: '1s'}}>
            <button className="px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold rounded-2xl shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300 pulse-glow">
              🗺️ Explorer la carte
            </button>
            <button className="px-8 py-4 bg-white bg-opacity-80 backdrop-blur-xl border border-white border-opacity-20 shadow-xl text-gray-700 font-semibold rounded-2xl hover:bg-white hover:bg-opacity-90 transition-all duration-300">
              📊 Voir les statistiques
            </button>
          </div>
        </div>
      </div>

      {/* Bottom Wave */}
      <div className="absolute bottom-0 left-0 right-0">
        <svg viewBox="0 0 1200 120" preserveAspectRatio="none" className="w-full h-16 fill-white">
          <path d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z"></path>
        </svg>
      </div>
    </div>
  );
};

export default HeroSection;
