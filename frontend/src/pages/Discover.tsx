import React, { useState } from 'react';
import { postEdge } from "../lib/api";
import { toCSV } from "../lib/csv";

type Company = {
  company_id: string;
  name: string;
  siren: string;
  ape: string;
  region: string;
  department: string;
  win_score: number;
  band: string;
  confidence_badge: string;
  neighbors: Array<{name: string, sim: number, outcome: string}>;
  reasons: string[];
  source: string;
};

const Discover: React.FC = () => {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  const run = async () => {
    setLoading(true);
    setError("");
    try {
      const payload = {
        tenant_id: "dev-tenant",
        filters: { 
          ape_codes: ["6201Z"], 
          regions: ["Île-de-France"],
          age_buckets: ["0-5"],
          headcount_buckets: ["1-10"]
        }
      };
      const response = await postEdge("/discover", payload);
      
      if (response.ok) {
        setCompanies(response.items || []);
      } else {
        setError(response.error || "Discovery failed");
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const download = () => {
    if (!companies.length) return;
    const cols = ["name", "siren", "ape", "region", "department", "win_score", "band", "confidence_badge", "source"];
    const csv = toCSV(companies, cols);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "chyll_discover_results.csv";
    a.click();
  };

  return (
    <div className="bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-extrabold text-gray-900">
              Discover & Score Companies
            </h2>
            <p className="mt-2 text-sm text-gray-600">
              Build and score companies using ML predictions
            </p>
          </div>
          <div className="text-sm text-gray-500">
            Enhanced ML Pipeline
          </div>
        </div>
        
        <div className="flex gap-4 mb-8">
          <button
            onClick={run}
            disabled={loading}
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Scoring...
              </>
            ) : (
              'Build & Score Universe'
            )}
          </button>
          <button 
            className="px-6 py-3 border border-gray-300 rounded-md shadow-sm text-base font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
            onClick={download} 
            disabled={!companies.length}
          >
            Download CSV
          </button>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
            <div className="text-red-800">{error}</div>
          </div>
        )}

        {companies.length > 0 && (
          <div className="bg-white shadow rounded-lg overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">
                Top Scored Companies ({companies.length} results)
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Company</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">SIREN</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">APE</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Region</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Department</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Band</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Why</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Similar Companies</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Source</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {companies.map((company, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{company.name}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{company.siren}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{company.ape}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{company.region}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{company.department}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-medium">{company.win_score.toFixed(3)}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          company.band === 'High' ? 'bg-green-100 text-green-800' :
                          company.band === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {company.band}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-500 max-w-xs">
                        <div className="flex flex-wrap gap-1">
                          {(company.reasons || []).slice(0, 3).map((reason, idx) => (
                            <span key={idx} className="inline-flex px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded">
                              {reason}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-500 max-w-xs">
                        {(company.neighbors || []).slice(0, 2).map((neighbor, idx) => (
                          <div key={idx} className="text-xs">
                            {neighbor.name} ({neighbor.outcome}) - {neighbor.sim.toFixed(2)}
                          </div>
                        ))}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          company.source === 'sirene' ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'
                        }`}>
                          {company.source}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Discover;
