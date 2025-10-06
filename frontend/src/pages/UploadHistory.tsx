import React, { useState } from 'react';
import { postEdge } from "../lib/api";

const UploadHistory: React.FC = () => {
  const [rows, setRows] = useState<any[]>([]);
  const [logs, setLogs] = useState<string>("");
  const [discoveredLeads, setDiscoveredLeads] = useState<any[]>([]);
  const [showLeads, setShowLeads] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [modelTrained, setModelTrained] = useState(false);
  
  // Enhanced training results
  const [hypotheses, setHypotheses] = useState<any>(null);
  const [discoveryCriteria, setDiscoveryCriteria] = useState<any>(null);
  const [showTrainingResults, setShowTrainingResults] = useState(false);

  const onFile = async (file: File) => {
    const text = await file.text();
    const lines = text.split(/\r?\n/).filter(Boolean);
    const headers = lines[0].split(",").map(h=>h.trim());
    const data = lines.slice(1).map((l) => {
      const cols = l.split(",");
      const obj: any = {};
      headers.forEach((h,i)=>obj[h]= (cols[i]||"").trim());
      return obj;
    });
    setRows(data);
  };

  const train = async () => {
    setIsTraining(true);
    setLogs("🔄 Training model with your data...");
    setShowLeads(false);
    setModelTrained(false);
    setShowTrainingResults(false);
    
    try {
      const res = await postEdge("/train", { tenant_id: "dev-tenant", rows });
      console.log("Training response:", res);
      
      if (res.ok && res.enriched_data) {
        setModelTrained(true);
        setHypotheses(res.hypotheses);
        setDiscoveryCriteria(res.discovery_criteria);
        setShowTrainingResults(true);
        
        const analysis = res.analysis;
        setLogs(`✅ Model trained successfully!\n📊 Enriched ${analysis.total_companies} companies\n🎯 ${analysis.won_companies} won, ${analysis.lost_companies} lost\n📈 Enrichment success: ${analysis.enrichment_success_rate}\n\n🔍 Click "Discover Companies" to find new leads!`);
      } else {
        setLogs(`❌ Error: ${res.error || 'Unknown error'}`);
      }
    } catch (e:any) {
      setLogs(`❌ Error: ${e.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const discover = async () => {
    setIsDiscovering(true);
    setLogs("🔍 Discovering companies from SIRENE database...");
    setShowLeads(false);
    
    try {
      const res = await postEdge("/discover", { 
        tenant_id: "dev-tenant", 
        filters: {}, // Empty filters - backend will use learned patterns
        limit: 20 
      });
      console.log("Discovery response:", res);
      
      if (res.ok && res.companies && res.companies.length > 0) {
        setDiscoveredLeads(res.companies);
        setShowLeads(true);
        setLogs(`🎉 Found ${res.companies.length} companies matching your winning patterns!\n\n📈 Companies are sorted by win probability (highest first).`);
      } else {
        setLogs(`⚠️ No companies found matching your patterns. Try training with more diverse data.`);
      }
    } catch (e:any) {
      setLogs(`❌ Error: ${e.message}`);
    } finally {
      setIsDiscovering(false);
    }
  };

  return (
    <div className="bg-black py-12 px-4 sm:px-6 lg:px-8 min-h-screen">
      {/* Dark theme - updated - v2 */}
      <div className="max-w-md mx-auto">
        <div className="text-center">
          <h2 className="mt-6 text-3xl font-extrabold text-white">
            Upload Your History
          </h2>
          <p className="mt-2 text-sm text-gray-300">
            Upload your won/lost deals to train your predictive model
          </p>
        </div>
        
        <div className="mt-8 space-y-6">
          <div>
            <label htmlFor="file-upload" className="block text-sm font-medium text-gray-300">
              CSV File
            </label>
            <input
              id="file-upload"
              type="file"
              accept=".csv"
              onChange={(e) => e.target.files && onFile(e.target.files[0])}
              className="mt-1 block w-full px-3 py-2 border border-gray-600 rounded-md shadow-sm bg-gray-800 text-white focus:outline-none focus:ring-white focus:border-white"
            />
          </div>

          {rows.length > 0 && (
            <div className="bg-gray-900 shadow rounded-lg p-4 border border-gray-700">
              <h3 className="text-lg font-medium text-white mb-3">
                Preview ({rows.length} rows)
              </h3>
              <div className="space-y-2">
                {rows.slice(0, 3).map((row, index) => (
                  <div key={index} className="text-sm text-gray-300">
                    {row.company_name} - {row.deal_status} - {row.website || 'No website'}
                  </div>
                ))}
                {rows.length > 3 && (
                  <div className="text-sm text-gray-400">
                    ... and {rows.length - 3} more rows
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="space-y-3">
            <button
              onClick={train}
              disabled={!rows.length || isTraining}
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-black bg-white hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-white disabled:bg-gray-600 disabled:cursor-not-allowed"
            >
              {isTraining ? "🔄 Training..." : "🧠 Train My Model"}
            </button>
            
            <button
              onClick={discover}
              disabled={!modelTrained || isDiscovering}
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-gray-700 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 disabled:bg-gray-800 disabled:cursor-not-allowed"
            >
              {isDiscovering ? "🔍 Discovering..." : "🎯 Discover Companies"}
            </button>
          </div>

          {logs && (
            <div className="bg-gray-800 p-3 rounded-md border border-gray-700">
              <pre className="text-sm whitespace-pre-wrap text-white">{logs}</pre>
            </div>
          )}

          {/* Training Results Section */}
          {showTrainingResults && (
            <div className="mt-6 space-y-4">
              {/* Hypotheses Display */}
              {hypotheses && (
                <div className="bg-gray-900 shadow rounded-lg p-4 border border-gray-700">
                  <h3 className="text-lg font-medium text-white mb-3">
                    🧠 Discovered Patterns & Hypotheses
                  </h3>
                  
                  {/* Strong Patterns */}
                  {hypotheses.strong_patterns && hypotheses.strong_patterns.length > 0 && (
                    <div className="mb-4">
                      <h4 className="text-sm font-medium text-green-400 mb-2">High Confidence Patterns:</h4>
                      <div className="space-y-2">
                        {hypotheses.strong_patterns.map((pattern: any, idx: number) => (
                          <div key={idx} className="bg-green-900/20 border border-green-700 rounded p-2">
                            <div className="text-sm text-green-200 font-medium">{pattern.pattern}</div>
                            <div className="text-xs text-green-300 mt-1">{pattern.evidence}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Moderate Patterns */}
                  {hypotheses.moderate_patterns && hypotheses.moderate_patterns.length > 0 && (
                    <div className="mb-4">
                      <h4 className="text-sm font-medium text-yellow-400 mb-2">Medium Confidence Patterns:</h4>
                      <div className="space-y-2">
                        {hypotheses.moderate_patterns.map((pattern: any, idx: number) => (
                          <div key={idx} className="bg-yellow-900/20 border border-yellow-700 rounded p-2">
                            <div className="text-sm text-yellow-200 font-medium">{pattern.pattern}</div>
                            <div className="text-xs text-yellow-300 mt-1">{pattern.evidence}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Weak Patterns */}
                  {hypotheses.weak_patterns && hypotheses.weak_patterns.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-400 mb-2">Low Confidence Patterns:</h4>
                      <div className="space-y-2">
                        {hypotheses.weak_patterns.map((pattern: any, idx: number) => (
                          <div key={idx} className="bg-gray-800 border border-gray-600 rounded p-2">
                            <div className="text-sm text-gray-300 font-medium">{pattern.pattern}</div>
                            <div className="text-xs text-gray-400 mt-1">{pattern.evidence}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Discovery Criteria */}
              {discoveryCriteria && (
                <div className="bg-gray-900 shadow rounded-lg p-4 border border-gray-700">
                  <h3 className="text-lg font-medium text-white mb-3">
                    🎯 Discovery Criteria Generated
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="text-sm font-medium text-blue-400 mb-2">Primary Filters:</h4>
                      <div className="text-sm text-gray-300 space-y-1">
                        {discoveryCriteria.primary_filters && Object.entries(discoveryCriteria.primary_filters).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="capitalize">{key.replace(/_/g, ' ')}:</span>
                            <span className="text-blue-300">{Array.isArray(value) ? value.join(', ') : String(value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-purple-400 mb-2">Secondary Filters:</h4>
                      <div className="text-sm text-gray-300 space-y-1">
                        {discoveryCriteria.secondary_filters && Object.entries(discoveryCriteria.secondary_filters).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="capitalize">{key.replace(/_/g, ' ')}:</span>
                            <span className="text-purple-300">{Array.isArray(value) ? value.join(', ') : String(value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {showLeads && discoveredLeads.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-medium text-white mb-4">
                🎯 Discovered Companies ({discoveredLeads.length} found)
              </h3>
              <div className="bg-gray-900 shadow rounded-lg overflow-hidden border border-gray-700">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-700">
                    <thead className="bg-gray-800">
                      <tr>
                        {/* Core Information */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Company Name</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">SIREN</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">SIRET</th>
                        
                        {/* Business Activity */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">APE Code</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Activity</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Legal Form</th>
                        
                        {/* Location */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Full Address</th>
                        
                        {/* Company Characteristics */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Employee Range</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Age (Years)</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Age Category</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Company Category</th>
                        
                        {/* Status */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Active</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">ESS</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Mission</th>
                        
                        {/* Scoring */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Win Score</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Band</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Reasons</th>
                      </tr>
                    </thead>
                    <tbody className="bg-gray-900 divide-y divide-gray-700">
                      {discoveredLeads.map((lead, i) => (
                        <tr key={i} className="hover:bg-gray-800">
                          {/* Core Information */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs font-medium text-white">
                            <div className="max-w-24 truncate" title={lead.company_name || 'N/A'}>
                              {lead.company_name || 'N/A'}
                            </div>
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300 font-mono">{lead.siren || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300 font-mono">{lead.siret || 'N/A'}</td>
                          
                          {/* Business Activity */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300 font-mono">{lead.ape || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <div className="max-w-20 truncate" title={lead.ape_description || 'N/A'}>
                              {lead.ape_description || 'N/A'}
                            </div>
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <div className="max-w-20 truncate" title={lead.legal_form_description || 'N/A'}>
                              {lead.legal_form_description || 'N/A'}
                            </div>
                          </td>
                          
                          {/* Location */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <div className="max-w-40 truncate" title={lead.full_address || `${lead.postal_code || ''} ${lead.city || ''}`.trim() || 'N/A'}>
                              {lead.full_address || `${lead.postal_code || ''} ${lead.city || ''}`.trim() || 'N/A'}
                            </div>
                          </td>
                          
                          {/* Company Characteristics */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <div className="max-w-20 truncate" title={lead.employee_description || 'N/A'}>
                              {lead.employee_description || 'N/A'}
                            </div>
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.age_years || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.age_category || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.company_category || 'N/A'}</td>
                          
                          {/* Status */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <span className={`inline-flex px-1 py-0.5 text-xs font-semibold rounded-full ${
                              lead.is_active ? 'bg-green-800 text-green-200' : 'bg-red-800 text-red-200'
                            }`}>
                              {lead.is_active ? 'Yes' : 'No'}
                            </span>
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <span className={`inline-flex px-1 py-0.5 text-xs font-semibold rounded-full ${
                              lead.is_ess ? 'bg-blue-800 text-blue-200' : 'bg-gray-800 text-gray-200'
                            }`}>
                              {lead.is_ess ? 'Yes' : 'No'}
                            </span>
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <span className={`inline-flex px-1 py-0.5 text-xs font-semibold rounded-full ${
                              lead.is_mission_company ? 'bg-purple-800 text-purple-200' : 'bg-gray-800 text-gray-200'
                            }`}>
                              {lead.is_mission_company ? 'Yes' : 'No'}
                            </span>
                          </td>
                          
                          {/* Scoring */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-white font-medium">
                            {typeof lead.win_score === 'number' ? lead.win_score.toFixed(3) : lead.win_score || 'N/A'}
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap">
                            <span className={`inline-flex px-1 py-0.5 text-xs font-semibold rounded-full ${
                              lead.band === 'High' ? 'bg-green-800 text-green-200' :
                              lead.band === 'Medium' ? 'bg-yellow-800 text-yellow-200' :
                              'bg-red-800 text-red-200'
                            }`}>
                              {lead.band || 'N/A'}
                            </span>
                          </td>
                          <td className="px-2 py-2 text-xs text-gray-300 max-w-32">
                            <div className="flex flex-wrap gap-1">
                              {(lead.reasons || []).slice(0, 1).map((reason: string, idx: number) => (
                                <span key={idx} className="inline-flex px-1 py-0.5 text-xs bg-gray-700 text-gray-200 rounded">
                                  {reason}
                                </span>
                              ))}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <div className="mt-4 text-center">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-800 text-green-200">
                  ✅ Real SIRENE Data • Sorted by Win Probability
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UploadHistory;
