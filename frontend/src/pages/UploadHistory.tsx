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
    
    try {
      const res = await postEdge("/train", { tenant_id: "dev-tenant", rows });
      console.log("Training response:", res);
      
      if (res.ok && res.stats) {
        setModelTrained(true);
        setLogs(`✅ Model trained successfully!\n📊 Stats: ${res.stats.rows} rows, ${res.stats.wins} wins, ${res.stats.losses} losses\n\n🎯 Ready to discover companies! Click "Discover Companies" below.`);
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
                        {/* Basic Info */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Company</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">SIREN</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">SIRET</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">APE</th>
                        
                        {/* Names */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Official Name</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Usual Name</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Acronym</th>
                        
                        {/* Location */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Postal Code</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">City</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Street</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Street Number</th>
                        
                        {/* Business Info */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Legal Form</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Activity</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Category</th>
                        
                        {/* Size & Employees */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Employee Range</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Employee Year</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Size Label</th>
                        
                        {/* Status */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Admin Status</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Employer</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Head Office</th>
                        
                        {/* Dates */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Creation Date</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Last Update</th>
                        
                        {/* Special Status */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Social Economy</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Mission Company</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Association ID</th>
                        
                        {/* Coordinates */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">X Coord</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Y Coord</th>
                        
                        {/* Scoring */}
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Score</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Band</th>
                        <th className="px-2 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Why</th>
                      </tr>
                    </thead>
                    <tbody className="bg-gray-900 divide-y divide-gray-700">
                      {discoveredLeads.map((lead, i) => (
                        <tr key={i} className="hover:bg-gray-800">
                          {/* Basic Info */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs font-medium text-white">
                            <div className="max-w-24 truncate" title={lead.name || lead.denominationUniteLegale || 'N/A'}>
                              {lead.name || lead.denominationUniteLegale || 'N/A'}
                            </div>
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300 font-mono">{lead.siren || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300 font-mono">{lead.siret || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300 font-mono">{lead.ape || 'N/A'}</td>
                          
                          {/* Names */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <div className="max-w-24 truncate" title={lead.denominationUniteLegale || 'N/A'}>
                              {lead.denominationUniteLegale || 'N/A'}
                            </div>
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <div className="max-w-24 truncate" title={lead.denominationUsuelle1UniteLegale || 'N/A'}>
                              {lead.denominationUsuelle1UniteLegale || 'N/A'}
                            </div>
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <div className="max-w-16 truncate" title={lead.sigleUniteLegale || 'N/A'}>
                              {lead.sigleUniteLegale || 'N/A'}
                            </div>
                          </td>
                          
                          {/* Location */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300 font-mono">{lead.postal_code || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <div className="max-w-20 truncate" title={lead.city || 'N/A'}>
                              {lead.city || 'N/A'}
                            </div>
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <div className="max-w-20 truncate" title={lead.libelleVoieEtablissement || 'N/A'}>
                              {lead.libelleVoieEtablissement || 'N/A'}
                            </div>
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.numeroVoieEtablissement || 'N/A'}</td>
                          
                          {/* Business Info */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <div className="max-w-20 truncate" title={lead.categorieJuridiqueUniteLegale || 'N/A'}>
                              {lead.categorieJuridiqueUniteLegale || 'N/A'}
                            </div>
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300 font-mono">{lead.activitePrincipaleUniteLegale || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <div className="max-w-20 truncate" title={lead.categorieEntreprise || 'N/A'}>
                              {lead.categorieEntreprise || 'N/A'}
                            </div>
                          </td>
                          
                          {/* Size & Employees */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.trancheEffectifsUniteLegale || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.anneeEffectifsUniteLegale || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">
                            <div className="max-w-20 truncate" title={lead.companySizeLabel || 'N/A'}>
                              {lead.companySizeLabel || 'N/A'}
                            </div>
                          </td>
                          
                          {/* Status */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.etatAdministratifUniteLegale || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.caractereEmployeurUniteLegale || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.etablissementSiege ? 'Yes' : 'No'}</td>
                          
                          {/* Dates */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.dateCreationUniteLegale || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.dateDernierTraitementUniteLegale || 'N/A'}</td>
                          
                          {/* Special Status */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.economieSocialeSolidaireUniteLegale || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.societeMissionUniteLegale || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300 font-mono">{lead.identifiantAssociationUniteLegale || 'N/A'}</td>
                          
                          {/* Coordinates */}
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.coordonneeLambertAbscisseEtablissement || 'N/A'}</td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs text-gray-300">{lead.coordonneeLambertOrdonneeEtablissement || 'N/A'}</td>
                          
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
