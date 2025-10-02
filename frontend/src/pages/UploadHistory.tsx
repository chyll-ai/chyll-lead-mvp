import React, { useState } from 'react';
import { postEdge } from "../lib/api";

const UploadHistory: React.FC = () => {
  const [rows, setRows] = useState<any[]>([]);
  const [logs, setLogs] = useState<string>("");
  const [discoveredLeads, setDiscoveredLeads] = useState<any[]>([]);
  const [showLeads, setShowLeads] = useState(false);

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
    setLogs("Training...");
    setShowLeads(false);
    try {
      const res = await postEdge("/train", { tenant_id: "dev-tenant", rows });
      console.log("Training response:", res); // Debug log
      console.log("Response keys:", Object.keys(res)); // Debug log
      console.log("discovered_leads:", res.discovered_leads); // Debug log
      console.log("discovered_leads length:", res.discovered_leads?.length); // Debug log
      
      if (res.ok && res.stats) {
        setLogs(`OK: rows=${res.stats.rows} wins=${res.stats.wins} losses=${res.stats.losses}`);
        
        // Show discovered leads if available
        if (res.discovered_leads && res.discovered_leads.length > 0) {
          console.log("Setting discovered leads:", res.discovered_leads); // Debug log
          setDiscoveredLeads(res.discovered_leads);
          setShowLeads(true);
          setLogs(`${res.message || 'Model trained successfully!'} Showing ${res.discovered_leads.length} discovered leads.`);
        } else {
          console.log("No discovered leads found"); // Debug log
          setLogs(`${res.message || 'Model trained successfully!'} No companies found matching your patterns.`);
        }
      } else {
        setLogs(`Error: ${res.error || 'Unknown error'}`);
      }
    } catch (e:any) {
      setLogs(`Error: ${e.message}`);
    }
  };

  return (
    <div className="bg-black py-12 px-4 sm:px-6 lg:px-8 min-h-screen">
      {/* Dark theme - updated */}
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

          <button
            onClick={train}
            disabled={!rows.length}
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-black bg-white hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-white disabled:bg-gray-600 disabled:cursor-not-allowed"
          >
            Train My Model
          </button>

          {logs && (
            <div className="bg-gray-800 p-3 rounded-md border border-gray-700">
              <pre className="text-sm whitespace-pre-wrap text-white">{logs}</pre>
            </div>
          )}

          {showLeads && discoveredLeads.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-medium text-white mb-4">
                🎯 Discovered Leads (Real SIRENE Data)
              </h3>
              <div className="bg-gray-900 shadow rounded-lg overflow-hidden border border-gray-700">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-700">
                    <thead className="bg-gray-800">
                      <tr>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Company</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">SIREN</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">APE</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Location</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Legal Form</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Size</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Category</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Active</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Score</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Band</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Why</th>
                      </tr>
                    </thead>
                    <tbody className="bg-gray-900 divide-y divide-gray-700">
                      {discoveredLeads.map((lead, i) => (
                        <tr key={i} className="hover:bg-gray-800">
                          <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-white">
                            <div className="max-w-xs truncate" title={lead.name}>
                              {lead.name}
                            </div>
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-300 font-mono">{lead.siren}</td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-300 font-mono">{lead.ape}</td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-300">
                            {lead.location || 'N/A'}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-300">
                            {lead.categorieJuridiqueUniteLegale || 'N/A'}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-300">
                            {lead.headcountLabelUL || lead.trancheEffectifsUniteLegale || 'N/A'}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-300">
                            {lead.companySizeLabel || lead.categorieEntreprise || 'N/A'}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap">
                            <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                              lead.isActive ? 'bg-green-800 text-green-200' : 'bg-red-800 text-red-200'
                            }`}>
                              {lead.isActive ? 'Active' : 'Inactive'}
                            </span>
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-white font-medium">
                            {typeof lead.win_score === 'number' ? lead.win_score.toFixed(3) : lead.win_score}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap">
                            <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                              lead.band === 'High' ? 'bg-green-800 text-green-200' :
                              lead.band === 'Medium' ? 'bg-yellow-800 text-yellow-200' :
                              'bg-red-800 text-red-200'
                            }`}>
                              {lead.band}
                            </span>
                          </td>
                          <td className="px-3 py-2 text-sm text-gray-300 max-w-xs">
                            <div className="flex flex-wrap gap-1">
                              {(lead.reasons || []).slice(0, 2).map((reason: string, idx: number) => (
                                <span key={idx} className="inline-flex px-2 py-1 text-xs bg-gray-700 text-gray-200 rounded">
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
                  ✅ Real SIRENE Data
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
