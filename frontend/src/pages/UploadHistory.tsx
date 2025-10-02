import React, { useState } from 'react';
import { postEdge } from "../lib/api";

const UploadHistory: React.FC = () => {
  const [rows, setRows] = useState<any[]>([]);
  const [logs, setLogs] = useState<string>("");

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
    try {
      const res = await postEdge("train", { tenant_id: "dev-tenant", rows });
      setLogs(`OK: rows=${res.stats?.rows} wins=${res.stats?.wins} losses=${res.stats?.losses}`);
    } catch (e:any) {
      setLogs(`Error: ${e.message}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md mx-auto">
        <div className="text-center">
          <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
            Upload Your History
          </h2>
          <p className="mt-2 text-sm text-gray-600">
            Upload your won/lost deals to train your predictive model
          </p>
        </div>
        
        <div className="mt-8 space-y-6">
          <div>
            <label htmlFor="file-upload" className="block text-sm font-medium text-gray-700">
              CSV File
            </label>
            <input
              id="file-upload"
              type="file"
              accept=".csv"
              onChange={(e) => e.target.files && onFile(e.target.files[0])}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          {rows.length > 0 && (
            <div className="bg-white shadow rounded-lg p-4">
              <h3 className="text-lg font-medium text-gray-900 mb-3">
                Preview ({rows.length} rows)
              </h3>
              <div className="space-y-2">
                {rows.slice(0, 3).map((row, index) => (
                  <div key={index} className="text-sm text-gray-600">
                    {row.company_name} - {row.deal_status} - {row.website || 'No website'}
                  </div>
                ))}
                {rows.length > 3 && (
                  <div className="text-sm text-gray-500">
                    ... and {rows.length - 3} more rows
                  </div>
                )}
              </div>
            </div>
          )}

          <button
            onClick={train}
            disabled={!rows.length}
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Train My Model
          </button>

          {logs && (
            <div className="bg-gray-100 p-3 rounded-md">
              <pre className="text-sm whitespace-pre-wrap">{logs}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UploadHistory;
