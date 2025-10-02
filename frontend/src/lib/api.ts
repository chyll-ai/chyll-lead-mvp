// Get API base URL from environment or default to Railway
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://chyll-lead-mvp-production.up.railway.app';

export async function postEdge(path: string, body: any) {
  const url = `${API_BASE_URL}${path}`;
  const r = await fetch(url, { 
    method: "POST", 
    headers: { "content-type": "application/json" }, 
    body: JSON.stringify(body) 
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
