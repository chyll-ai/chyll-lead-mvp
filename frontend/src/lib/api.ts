export async function postEdge(path: string, body: any) {
  // Direct FastAPI call - working solution for local development
  const url = `http://127.0.0.1:8000${path}`;
  const r = await fetch(url, { 
    method: "POST", 
    headers: { "content-type": "application/json" }, 
    body: JSON.stringify(body) 
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
