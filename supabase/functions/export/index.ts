import { proxyJson } from "../_shared.ts";

Deno.serve(async (req) => {
  // For local development, bypass auth check
  const url = new URL(req.url);
  if (url.hostname === 'localhost' || url.hostname === '127.0.0.1') {
    return proxyJson("/export", req);
  }
  
  // For production, check auth
  const authHeader = req.headers.get('Authorization');
  if (!authHeader) {
    return new Response(JSON.stringify({ msg: "Error: Missing authorization header" }), { 
      status: 401,
      headers: { "Content-Type": "application/json" }
    });
  }
  
  return proxyJson("/export", req);
});
