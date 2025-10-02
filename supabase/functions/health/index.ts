Deno.serve(async (req) => {
  // Always return success for local development - bypass auth entirely
  return new Response(JSON.stringify({ ok: true, edge: "up" }), { 
    headers: { "content-type": "application/json" }
  });
});
