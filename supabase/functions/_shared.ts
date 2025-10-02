export async function proxyJson(route: string, req: Request) {
  const FASTAPI_URL = Deno.env.get("FASTAPI_URL")!;
  if (!FASTAPI_URL) {
    return new Response(JSON.stringify({ ok:false, error:"FASTAPI_URL not configured" }), { status: 400 });
  }
  
  // For local development, allow requests without authorization
  const body = await req.text();
  const r = await fetch(`${FASTAPI_URL}${route}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body,
  });
  
  return new Response(await r.text(), {
    status: r.status,
    headers: { "content-type": r.headers.get("content-type") ?? "application/json" }
  });
}
