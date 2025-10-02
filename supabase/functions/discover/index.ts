import { proxyJson } from "../_shared.ts";

Deno.serve(async (req) => {
  // Always proxy for local development - bypass auth entirely
  return proxyJson("/discover", req);
});
