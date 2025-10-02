import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { proxyJson } from "../_shared.ts";

Deno.serve(async (req) => {
  try {
    // Create Supabase client with service role key
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? 'http://127.0.0.1:54321',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? 'sb_secret_N7UND0UgjKTVK-Uodkm0Hg_xSvEMPvz'
    )
    
    // Always proxy for local development
    return proxyJson("/train", req);
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), { 
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
});
