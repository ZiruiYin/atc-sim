// Creates the single Supabase client (window.sb) from window.TRACON_CONFIG.
// If the URL/key are blank or the supabase-js CDN didn't load, window.sb stays
// null and the rest of the app degrades to guest-only play — no crashes.
(function () {
  const cfg = window.TRACON_CONFIG || {};
  const hasCreds = !!(cfg.SUPABASE_URL && cfg.SUPABASE_ANON_KEY);
  const hasLib = !!(window.supabase && window.supabase.createClient);

  if (hasCreds && hasLib) {
    window.sb = window.supabase.createClient(cfg.SUPABASE_URL, cfg.SUPABASE_ANON_KEY, {
      auth: { persistSession: true, autoRefreshToken: true },
    });
  } else {
    window.sb = null;
    if (hasCreds && !hasLib) {
      console.warn('[tracon] supabase-js failed to load — accounts/leaderboard disabled.');
    }
  }

  // True when the leaderboard backend is usable; the UI checks this to decide
  // whether to show login vs guest-only.
  window.TRACON_LEADERBOARD_ENABLED = !!window.sb;
})();
