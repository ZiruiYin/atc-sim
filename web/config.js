// ---------------------------------------------------------------------------
// Client config for the leaderboard / accounts. EDIT THIS FILE after creating
// your Supabase project (see DEPLOY.md).
//
// The anon key is PUBLIC by design — it ships in the browser and is safe to
// commit. Row-level security (see supabase/schema.sql) is what protects the
// data, not the secrecy of this key. Leave the two strings empty to run the
// simulator with NO accounts/leaderboard (guests only) — handy before Supabase
// is set up.
// ---------------------------------------------------------------------------
window.TRACON_CONFIG = {
  // From Supabase → Project Settings → API.
  SUPABASE_URL: 'https://gxcpozynkqrxecmjpqbl.supabase.co',
  SUPABASE_ANON_KEY: 'sb_publishable_iCjZ7zcJ9TI4zvr_hsZ3jQ_t9PRA0o4',   // publishable (browser-safe) key

  // Usernames are mapped to synthetic emails for Supabase Auth (we never store
  // passwords ourselves — GoTrue hashes them). The domain only needs to be a
  // valid email format; it is never emailed (confirmation is disabled).
  USERNAME_EMAIL_DOMAIN: 'users.traconsim.app',

  // Internal airport id  ->  leaderboard label. The sim uses 'test'/'egll';
  // the board groups by 'SIMULATOR'/'EGLL'.
  AIRPORTS: [
    { value: 'test', label: 'SIMULATOR' },
    { value: 'egll', label: 'EGLL' },
  ],
};
