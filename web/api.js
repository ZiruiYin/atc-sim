// ---------------------------------------------------------------------------
// TraconAPI — the only place that talks to Supabase. Pure data layer, no DOM.
//
//   Accounts use Supabase Auth (GoTrue). Usernames are mapped to a synthetic
//   email so we never store or check passwords ourselves — GoTrue hashes them
//   server-side. The leaderboard is one row per (user, airport): an upsert
//   overwrites the player's latest saved run for that airport.
// ---------------------------------------------------------------------------
window.TraconAPI = (function () {
  const cfg = window.TRACON_CONFIG || {};
  const sb = window.sb;                       // may be null (guest-only mode)
  const DOMAIN = cfg.USERNAME_EMAIL_DOMAIN || 'users.traconsim.app';

  let _user = null;   // {id, username} when logged in; null for guest/anon

  // ---- helpers ------------------------------------------------------------
  function usernameToEmail(username) {
    return `${username.trim().toLowerCase()}@${DOMAIN}`;
  }
  function toUser(u) {
    if (!u) return null;
    const md = u.user_metadata || {};
    const username = md.display_name || (u.email || '').split('@')[0];
    return { id: u.id, username };
  }
  function airportLabel(value) {
    const a = (cfg.AIRPORTS || []).find(x => x.value === value);
    return a ? a.label : String(value || '').toUpperCase();
  }
  function airports() { return (cfg.AIRPORTS || []).slice(); }

  // Client-side credential rules (Supabase enforces its own minimums too).
  function validateCredentials(username, password) {
    if (!/^[a-zA-Z0-9_]{3,20}$/.test(username || '')) {
      return 'Username must be 3–20 characters: letters, numbers, or underscore.';
    }
    if ((password || '').length < 8) {
      return 'Password must be at least 8 characters.';
    }
    if ((password || '').length > 72) {
      return 'Password must be at most 72 characters.';
    }
    return null;
  }

  // ---- session ------------------------------------------------------------
  async function init() {
    if (!sb) return null;
    try {
      const { data } = await sb.auth.getSession();
      _user = data && data.session ? toUser(data.session.user) : null;
    } catch (e) { _user = null; }
    sb.auth.onAuthStateChange((_event, session) => {
      _user = session ? toUser(session.user) : null;
    });
    return _user;
  }

  function currentUser() { return _user; }
  function enabled() { return !!sb; }

  // Combined login / signup, matching the landing-page UX:
  //   - username exists + right password  -> log in
  //   - username exists + wrong password  -> {ok:false, message:'Invalid password.'}
  //   - username is new                   -> sign up + log in
  async function signInOrUp(username, password) {
    if (!sb) return { ok: false, message: 'Leaderboard is not configured.' };
    const bad = validateCredentials(username, password);
    if (bad) return { ok: false, message: bad };

    const email = usernameToEmail(username);

    // 1) Try to log in.
    const signIn = await sb.auth.signInWithPassword({ email, password });
    if (!signIn.error && signIn.data && signIn.data.user) {
      _user = toUser(signIn.data.user);
      return { ok: true, isNew: false, user: _user };
    }

    // 2) Sign-in failed — either the username is new or the password is wrong.
    //    Attempt a signup: if the account already exists GoTrue rejects it,
    //    which means the earlier failure was a bad password.
    const signUp = await sb.auth.signUp({
      email, password,
      options: { data: { display_name: username.trim() } },
    });

    if (signUp.error) {
      const msg = (signUp.error.message || '').toLowerCase();
      if (msg.includes('already') || msg.includes('registered') ||
          msg.includes('exists') || signUp.error.status === 422) {
        return { ok: false, message: 'Invalid password.' };
      }
      return { ok: false, message: signUp.error.message || 'Could not sign up.' };
    }

    // Signup OK. With email confirmation disabled, a session is returned and
    // the user is logged in immediately.
    if (signUp.data && signUp.data.session) {
      _user = toUser(signUp.data.user);
      return { ok: true, isNew: true, user: _user };
    }
    // No session => email confirmation is still ON in Supabase (misconfig).
    return {
      ok: false,
      message: 'Account created, but email confirmation is enabled. Disable it in Supabase → Authentication → Sign In / Providers.',
    };
  }

  async function signOut() {
    if (sb) { try { await sb.auth.signOut(); } catch (e) {} }
    _user = null;
  }

  // ---- leaderboard --------------------------------------------------------
  // Upsert this player's run for `airportValue` ('test'/'egll'); overwrites any
  // existing row for the same (user, airport). No-op for guests.
  async function saveRun(airportValue, stats) {
    if (!sb) return { ok: false, message: 'Leaderboard is not configured.' };
    if (!_user) return { ok: false, message: 'Guests cannot save runs.' };
    const row = {
      user_id: _user.id,
      username: _user.username,
      airport: airportLabel(airportValue),
      landed: Math.max(0, Math.round(stats.landed || 0)),
      violation_secs: Math.max(0, Math.round((stats.violation_secs || 0) * 10) / 10),
      exits: Math.max(0, Math.round(stats.exits || 0)),
      play_secs: Math.max(0, Math.round(stats.play_secs || 0)),
      updated_at: new Date().toISOString(),
    };
    const { error } = await sb.from('runs').upsert(row, { onConflict: 'user_id,airport' });
    if (error) return { ok: false, message: error.message };
    return { ok: true };
  }

  // Top-N for an airport, ranked: most landed, then fewest violation secs,
  // then fewest improper exits, then fastest (least sim time).
  async function fetchLeaderboard(airportValue, limit = 10) {
    if (!sb) return { ok: false, message: 'Leaderboard is not configured.', rows: [] };
    const { data, error } = await sb.from('runs')
      .select('username, landed, violation_secs, exits, play_secs')
      .eq('airport', airportLabel(airportValue))
      .order('landed', { ascending: false })
      .order('violation_secs', { ascending: true })
      .order('exits', { ascending: true })
      .order('play_secs', { ascending: true })
      .limit(limit);
    if (error) return { ok: false, message: error.message, rows: [] };
    return { ok: true, rows: data || [] };
  }

  return {
    init, enabled, currentUser, signInOrUp, signOut,
    saveRun, fetchLeaderboard,
    airports, airportLabel, validateCredentials,
  };
})();
