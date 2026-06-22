# Deployment

Two independent pieces, hosted separately and for free:

| Concern | Needs | Host | Works on |
|---|---|---|---|
| **Accounts + leaderboard** | persistence (a database) | **Supabase** (free) | both builds below |
| **AUTO mode** | Python + torch compute | **Hugging Face Space** (free) | the HF build only |
| **The game itself (human play)** | nothing (runs in-browser via Pyodide) | **GitHub Pages** (free, current) | — |

The accounts/leaderboard talk to Supabase **directly from the browser**, so they
work on *both* the GitHub Pages (Pyodide) build and the Hugging Face (Flask)
build. AUTO mode (torch) cannot run in Pyodide, so it only exists on the HF
build.

---

## 1. Supabase — accounts + leaderboard (~5 min)

1. Create a free project at <https://supabase.com> (New project; pick any region;
   wait for it to provision).
2. **Create the table + security policies.** Open **SQL Editor → New query**,
   paste the contents of [`supabase/schema.sql`](supabase/schema.sql), and click
   **Run**. This creates the `runs` table with row-level security (public read,
   owner-only write).
3. **Turn OFF email confirmation** (we use usernames, not real emails):
   **Authentication → Sign In / Providers → Email** → disable *"Confirm email"*
   → Save. (If you skip this, sign-up will create accounts that can't log in.)
4. **Copy your keys:** **Project Settings → API** →
   - *Project URL* (e.g. `https://abcdefgh.supabase.co`)
   - *anon public* key (safe to ship in the browser — RLS is what protects data)
5. **Paste them into [`web/config.js`](web/config.js):**
   ```js
   SUPABASE_URL: 'https://abcdefgh.supabase.co',
   SUPABASE_ANON_KEY: 'eyJhbGc...the anon public key...',
   ```
   Commit and push. The leaderboard now works on the live GitHub Pages site.

That's it — no backend needed for accounts/leaderboard. Until step 5 is done the
site runs in **guest-only** mode (no login box, no board), so nothing breaks in
the meantime.

> **Note on usernames:** each username is stored in Supabase Auth as a synthetic
> email `username@<USERNAME_EMAIL_DOMAIN>` (see `web/config.js`). The domain is
> never emailed; it just has to be a valid email format. Passwords are hashed by
> Supabase — we never see or store them.

---

## 2. Hugging Face — AUTO mode (the full Flask build)

This stands up the Flask backend (with the torch AUTO planner) as a Docker
Space. The frontend auto-detects the backend (`GET /state` returns JSON →
`HttpEngine`), so the **AUTO** button appears and works with no extra wiring.

1. Create a free account at <https://huggingface.co>.
2. **New → Space.** Name it (e.g. `tracon-sim`), **SDK = Docker**, **blank**
   template, visibility public. Hardware: **CPU basic (free)**.
3. Push this repo to the Space's git remote. From this repo:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/tracon-sim
   git push hf main
   ```
   (You'll authenticate with an HF **access token** — Settings → Access Tokens →
   make one with *write* scope — as the password.)
4. The Space builds from the [`Dockerfile`](Dockerfile) (installs CPU-only torch,
   runs `python main.py --host 0.0.0.0 --port 7860`). First build takes a few
   minutes. When it's "Running", open the Space URL — you get the full sim with
   the **AUTO** button and the same leaderboard.

**Cold starts:** a free Space sleeps after ~48 h with no visitors; the next visit
wakes it (≈1–3 min one-time, with torch in the image). To avoid this, point a
free uptime pinger (e.g. UptimeRobot) at the Space URL every few hours so it
never goes idle.

> The leaderboard config (`web/config.js`) is shared, so the HF build reads the
> same Supabase leaderboard as the GitHub Pages build automatically.

---

## What runs where (summary)

- **GitHub Pages** (`https://ziruiyin.github.io/atc-sim/`): human play + accounts
  + leaderboard. No AUTO.
- **Hugging Face Space**: everything above **plus** AUTO mode.
- **Local** (`python main.py`): same as the HF build.

## Local dev

```bash
pip install -r requirements.txt   # includes torch, for AUTO
python main.py                    # http://127.0.0.1:5000
```
The leaderboard uses whatever is in `web/config.js`. To test against a throwaway
Supabase project, point it there; to test guest-only, leave the keys blank.
