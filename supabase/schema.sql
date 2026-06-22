-- ===========================================================================
-- TRACON simulator — leaderboard schema.
-- Paste this into Supabase → SQL Editor → New query → Run. (See DEPLOY.md.)
--
-- One row per (player, airport): the player's most recently SAVED run for that
-- airport. The client upserts on (user_id, airport), so a new save overwrites
-- the old one. Anyone can READ the board; only the authenticated owner can
-- write their own row. Accounts themselves live in Supabase Auth (auth.users)
-- — we never store passwords here.
-- ===========================================================================

create table if not exists public.runs (
  user_id        uuid not null references auth.users(id) on delete cascade,
  airport        text not null check (airport in ('SIMULATOR', 'EGLL')),
  username       text not null,
  landed         integer       not null default 0 check (landed >= 0 and landed <= 100000),
  violation_secs numeric(10,1) not null default 0 check (violation_secs >= 0),
  exits          integer       not null default 0 check (exits >= 0 and exits <= 100000),
  play_secs      integer       not null default 0 check (play_secs >= 0),
  updated_at     timestamptz   not null default now(),
  primary key (user_id, airport)
);

-- Fast top-N per airport (matches the client's ORDER BY).
create index if not exists runs_board_idx
  on public.runs (airport, landed desc, violation_secs asc, exits asc, play_secs asc);

alter table public.runs enable row level security;

-- Public read (guests included) so the landing-page board renders for everyone.
drop policy if exists "runs are publicly readable" on public.runs;
create policy "runs are publicly readable"
  on public.runs for select
  using (true);

-- A logged-in user may insert only a row owned by them.
drop policy if exists "users insert their own run" on public.runs;
create policy "users insert their own run"
  on public.runs for insert
  with check (auth.uid() = user_id);

-- ...and update only their own row.
drop policy if exists "users update their own run" on public.runs;
create policy "users update their own run"
  on public.runs for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
