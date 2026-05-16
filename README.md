# ATC Simulator — User Manual

A browser-based approach controller simulator. You vector inbound traffic onto the ILS and land them. Default airport is `test` (a compact training airport); EGLL (London Heathrow) is kept in the repo as legacy reference but is not deployed.

For internals and architecture, see `doc/architecture.md`, `doc/behavior.md`, and `doc/logger.md`.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Then open <http://127.0.0.1:5000>.

### CLI flags

| Flag | Meaning |
|---|---|
| `--airport <icao>` | Airport to load (default `test`; `egll` is legacy/local-only) |
| `--free_mode` | Disable STAR procedures and spawn from radar edges (free vectoring). Default is STAR mode. |
| `--host`, `--port` | Network bind options |

Single-aircraft mode and CSV recording were CLI flags previously — they are now in-game buttons (see "Bottom — command bar" below).

## Deploying to GitHub Pages

The repo root is also the Pages deployment root — there is no separate build folder. The browser loads `index.html` and Pyodide fetches the `environment/` Python sources directly from the same paths Flask serves locally. One tree, two front-ends.

Step by step:

1. **Regenerate the manifest** if you added/removed files in `environment/`:
   ```bash
   python build_pages.py
   ```
   This rewrites `env_manifest.json` (the list of files Pyodide fetches at boot). Skip this step if you only changed code *inside* existing files — only the file list matters.

2. **Commit and push:**
   ```bash
   git add -A
   git commit -m "<message>"
   git push origin main
   ```

3. **Configure GitHub Pages (one-time).** In the repo on github.com:
   - Settings → Pages
   - Source: **Deploy from a branch**
   - Branch: **`main`**, Folder: **`/ (root)`**
   - Save

4. **Wait ~1 minute** for the first deploy. The Pages status indicator in the Settings page turns green and shows the URL.

5. **Visit your site** at `https://<your-github-username>.github.io/<repo-name>/`.

Future updates: edit, optionally re-run step 1, commit, push. Pages rebuilds automatically on push to `main`.

First load downloads Pyodide (~10 MB, cached by the browser thereafter) and takes a few seconds to initialize. EGLL data is in the repo as legacy reference but excluded from the manifest — only the `test` airport ships to the browser.

---

## The display

The browser window has three columns.

### Center — radar scope

The radar is an 800×800 backing canvas scaled to your viewport. North is up, distances are in nautical miles, and the visible range is 60 nm across (a 30 nm radius).

What you see on the scope:

- **Airport label**, runways, range rings every 5 nm
- **VORs, NDBs, named waypoints** (toggleable)
- **Aircraft**: a white dot, a fading trail (last 10 positions), and a data tag
- **Red** dot/tag means the aircraft is in a separation conflict
- **Crash overlay** ends the session if two aircraft collide

### Aircraft tag (data block)

Toggle detailed tags with the **D** key. The tag format varies with state:

```
CALLSIGN  IAS  ALT  HDG-or-WPT-or-APPROACH-INFO
```

- **IAS**: knots. If a different speed has been commanded, you see `current->target` (e.g., `220->180`).
- **ALT**: hundreds of feet (so `30` = 3000 ft). Same `current->target` rule when climbing/descending.
- **Last field** depends on what the aircraft is doing:
  - `270` — flying heading 270
  - `BIG` — direct to waypoint BIG
  - `H` — in a holding pattern
  - `LOC 27` / `GS 27` — captured localizer / glideslope for runway 27
  - `<hdg> 27 loc gs` — cleared for runway 27 but not yet captured (lowercase tokens = not yet captured)

Hold **Ctrl** to swap the tag to `CALLSIGN STAR_NAME` for any aircraft following a STAR (and to hide LOC/GS substitutions, so you see the underlying numbers).

When detailed tags are off (D), only the callsign is shown.

### Right column — control panel

- **Score row**: `Landed: N  Violation: Ns  Exits: N`
  - *Landed*: successful arrivals
  - *Violation*: cumulative seconds any aircraft has spent in a separation conflict
  - *Exits*: aircraft that left the radar boundary without landing
- **Display toggles**: A, R, V, N, W, U, D — described below
- **STAR toggles** (`1`–`6`): overlay one or more published STAR procedures on the scope (NORTH1/2/3, SOUTH1/2/3 on EGLL)
- **Spawn directions** (N/E/S/W): which radar edges new aircraft come from (disabled in `--star` mode)
- **Spawn rate**: seconds between spawns; use `−` / `+` to make it faster/slower
- **Flight strips**: one strip per active aircraft — click a strip to load that callsign into the command box

### Left column — script log

Live transcript of ATC and pilot radio calls, plus rejection messages:

- **ATC** lines (you / the controller)
- **Pilot** readbacks
- **INVALID** — command was malformed
- **UNABLE** — command was well-formed but disallowed in the aircraft's current state

### Bottom — command bar

- **Command box**: type `CALLSIGN COMMAND` and press `Enter`. Press `Esc` to release focus. Clicking an aircraft on the radar, or a flight strip, pre-fills the callsign.
- **Speed**: toggles 1× ⇄ 10× (also `Tab`). In 10×, the radar updates the same once-per-second cadence but the simulation advances 10 simulated seconds per update.
- **Pause**: pauses the simulation (also `P`).
- **Single**: toggles single-aircraft mode. Clicking **restarts the simulation** (wipes the current aircraft list and score) and spawns one aircraft at a time — the next plane only appears after the previous lands or exits the radar. Click again to restart in normal multi-aircraft mode. Button label shows `Single: on` / `Single: off`.
- **Record**: starts CSV recording from the next simulated second. Click again to stop; the CSV file downloads automatically to your browser's downloads folder, named `YYYYMMDD_HHMMSS_{single|multiple}.csv`. The schema is the same as `doc/logger.md`.

### Keyboard shortcuts

Active only when the command box is **not** focused (press `Esc` to leave the box).

| Key | Effect |
|---|---|
| `Tab` | Toggle 1× / 10× speed |
| `P` | Pause / resume |
| `A` | Toggle airport label |
| `R` | Toggle range rings |
| `V` | Toggle VOR stations |
| `N` | Toggle NDB stations |
| `W` | Toggle waypoints |
| `U` | Toggle runway names |
| `D` | Toggle detailed aircraft tags |
| `1`–`6` | Toggle published STAR overlays |
| `↑` `↓` `←` `→` | Toggle North / South / West / East spawn edges |
| `+` / `−` | Slower / faster spawn rate (10 s steps) |
| `Esc` | Release focus from the command box |
| Hold `Ctrl` | Show STAR name and hide LOC/GS substitution in aircraft tags |

---

## Issuing commands

Format: `CALLSIGN COMMAND` (case-insensitive). Multiple commands can be chained in one line, separated by spaces.

```
BA42 C 270
BA42 C 5
BA42 S 200
BA42 C 270 C 5 S 200 L 27R
```

Click an aircraft (or its strip) to autofill the callsign, then type the rest. Press `Enter` to send.

### Command summary

| Cmd | Param | Modifiers | Example | Effect |
|---|---|---|---|---|
| `A` | — | — | `BA42 A` | Abort approach (go around) |
| `C` | heading (3 digits, 0–360) | `;L` or `;R` for turn direction | `BA42 C 270;L` | Turn to heading |
| `C` | altitude (1–2 digits, 1–18 = ×1000 ft) | `;X` to expedite | `BA42 C 5;X` | Climb/descend to altitude |
| `C` | waypoint name | `;L` or `;R` | `BA42 C BIG` | Proceed direct to waypoint |
| `S` | speed (140–280 kt) | `;X` to expedite | `BA42 S 200;X` | Set indicated airspeed |
| `H` | waypoint name | `;L` or `;R` (default `R`) | `BA42 H CPT` | Hold at waypoint (racetrack) |
| `L` | runway name | — | `BA42 L 27R` | Clear for ILS approach |

### Modifiers (suffixes after `;`)

- `;L` / `;R` — force left/right turn (otherwise the shorter side is chosen). Applies to `C [heading]`, `C [waypoint]`, `H [waypoint]`.
- `;X` — expedite. Applies to `C [altitude]` and `S [speed]`; uses higher climb/descent or acceleration rates.

### Chain ordering rules

- `A` (abort), if used, must be **first** in the chain.
- `L` (land), if used, must be **last** in the chain.
- `H` (hold) cannot be combined with `C` or `L` in the same chain.

### Heading vs altitude vs waypoint disambiguation for `C`

`C` is overloaded by the shape of its argument:

- Three digits, 0–360 → **heading** (e.g., `C 090`)
- One or two digits, 1–18 → **altitude** in thousands of feet (e.g., `C 5` = 5000 ft)
- Anything else → looked up as a **waypoint name**

So `C 5` climbs/descends; `C 050` is a heading change; `C BIG` is direct to BIG.

### Abort behavior (`A`)

Aborting clears any ILS clearance and recovers the aircraft:

- Heading → current heading (no turn)
- Altitude → `max(current, 3000 ft)`
- Speed → `max(current, 180 kt)`

You can chain commands after `A` to redirect immediately, e.g. `BA42 A C 360 C 5`.

### Landing (`L`) and the ILS

`L 27R` clears an aircraft for the ILS approach to runway 27R. The aircraft will capture the localizer and glideslope **only when** all four conditions are met simultaneously:

- Airspeed below **240 kt**
- Altitude below **5000 ft**
- Distance to threshold within **20 nm**
- Heading within **±30°** of the runway centerline (tightened to **±20°** within 5 nm)

If you clear an aircraft early, the clearance stays active and capture happens once the geometry comes into the window. Once captured, the aircraft locks onto LOC and GS; you can't vector or change altitude until you abort (`A`). Within 5 nm of the threshold it automatically slows to 140 kt.

### Holding (`H`)

`H BIG` puts the aircraft into a right-hand racetrack hold over BIG. Use `;L` for a left-hand hold. The aircraft remains in the hold until you give it a new heading, waypoint, or land clearance.

---

## Separation rules

You must keep aircraft apart. Two thresholds are enforced:

### Separation warning (red tag, counts against score)

- **Less than 1000 ft vertical AND less than 3 nm lateral**, *or*
- Two aircraft on the **same runway** (runway incursion)

Suppressed when either aircraft is established on the ILS, or when one is already on the ground after landing.

### Crash (game over)

- **50 ft vertical or less AND 0.2 nm lateral or less** at the same time.

The simulation stops; reload the page to start over.

---

## Outcomes per aircraft

- **Landed** — touched down and decelerated to a stop. Counted in *Landed*.
- **Improper exit** — left the radar area without being cleared for landing. Counted in *Exits*.

A successful run is one with many landings, zero crashes, low *Violation* seconds, and few improper exits.

---

## Tips

- Clicking on a busy area selects whichever aircraft is closest to your click within ~50 px.
- The 1×/10× speed toggle (`Tab`) is useful for long vectors; switch back to 1× when sequencing or vectoring to final.
- The flight strip panel shows the same data as the radar tag and is easier to read in heavy traffic.
- Toggle off layers you don't need (`V` to drop VORs, etc.) to declutter.
- A command that's grammatically correct but currently disallowed (e.g., changing altitude after GS capture) is reported as **UNABLE** in the script log, not **INVALID**.
