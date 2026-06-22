---
title: TRACON Simulator
emoji: 🛬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ATC Simulator — User Manual

A browser-based approach controller simulator. You vector inbound traffic onto the ILS and land them. Default airport is called SIMULATOR (a simulator airport named TEST). EGLL (London Heathrow) is also available but with no STAR implemented yet.

## Run

Play in the browser: <https://ziruiyin.github.io/atc-sim/>
The Hugging Face deployment also has **AUTO** mode — a model that flies the aircraft for you (see the [AUTO](#auto) section): <https://huggingface.co/spaces/JerryYin14/ATC-SIM>

Or run locally:

```bash
pip install -r requirements.txt
python main.py
```

This is hosted locally on port 5000.

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

- **Score row**: `Landed: N  Violation: Ns  Exits: N` with `Time: M:SS` on the next line
  - *Landed*: successful arrivals
  - *Violation*: cumulative seconds any aircraft has spent in a separation conflict
  - *Exits*: aircraft that left the radar boundary without landing
  - *Time*: simulator seconds elapsed this run (sim-clock, not wall-clock — a 10× session and a 1× session are scored on the same footing). Recorded as your run's play time on the leaderboard.
- **Display toggles**: A, R, V, N, W, U, D — described below
- **STAR toggles** (`1`–`6`): overlay published STAR procedures on the scope (NORTH1/2/3, SOUTH1/2/3). These belong to the **SIMULATED** airport, which spawns its traffic on STARs; EGLL has no STARs yet.
- **Spawn directions** (N/E/S/W): which radar edges new aircraft come from — used on **EGLL**. Disabled on the SIMULATED airport, which spawns aircraft onto STARs instead.
- **Spawn rate**: seconds between spawns; use `−` / `+` to make it faster/slower
- **Flight strips**: one strip per active aircraft — click a strip to load that callsign into the command box

### Left column — airport selector + script log

The top of the left column has the **airport selector** (SIMULATED / EGLL) and an **EXIT** button, which returns you to the title screen (offering to save your run first if you're logged in). Below it is a live transcript of ATC and pilot radio calls, plus rejection messages:

- **ATC** lines (you / the controller)
- **Pilot** readbacks
- **INVALID** — command was malformed
- **UNABLE** — command was well-formed but disallowed in the aircraft's current state

### Bottom — command bar

- **Command box**: type `CALLSIGN COMMAND` and press `Enter`. Press `Esc` to release focus. Clicking an aircraft on the radar, or a flight strip, pre-fills the callsign.
- **Speed**: toggles 1× ⇄ 10× (also `Tab`). In 10×, the radar updates the same once-per-second cadence but the simulation advances 10 simulated seconds per update.
- **Pause**: pauses the simulation (also `P`).
- **Restart**: clears all aircraft and the score and starts a fresh run on the current airport (offers to save your run first if you're logged in).

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

You must keep aircraft apart. A red dot/tag marks a **separation warning**, and every second any aircraft spends in conflict adds to your *Violation* count. The two airports use **different** rules.

### SIMULATED airport

A warning is raised for any two aircraft in the **same medium** (both airborne, or both on the ground) that are within:

- **less than 2 nm lateral** *and* **less than 1000 ft vertical**.

### EGLL

A warning is raised when two airborne aircraft are within:

- **less than 3 nm lateral** *and* **less than 1000 ft vertical**,

**unless** either aircraft is established on the ILS, or one is already on the ground after landing. Separately, two aircraft on the **same runway** raise a runway-incursion warning.

### Crash

If two aircraft are within **0.2 nm lateral and 50 ft vertical or less** at the same instant, they collide and the simulation stops. The crash overlay offers **Restart** and **Exit** buttons.

---

## AUTO

On the **SIMULATED** airport you can hand all traffic to an autonomous controller with the **AUTO** button on the radar. AUTO needs PyTorch, so it runs only on the backend deployments (the [Hugging Face Space](https://huggingface.co/spaces/JerryYin14/ATC-SIM) or a local `python main.py`) — not the in-browser GitHub Pages build. Engaging AUTO means the run is no longer flown by you, so it forfeits leaderboard saving for the rest of the session.

How it works:

- **Policy** — a lightweight Gaussian Mixture Model with 6,752 parameters. It was first trained on past human play data, then refined with [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)).
- **Planning** — a multi-aircraft planner rolls the policy forward, planning 400 steps ahead and resolving conflicts between aircraft before issuing commands each tick.
- **Performance** — over a test of **512 rollouts**, AUTO lands aircraft with a success rate of **nearly 98%**, with low loss of separation and **no crashes**.

The full training pipeline and algorithms are described [here](https://cs224r.stanford.edu/projects/pdfs/Jerry%20Yin%20submission_416287340/224R_Final_Project.pdf).

---

## Outcomes per aircraft

- **Landed** — touched down and decelerated to a stop. Counted in *Landed*.
- **Improper exit** — left the radar area without being cleared for landing. Counted in *Exits*.

A successful run is one with many landings, zero crashes, low *Violation* seconds, and few improper exits.

---

## Tips

- The flight strip panel shows the same data as the radar tag and is easier to read in heavy traffic.
- If you encounter issues with the command, be sure to check the log on the left of the display, which would show you the reason why the command was not accepted.