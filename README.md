# ATC Simulator (EGLL/LHR)

This is an ATC simulator game (currently only for EGLL/LHR), inspired by [atc-sim.com](http://atc-sim.com).

As an approach controller, you guide inbound traffic to intercept the ILS and land safely.

The implementation is still in progress. TODOs (mostly in this order):

- [x] Fix bugs with ILS LOC  
- [x] Scoring system  
- [x] Improper exits detection  
- [x] Fast forwarding  
- [ ] Text-to-speech models deployed for sound  
- [ ] Speech-to-text + LM models deployed for verbal commands  
- [ ] Game crash protections for invalid commands  
- [ ] More realistic constraints (speed, altitude, ILS interception restrictions, SID/STAR procedures, terrain/winds, etc.)  
- [ ] Departure aircrafts (all aircrafts right now are arrival traffic)  
- [ ] Aircraft types  

---

## Requirements
- Python with pygame:  
  ```bash
  pip install pygame
  ```
- An executable package will also be offered

---

## How to run
```bash
python main.py
```
(or double-click the packaged `.exe` if using the binary release)

---

## Display

The main display shows:  
- Radar scope with the airport, all navaids, and 5nm radar rings  
- Aircraft labels with speed, altitude, heading/waypoint/hold, and ILS intercept status (LOC/GS)  
- Aircraft spacing and separation logic:  
  - Maintain **1000 ft vertical** or **3 nm lateral** separation (except on glidepath)  
  - On the ground: runway must remain clear (no incursions)  
  - Violations mark aircraft in **red**  
  - Collisions terminate the game  

At the **top-right**:  
- Current aircraft spawn interval and inbound direction  
- **Scoring stats**: total landed, violation time (in seconds), total improper exits  

At the **top-left**:  
- **Fast forwarding indicator**  
- Press `TAB` (regardless of **LOCKED/UNLOCKED**) to toggle between ×1 and ×10 speed  
- When ×10 is active, it will be shown here  

At the bottom: the command textbox  

---

### Display Toggles

While the textbox is **UNLOCKED**, you can use keyboard shortcuts:

| Key         | Effect                                   |
|-------------|------------------------------------------|
| `A`         | Toggle airport label display             |
| `R`         | Toggle radar rings                       |
| `V`         | Toggle VOR stations                      |
| `N`         | Toggle NDB stations                      |
| `W`         | Toggle waypoints                         |
| `U`         | Toggle runway names                      |
| `D`         | Toggle detailed aircraft tags            |
| `+` / `=`   | Increase aircraft spawn interval         |
| `-`         | Decrease aircraft spawn interval         |
| `↑`         | Toggle spawning from North               |
| `↓`         | Toggle spawning from South               |
| `←`         | Toggle spawning from West                |
| `→`         | Toggle spawning from East                |
| `L`         | Lock the textbox for command entry       |

- When the textbox is **LOCKED**, you can type commands for aircraft  
- To **UNLOCK** the textbox, type `unlock` and press `Enter`  

---

## Aircraft Command Reference

Commands are **not case-sensitive**.  
Enter the aircraft’s **callsign** (by typing or clicking it), then a space, then the command(s).  
Multiple commands can be chained with semicolons (e.g., `C 090;L S 220;X`).  

---

### Course Commands (C)
- `C xxx` — Set heading (xxx = degrees, e.g., `C 090`)  
  - Optional: add `;L` or `;R` for specific turn direction (e.g., `C 090;L`)  
- `C x/xx` — Set altitude (in thousands of feet, e.g., `C 5` = 5000 ft)  
  - Optional: add `;X` for expedited climb/descent (e.g., `C 5;X`)  
- `C [waypoint]` — Direct to specified waypoint (e.g., `C BIG`)  
  - Optional: add `;L` or `;R` for turn direction  

---

### Speed Commands (S)
- `S xxx` — Set airspeed (xxx = knots, e.g., `S 220`)  
  - Optional: add `;X` for expedited change (e.g., `S 220;X`)  

---

### Hold Commands (H)
- `H [waypoint]` — Hold at waypoint (e.g., `H CPT`)  
  - Optional: `;L` or `;R` for turn direction (default: right)  
  - Exit hold by issuing new course/heading/waypoint command  

---

### Landing Commands (L)
- `L [runway]` — Clear for ILS approach (e.g., `L 27R`)  
  - Requirements:  
    - Within 15 nm of airport  
    - Below 5000 ft  
    - Below 240 knots  
  - ILS intercept:  
    - 5–15 nm out: heading within ±30° of runway  
    - ≤5 nm out: heading within ±20° of runway  
  - On final (last 5 nm), aircraft auto-slows to 140 kts  

---

### Abort Command (A)
- `A` — Abort approach (only valid when cleared for ILS)  
  - Aircraft executes missed approach:  
    - Maintains current heading  
    - If speed < 180 kts: accelerates to 180 kts  
    - If altitude < 3000 ft: climbs to 3000 ft  

---

## Notes
- Use semicolons (`;`) to specify options (e.g., turn direction, expedite)  
- Command `A` should come **before** all other commands in a chain  
- Command `L` should come **after** all other commands  
