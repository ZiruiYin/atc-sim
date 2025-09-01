This is an ATC simulator game (currently only for EGLL/LHR), inspired by atc-sim.com.

The implementation is in progress. TODOs (mostly in this order):
- [ ] Text-to-speech models deployed for sound
- [ ] Speech-to-text + LM models deployed for verbal commands
- [ ] More realistic constraints (speed, altitude, ILS interception restrictions, etc.)
- [ ] Game crash protections for invalid commands
- [ ] Fix bugs with certain ILS angles
- [ ] Scoring system
- [ ] Departure aircraft functionality (all aircrafts right now are arrival traffic)
- [ ] Aircraft types

## Requirements
- Python with pygame `pip install pygame`

## How to run
`python main.py`

## Aircraft Command Reference

Control aircraft using the following commands. Enter the aircraft's callsign first (by typing or clicking the aircraft), followed by a space and the command(s).  
Multiple commands can be chained (e.g., `C 090;L S 220;X`).

### Course Commands (C)
- `C xxx` — Set heading (xxx = heading in degrees, e.g., `C 090`)
  - Optional: Add `;L` or `;R` (semicolon, no space) for specific turn direction (e.g., `C 090;L`)
- `C x/xx` — Set altitude (x/xx = altitude in thousands of feet, e.g., `C 5` for 5000 ft)
  - Optional: Add `;X` for expedited climb/descent (e.g., `C 5;X`)
- `C [waypoint]` — Direct aircraft to waypoint (e.g., `C CPT`)
  - Optional: Add `;L` or `;R` for turn direction

### Speed Commands (S)
- `S xxx` — Set airspeed (xxx = speed in knots, e.g., `S 220`)
  - Optional: Add `;X` for expedited speed change (e.g., `S 220;X`)

### Hold Commands (H)
- `H [waypoint]` — Hold at specified waypoint (e.g., `H CPT`)
  - Optional: Add `;L` or `;R` for turn direction (default: right turns)
  - Use course/hold commands to exit hold

### Landing Commands (L)
- `L [runway]` — Clear for ILS approach to specified runway (e.g., `L 27R`)
  - Must be within 15nm of airport, below 5000 feet, and below 240 knots
  - ILS interception requires heading within 30° if 5–15nm from runway, or within 20° if within 5nm
  - Aircraft will automatically slow to 140kts on short final (within 5nm)

### Abort Command (A)
- `A` — Abort approach (only when cleared for ILS)
  - Aircraft will climb and accelerate, exiting ILS

**Notes:**
- Use semicolons (`;`) to specify options (e.g., turn direction or expedite).
- Command `A` should be before all other commands.
- Command `L` should follow all other commands.