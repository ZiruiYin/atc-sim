This is an ATC simulator game for London Heathrow airport (EGLL/LHR), inspired by atc-sim.com.

The implementation is in progress. TODOs:
- [ ] More realistic constraints (speed, altitude, ILS interception angle restrictions, etc.)
- [ ] Collision warning system
- [ ] Scoring system
- [ ] Departure aircraft functionality (all aircrafts right now are arrival traffic)
- [ ] Aircraft types (they all have the same performance for now)
- [ ] Code refactoring (LOTS of hard-coded stuff! Also, the architecture is a mess...)
- [ ] A side page to show the current aircrafts

## Requirements
- Python with pygame (`pip install pygame`)

## Commands
Control aircraft using the following commands:

### Course Commands (C)
- `C xxx` - Set heading (xxx = heading in degrees)
  - Optional: Add L/R for specific turn direction (e.g., `C 090 L`)
- `C x/xx` - Set altitude (x/xx = altitude in thousands of feet)
  - Optional: Add X for expedited climb/descent (e.g., `C 5 X`)
- `C [waypoint]` - Direct aircraft to waypoint
  - Optional: Add L/R for specific turn direction

### Landing Commands (L)
- `L [runway]` - Clear for ILS approach to specified runway
  - Must be within 15nm of airport
  - Must be below 5000 feet AGL
  - Must be below 200 knots

### Hold Commands (H)
- `H [waypoint]` - Hold at specified waypoint
  - Optional: Add L/R for turn direction (default: right turns)
  - Use course/hold commands to exit hold

### Speed Commands (S)
- `S xxx` - Set airspeed (xxx = speed in knots)

### Abort Command (A)
- `A` - Abort approach (only when cleared for ILS)
  - Aircraft will climb to 5000 feet and fly runway heading