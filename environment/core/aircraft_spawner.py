import random

from environment.params import *
from environment.utils import get_bearing_from_coords
from environment.core.aircraft import Aircraft

class AircraftSpawner:
    def __init__(self, screen_width, screen_height, spawn_rate, spawn_directions,
                 nm_per_pixel, coords, spawn_single=False,
                 star_mode=False, procedures=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.spawn_rate = spawn_rate
        self.spawn_directions = spawn_directions
        self.nm_per_pixel = nm_per_pixel
        self.coords = coords
        self.spawn_single = spawn_single
        self.star_mode = star_mode
        self.procedures = procedures or {}
        self.last_spawned_direction = None
        self.last_spawned_star = None

        self.spawn_timer = spawn_rate

    def update(self, delta_t):
        if self.spawn_single:
            return None
        self.spawn_timer += delta_t
        new_aircraft = None
        if self.spawn_timer >= self.spawn_rate:
            self.spawn_timer = 0.0
            new_aircraft = self.spawn_aircraft()
        return new_aircraft if new_aircraft else None

    def spawn_aircraft(self):
        callsign = random.choice(CALLSIGN_PREFIXES) + str(random.randint(1, 999))
        if self.star_mode and self.procedures:
            return self._spawn_star(callsign)
        return self._spawn_edge(callsign)

    def _spawn_edge(self, callsign):
        if len(self.spawn_directions) == 1:
            spawn_direction = self.spawn_directions[0]
        else:
            spawn_direction = random.choice([dir for dir in self.spawn_directions if dir != self.last_spawned_direction])
        if spawn_direction == "N":
            initial_x = random.uniform(self.screen_width * 0.25, self.screen_width * 0.75)
            initial_y = 0
            heading = 180 + random.uniform(-15, 15)
        elif spawn_direction == "S":
            initial_x = random.uniform(self.screen_width * 0.25, self.screen_width * 0.75)
            initial_y = self.screen_height
            heading = (0 + random.uniform(-15, 15)) % 360
        elif spawn_direction == "E":
            initial_x = self.screen_width
            initial_y = random.uniform(self.screen_height * 0.25, self.screen_height * 0.75)
            heading = 270 + random.uniform(-15, 15)
        elif spawn_direction == "W":
            initial_x = 0
            initial_y = random.uniform(self.screen_height * 0.25, self.screen_height * 0.75)
            heading = 90 + random.uniform(-15, 15)

        altitude = random.randint(5, 10) * 1000
        airspeed = 250

        new_aircraft = Aircraft(callsign, initial_x, initial_y, heading, altitude, airspeed,
                                self.nm_per_pixel, self.coords)

        self.last_spawned_direction = spawn_direction
        return new_aircraft

    def _spawn_star(self, callsign):
        candidates = list(self.procedures.keys())
        if len(candidates) > 1 and self.last_spawned_star in candidates:
            candidates.remove(self.last_spawned_star)
        star_name = random.choice(candidates)
        return self._build_star_aircraft(callsign, star_name, self.procedures[star_name])

    def _build_star_aircraft(self, callsign, star_name, steps):
        head = steps[0]
        head_wp = self.coords[head['waypoint']]
        initial_x = head_wp['x']
        initial_y = head_wp['y']
        if len(steps) >= 2:
            next_wp = self.coords[steps[1]['waypoint']]
            initial_heading = get_bearing_from_coords(initial_x, initial_y, next_wp['x'], next_wp['y'])
        else:
            initial_heading = 0
        new_aircraft = Aircraft(callsign, initial_x, initial_y, initial_heading,
                                head['altitude'], head['speed'],
                                self.nm_per_pixel, self.coords)
        new_aircraft.assign_star(steps, name=star_name)
        self.last_spawned_star = star_name
        return new_aircraft