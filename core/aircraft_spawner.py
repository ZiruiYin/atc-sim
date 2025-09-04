import random

from params import *
from core.aircraft import Aircraft

class AircraftSpawner:
    def __init__(self, screen_width, screen_height, spawn_rate, spawn_directions):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.spawn_rate = spawn_rate
        self.spawn_directions = spawn_directions
        self.last_spawned_direction = None

        self.spawn_timer = spawn_rate

    def update(self, delta_t):
        self.spawn_timer += delta_t
        new_aircraft = None
        if self.spawn_timer >= self.spawn_rate:
            self.spawn_timer = 0.0
            new_aircraft = self.spawn_aircraft()
        return new_aircraft if new_aircraft else None

    def spawn_aircraft(self):
        callsign = random.choice(CALLSIGN_PREFIXES) + str(random.randint(1, 999))
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

        new_aircraft = Aircraft(callsign, initial_x, initial_y, heading, altitude, airspeed)

        self.last_spawned_direction = spawn_direction
        return new_aircraft
    
    def spawn_text_example(self): #For testing purposes only
        aircraft_1 = Aircraft("TEST1", self.screen_width / 2 - 150, self.screen_height / 2 + 20, 90, 1000, 220)
        aircraft_2 = Aircraft("TEST2", self.screen_width / 2 - 160, self.screen_height / 2 + 20, 90, 1000, 220)
        aircraft_3 = Aircraft("TEST3", self.screen_width / 2 + 150, self.screen_height / 2 + 20, 270, 1000, 220)

        return [aircraft_1, aircraft_2, aircraft_3]