from environment.display.generate_game_coordinates import generate_game_coordinates
from environment.core.aircraft_spawner import AircraftSpawner
from environment.core.collision_monitor import CollisionMonitor


class SimulationEnv:
    def __init__(self, radar_side=800, nm_range=60, airport_name="egll", spawn_rate=60,
                 spawn_directions=None, spawn_single=False, star_mode=False, recorder=None):
        self.radar_side = radar_side
        self.airport_name = airport_name.lower()
        self.spawn_single = spawn_single
        self.star_mode = star_mode
        self.recorder = recorder

        self.data = generate_game_coordinates(radar_side, radar_side, nm_range, self.airport_name)
        self.nm_per_pixel = self.data['screen_info']['nm_per_pixel']
        self.airport_x = self.data['airport']['coordinates']['x']
        self.airport_y = self.data['airport']['coordinates']['y']
        self.coords = self._flatten_coords(self.data)

        self.spawn_rate = spawn_rate
        self.spawn_directions = list(spawn_directions) if spawn_directions else ["N", "S", "E", "W"]
        self.spawner = AircraftSpawner(radar_side, radar_side, self.spawn_rate,
                                      self.spawn_directions, self.nm_per_pixel, self.coords,
                                      spawn_single=self.spawn_single,
                                      star_mode=self.star_mode,
                                      procedures=self.data.get('star_procedures', {}))
        self.collision_monitor = CollisionMonitor(radar_side, radar_side, self.nm_per_pixel)

        self.aircraft_list = {}

        self.num_landed = 0
        self.improper_exits = 0
        self.has_violation = False
        self.violation_seconds = 0.0

        self.crash_occurred = False
        self.crash_message = ""

        self.fast_forward = 1
        self.sim_time = 0.0

    @staticmethod
    def _flatten_coords(data):
        coords = {'airport': data['airport']['coordinates']}
        for rwy_data in data['runways'].values():
            for thr, thr_data in rwy_data['thresholds'].items():
                coords[thr] = thr_data
        for vor, vor_data in data['vor_stations'].items():
            coords[vor] = vor_data['coordinates']
        for ndb, ndb_data in data['ndb_stations'].items():
            coords[ndb] = ndb_data['coordinates']
        for wpt, wpt_data in data['rnav_waypoints'].items():
            coords[wpt] = wpt_data['coordinates']
        return coords

    def _add_spawned_aircraft(self, new_aircraft):
        if new_aircraft.callsign not in self.aircraft_list:
            self.aircraft_list[new_aircraft.callsign] = new_aircraft
        else:
            while True:
                retry = self.spawner.spawn_aircraft()
                if retry.callsign not in self.aircraft_list:
                    self.aircraft_list[retry.callsign] = retry
                    break

    def step(self, delta_t=1.0):
        if self.crash_occurred:
            return self.get_state()

        aircraft_list = list(self.aircraft_list.values())
        self.collision_monitor.check_collisions(aircraft_list)

        for aircraft in aircraft_list:
            if aircraft.crash is not None:
                self.crash_occurred = True
                self.crash_message = f"CRASH: {aircraft.callsign} {aircraft.crash}"
                break

        self.has_violation = False
        to_remove = []
        removal_reason = {}

        for aircraft in list(self.aircraft_list.values()):
            if not self.crash_occurred:
                aircraft.update(delta_t)

            if aircraft.collision_warning:
                self.has_violation = True

            x, y = aircraft.x, aircraft.y
            cs = aircraft.callsign
            if x < 0 or x > self.radar_side or y < 0 or y > self.radar_side:
                to_remove.append(cs)
                removal_reason[cs] = 'IMPROPER_EXIT'
                self.improper_exits += 1
                continue

            if aircraft.landed:
                to_remove.append(cs)
                removal_reason[cs] = 'LANDED'
                self.num_landed += 1

        self.sim_time += delta_t

        if self.recorder and not self.crash_occurred:
            rt = {}
            for cs in to_remove:
                rt[cs] = removal_reason[cs]
            self.recorder.log_timestep(self, rt)

        for cs in to_remove:
            del self.aircraft_list[cs]

        if not self.crash_occurred:
            if self.spawn_single:
                if len(self.aircraft_list) == 0:
                    self._add_spawned_aircraft(self.spawner.spawn_aircraft())
            else:
                new_aircraft = self.spawner.update(delta_t)
                if new_aircraft:
                    self._add_spawned_aircraft(new_aircraft)

        if self.has_violation:
            self.violation_seconds += delta_t

        return self.get_state()

    def command(self, callsign, cmd_string):
        callsign = callsign.upper()
        if callsign not in self.aircraft_list:
            return {"ok": False, "category": "invalid",
                    "message": f"unknown callsign: {callsign}",
                    "callsign_valid": False}
        result = self.aircraft_list[callsign].process_command(cmd_string.upper())
        result["callsign_valid"] = True
        result["callsign"] = callsign
        if self.recorder:
            # Record any attempted command on a valid callsign, even if rejected,
            # so the training data captures the agent's intent.
            self.recorder.enqueue_command_action(callsign, cmd_string)
        return result

    def set_speed(self, multiplier):
        if multiplier in (1, 10):
            self.fast_forward = multiplier

    def set_spawn_rate(self, rate):
        self.spawn_rate = max(10, int(rate))
        self.spawner.spawn_rate = self.spawn_rate

    def set_spawn_directions(self, directions):
        if self.star_mode:
            return
        valid = [d for d in directions if d in ("N", "S", "E", "W")]
        if valid:
            self.spawn_directions = valid
            self.spawner.spawn_directions = self.spawn_directions

    def get_state(self):
        aircraft_state = []
        for ac in self.aircraft_list.values():
            aircraft_state.append({
                "callsign": ac.callsign,
                "x": ac.x,
                "y": ac.y,
                "trajectory": list(ac.trajectory),
                "heading": ac.heading,
                "altitude": ac.altitude,
                "airspeed": ac.airspeed,
                "target_heading": ac.target_heading,
                "target_altitude": ac.target_altitude,
                "target_airspeed": ac.target_airspeed,
                "target_wpt": ac.target_wpt,
                "runway": ac.ils_runway,
                "loc": ac.loc_intercepted,
                "gs": ac.gs_intercepted,
                "can_intercept": ac.can_intercept,
                "holding": ac.holding,
                "collision_warning": ac.collision_warning,
                "landed": ac.landed,
                "star": ac.star_name if ac.star else None,
            })

        return {
            "static": {
                "airport": self.data['airport'],
                "runways": self.data['runways'],
                "vor_stations": self.data['vor_stations'],
                "ndb_stations": self.data['ndb_stations'],
                "rnav_waypoints": self.data['rnav_waypoints'],
                "star_procedures": self.data.get('star_procedures', {}),
                "nm_per_pixel": self.nm_per_pixel,
                "radar_side": self.radar_side,
            },
            "aircraft": aircraft_state,
            "scoring": {
                "num_landed": self.num_landed,
                "improper_exits": self.improper_exits,
                "violation_seconds": round(self.violation_seconds, 2),
            },
            "crash": {
                "occurred": self.crash_occurred,
                "message": self.crash_message,
            },
            "fast_forward": self.fast_forward,
            "spawn_rate": self.spawn_rate,
            "spawn_directions": list(self.spawn_directions),
            "star_mode": self.star_mode,
        }
