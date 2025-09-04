import math
from params import *
from utils import *

class Aircraft:
    def __init__(self, callsign, initial_x, initial_y, heading, altitude, airspeed):
        self.callsign = callsign
        self.x = initial_x
        self.y = initial_y
        self.heading = heading
        self.altitude = altitude
        self.airspeed = airspeed

        self.trajectory = []
        self._traj_counter = 0

        self.target_heading = heading
        self.target_altitude = altitude
        self.target_airspeed = airspeed
        self.target_wpt = None

        self.turn_direction = None
        self.expedite_speed = None
        self.expedite_altitude = None

        self.holding = False
        self.hold_direction = "R"
        self.holding_outbound = True
        self.initial_heading = heading

        self.ils_runway = None
        self.loc_intercepted = False
        self.gs_intercepted = False
        self.short_final = False
        self.on_ground = None
        self.go_around = False

        self.landed = False

        self.selected = False
        self.collision_warning = False
        self.crash = None
        
        self.nm_per_pixel = get_nm_per_pixel()
        self.coords = get_coords()

    def get_info(self):
        return {
            "callsign": self.callsign,
            "position": (self.x, self.y),
            "trajectory": self.trajectory,
            "heading": self.heading,
            "altitude": self.altitude,
            "airspeed": self.airspeed,
            "runway": self.ils_runway,
            "loc": self.loc_intercepted,
            "gs": self.gs_intercepted,
            "holding": self.holding,
            "target_heading": self.target_heading,
            "target_altitude": self.target_altitude,
            "target_airspeed": self.target_airspeed,
            "target_wpt": self.target_wpt,
            "landed": self.landed,
            "collision_warning": self.collision_warning
        }

    def update(self, delta_t, fast_forward):
        self.update_landed()
        self.update_final()
        self.update_wpt_nav()
        self.update_holding()
        self.update_ils()
        self.update_airspeed(delta_t)
        self.update_altitude(delta_t)
        self.update_heading(delta_t)
        self.update_position(delta_t, fast_forward)

    def update_landed(self):
        if self.altitude <= 20:
            if not self.on_ground:
                threshold_x, threshold_y = self.coords[self.ils_runway]['x'], self.coords[self.ils_runway]['y']
                runway_number = ''.join(filter(str.isdigit, self.ils_runway))
                runway_heading = int(runway_number) * 10
                if distance_between_coords_pixels(self.x, self.y, threshold_x, threshold_y) < 5:
                    if self.ils_runway:
                        self.on_ground = self.ils_runway
                    self.target_airspeed = 0
                    self.target_altitude = 0
                    self.altitude = 0
                    self.heading = runway_heading
                    
                    self.ils_runway = None
                    self.loc_intercepted = False
                    self.gs_intercepted = False
                    self.short_final = False

            elif self.airspeed == 0:
                self.landed = True

    def update_final(self):
        if self.short_final:
            self.target_airspeed = SHORT_FINAL_IAS

        elif self.ils_runway:
            threshold_x, threshold_y = self.coords[self.ils_runway]['x'], self.coords[self.ils_runway]['y']
            distance_to_threshold = self.nm_per_pixel * distance_between_coords_pixels(self.x, self.y, threshold_x, threshold_y)

            if distance_to_threshold < 5:
                self.short_final = True

    def update_airspeed(self, delta_t):
        if self.on_ground:
            self.airspeed = max(0, self.airspeed - GROUND_DECELERATION_RATE * delta_t)
            return

        if self.airspeed == self.target_airspeed:
            self.expedite_speed = None
            return
        
        if self.expedite_speed:
            if self.target_altitude < self.altitude and self.target_airspeed < self.airspeed:
                rate = EXP_SPEED_CHANGE_RATE * 0.8
            else:
                rate = EXP_SPEED_CHANGE_RATE
        else:
            rate = SPEED_CHANGE_RATE
        
        change = rate * delta_t
        
        if self.airspeed < self.target_airspeed:
            self.airspeed = min(self.airspeed + change, self.target_airspeed)
        else:
            self.airspeed = max(self.airspeed - change, self.target_airspeed)
        
        if self.airspeed == self.target_airspeed:
            self.expedite_speed = None

    def update_altitude(self, delta_t):
        if self.altitude == self.target_altitude:
            self.expedite_altitude = None
            return
        
        if self.expedite_altitude:
            if self.altitude < self.target_altitude:
                rate = CLIMB_RATE
            else:
                rate = EXP_DESCENT_RATE
        else:
            if self.altitude < self.target_altitude:
                rate = CLIMB_RATE
            else:
                rate = DESCENT_RATE
        
        change = rate * delta_t
        
        if self.altitude < self.target_altitude:
            self.altitude = min(self.altitude + change, self.target_altitude)
        else:
            self.altitude = max(self.altitude - change, self.target_altitude)
        
        if self.altitude == self.target_altitude:
            self.expedite_altitude = None

    def update_heading(self, delta_t):
        if self.heading == self.target_heading:
            self.turn_direction = None
            return

        max_turn = TURN_RATE * delta_t

        if self.turn_direction is None:
            diff = (self.target_heading - self.heading) % 360
            if diff > 180:
                self.turn_direction = "L"
            else:
                self.turn_direction = "R"
    
        if self.turn_direction == "L":
            new_heading = self.heading - max_turn
        else:
            new_heading = self.heading + max_turn
        
        new_heading = new_heading % 360
        
        if heading_diff(self.heading, self.target_heading) <= max_turn:
            self.heading = self.target_heading
        else:
            self.heading = new_heading
        self.turn_direction = None

    def update_wpt_nav(self):
        if not self.target_wpt:
            return
        wpt_coords = self.coords.get(self.target_wpt)
        wx, wy = wpt_coords['x'], wpt_coords['y']
        bearing = get_bearing_from_coords(self.x, self.y, wx, wy)
        self.target_heading = bearing
        dist_pixels = distance_between_coords_pixels(self.x, self.y, wx, wy)
        dist_nm = dist_pixels * self.nm_per_pixel
        if dist_nm <= 0.5:
            self.target_heading = self.heading
            if self.holding:
                self.initial_heading = self.heading
                self.target_heading = (self.initial_heading + 180) % 360
            self.target_wpt = None

    def update_holding(self):
        if not self.holding:
            self.holding_outbound = True
            return
        if not self.target_wpt:
            self.turn_direction = self.hold_direction
            if self.holding_outbound:
                if self.heading == self.target_heading:
                    self.holding_outbound = False
                    self.target_heading = self.initial_heading
            else:
                if self.heading == self.target_heading:
                    self.holding_outbound = True
                    self.target_heading = (self.initial_heading + 180) % 360

    def update_ils(self):
        if not self.ils_runway or not self._can_intercept:
            return
        
        self._update_ils_loc()
        self._update_ils_gs()

    def _can_intercept(self):
        if self.airspeed >= 240 or self.altitude >= 5000:
            return False
        threshold_x, threshold_y = self.coords[self.ils_runway]['x'], self.coords[self.ils_runway]['y']
        runway_number = ''.join(filter(str.isdigit, self.ils_runway))
        runway_heading = int(runway_number) * 10
        distance_between_coords = distance_between_coords_pixels(self.x, self.y, threshold_x, threshold_y)
        dist_nm = self.nm_per_pixel * distance_between_coords
        angle_diff = heading_diff(self.heading, runway_heading)

        if dist_nm <= 5:
            max_angle = 20
        elif dist_nm <= 15:
            max_angle = 30
        else:
            return False

        if angle_diff > max_angle:
            return False
        return True
    
    def _update_ils_loc(self):
        runway_number = ''.join(filter(str.isdigit, self.ils_runway))
        runway_heading = int(runway_number) * 10
        
        threshold_x, threshold_y = self.coords[self.ils_runway]['x'], self.coords[self.ils_runway]['y']
        
        projection_distance_nm = 0.3
        projection_distance_pixels = projection_distance_nm / self.nm_per_pixel
        
        heading_rad = math.radians(self.heading)
        projected_x = self.x + projection_distance_pixels * math.sin(heading_rad)
        projected_y = self.y - projection_distance_pixels * math.cos(heading_rad)

        aircraft_bearing_to_threshold = get_bearing_from_coords(self.x, self.y, threshold_x, threshold_y)
        
        if not self.loc_intercepted:
            projected_bearing_to_threshold = get_bearing_from_coords(projected_x, projected_y, threshold_x, threshold_y)
            
            if opposite_sides(aircraft_bearing_to_threshold, projected_bearing_to_threshold, runway_heading):
                self.loc_intercepted = True
        else:
            rwy_extension_hdg = (runway_heading + 180) % 360

            aircraft_to_extension_proj = self.nm_per_pixel * projection_distance(self.x, self.y, rwy_extension_hdg, threshold_x, threshold_y)
            angle_diff = abs(math.degrees(math.asin(aircraft_to_extension_proj / projection_distance_nm)))
            candidate_heading_1 = (runway_heading - angle_diff) % 360
            candidate_heading_2 = (runway_heading + angle_diff) % 360

            if heading_diff(aircraft_bearing_to_threshold, candidate_heading_1) < heading_diff(aircraft_bearing_to_threshold, candidate_heading_2):
                self.target_heading = candidate_heading_1
            else:
                self.target_heading = candidate_heading_2

    def _update_ils_gs(self):
        if not self.loc_intercepted:
            return
        threshold_x, threshold_y = self.coords[self.ils_runway]['x'], self.coords[self.ils_runway]['y']
        distance_to_threshold = self.nm_per_pixel * distance_between_coords_pixels(self.x, self.y, threshold_x, threshold_y)
        projected_alt = distance_to_threshold * 300

        if not self.gs_intercepted:
            if self.altitude <= projected_alt:
                if abs(self.altitude - projected_alt) <= 50:
                    self.gs_intercepted = True
                    self.target_altitude = projected_alt
        else:
            self.target_altitude = projected_alt

    def update_position(self, delta_t, fast_forward):
        nm_per_second = ias_to_gs(self.airspeed, self.altitude) / 3600
        nm_traveled = nm_per_second * delta_t
        pixels_traveled = nm_traveled / self.nm_per_pixel

        heading_rad = math.radians(self.heading)
        dx = pixels_traveled * math.sin(heading_rad)
        dy = -pixels_traveled * math.cos(heading_rad)

        self.x += dx
        self.y += dy

        self._traj_counter += 1
        if self._traj_counter >= 10/fast_forward:
            self.trajectory.append((self.x, self.y))
            if len(self.trajectory) > TRAJ_LENGTH:
                self.trajectory = self.trajectory[-TRAJ_LENGTH:]
            self._traj_counter = 0

    def process_command(self, cmd):
        try:
            commands = cmd.strip().split()
            if commands[0].upper() == "A":
                commands.insert(1, None)
            if len(commands) % 2 != 0:
                return
            
            command_pairs = []
            for i in range(0, len(commands), 2):
                command_pairs.append((commands[i], commands[i + 1]))
            
            command_types = [pair[0] for pair in command_pairs]

            if 'H' in command_types and ('C' in command_types or 'L' in command_types):
                return
            
            a_index = command_types.index('A') if 'A' in command_types else -1
            l_index = command_types.index('L') if 'L' in command_types else len(command_types)

            
            for i, cmd_type in enumerate(command_types):
                if cmd_type in ['C', 'S', 'H']:
                    if a_index != -1 and i < a_index:
                        return
                    if i >= l_index:
                        return
            
            for cmd_type, param in command_pairs:
                if cmd_type == 'A':
                    if self.ils_runway and not self.on_ground and not self.landed:
                        self.ils_runway = None
                        self.loc_intercepted = False
                        self.gs_intercepted = False
                        self.short_final = False
                        self.target_airspeed = max(self.airspeed, 180)
                        self.target_altitude = max(self.altitude, 3000)
                        self.target_heading = self.heading
                elif cmd_type == 'C':
                    parts = param.split(';')
                    if len(parts[0]) == 3 and parts[0].isdigit():
                        if not self.loc_intercepted and not self.on_ground and not self.landed:
                            self.target_heading = int(parts[0])
                            self.target_wpt = None
                            self.holding = False
                            self.turn_direction = None
                            if len(parts) > 1 and parts[1] in ['L', 'R']:
                                self.turn_direction = parts[1]
                    elif len(parts[0]) == 1 or len(parts[0]) == 2:
                        if not self.gs_intercepted and not self.on_ground and not self.landed:
                            self.target_altitude = int(parts[0]) * 1000
                            if len(parts) > 1 and parts[1] == 'X':
                                self.expedite_altitude = True
                    else:
                        if not self.loc_intercepted and not self.on_ground and not self.landed:
                            wpt = parts[0]
                            self.target_wpt = wpt
                            self.holding = False
                            self.turn_direction = None
                            if len(parts) > 1 and parts[1] in ['L', 'R']:
                                self.turn_direction = parts[1]
                
                elif cmd_type == 'S':
                    parts = param.split(';')
                    if not self.short_final and not self.on_ground and not self.landed:
                        self.target_airspeed = int(parts[0])
                        if len(parts) > 1 and parts[1] == 'X':
                            self.expedite_speed = True
                
                elif cmd_type == 'H':
                    parts = param.split(';')
                    if self.ils_runway is None:
                        wpt = parts[0]
                        turn_dir = 'R'
                        if len(parts) > 1 and parts[1] in ['L', 'R']:
                            turn_dir = parts[1]
                        self.target_wpt = wpt
                        self.holding = True
                        self.hold_direction = turn_dir
                        self.holding_outbound = True
                
                elif cmd_type == 'L':
                    if not self.ils_runway:
                        self.ils_runway = param
                        self.loc_intercepted = False
                        self.gs_intercepted = False
                        
        except Exception as e:
            pass