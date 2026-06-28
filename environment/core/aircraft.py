import math
import re
from environment.params import *
from environment.utils import *

class Aircraft:
    def __init__(self, callsign, initial_x, initial_y, heading, altitude, airspeed,
                 nm_per_pixel, coords):
        self.callsign = callsign
        self.x = initial_x
        self.y = initial_y
        self.heading = heading
        self.altitude = altitude
        self.airspeed = airspeed

        self.trajectory = []

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
        self.can_intercept = False
        self.short_final = False
        self.on_ground = None
        self.go_around = False

        self.landed = False

        self.selected = False
        self.collision_warning = False
        self.crash = None

        self.star = None
        self.star_name = None
        self.star_apply_alt = True
        self.star_apply_spd = True

        self.nm_per_pixel = nm_per_pixel
        self.coords = coords

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

    def update(self, delta_t):
        self.update_landed()
        self.update_final()
        self.update_wpt_nav()
        self.update_holding()
        self.update_ils()
        self.update_airspeed(delta_t)
        self.update_altitude(delta_t)
        self.update_heading(delta_t)
        self.update_position(delta_t)

    def update_landed(self):
        if self.altitude <= 20:
            if not self.on_ground:
                if not self.ils_runway:
                    return
                threshold_x, threshold_y = self.coords[self.ils_runway]['x'], self.coords[self.ils_runway]['y']
                runway_number = ''.join(filter(str.isdigit, self.ils_runway))
                runway_heading = int(runway_number) * 10
                if distance_between_coords_pixels(self.x, self.y, threshold_x, threshold_y) < 5:
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
            if self.star:
                self.star.pop(0)
                if not self.star:
                    self.star = None
                else:
                    self._apply_star_head()

    def assign_star(self, steps, name=None):
        if not steps:
            return
        self.star = list(steps)
        self.star_name = name
        self.star_apply_alt = True
        self.star_apply_spd = True
        self.holding = False
        self.turn_direction = None
        self._apply_star_head()

    def _apply_star_head(self):
        if not self.star:
            return
        step = self.star[0]
        self.target_wpt = step['waypoint']
        if self.star_apply_alt:
            self.target_altitude = step['altitude']
        if self.star_apply_spd:
            self.target_airspeed = step['speed']

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
        if not self.ils_runway:
            self.can_intercept = False
            return
        self.can_intercept = self._can_intercept()
        if not self.can_intercept:
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
        elif dist_nm <= 20:
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

    def update_position(self, delta_t):
        nm_per_second = ias_to_gs(self.airspeed, self.altitude) / 3600
        nm_traveled = nm_per_second * delta_t
        pixels_traveled = nm_traveled / self.nm_per_pixel

        heading_rad = math.radians(self.heading)
        dx = pixels_traveled * math.sin(heading_rad)
        dy = -pixels_traveled * math.cos(heading_rad)

        self.x += dx
        self.y += dy

        self.trajectory.append((self.x, self.y))
        if len(self.trajectory) > TRAJ_LENGTH:
            self.trajectory = self.trajectory[-TRAJ_LENGTH:]

    def process_command(self, cmd):
        try:
            return self._process_command_inner(cmd)
        except Exception as e:
            return {'ok': False, 'category': 'invalid', 'message': f'internal error: {e}'}

    def _process_command_inner(self, cmd):
        cmd = cmd.strip()
        if not cmd:
            return {'ok': False, 'category': 'invalid', 'message': 'empty command'}

        commands = cmd.split()
        if commands[0].upper() == "A":
            commands.insert(1, None)
        if len(commands) % 2 != 0:
            return {'ok': False, 'category': 'invalid', 'message': 'expected pairs of TYPE PARAM'}

        command_pairs = []
        for i in range(0, len(commands), 2):
            command_pairs.append((commands[i], commands[i + 1]))

        command_types = [pair[0] for pair in command_pairs]

        for ct in command_types:
            if ct not in ('A', 'C', 'S', 'H', 'L'):
                return {'ok': False, 'category': 'invalid', 'message': f'unknown command type: {ct}'}

        if 'H' in command_types and ('C' in command_types or 'L' in command_types):
            return {'ok': False, 'category': 'invalid', 'message': 'H cannot be combined with C or L'}

        a_index = command_types.index('A') if 'A' in command_types else -1
        l_index = command_types.index('L') if 'L' in command_types else len(command_types)

        for i, cmd_type in enumerate(command_types):
            if cmd_type in ['C', 'S', 'H']:
                if a_index != -1 and i < a_index:
                    return {'ok': False, 'category': 'invalid', 'message': 'A must come first in chain'}
                if i >= l_index:
                    return {'ok': False, 'category': 'invalid', 'message': 'L must come last in chain'}

        for cmd_type, param in command_pairs:
            err = self._validate_param(cmd_type, param)
            if err:
                return {'ok': False, 'category': 'unable', 'message': err}

        before = {
            'heading': self.heading,
            'altitude': self.altitude,
            'airspeed': self.airspeed,
        }
        for cmd_type, param in command_pairs:
            err = self._apply_command(cmd_type, param)
            if err:
                return {'ok': False, 'category': 'unable', 'message': err}

        atc, pilot = self._build_radio_messages(command_pairs, before)
        return {'ok': True, 'category': 'success', 'atc': atc, 'pilot': pilot}

    def _build_radio_messages(self, command_pairs, before):
        # bucket index for each phrase: A first, then vector, alt, speed, hold, land
        BUCKET = {'a': 0, 'vector': 1, 'alt': 2, 'speed': 3, 'hold': 1, 'land': 4}

        has_vector_cmd = False
        has_alt_cmd = False
        # On an ILS clearance (TRACON) the altitude is folded into the approach
        # clearance ("descend/climb and maintain X until established on the
        # localizer"), so a standalone altitude phrase from the same command is
        # suppressed to avoid stating the altitude twice.
        has_land = any(ct == 'L' for ct, _ in command_pairs)
        for ct, p in command_pairs:
            if ct == 'C':
                f = (p or '').split(';')[0]
                if f.isdigit() and len(f) == 3:
                    has_vector_cmd = True
                elif f.isdigit() and len(f) <= 2:
                    has_alt_cmd = True
                else:
                    has_vector_cmd = True

        buckets = []
        for cmd_type, param in command_pairs:
            if cmd_type == 'A':
                a_parts = ['go around']
                if not has_vector_cmd:
                    a_parts.append('fly current heading')
                buckets.append((BUCKET['a'], ', '.join(a_parts)))
                if not has_alt_cmd:
                    cur_alt = before['altitude']
                    if cur_alt < 3000:
                        buckets.append((BUCKET['alt'], 'climb and maintain 3000'))
                    else:
                        buckets.append((BUCKET['alt'], f'maintain {int(cur_alt)}'))
                continue
            parts = (param or '').split(';')
            first = parts[0]
            if cmd_type == 'C':
                if first.isdigit() and len(first) == 3:
                    hdg = int(first)
                    # Spoken/written heading uses 360 for north, never 000.
                    hdg_disp = hdg % 360 or 360
                    explicit = parts[1] if len(parts) > 1 and parts[1] in ('L', 'R') else None
                    turn_dir = explicit
                    if turn_dir is None:
                        cur = before['heading']
                        diff = (hdg - cur) % 360
                        if diff == 0 or diff == 360:
                            turn_dir = None
                        elif diff <= 180:
                            turn_dir = 'R'
                        else:
                            turn_dir = 'L'
                    if turn_dir:
                        word = 'left' if turn_dir == 'L' else 'right'
                        buckets.append((BUCKET['vector'], f'turn {word} heading {hdg_disp:03d}'))
                    else:
                        buckets.append((BUCKET['vector'], f'fly heading {hdg_disp:03d}'))
                elif first.isdigit() and len(first) <= 2:
                    if has_land:
                        pass  # altitude is folded into the ILS clearance below
                    else:
                        alt = int(first) * 1000
                        if alt > before['altitude']:
                            buckets.append((BUCKET['alt'], f'climb and maintain {alt}'))
                        elif alt < before['altitude']:
                            buckets.append((BUCKET['alt'], f'descend and maintain {alt}'))
                        else:
                            buckets.append((BUCKET['alt'], f'maintain {alt}'))
                else:
                    explicit = parts[1] if len(parts) > 1 and parts[1] in ('L', 'R') else None
                    if explicit:
                        word = 'left' if explicit == 'L' else 'right'
                        buckets.append((BUCKET['vector'], f'{word} turn direct to {first}'))
                    else:
                        buckets.append((BUCKET['vector'], f'direct to {first}'))
            elif cmd_type == 'S':
                spd = int(first)
                if spd < before['airspeed']:
                    buckets.append((BUCKET['speed'], f'reduce speed to {spd} knots'))
                elif spd > before['airspeed']:
                    buckets.append((BUCKET['speed'], f'increase speed to {spd} knots'))
                else:
                    buckets.append((BUCKET['speed'], f'maintain {spd} knots'))
            elif cmd_type == 'H':
                turn = parts[1] if len(parts) > 1 and parts[1] in ('L', 'R') else 'R'
                word = 'left' if turn == 'L' else 'right'
                buckets.append((BUCKET['hold'], f'hold over {first} {word} turn'))
            elif cmd_type == 'L':
                # TRACON ILS clearance (FAA JO 7110.65 5-9-1): bring the aircraft
                # to the platform altitude and hold it until established on the
                # localizer, stated in one phrase. self.target_altitude is the
                # assigned altitude -- from an altitude command in this chain OR
                # carried over from the STAR. The verb keys off the *current*
                # altitude, so a descent/climb is voiced whenever the aircraft is
                # not already at it (a STAR target while still descending -> say
                # "descend and maintain"); plain "maintain" only when already there.
                maintain_alt = int(self.target_altitude)
                if maintain_alt < before['altitude']:
                    verb = 'descend and maintain'
                elif maintain_alt > before['altitude']:
                    verb = 'climb and maintain'
                else:
                    verb = 'maintain'
                buckets.append((BUCKET['land'],
                                f'{verb} {maintain_alt} until established on the '
                                f'localizer, cleared ILS runway {param} approach'))

        buckets.sort(key=lambda b: b[0])
        phrases = [text for _, text in buckets]
        if not phrases:
            return None, None

        atc_body = ', '.join(phrases)
        atc = f'{self.callsign}, {atc_body}'

        # Pilots read the instruction back as given, including the turn direction
        # ("turn right heading 240"); only the callsign moves to the end.
        pilot_body = atc_body.replace('go around', 'going around')
        pilot_body = pilot_body[:1].upper() + pilot_body[1:]
        pilot = f'{pilot_body}, {self.callsign}'
        return atc, pilot

    def _validate_param(self, cmd_type, param):
        if cmd_type == 'A':
            return None
        if param is None or param == '':
            return f"{cmd_type} requires a parameter"
        parts = param.split(';')
        first = parts[0]
        if cmd_type == 'C':
            if not first:
                return "C requires a value"
            if first.isdigit() and len(first) == 3:
                hdg = int(first)
                if hdg < 0 or hdg > 360:
                    return f"heading must be 0-360 (got {hdg})"
            elif first.isdigit() and len(first) <= 2:
                alt = int(first)
                if alt < 1 or alt > 18:
                    return f"altitude must be 1-18 thousands of feet (got {alt})"
            else:
                if first not in self.coords:
                    return f"unknown waypoint: {first}"
        elif cmd_type == 'S':
            if not first.lstrip('-').isdigit():
                return f"speed must be integer (got {first})"
            spd = int(first)
            if spd < 140 or spd > 280:
                return f"speed must be 140-280 kts (got {spd})"
        elif cmd_type == 'H':
            if first not in self.coords:
                return f"unknown waypoint: {first}"
        elif cmd_type == 'L':
            if param not in self.coords or not any(c.isdigit() for c in param):
                return f"unknown runway: {param}"
        return None

    def _apply_command(self, cmd_type, param):
        if cmd_type == 'A':
            if not self.ils_runway:
                return "abort: not cleared for ILS"
            if self.on_ground or self.landed:
                return "abort: aircraft on ground"
            self.ils_runway = None
            self.loc_intercepted = False
            self.gs_intercepted = False
            self.short_final = False
            self.target_airspeed = max(self.airspeed, 180)
            self.target_altitude = max(self.altitude, 3000)
            self.target_heading = self.heading
            return None

        if cmd_type == 'C':
            parts = param.split(';')
            first = parts[0]
            if first.isdigit() and len(first) == 3:
                if self.loc_intercepted:
                    return "cannot vector: already on localizer"
                if self.on_ground or self.landed:
                    return "cannot vector: aircraft on ground"
                self.target_heading = int(first)
                self.target_wpt = None
                self.holding = False
                self.turn_direction = None
                self.star = None
                if len(parts) > 1 and parts[1] in ['L', 'R']:
                    self.turn_direction = parts[1]
            elif first.isdigit() and len(first) <= 2:
                if self.gs_intercepted:
                    return "cannot change altitude: on glideslope"
                if self.on_ground or self.landed:
                    return "cannot change altitude: aircraft on ground"
                self.target_altitude = int(first) * 1000
                self.star_apply_alt = False
                if len(parts) > 1 and parts[1] == 'X':
                    self.expedite_altitude = True
            else:
                if self.loc_intercepted:
                    return "cannot navigate: already on localizer"
                if self.on_ground or self.landed:
                    return "cannot navigate: aircraft on ground"
                self.target_wpt = first
                self.holding = False
                self.turn_direction = None
                self.star = None
                if len(parts) > 1 and parts[1] in ['L', 'R']:
                    self.turn_direction = parts[1]
            return None

        if cmd_type == 'S':
            if self.short_final:
                return "cannot change speed: on short final"
            if self.on_ground or self.landed:
                return "cannot change speed: aircraft on ground"
            parts = param.split(';')
            self.target_airspeed = int(parts[0])
            self.star_apply_spd = False
            if len(parts) > 1 and parts[1] == 'X':
                self.expedite_speed = True
            return None

        if cmd_type == 'H':
            if self.ils_runway is not None:
                return "cannot hold: cleared for ILS approach"
            parts = param.split(';')
            wpt = parts[0]
            turn_dir = 'R'
            if len(parts) > 1 and parts[1] in ['L', 'R']:
                turn_dir = parts[1]
            self.target_wpt = wpt
            self.holding = True
            self.hold_direction = turn_dir
            self.holding_outbound = True
            self.star = None
            return None

        if cmd_type == 'L':
            if self.ils_runway:
                return "already cleared for ILS approach"
            if self.star:
                return "on STAR; vector before clearing for ILS"
            self.ils_runway = param
            self.loc_intercepted = False
            self.gs_intercepted = False
            return None

        return f"unknown command type: {cmd_type}"