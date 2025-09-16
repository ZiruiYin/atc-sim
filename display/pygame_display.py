import pygame
import json
import os
import math

from display.generate_game_coordinates import generate_game_coordinates
from core.aircraft_spawner import AircraftSpawner
from core.collision_monitor import CollisionMonitor

class RadarDisplay:
    def __init__(self, screen_width, screen_height, airport_name="egll"):
        self.textbox_height = 80
        
        self.radar_height = screen_height - self.textbox_height
        
        generate_game_coordinates(screen_width, self.radar_height)
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.airport_name = airport_name.lower()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("ATC Radar Display")
        
        pygame.font.init()
        self.font = pygame.font.Font(None, 20)
        
        self.textbox_locked = False
        self.textbox_text = ""
        
        self.show_airport = False
        self.show_radar_rings = True
        self.show_vor = True
        self.show_ndb = True
        self.show_waypoints = True
        self.show_runway_names = True
        self.aircraft_details = True

        self.aircraft_list = {}

        self.load_data()
        self.spawn_rate = 60
        self.spawn_directions = ["N", "S", "E", "W"]
        self.spawner = AircraftSpawner(self.screen_width, self.radar_height, self.spawn_rate, self.spawn_directions)
        
        # TESTING ONLY
        # test_aircrafts = self.spawner.spawn_text_example()
        # for aircraft in test_aircrafts:
        #     self.aircraft_list[aircraft.callsign] = aircraft
        
        self.collision_monitor = CollisionMonitor(self.screen_width, self.radar_height)

        self.num_landed = 0
        self.improper_exits = 0
        self.has_violation = False
        self.violation_seconds = 0.0

        self.crash_occurred = False
        self.crash_message = ""

        self.fast_forward = 1
        
    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(script_dir), 'data')
        game_file = os.path.join(data_dir, f'{self.airport_name}_game.json')
        
        with open(game_file, 'r') as f:
            self.data = json.load(f)
        
        self.nm_per_pixel = self.data['screen_info']['nm_per_pixel']
        self.airport_x = self.data['airport']['coordinates']['x']
        self.airport_y = self.data['airport']['coordinates']['y']
        
    def draw_radar_rings(self):
        if self.show_radar_rings:
            ring_distances = [5, 10, 15, 20]
            
            for distance_nm in ring_distances:
                radius_pixels = distance_nm / self.nm_per_pixel
                pygame.draw.circle(self.screen, (0, 255, 0), 
                                 (int(self.airport_x), int(self.airport_y)), 
                                 int(radius_pixels), 1)
    
    def draw_airport(self):
        self.draw_runways()
        
        if self.show_airport:
            airport_icao = self.data['airport']['icao']
            text = self.font.render(airport_icao, True, (255, 255, 255))
            text_rect = text.get_rect(center=(int(self.airport_x), int(self.airport_y - 15)))
            self.screen.blit(text, text_rect)
    
    def draw_runways(self):
        for _, runway_data in self.data['runways'].items():
            if 'thresholds' in runway_data:
                thresholds = runway_data['thresholds']
                threshold_keys = list(thresholds.keys())
                if len(threshold_keys) >= 2:
                    t1_name = threshold_keys[0]
                    t2_name = threshold_keys[1]
                    t1 = thresholds[t1_name]
                    t2 = thresholds[t2_name]
                    pygame.draw.line(self.screen, (255, 255, 255),
                                     (int(t1['x']), int(t1['y'])),
                                     (int(t2['x']), int(t2['y'])), 1)
                    if self.show_runway_names:
                        for t_name, t in [(t1_name, t1), (t2_name, t2)]:
                            if t_name[:2].isdigit():
                                heading = int(t_name[:2])
                                angle_rad = math.radians((heading * 10 + 180) % 360)
                                offset_dist = 50
                                offset_x = offset_dist * math.sin(angle_rad)
                                offset_y = offset_dist * math.cos(angle_rad)
                            else:
                                offset_x, offset_y = 40, 0
                            text = self.font.render(t_name, True, (200, 200, 200))
                            text_rect = text.get_rect(center=(int(t['x'] + offset_x), int(t['y'] + offset_y)))
                            self.screen.blit(text, text_rect)
    
    def draw_vor_stations(self):
        if self.show_vor:
            for vor_id, vor_data in self.data['vor_stations'].items():
                x = vor_data['coordinates']['x']
                y = vor_data['coordinates']['y']
                pygame.draw.circle(self.screen, (0, 100, 255), (int(x), int(y)), 4)
                
                text = self.font.render(vor_id, True, (0, 100, 255))
                text_rect = text.get_rect(center=(int(x), int(y - 20)))
                self.screen.blit(text, text_rect)
    
    def draw_ndb_stations(self):
        if self.show_ndb:
            for ndb_id, ndb_data in self.data['ndb_stations'].items():
                x = ndb_data['coordinates']['x']
                y = ndb_data['coordinates']['y']
                pygame.draw.circle(self.screen, (255, 0, 255), (int(x), int(y)), 4)
                
                pygame.draw.circle(self.screen, (255, 0, 255), (int(x), int(y)), 8, 2)
                
                text = self.font.render(ndb_id, True, (255, 0, 255))
                text_rect = text.get_rect(center=(int(x), int(y - 20)))
                self.screen.blit(text, text_rect)
    
    def draw_waypoints(self):
        if self.show_waypoints:
            for wpt_id, wpt_data in self.data['rnav_waypoints'].items():
                x = wpt_data['coordinates']['x']
                y = wpt_data['coordinates']['y']
                
                size = 6
                points = [
                    (x - size, y),
                    (x, y - size),
                    (x + size, y),
                    (x, y + size)
                ]
                pygame.draw.polygon(self.screen, (255, 255, 0), points, 2)
                
                text = self.font.render(wpt_id, True, (255, 255, 0))
                text_rect = text.get_rect(center=(int(x), int(y - 18)))
                self.screen.blit(text, text_rect)

    def draw_aircraft(self, delta_t):
        aircraft_list = list(self.aircraft_list.values())
        self.collision_monitor.check_collisions(aircraft_list)
        self.has_violation = False
        
        for aircraft in aircraft_list:
            if aircraft.crash is not None:
                self.crash_occurred = True
                self.crash_message = f"CRASH: {aircraft.callsign} {aircraft.crash}"
                break
        
        to_remove = []
        for aircraft in self.aircraft_list.values():
            if not self.crash_occurred:
                aircraft.update(delta_t, self.fast_forward)
            
            info = aircraft.get_info()

            x, y = info['position'][0], info['position'][1]

            if x < 0 or x > self.screen_width or y < 0 or y > self.screen_height:
                to_remove.append(info['callsign'])
                self.improper_exits += 1
                continue

            if info['landed']:
                to_remove.append(info['callsign'])
                self.num_landed += 1
                continue

            traj = info['trajectory']
            n = len(traj)
            for i, (x, y) in enumerate(traj):
                alpha = int(255 * (i + 1) / n) if n > 1 else 255
                color = (255, 255, 255, alpha)
                surf = pygame.Surface((6, 6), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (3, 3), 3)
                self.screen.blit(surf, (int(x) - 3, int(y) - 3))
            
            aircraft_color = (255, 255, 255)
            if aircraft.collision_warning:
                aircraft_color = (255, 0, 0)
            
            pygame.draw.circle(self.screen, aircraft_color, (int(x), int(y)), 5)
            if aircraft.selected:
                pygame.draw.circle(self.screen, (255, 255, 0), (int(x), int(y)), 7, 2)

            tag = ""
            if info:
                cs = info['callsign']
                ias = int(info['airspeed'])
                alt = int(info['altitude']//100)
                hdg = int(info['heading'])
                tgt_ias = int(info['target_airspeed'])
                tgt_alt = int(info['target_altitude']//100)
                tgt_hdg = int(info['target_heading'])
                wpt = info['target_wpt']
                rwy = info['runway']
                loc = info['loc']
                gs = info['gs']
                holding = info['holding']
                ias_str = f"{ias}->{tgt_ias}" if ias != tgt_ias else f"{ias}"
                alt_str = f"{alt}->{tgt_alt}" if alt != tgt_alt else f"{alt}"
                hdg_str = f"{hdg:03d}->{tgt_hdg:03d}" if hdg != tgt_hdg else f"{hdg:03d}"

                if holding:
                    tag = f"{cs} {ias_str} {alt_str} H"
                elif wpt:
                    tag = f"{cs} {ias_str} {alt_str} {wpt}"
                elif rwy:
                    loc_str = "LOC" if loc else "loc"
                    gs_str = "GS" if gs else "gs"
                    tag = f"{cs} {ias_str} {alt_str} {hdg_str} {rwy} {loc_str} {gs_str}"
                else:
                    tag = f"{cs} {ias_str} {alt_str} {hdg_str}"
            
            text_color = (255, 255, 255)
            if info['collision_warning']:
                text_color = (255, 0, 0)
                self.has_violation = True
            
            if self.aircraft_details:
                text = self.font.render(tag, True, text_color)
            else:
                text = self.font.render(cs, True, text_color)
            text_rect = text.get_rect(center=(int(x), int(y - 18)))
            self.screen.blit(text, text_rect)

        for cs in to_remove:
            del self.aircraft_list[cs]

    def draw_textbox(self):
        textbox_font = pygame.font.Font(None, 28)
        textbox_x = 20
        textbox_y = self.radar_height + 10
        textbox_w = self.screen_width - 40
        textbox_h = 30
        color = (255, 255, 255) if self.textbox_locked else (100, 100, 100)
        rect = pygame.Rect(textbox_x, textbox_y, textbox_w, textbox_h)
        pygame.draw.rect(self.screen, (30, 30, 30), rect)
        pygame.draw.rect(self.screen, color, rect, 2)
        txt_surface = textbox_font.render(self.textbox_text, True, color)
        self.screen.blit(txt_surface, (rect.x + 8, rect.y + 4))
        status = "LOCKED" if self.textbox_locked else "UNLOCKED"
        status_surface = self.font.render(status, True, color)
        self.screen.blit(status_surface, (rect.x, rect.y - 22))

    def draw_misc_display(self):
        spawn_text = f"Spawn rate: {self.spawn_rate}s/aircraft"
        text_surface = self.font.render(spawn_text, True, (200, 200, 200))
        text_rect = text_surface.get_rect()
        text_rect.topright = (self.screen_width - 10, 10)
        self.screen.blit(text_surface, text_rect)
        
        directions_order = ['N', 'S', 'W', 'E']
        directions_display = []
        for direction in directions_order:
            if direction in self.spawn_directions:
                directions_display.append(direction)
        
        directions_text = f"Spawn direction(s): {' '.join(directions_display)}"
        directions_surface = self.font.render(directions_text, True, (200, 200, 200))
        directions_rect = directions_surface.get_rect()
        directions_rect.topright = (self.screen_width - 10, 30)
        self.screen.blit(directions_surface, directions_rect)

        scoring_text = f"Total aircrafts landed: {self.num_landed}, violation: {self.violation_seconds}s, improper exits: {self.improper_exits}"
        scoring_surface = self.font.render(scoring_text, True, (200, 200, 200))
        scoring_rect = scoring_surface.get_rect()
        scoring_rect.topright = (self.screen_width - 10, 50)
        self.screen.blit(scoring_surface, scoring_rect)

        if self.fast_forward - 1:
            fast_forward_text = f"X{self.fast_forward}"
            fast_forward_surface = self.font.render(fast_forward_text, True, (255, 255, 0))
            fast_forward_rect = fast_forward_surface.get_rect()
            fast_forward_rect.topleft = (10, 10)
            self.screen.blit(fast_forward_surface, fast_forward_rect)

    def draw_radar_separator(self):
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (0, self.radar_height), 
                        (self.screen_width, self.radar_height), 2)

    def draw_crash_message(self):
        if self.crash_occurred:
            overlay = pygame.Surface((self.screen_width, self.radar_height))
            overlay.set_alpha(128)
            overlay.fill((255, 0, 0))
            self.screen.blit(overlay, (0, 0))
            
            box_width = 600
            box_height = 100
            box_x = (self.screen_width - box_width) // 2
            box_y = (self.radar_height - box_height) // 2
            
            pygame.draw.rect(self.screen, (200, 0, 0), 
                           (box_x, box_y, box_width, box_height))
            
            pygame.draw.rect(self.screen, (0, 0, 0), 
                           (box_x, box_y, box_width, box_height), 3)
            
            crash_font = pygame.font.Font(None, 36)
            crash_text = crash_font.render(self.crash_message, True, (255, 255, 255))
            crash_rect = crash_text.get_rect(center=(box_x + box_width // 2, box_y + box_height // 2))
            self.screen.blit(crash_text, crash_rect)

    def update(self, delta_t):
        delta_t *= self.fast_forward

        self.screen.fill((25, 25, 25))
        
        radar_clip = pygame.Rect(0, 0, self.screen_width, self.radar_height)
        self.screen.set_clip(radar_clip)
        
        self.draw_radar_rings()
        self.draw_airport()
        self.draw_vor_stations()
        self.draw_ndb_stations()
        self.draw_waypoints()
        self.draw_aircraft(delta_t)

        self.screen.set_clip(None)
        
        self.draw_crash_message()
        
        self.draw_radar_separator()
        self.draw_textbox()
        self.draw_misc_display()

        if not self.crash_occurred:
            aircraft = self.spawner.update(delta_t)
            if aircraft:
                if aircraft.callsign not in self.aircraft_list:
                    self.aircraft_list[aircraft.callsign] = aircraft
                else:
                    while True:
                        new_aircraft = self.spawner.spawn_aircraft()
                        if new_aircraft.callsign not in self.aircraft_list:
                            self.aircraft_list[new_aircraft.callsign] = new_aircraft
                            break
        
        if self.has_violation:
            self.violation_seconds += delta_t
    
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                if self.fast_forward == 1:
                    self.fast_forward = 10
                    return
                self.fast_forward = 1
            if not self.textbox_locked:
                if event.key == pygame.K_l:
                    self.textbox_locked = True
                    self.textbox_text = ""
                    return
                if event.key == pygame.K_a:
                    self.show_airport = not self.show_airport
                elif event.key == pygame.K_r:
                    self.show_radar_rings = not self.show_radar_rings
                elif event.key == pygame.K_v:
                    self.show_vor = not self.show_vor
                elif event.key == pygame.K_n:
                    self.show_ndb = not self.show_ndb
                elif event.key == pygame.K_w:
                    self.show_waypoints = not self.show_waypoints
                elif event.key == pygame.K_u:
                    self.show_runway_names = not self.show_runway_names
                elif event.key == pygame.K_d:
                    self.aircraft_details = not self.aircraft_details
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.spawn_rate += 10
                    self.spawner.spawn_rate = self.spawn_rate
                elif event.key == pygame.K_MINUS:
                    if self.spawn_rate > 20:
                        self.spawn_rate -= 10
                        self.spawner.spawn_rate = self.spawn_rate
                elif event.key == pygame.K_UP:
                    if 'N' in self.spawn_directions and len(self.spawn_directions) > 1:
                        self.spawn_directions.remove('N')
                    else:
                        self.spawn_directions.append('N')
                    self.spawner.spawn_directions = self.spawn_directions
                elif event.key == pygame.K_DOWN:
                    if 'S' in self.spawn_directions and len(self.spawn_directions) > 1:
                        self.spawn_directions.remove('S')
                    else:
                        self.spawn_directions.append('S')
                    self.spawner.spawn_directions = self.spawn_directions
                elif event.key == pygame.K_LEFT:
                    if 'W' in self.spawn_directions and len(self.spawn_directions) > 1:
                        self.spawn_directions.remove('W')
                    else:
                        self.spawn_directions.append('W')
                    self.spawner.spawn_directions = self.spawn_directions
                elif event.key == pygame.K_RIGHT:
                    if 'E' in self.spawn_directions and len(self.spawn_directions) > 1:
                        self.spawn_directions.remove('E')
                    else:
                        self.spawn_directions.append('E')
                    self.spawner.spawn_directions = self.spawn_directions
            else:
                if event.key == pygame.K_TAB:
                    return
                if event.key == pygame.K_RETURN:
                    for aircraft in self.aircraft_list.values():
                        aircraft.selected = False
                    if self.textbox_text == "unlock":
                        self.textbox_locked = False
                        self.textbox_text = ""
                    else:
                        parts = self.textbox_text.strip().upper().split()
                        if len(parts) > 1:
                            callsign = parts[0]
                            if callsign in self.aircraft_list:
                                command_part = ' '.join(parts[1:])
                                self.aircraft_list[callsign].process_command(command_part)
                        self.textbox_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    self.textbox_text = self.textbox_text[:-1]
                else:
                    self.textbox_text += event.unicode
                    
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.textbox_locked:
                mouse_x, mouse_y = event.pos
                if mouse_y < self.radar_height:
                    for aircraft in self.aircraft_list.values():
                        dx = mouse_x - int(aircraft.x)
                        dy = mouse_y - int(aircraft.y)
                        if dx * dx + dy * dy <= 50:
                            self.textbox_text = aircraft.callsign.upper() + " "
                            aircraft.selected = True
                        else:
                            aircraft.selected = False