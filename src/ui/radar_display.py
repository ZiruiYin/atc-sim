import pygame
import math
from typing import Dict, List, Tuple, Optional

from core.airport import Airport
from core.navigation import Navigation
from core.aircraft import Aircraft
from core.aircraft_spawner import AircraftSpawner
from core.collision import CollisionDetector
from utils.math_utils import lat_lon_to_pixels, distance_between_points


class RadarDisplay:
    def __init__(self, screen_width: int = None, screen_height: int = None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scale_factor = 800  # Pixels per degree (zoom level)
        self.airport: Optional[Airport] = None
        self.navigation: Optional[Navigation] = None
        self.aircraft: List[Aircraft] = []
        self.aircraft_spawner = AircraftSpawner(screen_width, screen_height)
        
        # Colors
        self.COLOR_BACKGROUND = (0, 20, 0)  # Dark green
        self.COLOR_RUNWAY = (80, 80, 80)    # Gray
        self.COLOR_RUNWAY_OUTLINE = (200, 200, 200)  # Light gray
        self.COLOR_AIRPORT = (255, 255, 255)  # White
        self.COLOR_VOR = (0, 200, 255)       # Cyan
        self.COLOR_NDB = (255, 100, 100)     # Red
        self.COLOR_WAYPOINT = (255, 255, 0)  # Yellow
        self.COLOR_TEXT = (255, 255, 255)    # White
        self.COLOR_GRID = (0, 40, 0)         # Dark green
        self.COLOR_SID = (0, 255, 128)       # Green
        self.COLOR_STAR = (255, 128, 0)      # Orange
        self.COLOR_AIRCRAFT = (255, 255, 255)  # White
        self.COLOR_TRAJECTORY = (100, 100, 100)  # Gray
        self.COLOR_ILS = (255, 0, 255)       # Magenta
        self.COLOR_SELECTED = (255, 255, 0)  # Yellow for selected aircraft
        self.COLOR_WARNING = (255, 0, 0)     # Red for collision warnings
        
        # Fonts
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.font_callsign = None
        
        # Display options
        self.show_vor = True
        self.show_ndb = True
        self.show_waypoints = False
        self.show_grid = False
        self.show_range_rings = True
        self.show_airport_name = False
        self.show_ils = False
        self.show_legend = True
        self.show_info_panels = True
        self.current_procedure_index = -1  # -1 = OFF, 0+ = procedure index
        
        # Command input and lock
        self.command_input = ""
        self.is_locked = False
        
        # Aircraft selection
        self.selected_aircraft = None
        
        # Collision detection
        self.collision_detector = CollisionDetector()
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("ATC Simulator - EGLL Radar Display")
        
        # Initialize fonts
        self.font_large = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 14)
        self.font_callsign = pygame.font.Font(None, 20)  # Bigger font for callsigns
    
    def set_airport(self, airport: Airport):
        """Set the airport to display"""
        self.airport = airport
        print(f"Radar display set to airport: {airport.name}")
    
    def set_navigation(self, navigation: Navigation):
        """Set the navigation data to display"""
        self.navigation = navigation
        print(f"Navigation data loaded for radar display")
    
    def set_spawn_rate(self, aircraft_per_minute: float):
        """Set the aircraft spawn rate"""
        self.aircraft_spawner.set_spawn_rate(aircraft_per_minute)
        print(f"Aircraft spawn rate set to {aircraft_per_minute} aircraft per minute")
    
    def find_aircraft_at_position(self, mouse_x: int, mouse_y: int) -> Optional[Aircraft]:
        """Find aircraft at the given mouse position"""
        for aircraft in self.aircraft:
            x, y = aircraft.get_position()
            # Check if mouse click is within aircraft circle (radius 6 pixels)
            distance = math.sqrt((mouse_x - x) ** 2 + (mouse_y - y) ** 2)
            if distance <= 6:
                return aircraft
        return None
    
    def draw_grid(self):
        """Draw coordinate grid"""
        if not self.show_grid or not self.airport:
            return
        
        center_lat, center_lon = self.airport.get_coordinates()
        
        # Draw grid lines every 0.05 degrees (roughly 3 nm)
        grid_spacing = 0.05
        
        # Vertical lines
        for i in range(-10, 11):
            lon = center_lon + i * grid_spacing
            x, _ = lat_lon_to_pixels(center_lat, lon, center_lat, center_lon, 
                                   self.screen_width, self.screen_height, self.scale_factor)
            if 0 <= x <= self.screen_width:
                pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_height))
        
        # Horizontal lines
        for i in range(-10, 11):
            lat = center_lat + i * grid_spacing
            _, y = lat_lon_to_pixels(lat, center_lon, center_lat, center_lon, 
                                   self.screen_width, self.screen_height, self.scale_factor)
            if 0 <= y <= self.screen_height:
                pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y))
    
    def draw_range_rings(self):
        """Draw range rings around airport"""
        if not self.show_range_rings or not self.airport:
            return
        
        center_lat, center_lon = self.airport.get_coordinates()
        center_x, center_y = lat_lon_to_pixels(center_lat, center_lon, center_lat, center_lon,
                                             self.screen_width, self.screen_height, self.scale_factor)
        
        # Draw rings at 5, 10, 15, 20 NM
        ring_distances = [5, 10, 15, 20]  # nautical miles
        
        for distance in ring_distances:
            # Calculate radius in pixels (approximate)
            radius_pixels = int(distance * self.scale_factor / 60)  # 60 nm per degree approx
            
            if radius_pixels > 0 and radius_pixels < max(self.screen_width, self.screen_height):
                pygame.draw.circle(self.screen, self.COLOR_GRID, (center_x, center_y), radius_pixels, 1)
                
                # Draw distance label
                text = self.font_small.render(f"{distance}nm", True, self.COLOR_GRID)
                self.screen.blit(text, (center_x + radius_pixels - 20, center_y - 10))
    
    def draw_airport(self):
        """Draw airport symbol and info"""
        if not self.airport:
            return
        
        center_lat, center_lon = self.airport.get_coordinates()
        x, y = lat_lon_to_pixels(center_lat, center_lon, center_lat, center_lon,
                               self.screen_width, self.screen_height, self.scale_factor)
        
        # Draw airport symbol (circle)
        pygame.draw.circle(self.screen, self.COLOR_AIRPORT, (x, y), 8, 2)
        
        # Draw airport name only if enabled
        if self.show_airport_name:
            text = self.font_medium.render(f"{self.airport.icao}", True, self.COLOR_AIRPORT)
            self.screen.blit(text, (x + 12, y - 8))
    
    def draw_runway(self, runway_name: str):
        """Draw a single runway"""
        if not self.airport:
            return
        
        runway = self.airport.get_runway(runway_name)
        if not runway:
            return
        
        center_lat, center_lon = self.airport.get_coordinates()
        
        # Get runway coordinates
        threshold_coords = runway.get_threshold_coords()
        end_coords = runway.get_end_coords()
        
        # Convert to screen coordinates
        x1, y1 = lat_lon_to_pixels(threshold_coords[0], threshold_coords[1], 
                                 center_lat, center_lon, self.screen_width, self.screen_height, self.scale_factor)
        x2, y2 = lat_lon_to_pixels(end_coords[0], end_coords[1], 
                                 center_lat, center_lon, self.screen_width, self.screen_height, self.scale_factor)
        
        # Draw runway (thinner)
        pygame.draw.line(self.screen, self.COLOR_RUNWAY, (x1, y1), (x2, y2), 3)
        pygame.draw.line(self.screen, self.COLOR_RUNWAY_OUTLINE, (x1, y1), (x2, y2), 5)
        
        # Map runway names to display labels
        runway_labels = {
            "09L": "09L",
            "09R": "09R", 
            "27R": "27R",
            "27L": "27L"
        }
        
        # Draw runway labels at both ends
        # Calculate the direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            # Normalize direction
            dx /= length
            dy /= length
            
            # Offset for label placement
            offset = 30
            
            # Draw labels based on runway name
            if runway_name in runway_labels:
                label = runway_labels[runway_name]
                
                # Label at first end (threshold)
                text1 = self.font_small.render(label, True, self.COLOR_TEXT)
                self.screen.blit(text1, (x1 - dx*offset - 10, y1 - dy*offset - 8))
                
                # Label at second end (opposite direction)
                opposite_label = self.get_opposite_runway_label(label)
                text2 = self.font_small.render(opposite_label, True, self.COLOR_TEXT)
                self.screen.blit(text2, (x2 + dx*offset - 10, y2 + dy*offset - 8))
    
    def get_opposite_runway_label(self, runway_label: str) -> str:
        """Get the opposite runway label"""
        opposites = {
            "09L": "27R",
            "09R": "27L",
            "27R": "09L", 
            "27L": "09R"
        }
        return opposites.get(runway_label, runway_label)
    
    def draw_runways(self):
        """Draw all runways"""
        if not self.airport:
            return
        
        for runway_name in self.airport.get_all_runways():
            self.draw_runway(runway_name)
    
    def draw_vor_stations(self):
        """Draw VOR stations"""
        if not self.show_vor or not self.navigation or not self.airport:
            return
        
        center_lat, center_lon = self.airport.get_coordinates()
        
        for vor_id, vor_station in self.navigation.get_all_vor_stations().items():
            vor_coords = vor_station.get_coordinates()
            x, y = lat_lon_to_pixels(vor_coords[0], vor_coords[1], center_lat, center_lon,
                                   self.screen_width, self.screen_height, self.scale_factor)
            
            # Only draw if on screen
            if 0 <= x <= self.screen_width and 0 <= y <= self.screen_height:
                # Draw VOR symbol (hexagon)
                points = []
                for i in range(6):
                    angle = i * math.pi / 3
                    px = x + 8 * math.cos(angle)
                    py = y + 8 * math.sin(angle)
                    points.append((px, py))
                
                pygame.draw.polygon(self.screen, self.COLOR_VOR, points, 2)
                
                # Draw VOR label
                text = self.font_small.render(f"{vor_id}", True, self.COLOR_VOR)
                self.screen.blit(text, (x + 12, y - 8))
                
                # Draw frequency
                freq_text = self.font_small.render(f"{vor_station.frequency:.1f}", True, self.COLOR_VOR)
                self.screen.blit(freq_text, (x + 12, y + 5))
    
    def draw_ndb_stations(self):
        """Draw NDB stations"""
        if not self.show_ndb or not self.navigation or not self.airport:
            return
        
        center_lat, center_lon = self.airport.get_coordinates()
        
        for ndb_id, ndb_station in self.navigation.get_all_ndb_stations().items():
            ndb_coords = ndb_station.get_coordinates()
            x, y = lat_lon_to_pixels(ndb_coords[0], ndb_coords[1], center_lat, center_lon,
                                   self.screen_width, self.screen_height, self.scale_factor)
            
            # Only draw if on screen
            if 0 <= x <= self.screen_width and 0 <= y <= self.screen_height:
                # Draw NDB symbol (circle with dot)
                pygame.draw.circle(self.screen, self.COLOR_NDB, (x, y), 6, 2)
                pygame.draw.circle(self.screen, self.COLOR_NDB, (x, y), 2)
                
                # Draw NDB label (to the left)
                text = self.font_small.render(f"{ndb_id}", True, self.COLOR_NDB)
                text_width = text.get_width()
                self.screen.blit(text, (x - text_width - 10, y - 8))
                
                # Draw frequency (to the left)
                freq_text = self.font_small.render(f"{ndb_station.frequency}", True, self.COLOR_NDB)
                freq_width = freq_text.get_width()
                self.screen.blit(freq_text, (x - freq_width - 10, y + 5))
    
    def draw_waypoints(self):
        """Draw RNAV waypoints"""
        if not self.show_waypoints or not self.navigation or not self.airport:
            return
        
        center_lat, center_lon = self.airport.get_coordinates()
        
        for waypoint_id, waypoint in self.navigation.get_all_rnav_waypoints().items():
            waypoint_coords = waypoint.get_coordinates()
            x, y = lat_lon_to_pixels(waypoint_coords[0], waypoint_coords[1], center_lat, center_lon,
                                   self.screen_width, self.screen_height, self.scale_factor)
            
            # Only draw if on screen
            if 0 <= x <= self.screen_width and 0 <= y <= self.screen_height:
                # Draw waypoint symbol (triangle)
                points = [(x, y-6), (x-5, y+4), (x+5, y+4)]
                pygame.draw.polygon(self.screen, self.COLOR_WAYPOINT, points, 2)
                
                # Draw waypoint label
                text = self.font_small.render(f"{waypoint_id}", True, self.COLOR_WAYPOINT)
                self.screen.blit(text, (x + 8, y - 8))
    
    def draw_ils_approaches(self):
        """Draw ILS approach paths with intercept angles and altitude constraints"""
        if not self.show_ils or not self.airport:
            return
        
        center_lat, center_lon = self.airport.get_coordinates()
        
        # ILS approach parameters
        ils_approach_distance = 15  # Draw approaches up to 15 NM
        pixels_per_nm = self.scale_factor / 60  # Pixels per nautical mile
        approach_distance_pixels = ils_approach_distance * pixels_per_nm

        # Define constraint points (Distance in NM, Height AGL in ft)
        altitude_constraints = [
            (5, 1590),
            (10, 3180),
            (15, 4770)
        ]
        
        # Draw ILS for each runway
        for runway_name in self.airport.get_all_runways():
            runway = self.airport.get_runway(runway_name)
            if not runway:
                continue
                
            # Get runway threshold coordinates
            threshold_coords = runway.get_threshold_coords()
            end_coords = runway.get_end_coords()
            
            # Convert to screen coordinates
            threshold_x, threshold_y = lat_lon_to_pixels(threshold_coords[0], threshold_coords[1], 
                                                       center_lat, center_lon, self.screen_width, 
                                                       self.screen_height, self.scale_factor)
            end_x, end_y = lat_lon_to_pixels(end_coords[0], end_coords[1], 
                                           center_lat, center_lon, self.screen_width, 
                                           self.screen_height, self.scale_factor)
            
            # Calculate runway direction (approach is opposite direction)
            runway_dx = end_x - threshold_x
            runway_dy = end_y - threshold_y
            runway_length = math.sqrt(runway_dx*runway_dx + runway_dy*runway_dy)
            
            if runway_length > 0:
                # Normalize runway direction
                runway_unit_x = runway_dx / runway_length
                runway_unit_y = runway_dy / runway_length
                
                # Approach direction is opposite to runway direction
                approach_unit_x = -runway_unit_x
                approach_unit_y = -runway_unit_y
                
                # Calculate ILS centerline end point
                centerline_end_x = threshold_x + approach_unit_x * approach_distance_pixels
                centerline_end_y = threshold_y + approach_unit_y * approach_distance_pixels
                
                # Draw ILS centerline
                pygame.draw.line(self.screen, self.COLOR_ILS, 
                               (threshold_x, threshold_y), 
                               (centerline_end_x, centerline_end_y), 2)
                
                # Draw altitude constraints with better positioning
                for distance_nm, altitude_agl in altitude_constraints:
                    distance_pixels = distance_nm * pixels_per_nm
                    
                    # Calculate point along the centerline
                    point_x = threshold_x + approach_unit_x * distance_pixels
                    point_y = threshold_y + approach_unit_y * distance_pixels
                    
                    # Draw altitude constraint marker
                    pygame.draw.circle(self.screen, self.COLOR_ILS, (int(point_x), int(point_y)), 3)
    
    def draw_procedures(self):
        """Draw currently selected SID or STAR procedure"""
        if self.current_procedure_index == -1 or not self.navigation or not self.airport:
            return
        
        center_lat, center_lon = self.airport.get_coordinates()
        
        # Get all procedures in order
        all_procedures = []
        
        # Add SID procedures first
        for sid_id, sid_procedure in self.navigation.get_all_sid_procedures().items():
            all_procedures.append((sid_procedure, self.COLOR_SID, "SID"))
        
        # Add STAR procedures
        for star_id, star_procedure in self.navigation.get_all_star_procedures().items():
            all_procedures.append((star_procedure, self.COLOR_STAR, "STAR"))
        
        # Draw the currently selected procedure
        if 0 <= self.current_procedure_index < len(all_procedures):
            procedure, color, proc_type = all_procedures[self.current_procedure_index]
            self.draw_procedure_route(procedure, center_lat, center_lon, color)
    
    def draw_procedure_route(self, procedure, center_lat: float, center_lon: float, color: tuple):
        """Draw a single procedure route"""
        route_points = []
        
        for waypoint_name in procedure.route:
            # Check if waypoint is EGLL (airport)
            if waypoint_name == "EGLL":
                airport_coords = self.airport.get_coordinates()
                x, y = lat_lon_to_pixels(airport_coords[0], airport_coords[1], center_lat, center_lon,
                                       self.screen_width, self.screen_height, self.scale_factor)
                route_points.append((x, y))
            else:
                # Check in RNAV waypoints
                waypoint = self.navigation.get_rnav_waypoint(waypoint_name)
                if waypoint:
                    waypoint_coords = waypoint.get_coordinates()
                    x, y = lat_lon_to_pixels(waypoint_coords[0], waypoint_coords[1], center_lat, center_lon,
                                           self.screen_width, self.screen_height, self.scale_factor)
                    route_points.append((x, y))
                else:
                    # Check in VOR stations
                    vor_station = self.navigation.get_vor_station(waypoint_name)
                    if vor_station:
                        vor_coords = vor_station.get_coordinates()
                        x, y = lat_lon_to_pixels(vor_coords[0], vor_coords[1], center_lat, center_lon,
                                               self.screen_width, self.screen_height, self.scale_factor)
                        route_points.append((x, y))
        
        # Draw the route as connected lines
        if len(route_points) > 1:
            for i in range(len(route_points) - 1):
                start_point = route_points[i]
                end_point = route_points[i + 1]
                
                # Only draw if both points are on screen
                if (0 <= start_point[0] <= self.screen_width and 0 <= start_point[1] <= self.screen_height and
                    0 <= end_point[0] <= self.screen_width and 0 <= end_point[1] <= self.screen_height):
                    pygame.draw.line(self.screen, color, start_point, end_point, 2)
            
            # Draw procedure name at the midpoint
            if len(route_points) >= 2:
                mid_point = route_points[len(route_points) // 2]
                text = self.font_small.render(procedure.name, True, color)
                self.screen.blit(text, (mid_point[0] + 5, mid_point[1] - 15))
    
    def draw_info_panel(self):
        """Draw information panel"""
        if not self.airport:
            return
        
        # Background for info panel
        info_rect = pygame.Rect(10, 10, 300, 140)
        pygame.draw.rect(self.screen, (0, 0, 0), info_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, info_rect, 2)
        
        # Airport info
        y_offset = 20
        info_lines = [
            f"Airport: {self.airport.name}",
            f"ICAO: {self.airport.icao} / IATA: {self.airport.iata}",
            f"Coordinates: {self.airport.latitude:.4f}, {self.airport.longitude:.4f}",
            f"Elevation: {self.airport.elevation} ft",
            f"Runways: {len(self.airport.runways)}"
        ]
        
        # Add current procedure info
        if self.current_procedure_index == -1:
            info_lines.append("Procedure: OFF")
        else:
            # Get current procedure
            all_procedures = []
            if self.navigation:
                for sid_id, sid_procedure in self.navigation.get_all_sid_procedures().items():
                    all_procedures.append((sid_procedure, self.COLOR_SID, "SID"))
                for star_id, star_procedure in self.navigation.get_all_star_procedures().items():
                    all_procedures.append((star_procedure, self.COLOR_STAR, "STAR"))
                
                if 0 <= self.current_procedure_index < len(all_procedures):
                    procedure, color, proc_type = all_procedures[self.current_procedure_index]
                    info_lines.append(f"Procedure: {proc_type} {procedure.name}")
        
        for line in info_lines:
            text = self.font_small.render(line, True, self.COLOR_TEXT)
            self.screen.blit(text, (20, y_offset))
            y_offset += 18
    
    def draw_aircraft(self):
        """Draw all aircraft"""
        for aircraft in self.aircraft:
            if aircraft.is_on_screen(self.screen_width, self.screen_height):
                self.draw_single_aircraft(aircraft)
    
    def draw_single_aircraft(self, aircraft: Aircraft):
        """Draw a single aircraft with trajectory and flight data"""
        x, y = aircraft.get_position()
        
        # Draw trajectory trail
        self.draw_trajectory(aircraft)
        
        # Draw localizer guidance line if cleared for ILS
        self.draw_localizer_guidance(aircraft)
        
        # Choose color based on selection status and collision warnings
        if self.collision_detector.is_aircraft_in_warning(aircraft.callsign):
            aircraft_color = self.COLOR_WARNING
        elif aircraft == self.selected_aircraft:
            aircraft_color = self.COLOR_SELECTED
        else:
            aircraft_color = self.COLOR_AIRCRAFT
        
        # Draw aircraft symbol (circle)
        pygame.draw.circle(self.screen, aircraft_color, (int(x), int(y)), 6, 2)
        pygame.draw.circle(self.screen, aircraft_color, (int(x), int(y)), 3)
        
        # Draw selection highlight for selected aircraft
        if aircraft == self.selected_aircraft:
            pygame.draw.circle(self.screen, self.COLOR_SELECTED, (int(x), int(y)), 12, 2)
        
        # Draw target indicators
        self.draw_target_indicators(aircraft, x, y)
        
        # Draw flight data
        self.draw_flight_data(aircraft, x, y)
    
    def draw_localizer_guidance(self, aircraft: Aircraft):
        """Draw dashed guidance line for localizer interception"""
        guidance_line = aircraft.get_localizer_guidance_line()
        
        if len(guidance_line) == 2:
            start_point = guidance_line[0]
            end_point = guidance_line[1]
            
            # Draw dashed line
            self.draw_dashed_line(start_point, end_point, self.COLOR_AIRCRAFT, 2, 5)
    
    def draw_dashed_line(self, start_point: tuple, end_point: tuple, color: tuple, width: int, dash_length: int):
        """Draw a dashed line between two points"""
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Calculate line parameters
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance == 0:
            return
        
        # Unit vector
        unit_x = dx / distance
        unit_y = dy / distance
        
        # Draw dashed line
        current_distance = 0
        while current_distance < distance:
            # Calculate dash start point
            dash_start_x = x1 + current_distance * unit_x
            dash_start_y = y1 + current_distance * unit_y
            
            # Calculate dash end point
            dash_end_distance = min(current_distance + dash_length, distance)
            dash_end_x = x1 + dash_end_distance * unit_x
            dash_end_y = y1 + dash_end_distance * unit_y
            
            # Draw the dash
            pygame.draw.line(self.screen, color, 
                           (int(dash_start_x), int(dash_start_y)), 
                           (int(dash_end_x), int(dash_end_y)), width)
            
            # Move to next dash (skip gap)
            current_distance += dash_length * 2
    
    def draw_trajectory(self, aircraft: Aircraft):
        """Draw aircraft trajectory trail as dots"""
        trajectory = aircraft.get_trajectory()
        
        if len(trajectory) > 1:
            # Draw dots at previous positions (skip current position)
            for i in range(len(trajectory) - 1):
                pos = (int(trajectory[i][0]), int(trajectory[i][1]))
                
                # Fade older positions
                brightness = int(80 + 120 * (i + 1) / len(trajectory))
                dot_color = (brightness, brightness, brightness)
                
                # Draw dot (bigger and more visible)
                pygame.draw.circle(self.screen, dot_color, pos, 3)
                pygame.draw.circle(self.screen, dot_color, pos, 1)
    
    def draw_target_indicators(self, aircraft: Aircraft, x: float, y: float):
        """Draw target heading and altitude indicators"""
        # Draw target heading line
        if abs(aircraft.heading - aircraft.target_heading) > 1:
            target_rad = math.radians(aircraft.target_heading)
            line_length = 30
            end_x = x + line_length * math.sin(target_rad)
            end_y = y - line_length * math.cos(target_rad)
            pygame.draw.line(self.screen, (255, 255, 0), (x, y), (end_x, end_y), 2)
        
        # Draw altitude change indicator
        if abs(aircraft.altitude - aircraft.target_altitude) > 10:
            if aircraft.altitude < aircraft.target_altitude:
                # Climbing - draw down arrow (opposite direction)
                arrow_points = [(x + 20, y + 10), (x + 25, y + 5), (x + 30, y + 10)]
                pygame.draw.polygon(self.screen, (0, 255, 0), arrow_points)
            else:
                # Descending - draw up arrow (opposite direction)
                arrow_points = [(x + 20, y - 10), (x + 25, y - 5), (x + 30, y - 10)]
                pygame.draw.polygon(self.screen, (255, 0, 0), arrow_points)
    
    def draw_flight_data(self, aircraft: Aircraft, x: float, y: float):
        """Draw flight data above aircraft"""
        flight_data = aircraft.get_flight_data()
        
        # Show target airspeed if different from current
        airspeed_display = str(flight_data['airspeed'])
        if abs(flight_data['airspeed'] - flight_data['target_airspeed']) > 1:
            airspeed_display = f"{flight_data['airspeed']}->{flight_data['target_airspeed']}"
        
        # Create compact label: CALLSIGN/IAS/GS/ALT/HEADING (altitude in hundreds, heading 3 digits)
        altitude_hundreds = int(flight_data['altitude'] / 100)
        heading_3digit = f"{flight_data['heading']:03d}"
        
        # Add expedite indicator if active
        expedite_indicator = "X" if flight_data['expedite_altitude'] else ""
        
        # Add ILS status if cleared for ILS
        ils_status = ""
        if flight_data['ils_cleared']:
            loc_status = "LOC" if flight_data['loc_intercepted'] else "loc"
            gs_status = "GS" if flight_data['gs_intercepted'] else "gs"
            runway_info = f" {flight_data['ils_runway']}" if flight_data['ils_runway'] else ""
            ils_status = f" {loc_status} {gs_status}{runway_info}"
        
        label = f"{flight_data['callsign']}/{airspeed_display}/{flight_data['ground_speed']}/{altitude_hundreds}{expedite_indicator}/{heading_3digit}{ils_status}"
        
        # Render text with bigger font for callsign
        text = self.font_callsign.render(label, True, self.COLOR_AIRCRAFT)
        text_rect = text.get_rect()
        
        # Position above aircraft (centered)
        text_x = x - text_rect.width // 2
        text_y = y - 25
        
        # Draw background
        bg_rect = pygame.Rect(text_x - 2, text_y - 2, text_rect.width + 4, text_rect.height + 4)
        pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
        
        # Draw text
        self.screen.blit(text, (text_x, text_y))
    
    def update_aircraft(self, dt: float):
        """Update all aircraft positions and spawn new aircraft"""
        # Check if game is frozen due to crash
        if self.collision_detector.is_game_frozen():
            return
        
        # Spawn new aircraft
        new_aircraft = self.aircraft_spawner.update(self.aircraft)
        for aircraft in new_aircraft:
            self.aircraft.append(aircraft)
        
        # Create a list to track aircraft to remove
        aircraft_to_remove = []
        
        for aircraft in self.aircraft:
            # Set airport center coordinates for navigation
            if self.airport:
                aircraft.airport_center_lat, aircraft.airport_center_lon = self.airport.get_coordinates()
                
                # Set runway data for ILS approaches
                aircraft.airport_data = self.get_runway_data_for_aircraft()
            
            # Set screen dimensions and scale factor for navigation
            aircraft.screen_width = self.screen_width
            aircraft.screen_height = self.screen_height
            aircraft.scale_factor = self.scale_factor
            
            aircraft.update(dt)
            
            # Check if aircraft has landed and should be removed
            if hasattr(aircraft, 'landed') and aircraft.landed:
                aircraft_to_remove.append(aircraft)
            
            # Remove aircraft that have flown too far off screen
            elif not aircraft.is_on_screen(self.screen_width, self.screen_height):
                # Check if aircraft is really far off screen (not just slightly off)
                x, y = aircraft.get_position()
                margin = 200  # Larger margin for removal
                if (x < -margin or x > self.screen_width + margin or 
                    y < -margin or y > self.screen_height + margin):
                    aircraft_to_remove.append(aircraft)
        
        # Remove aircraft that should be removed
        for aircraft in aircraft_to_remove:
            self.aircraft.remove(aircraft)
            if hasattr(aircraft, 'landed') and aircraft.landed:
                print(f"{aircraft.callsign}: Removed from simulation (landed)")
            else:
                print(f"{aircraft.callsign}: Removed from simulation (left radar coverage)")
        
        # Update collision detection
        if self.airport:
            airport_center_lat, airport_center_lon = self.airport.get_coordinates()
            self.collision_detector.update_aircraft_positions(
                self.aircraft, airport_center_lat, airport_center_lon,
                self.screen_width, self.screen_height, self.scale_factor
            )
            
            # Check for collisions
            game_should_continue = self.collision_detector.check_collisions(self.aircraft)
            if not game_should_continue:
                # Crash occurred, game is frozen
                pass
    
    def get_runway_data_for_aircraft(self) -> dict:
        """Get runway data in format expected by aircraft"""
        runway_data = {}
        
        if self.airport:
            for runway_name in self.airport.get_all_runways():
                runway = self.airport.get_runway(runway_name)
                if runway:
                    threshold_coords = runway.get_threshold_coords()
                    end_coords = runway.get_end_coords()
                    
                    runway_data[runway_name] = {
                        'threshold_lat': threshold_coords[0],
                        'threshold_lon': threshold_coords[1],
                        'end_lat': end_coords[0],
                        'end_lon': end_coords[1],
                        'heading': runway.heading
                    }
        
        return runway_data
    
    def draw_legend(self):
        """Draw legend for symbols"""
        legend_rect = pygame.Rect(self.screen_width - 200, 10, 180, 210)
        pygame.draw.rect(self.screen, (0, 0, 0), legend_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, legend_rect, 2)
        
        legend_items = [
            ("Airport", self.COLOR_AIRPORT, "○"),
            ("Runway", self.COLOR_RUNWAY, "━"),
            ("VOR", self.COLOR_VOR, "⬡"),
            ("NDB", self.COLOR_NDB, "●"),
            ("Waypoint", self.COLOR_WAYPOINT, "△"),
            ("ILS", self.COLOR_ILS, "━"),
            ("SID", self.COLOR_SID, "━"),
            ("STAR", self.COLOR_STAR, "━"),
            ("Aircraft", self.COLOR_AIRCRAFT, "●"),
        ]
        
        title_text = self.font_small.render("Legend", True, self.COLOR_TEXT)
        self.screen.blit(title_text, (legend_rect.x + 10, legend_rect.y + 10))
        
        y_offset = legend_rect.y + 35
        for name, color, symbol in legend_items:
            symbol_text = self.font_small.render(symbol, True, color)
            name_text = self.font_small.render(name, True, self.COLOR_TEXT)
            self.screen.blit(symbol_text, (legend_rect.x + 10, y_offset))
            self.screen.blit(name_text, (legend_rect.x + 30, y_offset))
            y_offset += 20
    
    def draw_command_input(self):
        """Draw command input textbox at the bottom"""
        # Check if game is frozen due to crash
        if self.collision_detector.is_game_frozen():
            # Draw crash message instead of command input
            crash_message = self.collision_detector.get_crash_message()
            text_surface = self.font_large.render(crash_message, True, self.COLOR_WARNING)
            text_rect = text_surface.get_rect()
            text_rect.centerx = self.screen_width // 2
            text_rect.y = self.screen_height - 100
            
            # Draw background for crash message
            bg_rect = pygame.Rect(text_rect.x - 20, text_rect.y - 10, text_rect.width + 40, text_rect.height + 20)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
            pygame.draw.rect(self.screen, self.COLOR_WARNING, bg_rect, 3)
            
            self.screen.blit(text_surface, text_rect)
            
            # Draw restart instruction
            restart_text = "Press ESC to exit"
            restart_surface = self.font_medium.render(restart_text, True, self.COLOR_TEXT)
            restart_rect = restart_surface.get_rect()
            restart_rect.centerx = self.screen_width // 2
            restart_rect.y = self.screen_height - 60
            self.screen.blit(restart_surface, restart_rect)
            return
        
        # Command input box
        box_height = 30
        box_y = self.screen_height - box_height - 10
        input_rect = pygame.Rect(10, box_y, self.screen_width - 20, box_height)
        
        # Change color based on lock state
        if self.is_locked:
            border_color = (255, 255, 0)  # Yellow when locked
            bg_color = (40, 40, 0)  # Dark yellow background
        else:
            border_color = self.COLOR_TEXT
            bg_color = (0, 0, 0)
        
        pygame.draw.rect(self.screen, bg_color, input_rect)
        pygame.draw.rect(self.screen, border_color, input_rect, 2)
        
        # Draw prompt and input text
        prompt = "CMD> " if not self.is_locked else "LOCKED> "
        full_text = prompt + self.command_input
        
        # Add cursor if locked (typing mode)
        if self.is_locked:
            full_text += "_"
        
        text_surface = self.font_small.render(full_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 8))
        
        # Draw lock status
        if self.is_locked:
            lock_text = self.font_small.render("[LOCKED]", True, border_color)
        else:
            lock_text = self.font_small.render("[L] UNLOCKED", True, border_color)
        self.screen.blit(lock_text, (input_rect.x + input_rect.width - 100, input_rect.y - 20))
        
        # Draw selected aircraft indicator
        if self.selected_aircraft:
            selected_text = f"Selected: {self.selected_aircraft.callsign}"
            selected_surface = self.font_small.render(selected_text, True, self.COLOR_SELECTED)
            self.screen.blit(selected_surface, (input_rect.x + 5, input_rect.y - 40))
        
        # Draw instructions
        if not self.is_locked:
            help_text = "Press L to lock and type commands"
            help_surface = self.font_small.render(help_text, True, (150, 150, 150))
            self.screen.blit(help_surface, (input_rect.x + 5, input_rect.y - 20))
        else:
            help_text = 'Type "unlock" to unlock display controls'
            help_surface = self.font_small.render(help_text, True, (200, 200, 0))
            self.screen.blit(help_surface, (input_rect.x + 5, input_rect.y - 20))
    
    def draw_spawn_info(self):
        """Draw aircraft spawn information"""
        spawn_info = self.aircraft_spawner.get_spawn_info()
        
        # Display spawn rate and aircraft count
        info_text = f"Spawn Rate: {spawn_info['spawn_rate']:.1f}/min | Aircraft: {len(self.aircraft)} | Next: {spawn_info['next_spawn_in']:.1f}s"
        text = self.font_small.render(info_text, True, self.COLOR_TEXT)
        
        # Position in top-right corner
        text_rect = text.get_rect()
        x = self.screen_width - text_rect.width - 10
        y = 10
        
        # Draw background
        bg_rect = pygame.Rect(x - 5, y - 2, text_rect.width + 10, text_rect.height + 4)
        pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
        
        # Draw text
        self.screen.blit(text, (x, y))
        
        # Draw collision warnings if any
        warning_aircraft = self.collision_detector.get_warning_aircraft()
        if warning_aircraft:
            warning_text = f"COLLISION WARNINGS: {len(warning_aircraft)} aircraft"
            warning_surface = self.font_small.render(warning_text, True, self.COLOR_WARNING)
            warning_rect = warning_surface.get_rect()
            warning_x = self.screen_width - warning_rect.width - 10
            warning_y = y + text_rect.height + 5
            
            # Draw background for warning
            warning_bg_rect = pygame.Rect(warning_x - 5, warning_y - 2, warning_rect.width + 10, warning_rect.height + 4)
            pygame.draw.rect(self.screen, (0, 0, 0), warning_bg_rect)
            pygame.draw.rect(self.screen, self.COLOR_WARNING, warning_bg_rect, 2)
            
            # Draw warning text
            self.screen.blit(warning_surface, (warning_x, warning_y))
    
    def cycle_procedures(self):
        """Cycle through available procedures"""
        if not self.navigation:
            return
        
        # Get all procedures in order
        all_procedures = []
        
        # Add SID procedures first
        for sid_id, sid_procedure in self.navigation.get_all_sid_procedures().items():
            all_procedures.append((sid_procedure, self.COLOR_SID, "SID"))
        
        # Add STAR procedures
        for star_id, star_procedure in self.navigation.get_all_star_procedures().items():
            all_procedures.append((star_procedure, self.COLOR_STAR, "STAR"))
        
        # Cycle to next procedure
        self.current_procedure_index += 1
        if self.current_procedure_index >= len(all_procedures):
            self.current_procedure_index = -1  # Back to OFF
        
        # Display current procedure status
        if self.current_procedure_index == -1:
            print("Procedures: OFF")
        else:
            procedure, color, proc_type = all_procedures[self.current_procedure_index]
            print(f"Procedure: {proc_type} {procedure.name}")
    
    def render(self):
        """Render the complete radar display"""
        # Clear screen
        self.screen.fill(self.COLOR_BACKGROUND)
        
        # Draw grid and range rings
        self.draw_grid()
        self.draw_range_rings()
        
        # Draw airport elements
        self.draw_runways()
        self.draw_airport()
        
        # Draw navigation aids
        self.draw_vor_stations()
        self.draw_ndb_stations()
        self.draw_waypoints()
        self.draw_ils_approaches()
        self.draw_procedures()
        
        # Draw aircraft
        self.draw_aircraft()
        self.draw_spawn_info()
        
        # Draw UI elements
        if self.show_info_panels:
            self.draw_info_panel()
        if self.show_legend:
            self.draw_legend()
        self.draw_command_input()
        
        # Update display
        pygame.display.flip()
    
    def handle_event(self, event):
        """Handle pygame events"""
        # Check if game is frozen due to crash
        if self.collision_detector.is_game_frozen():
            # Only allow ESC key when game is frozen
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True  # Signal to exit
            return False  # Ignore all other events
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Handle mouse clicks for aircraft selection
            if event.button == 1:  # Left click
                mouse_x, mouse_y = event.pos
                clicked_aircraft = self.find_aircraft_at_position(mouse_x, mouse_y)
                
                if clicked_aircraft:
                    # Select the aircraft
                    self.selected_aircraft = clicked_aircraft
                    
                    # If locked (typing mode), load callsign into command input
                    if self.is_locked:
                        self.command_input = clicked_aircraft.callsign + " "
                else:
                    # Deselect if clicking on empty space
                    if self.selected_aircraft:
                        self.selected_aircraft = None
        
        elif event.type == pygame.KEYDOWN:
            # Handle lock (only when unlocked)
            if event.key == pygame.K_l and not self.is_locked:
                self.is_locked = True
                print(f"Display controls: LOCKED")
                return
            
            # Handle command input when locked
            if self.is_locked:
                if event.key == pygame.K_RETURN:
                    # Check if command is "unlock"
                    if self.command_input.strip().lower() == "unlock":
                        self.is_locked = False
                        print(f"Display controls: UNLOCKED")
                        self.command_input = ""
                    else:
                        self.process_command(self.command_input)
                        self.command_input = ""
                        # Deselect aircraft after command is processed
                        if self.selected_aircraft:
                            self.selected_aircraft = None
                elif event.key == pygame.K_BACKSPACE:
                    self.command_input = self.command_input[:-1]
                else:
                    # Add character to command input
                    if event.unicode.isprintable():
                        self.command_input += event.unicode
                return
            
            # Handle display controls when unlocked
            if event.key == pygame.K_v:
                self.show_vor = not self.show_vor
                print(f"VOR display: {'ON' if self.show_vor else 'OFF'}")
            elif event.key == pygame.K_n:
                self.show_ndb = not self.show_ndb
                print(f"NDB display: {'ON' if self.show_ndb else 'OFF'}")
            elif event.key == pygame.K_w:
                self.show_waypoints = not self.show_waypoints
                print(f"Waypoint display: {'ON' if self.show_waypoints else 'OFF'}")
            elif event.key == pygame.K_i:
                self.show_ils = not self.show_ils
                print(f"ILS display: {'ON' if self.show_ils else 'OFF'}")
            elif event.key == pygame.K_g:
                self.show_grid = not self.show_grid
                print(f"Grid display: {'ON' if self.show_grid else 'OFF'}")
            elif event.key == pygame.K_r:
                self.show_range_rings = not self.show_range_rings
                print(f"Range rings: {'ON' if self.show_range_rings else 'OFF'}")
            elif event.key == pygame.K_a:
                self.show_airport_name = not self.show_airport_name
                print(f"Airport name: {'ON' if self.show_airport_name else 'OFF'}")
            elif event.key == pygame.K_t:
                self.show_legend = not self.show_legend
                self.show_info_panels = not self.show_info_panels
                print(f"Info panels: {'ON' if self.show_info_panels else 'OFF'}")
            elif event.key == pygame.K_p:
                self.cycle_procedures()
            elif event.key == pygame.K_UP:
                # Increase spawn rate
                current_rate = self.aircraft_spawner.get_spawn_rate()
                new_rate = min(10.0, current_rate + 0.1)  # Max 10 aircraft per minute
                self.aircraft_spawner.set_spawn_rate(new_rate)
                print(f"Spawn rate: {new_rate:.1f} aircraft per minute")
            elif event.key == pygame.K_DOWN:
                # Decrease spawn rate
                current_rate = self.aircraft_spawner.get_spawn_rate()
                new_rate = max(0.1, current_rate - 0.1)  # Min 0.1 aircraft per minute
                self.aircraft_spawner.set_spawn_rate(new_rate)
                print(f"Spawn rate: {new_rate:.1f} aircraft per minute")
    
    def process_command(self, command: str):
        """Process typed command"""
        command = command.strip().upper()
        if not command:
            return
        
        parts = command.split()
        if len(parts) < 2:
            print(f"Invalid command format: '{command}'")
            return
        
        callsign = parts[0]
        command_parts = parts[1:]
        
        # Find aircraft by callsign
        aircraft = None
        for ac in self.aircraft:
            if ac.callsign == callsign:
                aircraft = ac
                break
        
        if not aircraft:
            print(f"Aircraft {callsign} not found")
            return
        
        # Process command
        success = aircraft.process_command(command_parts, self.navigation)