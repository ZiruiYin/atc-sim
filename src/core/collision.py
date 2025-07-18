"""
Collision Detection Module
Handles collision warnings and crash detection for the ATC simulator
"""

import math
from typing import Dict, List, Tuple, Set, Optional
from utils.math_utils import distance_between_points, pixels_to_lat_lon, lat_lon_to_pixels


class CollisionDetector:
    def __init__(self, grid_size_nm: float = 3.0):
        """
        Initialize collision detector with grid-based system
        
        Args:
            grid_size_nm: Size of each grid cell in nautical miles (default 3nm)
        """
        self.grid_size_nm = grid_size_nm
        self.grid: Dict[Tuple[int, int], Set[str]] = {}  # (grid_x, grid_y) -> set of aircraft callsigns
        self.aircraft_positions: Dict[str, Tuple[float, float, float]] = {}  # callsign -> (x, y, altitude)
        self.warnings: Set[Tuple[str, str]] = set()  # Set of (callsign1, callsign2) pairs in warning
        self.crashes: Set[Tuple[str, str]] = set()  # Set of (callsign1, callsign2) pairs that crashed
        self.game_frozen = False  # Game state when crash occurs
        self.crash_message = ""  # Message describing the crash
        
    def update_aircraft_positions(self, aircraft_list: List, airport_center_lat: float, 
                                 airport_center_lon: float, screen_width: int, 
                                 screen_height: int, scale_factor: float):
        """
        Update aircraft positions and grid assignments
        
        Args:
            aircraft_list: List of Aircraft objects
            airport_center_lat: Airport center latitude
            airport_center_lon: Airport center longitude
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            scale_factor: Scale factor for coordinate conversion
        """
        # Clear previous grid assignments
        self.grid.clear()
        self.aircraft_positions.clear()
        
        # Update aircraft positions and assign to grids
        for aircraft in aircraft_list:
            if hasattr(aircraft, 'landed') and aircraft.landed:
                continue  # Skip landed aircraft
                
            x, y = aircraft.get_position()
            altitude = aircraft.altitude
            
            # Convert pixel position to lat/lon for grid calculation
            lat, lon = pixels_to_lat_lon(x, y, airport_center_lat, airport_center_lon,
                                       screen_width, screen_height, scale_factor)
            
            # Calculate grid coordinates
            grid_x, grid_y = self.lat_lon_to_grid(lat, lon)
            
            # Store aircraft position
            self.aircraft_positions[aircraft.callsign] = (x, y, altitude)
            
            # Add aircraft to grid
            grid_key = (grid_x, grid_y)
            if grid_key not in self.grid:
                self.grid[grid_key] = set()
            self.grid[grid_key].add(aircraft.callsign)
    
    def lat_lon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Convert lat/lon coordinates to grid coordinates
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Tuple of (grid_x, grid_y) coordinates
        """
        # Convert lat/lon to grid coordinates
        # 1 degree ≈ 60 nautical miles
        grid_x = int(lon * 60 / self.grid_size_nm)
        grid_y = int(lat * 60 / self.grid_size_nm)
        return grid_x, grid_y
    
    def check_collisions(self, aircraft_list: List) -> bool:
        """
        Check for collisions and warnings
        
        Args:
            aircraft_list: List of Aircraft objects
            
        Returns:
            True if game should continue, False if crash occurred
        """
        if self.game_frozen:
            return False  # Game is frozen due to crash
            
        # Clear previous warnings
        old_warnings = self.warnings.copy()
        self.warnings.clear()
        
        # Check each grid for multiple aircraft
        for grid_key, aircraft_callsigns in self.grid.items():
            if len(aircraft_callsigns) < 2:
                continue
                
            # Check for multiple aircraft in same grid
            aircraft_in_grid = list(aircraft_callsigns)
            for i in range(len(aircraft_in_grid)):
                for j in range(i + 1, len(aircraft_in_grid)):
                    callsign1 = aircraft_in_grid[i]
                    callsign2 = aircraft_in_grid[j]
                    
                    # Get aircraft objects
                    aircraft1 = self.get_aircraft_by_callsign(aircraft_list, callsign1)
                    aircraft2 = self.get_aircraft_by_callsign(aircraft_list, callsign2)
                    
                    if not aircraft1 or not aircraft2:
                        continue
                    
                    # Check for exceptions (ILS or ground)
                    if self.is_exception_case(aircraft1, aircraft2):
                        continue
                    
                    # Get positions and altitudes
                    pos1 = self.aircraft_positions.get(callsign1)
                    pos2 = self.aircraft_positions.get(callsign2)
                    
                    if not pos1 or not pos2:
                        continue
                    
                    x1, y1, alt1 = pos1
                    x2, y2, alt2 = pos2
                    
                    # Check altitude separation
                    altitude_separation = abs(alt1 - alt2)
                    
                    if altitude_separation < 1000:  # Less than 1000ft separation
                        # Calculate distance in pixels
                        distance_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
                        # Convert to nautical miles (approximate)
                        # At scale factor 800, 1 pixel ≈ 54.4 meters
                        # 1 nm = 1852 meters, so 1 pixel ≈ 54.4/1852 ≈ 0.029 nm
                        distance_nm = distance_pixels * 0.029
                        
                        if distance_nm < 3.0:  # Less than 3nm separation
                            # Check for crash (same altitude and very close)
                            if altitude_separation < 100 and distance_nm < 0.1:
                                self.handle_crash(callsign1, callsign2)
                                return False
                            else:
                                # Add warning
                                warning_pair = tuple(sorted([callsign1, callsign2]))
                                self.warnings.add(warning_pair)
        
        # Check adjacent grids
        self.check_adjacent_grids(aircraft_list)
        
        # Check for resolved warnings
        resolved_warnings = old_warnings - self.warnings
        if resolved_warnings:
            for warning_pair in resolved_warnings:
                callsign1, callsign2 = warning_pair
                print(f"Collision warning resolved: {callsign1} and {callsign2}")
        
        return True
    
    def check_adjacent_grids(self, aircraft_list: List):
        """
        Check for collisions between aircraft in adjacent grids
        
        Args:
            aircraft_list: List of Aircraft objects
        """
        # Define adjacent grid offsets
        adjacent_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for grid_key, aircraft_callsigns in self.grid.items():
            grid_x, grid_y = grid_key
            
            for offset_x, offset_y in adjacent_offsets:
                adjacent_grid = (grid_x + offset_x, grid_y + offset_y)
                
                if adjacent_grid not in self.grid:
                    continue
                
                adjacent_aircraft = self.grid[adjacent_grid]
                
                # Check each aircraft in current grid against adjacent grid
                for callsign1 in aircraft_callsigns:
                    for callsign2 in adjacent_aircraft:
                        # Get aircraft objects
                        aircraft1 = self.get_aircraft_by_callsign(aircraft_list, callsign1)
                        aircraft2 = self.get_aircraft_by_callsign(aircraft_list, callsign2)
                        
                        if not aircraft1 or not aircraft2:
                            continue
                        
                        # Check for exceptions (ILS or ground)
                        if self.is_exception_case(aircraft1, aircraft2):
                            continue
                        
                        # Get positions and altitudes
                        pos1 = self.aircraft_positions.get(callsign1)
                        pos2 = self.aircraft_positions.get(callsign2)
                        
                        if not pos1 or not pos2:
                            continue
                        
                        x1, y1, alt1 = pos1
                        x2, y2, alt2 = pos2
                        
                        # Check altitude separation
                        altitude_separation = abs(alt1 - alt2)
                        
                        if altitude_separation < 1000:  # Less than 1000ft separation
                            # Calculate distance in pixels
                            distance_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            
                            # Convert to nautical miles (approximate)
                            distance_nm = distance_pixels * 0.029
                            
                            if distance_nm < 3.0:  # Less than 3nm separation
                                # Check for crash (same altitude and very close)
                                if altitude_separation < 100 and distance_nm < 0.1:
                                    self.handle_crash(callsign1, callsign2)
                                    return
                                else:
                                    # Add warning
                                    warning_pair = tuple(sorted([callsign1, callsign2]))
                                    self.warnings.add(warning_pair)
    
    def is_exception_case(self, aircraft1, aircraft2) -> bool:
        """
        Check if aircraft pair should be exempt from collision warnings
        
        Args:
            aircraft1: First Aircraft object
            aircraft2: Second Aircraft object
            
        Returns:
            True if exception applies, False otherwise
        """
        # Exception: if either aircraft is on ILS or ground
        aircraft1_exception = (hasattr(aircraft1, 'ils_cleared') and aircraft1.ils_cleared) or \
                             (hasattr(aircraft1, 'on_ground') and aircraft1.on_ground)
        
        aircraft2_exception = (hasattr(aircraft2, 'ils_cleared') and aircraft2.ils_cleared) or \
                             (hasattr(aircraft2, 'on_ground') and aircraft2.on_ground)
        
        return aircraft1_exception or aircraft2_exception
    
    def get_aircraft_by_callsign(self, aircraft_list: List, callsign: str):
        """
        Get aircraft object by callsign
        
        Args:
            aircraft_list: List of Aircraft objects
            callsign: Aircraft callsign
            
        Returns:
            Aircraft object or None if not found
        """
        for aircraft in aircraft_list:
            if aircraft.callsign == callsign:
                return aircraft
        return None
    
    def handle_crash(self, callsign1: str, callsign2: str):
        """
        Handle aircraft crash
        
        Args:
            callsign1: First aircraft callsign
            callsign2: Second aircraft callsign
        """
        self.game_frozen = True
        self.crashes.add(tuple(sorted([callsign1, callsign2])))
        self.crash_message = f"CRASH: {callsign1} and {callsign2} have collided!"
        print(f"\n{'='*50}")
        print(f"CRASH DETECTED!")
        print(f"{callsign1} and {callsign2} have collided!")
        print(f"All controls frozen.")
        print(f"{'='*50}\n")
    
    def is_aircraft_in_warning(self, callsign: str) -> bool:
        """
        Check if aircraft is involved in any collision warning
        
        Args:
            callsign: Aircraft callsign
            
        Returns:
            True if aircraft is in warning, False otherwise
        """
        for warning_pair in self.warnings:
            if callsign in warning_pair:
                return True
        return False
    
    def get_warning_aircraft(self) -> Set[str]:
        """
        Get set of all aircraft involved in warnings
        
        Returns:
            Set of aircraft callsigns in warnings
        """
        warning_aircraft = set()
        for warning_pair in self.warnings:
            warning_aircraft.update(warning_pair)
        return warning_aircraft
    
    def is_game_frozen(self) -> bool:
        """
        Check if game is frozen due to crash
        
        Returns:
            True if game is frozen, False otherwise
        """
        return self.game_frozen
    
    def get_crash_message(self) -> str:
        """
        Get crash message
        
        Returns:
            Crash message string
        """
        return self.crash_message
    
    def reset(self):
        """
        Reset collision detector state
        """
        self.grid.clear()
        self.aircraft_positions.clear()
        self.warnings.clear()
        self.crashes.clear()
        self.game_frozen = False
        self.crash_message = "" 