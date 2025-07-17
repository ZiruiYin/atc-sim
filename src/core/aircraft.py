"""
Aircraft Module
Handles aircraft objects for the ATC simulator
"""

import math
import time
from typing import List, Tuple, Optional

from utils.math_utils import bearing_between_points, distance_between_points, pixels_to_lat_lon


class Aircraft:
    def __init__(self, callsign: str, initial_x: float, initial_y: float, heading: float, altitude: float, airspeed: float):
        self.callsign = callsign
        self.x = initial_x
        self.y = initial_y
        self.heading = heading  # degrees (0-360)
        self.altitude = altitude  # feet
        self.airspeed = airspeed  # knots
        
        # No wind effects at airport (as requested)
        
        # Movement tracking
        self.trajectory: List[Tuple[float, float]] = []
        self.max_trajectory_length = 5  # Maximum trail length (last 10 positions)
        self.last_update = time.time()
        
        # Command state
        self.target_heading = heading
        self.target_altitude = altitude
        self.target_airspeed = airspeed  # Add target airspeed for gradual changes
        self.target_vor = None
        self.target_runway = None
        self.turn_direction = None  # 'L' or 'R' for forced turn direction
        self.expedite_altitude = False  # Flag for expedited altitude changes
        
        # Hold state
        self.holding = False
        self.hold_waypoint = None
        self.hold_turn_direction = 'R'  # Default right turns
        self.hold_leg = 0  # 0 = outbound, 1 = turn, 2 = inbound, 3 = turn
        self.hold_outbound_heading = None
        self.hold_inbound_heading = None
        self.hold_turn_start_heading = None
        self.hold_turn_target_heading = None
        
        # ILS state
        self.ils_runway = None  # Target runway for ILS approach
        self.loc_intercepted = False  # Localizer intercepted
        self.gs_intercepted = False  # Glideslope intercepted
        self.lower_than_gs = False  # Track when aircraft has gone below glideslope
        self.on_ground = False  # Track when aircraft is on ground
        self.landed = False  # Track when aircraft has landed and should be removed
        self.ils_cleared = False  # Cleared for ILS approach
        
        # Flight parameters
        self.turn_rate = 3  # degrees per second
        self.climb_rate = 25  # feet per second (1500 fpm)
        self.descent_rate = 25  # feet per second (1500 fpm)
        self.speed_change_rate = 2  # knots per second
        
        # Add initial position to trajectory
        self.trajectory.append((self.x, self.y))
    
    def update(self, dt: float):
        """Update aircraft position and trajectory"""
        # Update hold pattern first (can override target_heading)
        self.update_hold_pattern()
        
        # Update waypoint navigation (can override target_heading)
        self.update_waypoint_navigation()
        
        # Check if aircraft is on ground - if so, handle ground behavior
        self.update_ground(dt)

        self.update_ils_navigation()
        
        # Update heading towards target
        self.update_heading(dt)
        
        # Update altitude towards target
        self.update_altitude(dt)
        
        # Update airspeed towards target
        self.update_airspeed(dt)
        
        # Convert heading to radians
        heading_rad = math.radians(self.heading)
        
        # Calculate ground speed using altitude formula: GS = IAS + (IAS × Alt/1000 × 2%)
        ground_speed = self.get_ground_speed()
        
        # Convert speed to pixels per second
        # At scale factor 800, 1 pixel ≈ 54.4 meters
        # 1 knot = 0.514 m/s, so ground_speed knots = ground_speed * 0.514 m/s
        # pixels per second = (ground_speed * 0.514) / 54.4 ≈ ground_speed * 0.0094
        speed_pixels_per_second = ground_speed * 0.01  # Realistic scale factor for visual speed
        
        # Calculate movement (aviation coordinates: 0°=North, 90°=East, 180°=South, 270°=West)
        dx = speed_pixels_per_second * math.sin(heading_rad) * dt  # East is positive x
        dy = -speed_pixels_per_second * math.cos(heading_rad) * dt  # North is negative y (screen inverted)
        
        # Update position
        self.x += dx
        self.y += dy
        
        # Add to trajectory every few frames to make dots more visible
        if len(self.trajectory) == 0 or abs(self.x - self.trajectory[-1][0]) > 5 or abs(self.y - self.trajectory[-1][1]) > 5:
            self.trajectory.append((self.x, self.y))
        
        # Limit trajectory length
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
    
    def update_ground(self, dt: float):
        """Aircraft behavior when on ground"""
        # Get runway info for ground operations
        if not self.on_ground:
            return
        
        runway_info = self.get_runway_info()
        if not runway_info:
            return
        
        # Get airport elevation
        airport_elevation = getattr(self, 'airport_elevation', 83)
        
        # Ensure altitude is at airport elevation
        self.altitude = airport_elevation
        self.target_altitude = airport_elevation
        
        # Set runway heading
        runway_heading = runway_info.get('heading', 0)
        self.target_heading = runway_heading
        
        # Handle ground deceleration
        if self.airspeed > 0:
            self.airspeed = max(0, self.airspeed - 5 * 0.016)  # Assuming 60 FPS
            self.target_airspeed = 0
        if self.airspeed <= 0:
            # Aircraft has stopped, mark for removal
            print(f"{self.callsign}: Landed and stopped")
            self.landed = True
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position"""
        return (self.x, self.y)
    
    def get_ground_speed(self) -> float:
        """Calculate ground speed using altitude formula: GS = IAS + (IAS × Alt/1000 × 2%)"""
        # GS = IAS × (1 + Alt/1000 × 0.02)
        altitude_factor = 1 + (self.altitude / 1000) * 0.02
        ground_speed = self.airspeed * altitude_factor
        
        return max(0, ground_speed)  # Ensure non-negative
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get trajectory history"""
        return self.trajectory
    
    def get_flight_data(self) -> dict:
        """Get flight data for display"""
        return {
            'callsign': self.callsign,
            'altitude': int(self.altitude),
            'airspeed': int(self.airspeed),
            'target_airspeed': int(self.target_airspeed),
            'ground_speed': int(self.get_ground_speed()),
            'heading': int(self.heading),
            'expedite_altitude': self.expedite_altitude,
            'ils_cleared': self.ils_cleared,
            'ils_runway': self.ils_runway,
            'loc_intercepted': self.loc_intercepted,
            'gs_intercepted': self.gs_intercepted
        }
    
    def is_on_screen(self, screen_width: int, screen_height: int) -> bool:
        """Check if aircraft is visible on screen"""
        margin = 50  # Allow some margin for aircraft slightly off screen
        return (-margin <= self.x <= screen_width + margin and 
                -margin <= self.y <= screen_height + margin)
    
    def update_heading(self, dt: float):
        """Update heading towards target heading"""
        if abs(self.heading - self.target_heading) < 0.1:
            self.heading = self.target_heading
            self.turn_direction = None
            return
        
        # Calculate turn amount
        turn_amount = self.turn_rate * dt
        
        # Calculate shortest turn direction if not forced
        if self.turn_direction is None:
            # Find shortest path
            diff = self.target_heading - self.heading
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            
            if diff > 0:
                direction = 1  # Turn right
            else:
                direction = -1  # Turn left
        else:
            # Use forced turn direction
            if self.turn_direction == 'R':
                direction = 1
            else:
                direction = -1
        
        # Apply turn
        self.heading += direction * turn_amount
        
        # Normalize heading to 0-359
        if self.heading >= 360:
            self.heading -= 360
        elif self.heading < 0:
            self.heading += 360
    
    def update_altitude(self, dt: float):
        """Update altitude towards target altitude"""
        if abs(self.altitude - self.target_altitude) < 10:
            self.altitude = self.target_altitude
            self.expedite_altitude = False  # Clear expedite flag when target reached
            return
        
        # Use expedited rates if expedite flag is set
        climb_rate = self.climb_rate * (2 if self.expedite_altitude else 1)
        descent_rate = self.descent_rate * (2 if self.expedite_altitude else 1)
        
        if self.altitude < self.target_altitude:
            # Climb
            self.altitude += climb_rate * dt
            if self.altitude > self.target_altitude:
                self.altitude = self.target_altitude
                self.expedite_altitude = False  # Clear expedite flag when target reached
        else:
            # Descend
            self.altitude -= descent_rate * dt
            if self.altitude < self.target_altitude:
                self.altitude = self.target_altitude
                self.expedite_altitude = False  # Clear expedite flag when target reached
    
    def update_airspeed(self, dt: float):
        """Update airspeed towards target airspeed"""
        if abs(self.airspeed - self.target_airspeed) < 0.1:
            self.airspeed = self.target_airspeed
            return
        
        if self.airspeed < self.target_airspeed:
            # Accelerate
            self.airspeed += self.speed_change_rate * dt
            if self.airspeed > self.target_airspeed:
                self.airspeed = self.target_airspeed
        else:
            # Decelerate
            self.airspeed -= self.speed_change_rate * dt
            if self.airspeed < self.target_airspeed:
                self.airspeed = self.target_airspeed
    
    def update_hold_pattern(self):
        """Update hold pattern navigation"""
        if not self.holding or not self.hold_waypoint:
            return
        
        # Convert aircraft position from pixels to lat/lon
        if not hasattr(self, 'airport_center_lat') or not hasattr(self, 'airport_center_lon'):
            return
        
        # Use stored screen dimensions and scale factor
        screen_width = getattr(self, 'screen_width', 1600)
        screen_height = getattr(self, 'screen_height', 1000)
        scale_factor = getattr(self, 'scale_factor', 800)
        
        # Convert current position to lat/lon
        current_lat, current_lon = pixels_to_lat_lon(
            self.x, self.y, 
            self.airport_center_lat, self.airport_center_lon,
            screen_width, screen_height, scale_factor
        )
        
        # Find hold waypoint coordinates
        target_lat, target_lon = self.get_waypoint_coordinates(self.hold_waypoint)
        if target_lat is None or target_lon is None:
            return
        
        # Calculate distance to waypoint
        distance_nm = distance_between_points(current_lat, current_lon, target_lat, target_lon)
        
        # Check if we've reached the waypoint (within 0.5 nm)
        if distance_nm < 0.5:
            self.handle_hold_waypoint_reached()
            return
        
        # Handle different hold legs
        if self.hold_leg == 0:  # Outbound leg
            # Fly to the waypoint
            bearing_to_waypoint = bearing_between_points(current_lat, current_lon, target_lat, target_lon)
            self.target_heading = bearing_to_waypoint
        elif self.hold_leg == 1:  # First turn
            self.handle_hold_turn()
        elif self.hold_leg == 2:  # Inbound leg
            # Fly back to the waypoint
            bearing_to_waypoint = bearing_between_points(current_lat, current_lon, target_lat, target_lon)
            self.target_heading = bearing_to_waypoint
        elif self.hold_leg == 3:  # Second turn
            self.handle_hold_turn()
    
    def handle_hold_waypoint_reached(self):
        """Handle reaching the hold waypoint"""
        if self.hold_leg == 0:  # Starting outbound leg
            # Initialize hold pattern
            self.hold_outbound_heading = self.heading
            self.hold_inbound_heading = (self.heading + 180) % 360
            
            # Start first turn
            self.hold_leg = 1
            self.hold_turn_start_heading = self.heading
            self.hold_turn_target_heading = self.hold_inbound_heading
        elif self.hold_leg == 2:  # Completed inbound leg, start second turn
            self.hold_leg = 3
            self.hold_turn_start_heading = self.heading
            self.hold_turn_target_heading = self.hold_outbound_heading
    
    def handle_hold_turn(self):
        """Handle hold pattern turns"""
        if self.hold_turn_start_heading is None or self.hold_turn_target_heading is None:
            return
        
        # Check if turn is complete
        if abs(self.heading - self.hold_turn_target_heading) < 1.0:
            # Turn complete, move to next leg
            if self.hold_leg == 1:  # First turn complete
                self.hold_leg = 2  # Start inbound leg
            elif self.hold_leg == 3:  # Second turn complete
                self.hold_leg = 0  # Start new outbound leg
            return
        
        # Continue the turn
        self.target_heading = self.hold_turn_target_heading
        self.turn_direction = self.hold_turn_direction
    
    def clear_hold_pattern(self):
        """Clear hold pattern state"""
        self.holding = False
        self.hold_waypoint = None
        self.hold_turn_direction = 'R'
        self.hold_leg = 0
        self.hold_outbound_heading = None
        self.hold_inbound_heading = None
        self.hold_turn_start_heading = None
        self.hold_turn_target_heading = None

    def update_waypoint_navigation(self):
        """Update navigation to target waypoint"""
        if not self.target_vor or not hasattr(self, 'navigation_data') or not self.navigation_data:
            return
        
        # Convert aircraft position from pixels to lat/lon
        # We need to know the airport center coordinates for this conversion
        if not hasattr(self, 'airport_center_lat') or not hasattr(self, 'airport_center_lon'):
            return
        
        # Use stored screen dimensions and scale factor
        screen_width = getattr(self, 'screen_width', 1600)
        screen_height = getattr(self, 'screen_height', 1000)
        scale_factor = getattr(self, 'scale_factor', 800)
        
        # Convert current position to lat/lon
        current_lat, current_lon = pixels_to_lat_lon(
            self.x, self.y, 
            self.airport_center_lat, self.airport_center_lon,
            screen_width, screen_height, scale_factor
        )
        
        # Find target waypoint coordinates
        target_lat, target_lon = self.get_waypoint_coordinates(self.target_vor)
        if target_lat is None or target_lon is None:
            return
        
        # Calculate distance to waypoint
        distance_nm = distance_between_points(current_lat, current_lon, target_lat, target_lon)
        
        # Check if we've reached the waypoint (within 0.5 nm)
        if distance_nm < 0.5:
            self.target_vor = None
            return
        
        # Calculate bearing to waypoint
        bearing_to_waypoint = bearing_between_points(current_lat, current_lon, target_lat, target_lon)
        
        # Update target heading to fly toward waypoint
        self.target_heading = bearing_to_waypoint
    
    def update_ils_navigation(self):
        """Update ILS navigation and interception"""
        if not self.ils_cleared or not self.ils_runway:
            return
        
        # Get screen dimensions and scale factor from radar display
        screen_width = getattr(self, 'screen_width', 1600)
        screen_height = getattr(self, 'screen_height', 1000)
        scale_factor = getattr(self, 'scale_factor', 800)
        
        # Check if we have airport center coordinates
        if not hasattr(self, 'airport_center_lat') or not hasattr(self, 'airport_center_lon'):
            return
        
        # Convert current position to lat/lon
        current_lat, current_lon = pixels_to_lat_lon(
            self.x, self.y, 
            self.airport_center_lat, self.airport_center_lon,
            screen_width, screen_height, scale_factor
        )
        
        # Check interception conditions
        if not self.can_intercept_ils(current_lat, current_lon):
            return
        
        # Get runway information (would need to be passed from radar display)
        runway_info = self.get_runway_info()
        if not runway_info:
            return
        
        # Handle localizer interception
        if not self.loc_intercepted:
            self.handle_localizer_interception(runway_info)
        
        # Handle glideslope interception (after localizer)
        if self.loc_intercepted and not self.gs_intercepted:
            self.handle_glideslope_interception(runway_info)

        if self.gs_intercepted:
            self.handle_glideslope_descent(runway_info)
        
        # Handle short final and ground behavior
        self.handle_short_final_and_ground(runway_info)
    
    def handle_glideslope_descent(self, runway_info: dict):
        """Handle ongoing glideslope descent after interception"""
        # Get airport elevation from runway info
        airport_elevation = getattr(self, 'airport_elevation', 83)  # Assuming EGLL elevation
        
        # Calculate distance to runway threshold in x-axis (approximate)
        distance_nm = self.calculate_distance_to_runway_x()
        
        # Calculate glideslope altitude at current distance
        gs_altitude = self.calculate_glideslope_altitude(distance_nm, airport_elevation)
        
        # Set target altitude to follow the glideslope
        self.target_altitude = gs_altitude
    
    def can_intercept_ils(self, current_lat: float, current_lon: float) -> bool:
        """Check if aircraft meets ILS interception conditions"""
        # Distance check: must be at most 15 nm
        distance_nm = distance_between_points(
            current_lat, current_lon,
            self.airport_center_lat, self.airport_center_lon
        )
        if distance_nm > 15:
            return False
        
        # Altitude check: less than 5000 AGL
        # Assuming airport elevation is around 80 feet (EGLL)
        agl = self.altitude - 80
        if agl >= 5000:
            return False
        
        # Airspeed check: within 200kts
        if self.airspeed > 240:
            return False
        
        return True
    
    def get_runway_info(self) -> dict:
        """Get runway information for ILS approach"""
        if not self.ils_runway or not hasattr(self, 'airport_data'):
            return None
        
        # Get runway data from airport
        runway_data = self.airport_data.get(self.ils_runway)
        if not runway_data:
            return None
        
        return {
            'name': self.ils_runway,
            'threshold_lat': runway_data['threshold_lat'],
            'threshold_lon': runway_data['threshold_lon'],
            'end_lat': runway_data['end_lat'],
            'end_lon': runway_data['end_lon'],
            'heading': runway_data['heading']
        }
    
    def handle_localizer_interception(self, runway_info: dict):
        """Handle localizer interception logic"""
        # Calculate projected position 2nm ahead
        projected_lat, projected_lon = self.calculate_projected_position(2.0)
        
        # Check if localizer is intercepted based on the three cases
        if self.check_localizer_interception(projected_lat, projected_lon, runway_info):
            self.loc_intercepted = True
    
    def handle_glideslope_interception(self, runway_info: dict):
        """Handle glideslope interception logic"""
        if not self.loc_intercepted:
            return  # Can only intercept glideslope after localizer
        
        # Get airport elevation from runway info
        airport_elevation = getattr(self, 'airport_elevation', 83)  # Assuming EGLL elevation
        
        # Calculate distance to runway threshold in x-axis (approximate)
        distance_nm = self.calculate_distance_to_runway_x()
        
        # Calculate glideslope altitude at current distance
        gs_altitude = self.calculate_glideslope_altitude(distance_nm, airport_elevation)
        
        # Glideslope state machine
        if not self.gs_intercepted:
            if not self.lower_than_gs:
                if self.altitude > gs_altitude:
                    # Still above glideslope, keep waiting
                    pass
                else:
                    # Just went below or at glideslope, set lower_than_gs
                    self.lower_than_gs = True
            else:
                # lower_than_gs is true, check if we're now above the glideslope
                if self.altitude > gs_altitude:
                    # Now we can intercept the glideslope
                    self.gs_intercepted = True
    
        # If glideslope is intercepted, always follow the glideslope
        if self.gs_intercepted:
            # Standard 3-degree glideslope:318ft/nm descent rate
            # Calculate target altitude based on current distance and glideslope
            self.target_altitude = gs_altitude
    
    def calculate_distance_to_runway_x(self) -> float:
        """Calculate distance to runway threshold in x-axis (approximate)"""
        runway_info = self.get_runway_info()
        if not runway_info:
            return 0     
        from utils.math_utils import lat_lon_to_pixels
        
        screen_width = getattr(self, 'screen_width', 1600)
        screen_height = getattr(self, 'screen_height', 1000)
        scale_factor = getattr(self, 'scale_factor', 800)
        
        # Get runway threshold position
        threshold_x, threshold_y = lat_lon_to_pixels(
            runway_info['threshold_lat'], runway_info['threshold_lon'],
            self.airport_center_lat, self.airport_center_lon,
            screen_width, screen_height, scale_factor
        )
        
        # Calculate distance based on runway heading direction
        runway_heading = runway_info.get('heading', 0)
        
        if runway_heading == 90: # runway 09 - aircraft approach from west (negative x)
            distance_pixels = threshold_x - self.x
        elif runway_heading == 270: # runway 27 - aircraft approach from east (positive x)
            distance_pixels = self.x - threshold_x
        else:
            # Fallback to absolute distance for other headings
            distance_pixels = abs(self.x - threshold_x)
        
        # Convert to nautical miles
        pixels_per_nm = scale_factor / 60
        distance_nm = distance_pixels / pixels_per_nm
        
        return distance_nm
    
    def calculate_glideslope_altitude(self, distance_nm: float, airport_elevation: float) -> float:
        """Calculate glideslope altitude at given distance from runway"""
        # Standard 3-degree glideslope:318ft/nm descent rate
        gs_altitude = distance_nm * 318 + airport_elevation
        return gs_altitude
    
    def calculate_projected_position(self, distance_nm: float) -> tuple:
        """Calculate projected position based on current heading"""
        # Convert heading to radians
        heading_rad = math.radians(self.heading)
        
        # Calculate distance in degrees (approximate)
        distance_deg = distance_nm / 60  # 60 nm per degree
        
        # Calculate projected position
        current_lat, current_lon = pixels_to_lat_lon(
            self.x, self.y,
            self.airport_center_lat, self.airport_center_lon,
            getattr(self, 'screen_width', 1600),
            getattr(self, 'screen_height', 1000),
            getattr(self, 'scale_factor', 800)
        )
        
        # Calculate new position
        projected_lat = current_lat + distance_deg * math.cos(heading_rad)
        projected_lon = current_lon + distance_deg * math.sin(heading_rad)
        
        return projected_lat, projected_lon
    
    def check_localizer_interception(self, projected_lat: float, projected_lon: float, runway_info: dict) -> bool:
        """Check if localizer should be intercepted based on the three cases"""
        if not runway_info:
            return False
        
        # Get current position in lat/lon
        screen_width = getattr(self, 'screen_width', 1600)
        screen_height = getattr(self, 'screen_height', 1000)
        scale_factor = getattr(self, 'scale_factor', 800)
        
        current_lat, current_lon = pixels_to_lat_lon(
            self.x, self.y,
            self.airport_center_lat, self.airport_center_lon,
            screen_width, screen_height, scale_factor
        )
        
        # Convert runway coordinates to screen coordinates for easier y-value comparison
        from utils.math_utils import lat_lon_to_pixels
        
        threshold_x, threshold_y = lat_lon_to_pixels(
            runway_info['threshold_lat'], runway_info['threshold_lon'],
            self.airport_center_lat, self.airport_center_lon,
            screen_width, screen_height, scale_factor
        )
        
        end_x, end_y = lat_lon_to_pixels(
            runway_info['end_lat'], runway_info['end_lon'],
            self.airport_center_lat, self.airport_center_lon,
            screen_width, screen_height, scale_factor
        )
        
        # Convert projected position to screen coordinates
        projected_x, projected_y = lat_lon_to_pixels(
            projected_lat, projected_lon,
            self.airport_center_lat, self.airport_center_lon,
            screen_width, screen_height, scale_factor
        )
        
        # Calculate runway centerline (approach direction is opposite to runway)
        runway_center_y = (threshold_y + end_y) / 2
        
        # Get current aircraft screen position
        current_x, current_y = self.x, self.y
        
        aircraft_side = 1 if current_y < runway_center_y else -1
        tip_side = 1 if projected_y < runway_center_y else -1
        
        if aircraft_side != tip_side:
            return True
        
        return False
    
    def get_localizer_guidance_line(self) -> list:
        """Get the dashed line points for localizer guidance"""
        if not self.ils_cleared:
            return []
        
        from utils.math_utils import lat_lon_to_pixels
        runway_info = self.get_runway_info()
        if not runway_info:
            return []
        screen_width = getattr(self, 'screen_width', 1600)
        screen_height = getattr(self, 'screen_height', 1000)
        scale_factor = getattr(self, 'scale_factor', 800)
        # Calculate projected position 2nm ahead for analysis
        projected_lat, projected_lon = self.calculate_projected_position(2.0)
        projected_x, projected_y = lat_lon_to_pixels(
            projected_lat, projected_lon,
            self.airport_center_lat, self.airport_center_lon,
            screen_width, screen_height, scale_factor
        )
        # Check if localizer should be intercepted based on current heading
        should_intercept = self.check_localizer_interception(projected_lat, projected_lon, runway_info)
        # If not intercepted and should_intercept, set intercepted for next frame
        if not self.loc_intercepted and should_intercept:
            self.loc_intercepted = True  # If intercepted, guide toward runway centerline
        if self.loc_intercepted:
            threshold_x, threshold_y = lat_lon_to_pixels(
                runway_info['threshold_lat'], runway_info['threshold_lon'],
                self.airport_center_lat, self.airport_center_lon,
                screen_width, screen_height, scale_factor
            )
            end_x, end_y = lat_lon_to_pixels(
                runway_info['end_lat'], runway_info['end_lon'],
                self.airport_center_lat, self.airport_center_lon,
                screen_width, screen_height, scale_factor
            )
            runway_center_y = (threshold_y + end_y) / 2
            # Calculate guidance point2ward runway centerline
            guidance_distance_nm = 2.0
            guidance_distance_pixels = guidance_distance_nm * scale_factor / 60
            # Calculate direction from aircraft to runway centerline
            dx = projected_x - self.x  # Use projected_x as reference for distance
            dy = runway_center_y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            if distance > 0:
                # Normalize and scale to 2nm
                unit_x = dx / distance
                unit_y = dy / distance
                guide_x = self.x + unit_x * guidance_distance_pixels
                guide_y = self.y + unit_y * guidance_distance_pixels
            else:
                # Fallback to current heading if too close
                guide_x = projected_x
                guide_y = projected_y
            # Set aircraft heading to follow the guidance line
            guidance_heading = math.degrees(math.atan2(guide_x - self.x, self.y - guide_y))
            if guidance_heading < 0:
                guidance_heading += 360
            self.target_heading = guidance_heading
            return [(self.x, self.y), (guide_x, guide_y)]
        # If not intercepted, always show projected heading (current aircraft heading) - exactly 2
        return [(self.x, self.y), (projected_x, projected_y)]
    
    def get_waypoint_coordinates(self, waypoint_name: str) -> Tuple[float, float]:
        """Get coordinates for a waypoint (VOR, NDB, or RNAV)"""
        if not self.navigation_data:
            return None, None
        
        # Check VOR stations
        vor_station = self.navigation_data.get_vor_station(waypoint_name)
        if vor_station:
            return vor_station.get_coordinates()
        
        # Check NDB stations  
        ndb_station = self.navigation_data.get_ndb_station(waypoint_name)
        if ndb_station:
            return ndb_station.get_coordinates()
        
        # Check RNAV waypoints
        rnav_waypoint = self.navigation_data.get_rnav_waypoint(waypoint_name)
        if rnav_waypoint:
            return rnav_waypoint.get_coordinates()
        
        return None, None
    
    def check_ils_clearance_conditions(self) -> tuple[bool, list[str]]:
        """Check if aircraft meets ILS clearance conditions during command processing"""
        conditions_met = True
        failed_conditions = []
        
        # Check if we have airport center coordinates
        if not hasattr(self, 'airport_center_lat') or not hasattr(self, 'airport_center_lon'):
            return False, ["Airport reference data not available"]
        
        # Get screen dimensions and scale factor
        screen_width = getattr(self, 'screen_width', 1600)
        screen_height = getattr(self, 'screen_height', 1000)
        scale_factor = getattr(self, 'scale_factor', 800)
        
        # Convert current position to lat/lon
        current_lat, current_lon = pixels_to_lat_lon(
            self.x, self.y, 
            self.airport_center_lat, self.airport_center_lon,
            screen_width, screen_height, scale_factor
        )
        
        # Distance check: must be at most 15 nm
        distance_nm = distance_between_points(
            current_lat, current_lon,
            self.airport_center_lat, self.airport_center_lon
        )
        if distance_nm > 15:
            conditions_met = False
            failed_conditions.append(f"Aircraft too far from airport: {int(distance_nm)}nm (max 15nm)")
        
        # Altitude check: less than 5000 AGL
        # Assuming airport elevation is around 80 feet (EGLL)
        agl = self.altitude - 83
        if agl >= 5000:
            conditions_met = False
            failed_conditions.append(f"Aircraft too high: {int(agl)}ft AGL (max 5000ft)")
        
        # Airspeed check: within 200kts
        if self.airspeed > 220:
            conditions_met = False
            failed_conditions.append(f"Aircraft too fast: {int(self.airspeed)}kts (max 220kts)")
        
        return conditions_met, failed_conditions
    
    def handle_short_final_and_ground(self, runway_info: dict):
        """Handle short final approach and ground state transition"""
        if not runway_info:
            return
        
        # Get airport elevation
        airport_elevation = getattr(self, 'airport_elevation', 83)
        
        # Calculate distance to runway
        distance_nm = self.calculate_distance_to_runway_x()
        
        # Short final: 5nm out, slow to 140 knots (only if still on ILS approach)
        if distance_nm <= 5.0 and self.altitude > airport_elevation and self.ils_cleared and self.loc_intercepted and self.gs_intercepted:
            if self.target_airspeed > 140:
                self.target_airspeed = 140
        
        # Ground state transition: altitude below or at airport elevation
        if self.altitude <= airport_elevation and not self.on_ground:
            self.on_ground = True
            # Clear all ILS states since were on theground
            self.ils_cleared = False
            self.loc_intercepted = False
            self.gs_intercepted = False
            self.lower_than_gs = False

    def process_command(self, command_parts: List[str], navigation_data=None):
        """Process ATC command"""
        if not command_parts:
            return False
        
        # Store navigation data for waypoint navigation
        self.navigation_data = navigation_data
        
        i = 0
        while i < len(command_parts):
            # Check if aircraft is in hold pattern - only allow course commands
            if self.holding and command_parts[i] != 'C' and command_parts[i] != 'H':
                # If in hold pattern, only allow course commands
                print(f"{self.callsign}: Unable to comply, in hold pattern")
                i += 1
                continue
            
            if command_parts[i] == 'A':
                # Abort command - only valid after cleared for ILS
                if self.ils_cleared:
                    # Store runway heading before clearing runway assignments
                    runway_info = self.get_runway_info()
                    runway_heading = runway_info.get('heading', 0) if runway_info else 0
                    
                    # Reset all ILS conditions
                    self.ils_cleared = False
                    self.loc_intercepted = False
                    self.gs_intercepted = False
                    self.lower_than_gs = False
                    self.on_ground = False
                    
                    # Clear runway assignments
                    self.target_runway = None
                    self.ils_runway = None
                    
                    # Set heading to runway heading and climb to 5000
                    self.target_heading = runway_heading
                    self.target_altitude = 5000
                    self.target_vor = None  # Clear waypoint navigation
                    self.clear_hold_pattern()  # Clear hold pattern
                    
                    print(f"{self.callsign}: Abort approach, climbing to 5000 feet on runway heading {runway_heading}°")
                else:
                    print(f"{self.callsign}: Unable to abort - not cleared for ILS approach")
                i += 1
            elif command_parts[i] == 'C':
                # Course/Clearance command
                # Clear hold pattern when course command is given
                self.clear_hold_pattern()
                
                if i + 1 < len(command_parts):
                    param = command_parts[i + 1]
                    turn_dir = None
                    
                    # First check for a turn direction suffix in the param itself
                    turn_dir = None
                    heading_str = param
                    if param and param[-1].upper() in ['L', 'R']:
                        turn_dir = param[-1].upper()
                        heading_str = param[:-1]  # Remove the turn direction suffix
                    
                    if heading_str.isdigit():
                        if len(heading_str) == 3:
                            # 3 digits = heading - check if LOC intercepted
                            if self.loc_intercepted:
                                print(f"{self.callsign}: Unable to comply - localizer intercepted, heading commands disabled")
                            else:
                                self.target_heading = int(heading_str)
                                self.target_vor = None  # Clear waypoint navigation
                                self.turn_direction = turn_dir
                                turn_text = f" ({turn_dir})" if turn_dir else ""
                                print(f"{self.callsign}: Set heading {heading_str}°{turn_text}")
                        elif len(heading_str) <= 2:
                            # 1-2 digits = altitude in thousands - check if GS intercepted
                            if self.gs_intercepted:
                                print(f"{self.callsign}: Unable to comply - glideslope intercepted, altitude commands disabled")
                            else:
                                self.target_altitude = int(param) * 1000
                                self.expedite_altitude = False  # Reset expedite flag by default
                                # Check for expedite flag 'X' after altitude command
                                if i + 2 < len(command_parts) and command_parts[i + 2] == 'X':
                                    self.expedite_altitude = True
                                    i += 1  # Skip the 'X' token
                                    print(f"{self.callsign}: Cleared to {self.target_altitude} feet, expedite")
                                else:
                                    print(f"{self.callsign}: Cleared to {self.target_altitude} feet")
                    # Non-numeric input is treated as a waypoint/VOR identifier
                    else:
                        # Check if LOC intercepted
                        if self.loc_intercepted:
                            print(f"{self.callsign}: Unable to comply - localizer intercepted, course commands disabled")
                        else:
                            self.target_heading = None  # Clear any existing heading command
                            self.target_vor = heading_str.upper()  # Convert to uppercase for consistency
                            self.turn_direction = turn_dir
                            turn_text = f" ({turn_dir})" if turn_dir else ""
                            print(f"{self.callsign}: Set course to {heading_str.upper()} waypoint{turn_text}")
                    i += 2
                else:
                    i += 1
            elif command_parts[i] == 'L':
                # Landing command (ILS approach)
                if i + 1 < len(command_parts):
                    runway = command_parts[i + 1]
                    
                    # Check if aircraft meets ILS interception conditions
                    conditions_met, failed_conditions = self.check_ils_clearance_conditions()
                    if conditions_met:
                        self.target_runway = runway
                        self.ils_runway = runway
                        self.ils_cleared = True
                        self.loc_intercepted = False
                        self.gs_intercepted = False
                        self.clear_hold_pattern()  # Clear hold pattern
                        print(f"{self.callsign}: Cleared ILS approach runway {runway}")
                    else:
                        print(f"{self.callsign}: Unable to comply - ILS approach requirements not met:")
                        for condition in failed_conditions:
                            print(f"{self.callsign}: {condition}")
                    i += 2
                else:
                    i += 1
            elif command_parts[i] == 'H':
                # Hold command
                if i + 1 < len(command_parts):
                    waypoint = command_parts[i + 1]
                    turn_dir = 'R'  # Default right turns
                    
                    # Check for turn direction modifier
                    if i + 2 < len(command_parts) and command_parts[i + 2] in ['L', 'R']:
                        turn_dir = command_parts[i + 2]
                        i += 1
                    
                    # Clear any existing waypoint navigation
                    self.target_vor = None
                    self.holding = True
                    self.hold_waypoint = waypoint
                    self.hold_turn_direction = turn_dir
                    self.hold_leg = 0  # Start with outbound leg
                    
                    # Clear hold state variables
                    self.hold_outbound_heading = None
                    self.hold_inbound_heading = None
                    self.hold_turn_start_heading = None
                    self.hold_turn_target_heading = None
                    
                    turn_text = f" ({turn_dir})" if turn_dir else ""
                    print(f"{self.callsign}: Hold at {waypoint} waypoint{turn_text}")
                    i += 2
                else:
                    i += 1
            elif command_parts[i] == 'S':
                # Speed command (for chaining)
                if i + 1 < len(command_parts):
                    speed = command_parts[i + 1]
                    if speed.isdigit():
                        self.target_airspeed = int(speed)
                        print(f"{self.callsign}: Set speed {speed} knots")
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        return True