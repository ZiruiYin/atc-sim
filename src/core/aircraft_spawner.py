"""
Aircraft Spawner Module
Handles spawning aircraft at controlled rates for the ATC simulator
"""

import random
import time
from typing import List, Tuple, Optional
from core.aircraft import Aircraft


class AircraftSpawner:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Spawn configuration
        self.spawn_rate = 2.0  # aircraft per minute
        self.last_spawn_time = 0  # Start at 0 so first aircraft spawns immediately
        
        # Callsign prefixes for realistic aircraft callsigns
        self.airline_prefixes = ["BA", "LH", "AF", "KL", "UA", "AA", "DL", "VS", "EK", "QR", 
                                 "EY", "SQ", "CX", "JL", "NH", "TK", "IB", "AY", "SK", "AC"]

        
        # Entry points around the radar screen (further from screen edges)
        self.entry_margin = 20
        
        # Altitude ranges (feet) - in thousands from 6000 to 12000
        self.min_altitude = 6000
        self.max_altitude = 12000
        
        # Fixed airspeed (knots)
        self.airspeed = 250
        
        # Track last spawned altitude for separation
        self.last_spawned_altitude = None
        
        # Track used callsign numbers to ensure uniqueness
        self.used_callsigns = set()
    
    def set_spawn_rate(self, aircraft_per_minute: float):
        """Set the spawn rate in aircraft per minute"""
        self.spawn_rate = aircraft_per_minute
    
    def get_spawn_rate(self) -> float:
        """Get the current spawn rate in aircraft per minute"""
        return self.spawn_rate
    
    def should_spawn_aircraft(self) -> bool:
        """Check if it's time to spawn a new aircraft based on the spawn rate"""
        current_time = time.time()
        time_since_last_spawn = current_time - self.last_spawn_time
        
        # Calculate spawn interval in seconds
        spawn_interval = 60.0 / self.spawn_rate if self.spawn_rate > 0 else float('inf')
        
        # Add some randomness to make spawning more natural (±25% variation)
        spawn_interval *= random.uniform(0.75, 1.25)
        
        return time_since_last_spawn >= spawn_interval
    
    def generate_callsign(self) -> str:
        """Generate a realistic aircraft callsign with unique number"""
        prefix = random.choice(self.airline_prefixes)
        
        # Generate a random number from 1 to 999
        available_numbers = set(range(1, 1000)) - self.used_callsigns
        
        # If we've used all numbers, reset the used callsigns set
        if not available_numbers:
            self.used_callsigns.clear()
            available_numbers = set(range(1, 1000))
        
        # Pick a random available number
        number = random.choice(list(available_numbers))
        self.used_callsigns.add(number)
        
        return f"{prefix}{number}"
    
    def get_random_entry_point(self) -> Tuple[float, float, float]:
        """Get a random entry point and appropriate heading towards center"""
        # Choose which edge to spawn from
        edge = random.choice(['north', 'south', 'east', 'west'])
        
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        
        if edge == 'north':
            x = random.uniform(self.entry_margin, self.screen_width - self.entry_margin)
            y = self.entry_margin
            # Heading based on position relative to center
            if x < center_x:
                # Left side - heading 120 to 180 (southeast)
                heading = random.uniform(120, 180)
            else:
                # Right side - heading 180 to 240 (southwest)
                heading = random.uniform(180, 240)
        elif edge == 'south':
            x = random.uniform(self.entry_margin, self.screen_width - self.entry_margin)
            # Spawn higher to avoid text box at bottom
            y = self.screen_height - 50
            # Heading based on position relative to center
            if x < center_x:
                # Left side - heading 0 to 60 (northeast)
                heading = random.uniform(0, 60)
            else:
                # Right side - heading 300 to 360 (northwest)
                heading = random.uniform(300, 360)
        elif edge == 'east':
            x = self.screen_width + self.entry_margin
            y = random.uniform(self.entry_margin, self.screen_height - self.entry_margin)
            # Heading based on position relative to center
            if y < center_y:
                # Top side - heading 210 to 270 (southwest)
                heading = random.uniform(210, 270)
            else:
                # Bottom side - heading 270 to 330 (northwest)
                heading = random.uniform(270, 330)
        else:  # west
            x = -self.entry_margin
            y = random.uniform(self.entry_margin, self.screen_height - self.entry_margin)
            # Heading based on position relative to center
            if y < center_y:
                # Top side - heading 90 to 150 (southeast)
                heading = random.uniform(90, 150)
            else:
                # Bottom side - heading 30 to 90 (northeast)
                heading = random.uniform(30, 90)
        
        return x, y, heading
    
    def get_altitude_with_separation(self) -> float:
        """Get altitude ensuring at least 1000ft separation from last spawned aircraft"""
        if self.last_spawned_altitude is None:
            # First aircraft - choose random altitude in thousands
            altitude = random.choice([6000, 7000, 8000, 9000, 10000, 11000, 12000])
        else:
            # Find available altitudes with 1000ft separation
            available_altitudes = []
            for alt in [6000, 7000, 8000, 9000, 10000, 11000, 12000]:
                if abs(alt - self.last_spawned_altitude) >= 1000:
                    available_altitudes.append(alt)
            
            if available_altitudes:
                altitude = random.choice(available_altitudes)
            else:
                # Fallback - choose random altitude
                altitude = random.choice([6000, 7000, 8000, 9000, 10000, 11000, 12000])
        
        self.last_spawned_altitude = altitude
        return altitude
    
    def spawn_aircraft(self) -> Optional[Aircraft]:
        """Spawn a new aircraft if conditions are met"""
        if not self.should_spawn_aircraft():
            return None
        
        # Generate aircraft parameters
        callsign = self.generate_callsign()
        x, y, heading = self.get_random_entry_point()
        altitude = self.get_altitude_with_separation()
        airspeed = self.airspeed
        
        # Create aircraft
        aircraft = Aircraft(callsign, x, y, heading, altitude, airspeed)
        
        # Update last spawn time
        self.last_spawn_time = time.time()
        
        print(f"Spawned aircraft: {callsign} at ({x:.0f}, {y:.0f}), "
              f"heading {heading:.0f}°, altitude {altitude:.0f}ft, speed {airspeed:.0f}kts")
        
        return aircraft
    
    def update(self, current_aircraft: List[Aircraft]) -> List[Aircraft]:
        """Update spawner and return list of new aircraft to add"""
        new_aircraft = []
        
        # Try to spawn new aircraft
        aircraft = self.spawn_aircraft()
        if aircraft:
            new_aircraft.append(aircraft)
        
        return new_aircraft
    
    def get_spawn_info(self) -> dict:
        """Get current spawn configuration info"""
        return {
            'spawn_rate': self.spawn_rate,
            'time_since_last_spawn': time.time() - self.last_spawn_time,
            'next_spawn_in': max(0, (60.0 / self.spawn_rate) - (time.time() - self.last_spawn_time)),
            'used_callsigns': len(self.used_callsigns)
        } 