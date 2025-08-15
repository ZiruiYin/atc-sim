import json
import os
from typing import Dict, List, Optional, Tuple


class Runway:
    def __init__(self, name: str, data: Dict):
        self.name = name
        self.heading = data['heading']
        
        # Only store NM coordinates (no lat/lon)
        self.threshold_x = data['threshold']['x']
        self.threshold_y = data['threshold']['y']
        self.end_x = data['end']['x']
        self.end_y = data['end']['y']

    def get_threshold_coords(self) -> Tuple[float, float]:
        """Get runway threshold coordinates in screen pixels"""
        return (self.threshold_x, self.threshold_y)
    
    def get_end_coords(self) -> Tuple[float, float]:
        """Get runway end coordinates in screen pixels"""
        return (self.end_x, self.end_y)


class Airport:
    def __init__(self, game_coords_file: str):
        self.runways: Dict[str, Runway] = {}
        self.icao: str = ""
        self.name: str = ""
        self.x: float = 0.0
        self.y: float = 0.0
        
        self.load_game_data(game_coords_file)
    
    def load_game_data(self, game_coords_file: str):
        """Load airport data from game coordinates JSON file"""
        try:
            with open(game_coords_file, 'r') as f:
                data = json.load(f)
            
            airport_data = data['airport']
            
            # Basic airport info
            self.icao = airport_data['icao']
            self.name = airport_data['name']
            self.x = airport_data['coordinates']['x']
            self.y = airport_data['coordinates']['y']
            
            # Load runways
            for runway_name, runway_data in airport_data['runways'].items():
                self.runways[runway_name] = Runway(runway_name, runway_data)
            
            print(f"Loaded airport: {self.name} ({self.icao})")
            print(f"  - Coordinates: ({self.x:.0f}, {self.y:.0f}) pixels")
            print(f"  - Runways: {list(self.runways.keys())}")
            
        except Exception as e:
            print(f"Error loading airport data: {e}")
            raise
    
    def get_coordinates(self) -> Tuple[float, float]:
        """Get airport coordinates in screen pixels"""
        return (self.x, self.y)
    
    def get_runway(self, runway_name: str) -> Optional[Runway]:
        """Get runway by name"""
        return self.runways.get(runway_name)
    
    def get_all_runways(self) -> Dict[str, Runway]:
        """Get all runways"""
        return self.runways 