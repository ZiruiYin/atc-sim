import json
import os
from typing import Dict, List, Optional, Tuple


class Runway:
    def __init__(self, name: str, data: Dict):
        self.name = name
        self.heading = data['heading']
        self.opposite = data['opposite']
        self.length = data['length']  # meters
        self.width = data['width']    # meters
        self.surface = data['surface']
        self.threshold_lat = data['coordinates']['threshold']['latitude']
        self.threshold_lon = data['coordinates']['threshold']['longitude']
        self.end_lat = data['coordinates']['end']['latitude']
        self.end_lon = data['coordinates']['end']['longitude']
        self.displaced_threshold = data.get('displaced_threshold', 0)
        self.has_ils = data.get('ils', {}).get('available', False)
        self.ils_category = data.get('ils', {}).get('category', '')
        self.ils_frequency = data.get('ils', {}).get('frequency', '')

    def get_threshold_coords(self) -> Tuple[float, float]:
        """Get runway threshold coordinates"""
        return (self.threshold_lat, self.threshold_lon)
    
    def get_end_coords(self) -> Tuple[float, float]:
        """Get runway end coordinates"""
        return (self.end_lat, self.end_lon)


class Airport:
    def __init__(self, airport_file: str):
        self.runways: Dict[str, Runway] = {}
        self.icao: str = ""
        self.iata: str = ""
        self.name: str = ""
        self.city: str = ""
        self.country: str = ""
        self.latitude: float = 0.0
        self.longitude: float = 0.0
        self.elevation: int = 0
        self.magnetic_variation: float = 0.0
        self.frequencies: Dict[str, List[float]] = {}
        self.terminals: Dict[str, Dict] = {}
        self.operations: Dict = {}
        
        self.load_airport_data(airport_file)
    
    def load_airport_data(self, airport_file: str):
        """Load airport data from JSON file"""
        try:
            with open(airport_file, 'r') as f:
                data = json.load(f)
            
            airport_data = data['airport']
            
            # Basic airport info
            self.icao = airport_data['icao']
            self.iata = airport_data['iata']
            self.name = airport_data['name']
            self.city = airport_data['city']
            self.country = airport_data['country']
            self.latitude = airport_data['coordinates']['latitude']
            self.longitude = airport_data['coordinates']['longitude']
            self.elevation = airport_data['elevation']
            self.magnetic_variation = airport_data['magnetic_variation']
            
            # Frequencies
            self.frequencies = airport_data.get('frequencies', {})
            
            # Operations
            self.operations = airport_data.get('operations', {})
            
            # Terminals
            self.terminals = airport_data.get('terminals', {})
            
            # Load runways
            for runway_name, runway_data in airport_data['runways'].items():
                self.runways[runway_name] = Runway(runway_name, runway_data)
            
            print(f"Loaded airport: {self.name} ({self.icao})")
            print(f"  - Location: {self.latitude:.4f}, {self.longitude:.4f}")
            print(f"  - Runways: {list(self.runways.keys())}")
            
        except Exception as e:
            print(f"Error loading airport data: {e}")
            raise
    
    def get_coordinates(self) -> Tuple[float, float]:
        """Get airport coordinates"""
        return (self.latitude, self.longitude)
    
    def get_runway(self, runway_name: str) -> Optional[Runway]:
        """Get runway by name"""
        return self.runways.get(runway_name)
    
    def get_all_runways(self) -> Dict[str, Runway]:
        """Get all runways"""
        return self.runways
    
    def get_frequencies(self, freq_type: str) -> List[float]:
        """Get frequencies by type (ground, tower, approach, etc.)"""
        return self.frequencies.get(freq_type, []) 