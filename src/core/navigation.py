import json
import os
from typing import Dict, List, Optional, Tuple


class VORStation:
    def __init__(self, identifier: str, data: Dict):
        self.identifier = identifier
        self.name = data['name']
        self.frequency = data['frequency']
        self.x = data['coordinates']['x']
        self.y = data['coordinates']['y']
    
    def get_coordinates(self) -> Tuple[float, float]:
        """Get VOR station coordinates in screen pixels"""
        return (self.x, self.y)


class NDBStation:
    def __init__(self, identifier: str, data: Dict):
        self.identifier = identifier
        self.name = data['name']
        self.frequency = data['frequency']
        self.x = data['coordinates']['x']
        self.y = data['coordinates']['y']
    
    def get_coordinates(self) -> Tuple[float, float]:
        """Get NDB station coordinates in screen pixels"""
        return (self.x, self.y)


class Waypoint:
    def __init__(self, identifier: str, data: Dict):
        self.identifier = identifier
        self.name = data['name']
        self.x = data['coordinates']['x']
        self.y = data['coordinates']['y']
        self.waypoint_type = data['type']  # SID, STAR, INTERMEDIATE, etc.
    
    def get_coordinates(self) -> Tuple[float, float]:
        """Get waypoint coordinates in screen pixels"""
        return (self.x, self.y)


class Procedure:
    def __init__(self, name: str, data: Dict):
        self.name = name
        self.procedure_name = data['name']
        self.runway = data.get('runway', '')
        self.route = data.get('route', [])
        self.initial_altitude = data.get('initial_altitude', 0)
        self.final_altitude = data.get('final_altitude', 0)


class Navigation:
    def __init__(self, game_coords_file: str):
        self.vor_stations: Dict[str, VORStation] = {}
        self.ndb_stations: Dict[str, NDBStation] = {}
        self.waypoints: Dict[str, Waypoint] = {}
        self.sid_procedures: Dict[str, Procedure] = {}
        self.star_procedures: Dict[str, Procedure] = {}
        
        self.load_game_data(game_coords_file)
    
    def load_game_data(self, game_coords_file: str):
        """Load navigation data from game coordinates JSON file"""
        try:
            with open(game_coords_file, 'r') as f:
                data = json.load(f)
            
            nav_data = data['navigation']
            
            # Load VOR stations
            for vor_id, vor_data in nav_data.get('vor_stations', {}).items():
                self.vor_stations[vor_id] = VORStation(vor_id, vor_data)
            
            # Load NDB stations
            for ndb_id, ndb_data in nav_data.get('ndb_stations', {}).items():
                self.ndb_stations[ndb_id] = NDBStation(ndb_id, ndb_data)
            
            # Load waypoints
            for waypoint_id, waypoint_data in nav_data.get('waypoints', {}).items():
                self.waypoints[waypoint_id] = Waypoint(waypoint_id, waypoint_data)
            
            # Load SID procedures
            for sid_id, sid_data in nav_data.get('procedures', {}).get('sid', {}).items():
                self.sid_procedures[sid_id] = Procedure(sid_id, sid_data)
            
            # Load STAR procedures
            for star_id, star_data in nav_data.get('procedures', {}).get('star', {}).items():
                self.star_procedures[star_id] = Procedure(star_id, star_data)
            
            print(f"Loaded navigation data:")
            print(f"  - VOR stations: {len(self.vor_stations)}")
            print(f"  - NDB stations: {len(self.ndb_stations)}")
            print(f"  - Waypoints: {len(self.waypoints)}")
            print(f"  - SID procedures: {len(self.sid_procedures)}")
            print(f"  - STAR procedures: {len(self.star_procedures)}")
            
        except Exception as e:
            print(f"Error loading navigation data: {e}")
            raise
    
    def get_vor_station(self, identifier: str) -> Optional[VORStation]:
        """Get VOR station by identifier"""
        return self.vor_stations.get(identifier)
    
    def get_ndb_station(self, identifier: str) -> Optional[NDBStation]:
        """Get NDB station by identifier"""
        return self.ndb_stations.get(identifier)
    
    def get_waypoint(self, identifier: str) -> Optional[Waypoint]:
        """Get waypoint by identifier"""
        return self.waypoints.get(identifier)
    
    def get_all_vor_stations(self) -> Dict[str, VORStation]:
        """Get all VOR stations"""
        return self.vor_stations
    
    def get_all_ndb_stations(self) -> Dict[str, NDBStation]:
        """Get all NDB stations"""
        return self.ndb_stations
    
    def get_all_waypoints(self) -> Dict[str, Waypoint]:
        """Get all waypoints"""
        return self.waypoints
    
    def get_sid_procedure(self, identifier: str) -> Optional[Procedure]:
        """Get SID procedure by identifier"""
        return self.sid_procedures.get(identifier)
    
    def get_star_procedure(self, identifier: str) -> Optional[Procedure]:
        """Get STAR procedure by identifier"""
        return self.star_procedures.get(identifier)
    
    def get_all_sid_procedures(self) -> Dict[str, Procedure]:
        """Get all SID procedures"""
        return self.sid_procedures
    
    def get_all_star_procedures(self) -> Dict[str, Procedure]:
        """Get all STAR procedures"""
        return self.star_procedures 