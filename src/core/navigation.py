import json
import os
from typing import Dict, List, Optional, Tuple


class VORStation:
    def __init__(self, identifier: str, data: Dict):
        self.identifier = identifier
        self.name = data['name']
        self.frequency = data['frequency']
        self.latitude = data['coordinates']['latitude']
        self.longitude = data['coordinates']['longitude']
        self.distance_from_airport = data['distance_from_airport']
        self.bearing_from_airport = data['bearing_from_airport']
    
    def get_coordinates(self) -> Tuple[float, float]:
        """Get VOR station coordinates"""
        return (self.latitude, self.longitude)


class NDBStation:
    def __init__(self, identifier: str, data: Dict):
        self.identifier = identifier
        self.name = data['name']
        self.frequency = data['frequency']
        self.latitude = data['coordinates']['latitude']
        self.longitude = data['coordinates']['longitude']
        self.distance_from_airport = data['distance_from_airport']
        self.bearing_from_airport = data['bearing_from_airport']
    
    def get_coordinates(self) -> Tuple[float, float]:
        """Get NDB station coordinates"""
        return (self.latitude, self.longitude)


class RNAVWaypoint:
    def __init__(self, identifier: str, data: Dict):
        self.identifier = identifier
        self.name = data['name']
        self.latitude = data['coordinates']['latitude']
        self.longitude = data['coordinates']['longitude']
        self.waypoint_type = data['type']  # SID, STAR, INTERMEDIATE, etc.
        self.procedures = data.get('procedures', [])
    
    def get_coordinates(self) -> Tuple[float, float]:
        """Get waypoint coordinates"""
        return (self.latitude, self.longitude)


class Procedure:
    def __init__(self, name: str, data: Dict):
        self.name = name
        self.procedure_name = data['name']
        self.runway = data.get('runway', '')
        self.route = data.get('route', [])
        self.initial_altitude = data.get('initial_altitude', 0)
        self.final_altitude = data.get('final_altitude', 0)


class Navigation:
    def __init__(self, navigation_file: str):
        self.airport_icao: str = ""
        self.vor_stations: Dict[str, VORStation] = {}
        self.ndb_stations: Dict[str, NDBStation] = {}
        self.rnav_waypoints: Dict[str, RNAVWaypoint] = {}
        self.sid_procedures: Dict[str, Procedure] = {}
        self.star_procedures: Dict[str, Procedure] = {}
        
        self.load_navigation_data(navigation_file)
    
    def load_navigation_data(self, navigation_file: str):
        """Load navigation data from JSON file"""
        try:
            with open(navigation_file, 'r') as f:
                data = json.load(f)
            
            nav_data = data['navigation']
            self.airport_icao = nav_data['airport']
            
            # Load VOR stations
            for vor_id, vor_data in nav_data.get('vor_stations', {}).items():
                self.vor_stations[vor_id] = VORStation(vor_id, vor_data)
            
            # Load NDB stations
            for ndb_id, ndb_data in nav_data.get('ndb_stations', {}).items():
                self.ndb_stations[ndb_id] = NDBStation(ndb_id, ndb_data)
            
            # Load RNAV waypoints
            for waypoint_id, waypoint_data in nav_data.get('rnav_waypoints', {}).items():
                self.rnav_waypoints[waypoint_id] = RNAVWaypoint(waypoint_id, waypoint_data)
            
            # Load SID procedures
            for sid_id, sid_data in nav_data.get('sid_procedures', {}).items():
                self.sid_procedures[sid_id] = Procedure(sid_id, sid_data)
            
            # Load STAR procedures
            for star_id, star_data in nav_data.get('star_procedures', {}).items():
                self.star_procedures[star_id] = Procedure(star_id, star_data)
            
            print(f"Loaded navigation data for: {self.airport_icao}")
            print(f"  - VOR stations: {len(self.vor_stations)}")
            print(f"  - NDB stations: {len(self.ndb_stations)}")
            print(f"  - RNAV waypoints: {len(self.rnav_waypoints)}")
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
    
    def get_rnav_waypoint(self, identifier: str) -> Optional[RNAVWaypoint]:
        """Get RNAV waypoint by identifier"""
        return self.rnav_waypoints.get(identifier)
    
    def get_all_vor_stations(self) -> Dict[str, VORStation]:
        """Get all VOR stations"""
        return self.vor_stations
    
    def get_all_ndb_stations(self) -> Dict[str, NDBStation]:
        """Get all NDB stations"""
        return self.ndb_stations
    
    def get_all_rnav_waypoints(self) -> Dict[str, RNAVWaypoint]:
        """Get all RNAV waypoints"""
        return self.rnav_waypoints
    
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
    
    def get_rnav_waypoint(self, identifier: str) -> Optional[RNAVWaypoint]:
        """Get RNAV waypoint by identifier"""
        return self.rnav_waypoints.get(identifier)
    
    def get_vor_station(self, identifier: str) -> Optional[VORStation]:
        """Get VOR station by identifier"""
        return self.vor_stations.get(identifier) 