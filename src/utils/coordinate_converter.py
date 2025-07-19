#only need to be run once, all done in github repo.

import json
import math
import os
from typing import Dict, Any, Tuple


def latlon_to_xy_nm(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """
    Convert lat/lon to local x/y coordinates in nautical miles
    relative to center lat0/lon0.
    
    Parameters:
        lat, lon   : point to convert (in degrees)
        lat0, lon0 : center point (in degrees)
        
    Returns:
        (x, y) : coordinates in nautical miles
    """
    R = 3440  # Earth radius in nautical miles

    # Convert degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)

    # Equirectangular projection
    x = R * (lon_rad - lon0_rad) * math.cos(lat0_rad)
    y = R * (lat_rad - lat0_rad)

    return x, y


def add_nm_coordinates_to_airport_data(airport_file_path: str):
    """
    Add nautical mile coordinates to airport data file
    
    Args:
        airport_file_path: Path to the airport JSON file
    """
    # Read the airport data
    with open(airport_file_path, 'r') as f:
        data = json.load(f)
    
    airport = data['airport']
    center_lat = airport['coordinates']['latitude']
    center_lon = airport['coordinates']['longitude']
    
    # Add NM coordinates to airport center (should be 0,0)
    nm_coords = latlon_to_xy_nm(center_lat, center_lon, center_lat, center_lon)
    airport['coordinates']['nm_coordinates'] = {
        'x': round(nm_coords[0], 2),
        'y': round(nm_coords[1], 2)
    }
    
    # Add NM coordinates to runways
    for runway_name, runway_data in airport['runways'].items():
        # Threshold coordinates
        threshold_lat = runway_data['coordinates']['threshold']['latitude']
        threshold_lon = runway_data['coordinates']['threshold']['longitude']
        threshold_nm = latlon_to_xy_nm(threshold_lat, threshold_lon, center_lat, center_lon)
        runway_data['coordinates']['threshold']['nm_coordinates'] = {
            'x': round(threshold_nm[0], 2),
            'y': round(threshold_nm[1], 2)
        }
        
        # End coordinates
        end_lat = runway_data['coordinates']['end']['latitude']
        end_lon = runway_data['coordinates']['end']['longitude']
        end_nm = latlon_to_xy_nm(end_lat, end_lon, center_lat, center_lon)
        runway_data['coordinates']['end']['nm_coordinates'] = {
            'x': round(end_nm[0], 2),
            'y': round(end_nm[1], 2)
        }
    
    # Write back to file
    with open(airport_file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Added NM coordinates to {airport_file_path}")


def add_nm_coordinates_to_navigation_data(navigation_file_path: str):
    """
    Add nautical mile coordinates to navigation data file
    
    Args:
        navigation_file_path: Path to the navigation JSON file
    """
    # Read the navigation data
    with open(navigation_file_path, 'r') as f:
        data = json.load(f)
    
    # Get airport center coordinates (we'll need to read from airport file)
    airport_file_path = os.path.join(os.path.dirname(navigation_file_path), '..', 'airports', 'egll.json')
    with open(airport_file_path, 'r') as f:
        airport_data = json.load(f)
    
    center_lat = airport_data['airport']['coordinates']['latitude']
    center_lon = airport_data['airport']['coordinates']['longitude']
    
    # Add NM coordinates to VOR stations
    for vor_id, vor_data in data['navigation']['vor_stations'].items():
        lat = vor_data['coordinates']['latitude']
        lon = vor_data['coordinates']['longitude']
        nm_coords = latlon_to_xy_nm(lat, lon, center_lat, center_lon)
        vor_data['coordinates']['nm_coordinates'] = {
            'x': round(nm_coords[0], 2),
            'y': round(nm_coords[1], 2)
        }
    
    # Add NM coordinates to NDB stations
    for ndb_id, ndb_data in data['navigation']['ndb_stations'].items():
        lat = ndb_data['coordinates']['latitude']
        lon = ndb_data['coordinates']['longitude']
        nm_coords = latlon_to_xy_nm(lat, lon, center_lat, center_lon)
        ndb_data['coordinates']['nm_coordinates'] = {
            'x': round(nm_coords[0], 2),
            'y': round(nm_coords[1], 2)
        }
    
    # Add NM coordinates to RNAV waypoints
    if 'rnav_waypoints' in data['navigation']:
        for wp_id, wp_data in data['navigation']['rnav_waypoints'].items():
            lat = wp_data['coordinates']['latitude']
            lon = wp_data['coordinates']['longitude']
            nm_coords = latlon_to_xy_nm(lat, lon, center_lat, center_lon)
            wp_data['coordinates']['nm_coordinates'] = {
                'x': round(nm_coords[0], 2),
                'y': round(nm_coords[1], 2)
            }
    
    # Write back to file
    with open(navigation_file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Added NM coordinates to {navigation_file_path}")


def convert_all_coordinates():
    """
    Convert all coordinates in both airport and navigation files to nautical mile coordinates
    """
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths to data files
    airport_file = os.path.join(current_dir, '..', 'data', 'airports', 'egll.json')
    navigation_file = os.path.join(current_dir, '..', 'data', 'navigation', 'egll_navigation.json')
    
    print("Converting coordinates to nautical miles...")
    print(f"Airport file: {airport_file}")
    print(f"Navigation file: {navigation_file}")
    
    # Convert airport data
    add_nm_coordinates_to_airport_data(airport_file)
    
    # Convert navigation data
    add_nm_coordinates_to_navigation_data(navigation_file)
    
    print("Coordinate conversion complete!")


if __name__ == "__main__":
    # Run the conversion
    convert_all_coordinates() 