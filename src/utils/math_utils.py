import math
from typing import Tuple


def lat_lon_to_pixels(lat: float, lon: float, center_lat: float, center_lon: float, 
                     screen_width: int, screen_height: int, scale_factor: float = 800) -> Tuple[int, int]:
    """
    Convert latitude/longitude coordinates to screen pixels
    
    Args:
        lat, lon: Target coordinates
        center_lat, center_lon: Center of the display (usually airport coordinates)
        screen_width, screen_height: Screen dimensions
        scale_factor: Pixels per degree (adjust for zoom level)
    
    Returns:
        Tuple of (x, y) screen coordinates
    """
    # Calculate relative position from center
    dx = (lon - center_lon) * scale_factor
    dy = (center_lat - lat) * scale_factor  # Inverted Y for screen coordinates
    
    # Convert to screen coordinates
    x = int(screen_width / 2 + dx)
    y = int(screen_height / 2 + dy)
    
    return (x, y)


def pixels_to_lat_lon(x: int, y: int, center_lat: float, center_lon: float,
                     screen_width: int, screen_height: int, scale_factor: float = 1000) -> Tuple[float, float]:
    """
    Convert screen pixels to latitude/longitude coordinates
    
    Args:
        x, y: Screen coordinates
        center_lat, center_lon: Center of the display
        screen_width, screen_height: Screen dimensions
        scale_factor: Pixels per degree
    
    Returns:
        Tuple of (lat, lon) coordinates
    """
    # Convert screen coordinates to relative position
    dx = (x - screen_width / 2) / scale_factor
    dy = (y - screen_height / 2) / scale_factor
    
    # Calculate actual coordinates
    lon = center_lon + dx
    lat = center_lat - dy  # Inverted Y
    
    return (lat, lon)


def distance_between_points(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two lat/lon points using Haversine formula
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        Distance in nautical miles
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in nautical miles
    earth_radius_nm = 3440.065
    
    return earth_radius_nm * c


def bearing_between_points(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate bearing from point 1 to point 2
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        Bearing in degrees (0-360)
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    # Normalize to 0-360 degrees
    return (bearing_deg + 360) % 360


def normalize_heading(heading: float) -> float:
    """
    Normalize heading to 0-360 degrees
    
    Args:
        heading: Heading in degrees
    
    Returns:
        Normalized heading (0-360)
    """
    return heading % 360


def pixels_to_nautical_miles(pixel_distance: float, scale_factor: float = 800) -> float:
    """
    Convert pixel distance to nautical miles
    
    Args:
        pixel_distance: Distance in pixels
        scale_factor: Pixels per degree (default 800)
    
    Returns:
        Distance in nautical miles
    """
    # Convert pixels to degrees
    degrees = pixel_distance / scale_factor
    
    # At London latitude (51.5°N), convert degrees to nautical miles
    # For latitude: 1 degree = 60 nautical miles (constant)
    # For longitude: 1 degree ≈ 60 * cos(51.5°) ≈ 37.4 nautical miles
    lat_nm_per_degree = 60.0
    lon_nm_per_degree = 60.0 * math.cos(math.radians(51.5))
    
    # Use the average of lat/lon conversion for general distance
    nm_per_degree = (lat_nm_per_degree + lon_nm_per_degree) / 2
    
    return degrees * nm_per_degree


def nautical_miles_to_pixels(nm_distance: float, scale_factor: float = 800) -> float:
    """
    Convert nautical miles to pixel distance
    
    Args:
        nm_distance: Distance in nautical miles
        scale_factor: Pixels per degree (default 800)
    
    Returns:
        Distance in pixels
    """
    # Convert nautical miles to degrees
    lat_nm_per_degree = 60.0
    lon_nm_per_degree = 60.0 * math.cos(math.radians(51.5))
    nm_per_degree = (lat_nm_per_degree + lon_nm_per_degree) / 2
    
    degrees = nm_distance / nm_per_degree
    
    # Convert degrees to pixels
    return degrees * scale_factor


def nm_to_screen_coords(nm_x: float, nm_y: float, screen_width: int, screen_height: int, 
                       nm_per_pixel: float = 1.0) -> Tuple[int, int]:
    """
    Convert NM coordinates to Pygame screen coordinates
    
    Args:
        nm_x, nm_y: Coordinates in nautical miles (EGLL at 0,0)
        screen_width, screen_height: Screen dimensions in pixels
        nm_per_pixel: Scale factor (how many NM per pixel)
    
    Returns:
        (screen_x, screen_y) coordinates
    """
    # Convert NM to pixels
    pixel_x = nm_x / nm_per_pixel
    pixel_y = nm_y / nm_per_pixel
    
    # Center on screen (since NM coords are centered at 0,0)
    screen_x = int(screen_width / 2 + pixel_x)
    screen_y = int(screen_height / 2 - pixel_y)  # Invert Y for Pygame (positive Y = up in NM, down in screen)
    
    return (screen_x, screen_y)


def screen_to_nm_coords(screen_x: int, screen_y: int, screen_width: int, screen_height: int,
                       nm_per_pixel: float = 1.0) -> Tuple[float, float]:
    """
    Convert Pygame screen coordinates to NM coordinates
    
    Args:
        screen_x, screen_y: Screen coordinates in pixels
        screen_width, screen_height: Screen dimensions in pixels
        nm_per_pixel: Scale factor (how many NM per pixel)
    
    Returns:
        (nm_x, nm_y) coordinates
    """
    # Convert screen coordinates to relative position from center
    pixel_x = screen_x - screen_width / 2
    pixel_y = screen_height / 2 - screen_y  # Invert Y
    
    # Convert pixels to NM
    nm_x = pixel_x * nm_per_pixel
    nm_y = pixel_y * nm_per_pixel
    
    return (nm_x, nm_y)


def generate_game_coordinates(screen_width: int, screen_height: int, nm_per_pixel: float = 1.0):
    """
    Generate game coordinates from raw airport and navigation data
    
    Args:
        screen_width, screen_height: Screen dimensions in pixels
        nm_per_pixel: Scale factor (how many NM per pixel)
    """
    import json
    import os
    from pathlib import Path
    
    # Get paths
    current_dir = Path(__file__).parent
    airport_file = current_dir / '..' / 'data' / 'airports' / 'egll.json'
    navigation_file = current_dir / '..' / 'data' / 'navigation' / 'egll_navigation.json'
    output_file = current_dir / '..' / 'data' / 'game_coordinates.json'
    
    print(f"Processing coordinates with screen: {screen_width}x{screen_height}, scale: {nm_per_pixel} nm/pixel")
    
    # Load raw data
    with open(airport_file, 'r') as f:
        airport_data = json.load(f)
    
    with open(navigation_file, 'r') as f:
        nav_data = json.load(f)
    
    # Create game data structure
    game_data = {
        "metadata": {
            "screen_width": screen_width,
            "screen_height": screen_height,
            "nm_per_pixel": nm_per_pixel,
            "generated_from": ["egll.json", "egll_navigation.json"]
        },
        "airport": {
            "icao": airport_data["airport"]["icao"],
            "name": airport_data["airport"]["name"],
            "coordinates": {
                "x": screen_width // 2,  # Airport is always at screen center
                "y": screen_height // 2
            },
            "runways": {}
        },
        "navigation": {
            "vor_stations": {},
            "ndb_stations": {},
            "waypoints": {},
            "procedures": {
                "sid": {},
                "star": {}
            }
        }
    }
    
    # Process runways - convert NM to screen coordinates
    for runway_name, runway_data in airport_data["airport"]["runways"].items():
        if "nm_coordinates" in runway_data["coordinates"]["threshold"]:
            # Convert NM coordinates to screen coordinates
            threshold_nm_x = runway_data["coordinates"]["threshold"]["nm_coordinates"]["x"]
            threshold_nm_y = runway_data["coordinates"]["threshold"]["nm_coordinates"]["y"]
            threshold_screen_x, threshold_screen_y = nm_to_screen_coords(
                threshold_nm_x, threshold_nm_y, screen_width, screen_height, nm_per_pixel
            )
            
            end_nm_x = runway_data["coordinates"]["end"]["nm_coordinates"]["x"]
            end_nm_y = runway_data["coordinates"]["end"]["nm_coordinates"]["y"]
            end_screen_x, end_screen_y = nm_to_screen_coords(
                end_nm_x, end_nm_y, screen_width, screen_height, nm_per_pixel
            )
            
            game_data["airport"]["runways"][runway_name] = {
                "name": runway_name,
                "heading": runway_data["heading"],
                "threshold": {
                    "x": threshold_screen_x,
                    "y": threshold_screen_y
                },
                "end": {
                    "x": end_screen_x,
                    "y": end_screen_y
                }
            }
    
    # Process VOR stations - convert NM to screen coordinates
    for vor_id, vor_data in nav_data["navigation"]["vor_stations"].items():
        if "nm_coordinates" in vor_data["coordinates"]:
            nm_x = vor_data["coordinates"]["nm_coordinates"]["x"]
            nm_y = vor_data["coordinates"]["nm_coordinates"]["y"]
            screen_x, screen_y = nm_to_screen_coords(nm_x, nm_y, screen_width, screen_height, nm_per_pixel)
            
            game_data["navigation"]["vor_stations"][vor_id] = {
                "name": vor_data["name"],
                "frequency": vor_data["frequency"],
                "coordinates": {
                    "x": screen_x,
                    "y": screen_y
                }
            }
    
    # Process NDB stations - convert NM to screen coordinates
    for ndb_id, ndb_data in nav_data["navigation"]["ndb_stations"].items():
        if "nm_coordinates" in ndb_data["coordinates"]:
            nm_x = ndb_data["coordinates"]["nm_coordinates"]["x"]
            nm_y = ndb_data["coordinates"]["nm_coordinates"]["y"]
            screen_x, screen_y = nm_to_screen_coords(nm_x, nm_y, screen_width, screen_height, nm_per_pixel)
            
            game_data["navigation"]["ndb_stations"][ndb_id] = {
                "name": ndb_data["name"],
                "frequency": ndb_data["frequency"],
                "coordinates": {
                    "x": screen_x,
                    "y": screen_y
                }
            }
    
    # Process RNAV waypoints - convert NM to screen coordinates
    for wp_id, wp_data in nav_data["navigation"]["rnav_waypoints"].items():
        if "nm_coordinates" in wp_data["coordinates"]:
            nm_x = wp_data["coordinates"]["nm_coordinates"]["x"]
            nm_y = wp_data["coordinates"]["nm_coordinates"]["y"]
            screen_x, screen_y = nm_to_screen_coords(nm_x, nm_y, screen_width, screen_height, nm_per_pixel)
            
            game_data["navigation"]["waypoints"][wp_id] = {
                "name": wp_data["name"],
                "type": wp_data["type"],
                "coordinates": {
                    "x": screen_x,
                    "y": screen_y
                }
            }
    
    # Process SID procedures
    for sid_id, sid_data in nav_data["navigation"]["sid_procedures"].items():
        game_data["navigation"]["procedures"]["sid"][sid_id] = {
            "name": sid_data["name"],
            "runway": sid_data["runway"],
            "route": sid_data["route"],
            "initial_altitude": sid_data["initial_altitude"],
            "final_altitude": sid_data["final_altitude"]
        }
    
    # Process STAR procedures
    for star_id, star_data in nav_data["navigation"]["star_procedures"].items():
        game_data["navigation"]["procedures"]["star"][star_id] = {
            "name": star_data["name"],
            "runway": star_data["runway"],
            "route": star_data["route"],
            "initial_altitude": star_data["initial_altitude"],
            "final_altitude": star_data["final_altitude"]
        }
    
    # Write game coordinates file
    with open(output_file, 'w') as f:
        json.dump(game_data, f, indent=2)
    
    print(f"Game coordinates generated: {output_file}")
    print(f"  - Airport: {game_data['airport']['icao']} at screen center ({screen_width//2}, {screen_height//2})")
    print(f"  - Runways: {len(game_data['airport']['runways'])}")
    print(f"  - VOR stations: {len(game_data['navigation']['vor_stations'])}")
    print(f"  - NDB stations: {len(game_data['navigation']['ndb_stations'])}")
    print(f"  - Waypoints: {len(game_data['navigation']['waypoints'])}")
    print(f"  - SID procedures: {len(game_data['navigation']['procedures']['sid'])}")
    print(f"  - STAR procedures: {len(game_data['navigation']['procedures']['star'])}")
    
    return str(output_file)


def heading_difference(heading1: float, heading2: float) -> float:
    """
    Calculate the shortest angular difference between two headings
    
    Args:
        heading1, heading2: Headings in degrees
    
    Returns:
        Difference in degrees (-180 to +180)
    """
    diff = heading2 - heading1
    
    # Normalize to -180 to +180
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    
    return diff