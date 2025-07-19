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