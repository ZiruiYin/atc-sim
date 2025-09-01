import json
import os

import math

def get_nm_per_pixel(airport_name="egll"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    game_file = os.path.join(data_dir, f'{airport_name}_game.json')
    with open(game_file, 'r') as f:
        data = json.load(f)
    return data['screen_info']['nm_per_pixel']

def get_coords(airport_name="egll"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    game_file = os.path.join(data_dir, f'{airport_name}_game.json')
    with open(game_file, 'r') as f:
        data = json.load(f)
    coords = {}
    coords['airport'] = data['airport']['coordinates']
    for _, rwy_data in data['runways'].items():
        for thr, thr_data in rwy_data['thresholds'].items():
            coords[thr] = thr_data
    for vor, vor_data in data['vor_stations'].items():
        coords[vor] = vor_data['coordinates']
    for ndb, ndb_data in data['ndb_stations'].items():
        coords[ndb] = ndb_data['coordinates']
    for wpt, wpt_data in data['rnav_waypoints'].items():
        coords[wpt] = wpt_data['coordinates']
    return coords
    
def ias_to_gs(ias, alt):
    return ias * (1 + 0.02 * (alt / 1000))

def distance_between_coords_pixels(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.hypot(dx, dy)

def get_bearing_from_coords(x1, y1, x2, y2):
    angle = math.degrees(math.atan2(x2 - x1, y1 - y2))
    return angle % 360

def opposite_sides(b1, b2, b3):
    d1 = (b1 - b3 + 180) % 360 - 180
    d2 = (b2 - b3 + 180) % 360 - 180
    return (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)

def projection_distance(x1, y1, heading_deg, x2, y2):
    theta = math.radians(heading_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    vx, vy = x2 - x1, y2 - y1
    return vx * dx + vy * dy

def heading_diff(h1, h2):
    return abs((h1 - h2 + 180) % 360 - 180)