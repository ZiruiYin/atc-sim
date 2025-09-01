import math
import json
import os

from params import *

def nm_distance(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return 3440.065 * c

def latlon_to_xy(latlons, ref=None):
    if ref is None:
        ref = latlons[0]
    lat0, lon0 = math.radians(ref[0]), math.radians(ref[1])
    coords = []
    for lat, lon in latlons:
        dlat = math.radians(lat) - lat0
        dlon = math.radians(lon) - lon0
        x = dlon * math.cos(lat0) * 60 * (180/math.pi)
        y = dlat * 60 * (180/math.pi)
        coords.append((x, y))
    return coords

def generate_game_coordinates(
    screen_width=1600,
    screen_height=800,
    left_edge_ref_vor='CPT',
    distance_ref_vor_1='OCK',
    distance_ref_vor_2='BNN'
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    with open(os.path.join(data_dir, 'egll.json'), 'r') as f:
        airport_data = json.load(f)
    with open(os.path.join(data_dir, 'egll_navigation.json'), 'r') as f:
        navigation_data = json.load(f)
    all_coords = {}
    airport_lat = airport_data['airport']['coordinates']['latitude']
    airport_lon = airport_data['airport']['coordinates']['longitude']
    airport_ref = (airport_lat, airport_lon)
    all_coords['airport'] = {'coordinates': (airport_lat, airport_lon)}
    all_coords['runways'] = {}
    for pair_id, pair_data in airport_data['airport']['runways'].items():
        thresholds = pair_data.get('thresholds', {})
        th_dict = {}
        for end_id, end_data in thresholds.items():
            lat = end_data['latitude']
            lon = end_data['longitude']
            th_dict[end_id] = (lat, lon)
        all_coords['runways'][pair_id] = {'thresholds': th_dict}
    all_coords['vor_stations'] = {}
    dist_vor1_coord = None
    dist_vor2_coord = None
    for vor_id, vor_data in navigation_data['navigation']['vor_stations'].items():
        lat = vor_data['coordinates']['latitude']
        lon = vor_data['coordinates']['longitude']
        all_coords['vor_stations'][vor_id] = {
            'coordinates': (lat, lon),
            'name': vor_data['name']
        }
        if vor_id == distance_ref_vor_1:
            dist_vor1_coord = (lat, lon)
        if vor_id == distance_ref_vor_2:
            dist_vor2_coord = (lat, lon)
    all_coords['rnav_waypoints'] = {}
    for wpt_id, wpt_data in navigation_data['navigation']['rnav_waypoints'].items():
        all_coords['rnav_waypoints'][wpt_id] = {
            'coordinates': (
                wpt_data['coordinates']['latitude'],
                wpt_data['coordinates']['longitude']
            ),
            'name': wpt_data['name'],
            'type': wpt_data['type']
        }
    all_coords['ndb_stations'] = {}
    for ndb_id, ndb_data in navigation_data['navigation']['ndb_stations'].items():
        all_coords['ndb_stations'][ndb_id] = {
            'coordinates': (
                ndb_data['coordinates']['latitude'],
                ndb_data['coordinates']['longitude']
            ),
            'name': ndb_data['name']
        }
    all_latlon_pairs = [airport_ref]
    coord_mapping = {'airport': 0}
    runway_mapping = {}
    for pair_id, pair_data in all_coords['runways'].items():
        runway_mapping[pair_id] = {}
        for end_id, latlon in pair_data['thresholds'].items():
            runway_mapping[pair_id][end_id] = len(all_latlon_pairs)
            all_latlon_pairs.append(latlon)
    vor_mapping = {}
    for vor_id, vor_data in all_coords['vor_stations'].items():
        vor_mapping[vor_id] = len(all_latlon_pairs)
        all_latlon_pairs.append(vor_data['coordinates'])
    wpt_mapping = {}
    for wpt_id, wpt_data in all_coords['rnav_waypoints'].items():
        wpt_mapping[wpt_id] = len(all_latlon_pairs)
        all_latlon_pairs.append(wpt_data['coordinates'])
    ndb_mapping = {}
    for ndb_id, ndb_data in all_coords['ndb_stations'].items():
        ndb_mapping[ndb_id] = len(all_latlon_pairs)
        all_latlon_pairs.append(ndb_data['coordinates'])
    xy_coords = latlon_to_xy(all_latlon_pairs, airport_ref)
    ref_vor_xy = xy_coords[vor_mapping[left_edge_ref_vor]]
    ref_vor_target_x = 0.1 * screen_width - screen_width / 2
    scale_factor = ref_vor_target_x / ref_vor_xy[0] if ref_vor_xy[0] != 0 else 1.0
    scaled_coords = [(x * scale_factor, y * scale_factor) for x, y in xy_coords]
    game_coords = [(x + screen_width / 2, -y + screen_height / 2) for x, y in scaled_coords]
    dist_vor1_game = game_coords[vor_mapping[distance_ref_vor_1]]
    dist_vor2_game = game_coords[vor_mapping[distance_ref_vor_2]]
    pixel_distance = math.hypot(dist_vor1_game[0] - dist_vor2_game[0],
                                dist_vor1_game[1] - dist_vor2_game[1])
    nm_distance_vors = nm_distance(dist_vor1_coord[0], dist_vor1_coord[1],
                                   dist_vor2_coord[0], dist_vor2_coord[1])
    nm_per_pixel = nm_distance_vors / pixel_distance if pixel_distance != 0 else 0.0
    output_data = {
        'screen_info': {
            'width': screen_width,
            'height': screen_height,
            'nm_per_pixel': nm_per_pixel
        },
        'airport': {
            'icao': airport_data['airport']['icao'],
            'name': airport_data['airport']['name'],
            'coordinates': {
                'x': game_coords[coord_mapping['airport']][0],
                'y': game_coords[coord_mapping['airport']][1]
            }
        },
        'runways': {},
        'vor_stations': {},
        'ndb_stations': {},
        'rnav_waypoints': {}
    }
    for pair_id, ends_map in runway_mapping.items():
        output_data['runways'][pair_id] = {'thresholds': {}}
        for end_id, coord_idx in ends_map.items():
            output_data['runways'][pair_id]['thresholds'][end_id] = {
                'x': game_coords[coord_idx][0],
                'y': game_coords[coord_idx][1]
            }
    for vor_id, vor_data in all_coords['vor_stations'].items():
        coord_idx = vor_mapping[vor_id]
        output_data['vor_stations'][vor_id] = {
            'name': vor_data['name'],
            'coordinates': {
                'x': game_coords[coord_idx][0],
                'y': game_coords[coord_idx][1]
            }
        }
    for wpt_id, _ in all_coords['rnav_waypoints'].items():
        coord_idx = wpt_mapping[wpt_id]
        output_data['rnav_waypoints'][wpt_id] = {
            'name': navigation_data['navigation']['rnav_waypoints'][wpt_id]['name'],
            'coordinates': {
                'x': game_coords[coord_idx][0],
                'y': game_coords[coord_idx][1]
            }
        }
    for ndb_id, _ in all_coords['ndb_stations'].items():
        coord_idx = ndb_mapping[ndb_id]
        output_data['ndb_stations'][ndb_id] = {
            'name': navigation_data['navigation']['ndb_stations'][ndb_id]['name'],
            'coordinates': {
                'x': game_coords[coord_idx][0],
                'y': game_coords[coord_idx][1]
            }
        }
    output_file = os.path.join(data_dir, 'egll_game.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    return output_data

if __name__ == "__main__":
    generate_game_coordinates()
