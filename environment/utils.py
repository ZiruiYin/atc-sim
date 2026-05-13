import math


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