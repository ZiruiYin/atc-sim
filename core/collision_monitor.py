from utils import *
import math

class CollisionMonitor:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.nm_per_pixel = get_nm_per_pixel()

        self.min_separation_pixel = 3 / self.nm_per_pixel
        self.grid_width = self.min_separation_pixel / math.sqrt(2)

        self.grid_cols = math.ceil(self.screen_width / self.grid_width)
        self.grid_rows = math.ceil(self.screen_height / self.grid_width)
        
        self.grids = [[[] for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]

    def clear_grids(self):
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                self.grids[row][col].clear()

    def place_aircraft_in_grid(self, aircraft):
        x, y = aircraft.x, aircraft.y
        
        grid_col = max(0, min(int(x // self.grid_width), self.grid_cols - 1))
        grid_row = max(0, min(int(y // self.grid_width), self.grid_rows - 1))
        
        self.grids[grid_row][grid_col].append(aircraft)

    def get_neighboring_grids(self, grid_row, grid_col):
        neighbors = []
        
        for row_offset in [-1, 0, 1]:
            for col_offset in [-1, 0, 1]:
                neighbor_row = grid_row + row_offset
                neighbor_col = grid_col + col_offset
                
                if (0 <= neighbor_row < self.grid_rows and 
                    0 <= neighbor_col < self.grid_cols):
                    neighbors.append((neighbor_row, neighbor_col))
        
        return neighbors

    def check_collisions(self, aircraft_list):
        for aircraft in aircraft_list:
            aircraft.collision_warning = False
        
        self.clear_grids()
        for aircraft in aircraft_list:
            self.place_aircraft_in_grid(aircraft)
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                current_grid_aircraft = self.grids[row][col]
                
                self._check_aircraft_pairs_in_grid(current_grid_aircraft)
                neighbors = self.get_neighboring_grids(row, col)
                for neighbor_row, neighbor_col in neighbors:
                    if neighbor_row == row and neighbor_col == col:
                        continue
                    
                    neighbor_aircraft = self.grids[neighbor_row][neighbor_col]
                    self._check_aircraft_pairs_between_grids(current_grid_aircraft, neighbor_aircraft)

    def _check_aircraft_pairs_in_grid(self, aircraft_list):
        for i in range(len(aircraft_list)):
            for j in range(i + 1, len(aircraft_list)):
                aircraft1 = aircraft_list[i]
                aircraft2 = aircraft_list[j]
                self._check_aircraft_pair(aircraft1, aircraft2)

    def _check_aircraft_pairs_between_grids(self, grid1_aircraft, grid2_aircraft):
        for aircraft1 in grid1_aircraft:
            for aircraft2 in grid2_aircraft:
                distance = distance_between_coords_pixels(aircraft1.x, aircraft1.y, aircraft2.x, aircraft2.y)
                if distance < self.min_separation_pixel:
                    self._check_aircraft_pair(aircraft1, aircraft2)

    def _check_aircraft_pair(self, aircraft1, aircraft2):
        vertical_separation = abs(aircraft1.altitude - aircraft2.altitude)
        
        collision_warning = False
        
        if vertical_separation < 1000 and not aircraft1.ils_runway and not aircraft2.ils_runway:
            collision_warning = True
        
        if aircraft1.on_ground and aircraft2.on_ground and aircraft1.ils_runway == aircraft2.ils_runway and aircraft1.ils_runway is not None:
            collision_warning = True
        
        if collision_warning:
            aircraft1.collision_warning = True
            aircraft2.collision_warning = True
        
        if vertical_separation <= 40:
            pixel_distance = distance_between_coords_pixels(aircraft1.x, aircraft1.y, aircraft2.x, aircraft2.y)
            crash_threshold_pixels = 0.05 / self.nm_per_pixel
            
            if pixel_distance <= crash_threshold_pixels:
                aircraft1.crash = f"collided with {aircraft2.callsign}"
                aircraft2.crash = f"collided with {aircraft1.callsign}"