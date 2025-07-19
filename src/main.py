#!/usr/bin/env python3
"""
ATC Simulator - Main Application
London Heathrow (EGLL) Air Traffic Control Simulator
"""

import pygame
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.airport import Airport
from core.navigation import Navigation
from ui.radar_display import RadarDisplay


def main():
    """Main application entry point"""
    print("=" * 50)
    print("ATC Simulator - London Heathrow (EGLL)")
    print("=" * 50)
    
    # Initialize pygame
    pygame.init()
    
    try:
        # Load airport data
        print("Loading airport data...")
        airport_file = "data/airports/egll.json"
        if not os.path.exists(airport_file):
            print(f"Error: Airport data file not found: {airport_file}")
            return
        
        airport = Airport(airport_file)
        
        # Load navigation data
        print("Loading navigation data...")
        navigation_file = "data/navigation/egll_navigation.json"
        if not os.path.exists(navigation_file):
            print(f"Error: Navigation data file not found: {navigation_file}")
            return
        
        navigation = Navigation(navigation_file)
        
        # Create radar display
        print("Initializing radar display...")
        info = pygame.display.Info()
        screen_width = info.current_w
        screen_height = info.current_h
        radar = RadarDisplay(screen_width, screen_height)
        radar.set_airport(airport)
        radar.set_navigation(navigation)
        radar.set_spawn_rate(2.0)  # 2 aircraft per minute
        
        print("\nControls:")
        print("  V - Toggle VOR stations")
        print("  N - Toggle NDB stations")
        print("  W - Toggle waypoints")
        print("  I - Toggle ILS approaches")
        print("  G - Toggle grid")
        print("  R - Toggle range rings")
        print("  A - Toggle airport name")
        print("  T - Toggle legend")
        print("  P - Cycle procedures (SID/STAR)")
        print("  L - Lock/unlock display controls")
        print("  ↑/↓ - Increase/decrease spawn rate")
        print("  ESC - Exit")
        print("\nSpawn Rate: 2.0 aircraft per minute")
        print("\nCommand Input:")
        print("  Press L to lock controls and enable typing")
        print("  Type commands and press ENTER to execute")
        print('  Type "unlock" and press ENTER to unlock display controls')
        print("  Click on aircraft to select and auto-fill callsign")
        print("\nATC Commands:")
        print("  DL1 C 035    - Set heading to 035°")
        print("  DL1 C 3      - Cleared to 3,000 feet")
        print("  DL1 C 12     - Cleared to 12,000 feet")
        print("  DL1 C 3 X    - Cleared to 3,000 feet, expedite")
        print("  DL1 C OCK    - Set course to OCK VOR")
        print("  DL1 C 090 R  - Turn right to heading 090°")
        print("  DL1 L 27R    - Cleared to land runway 27R")
        print("  DL1 A        - Abort")
        print("  DL1 C OCK C 3 S 200 - Chained commands")
        print("\nRadar display initialized successfully!")
        
        # Main game loop
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Calculate delta time
            dt = clock.tick(60) / 1000.0  # Convert to seconds
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        # Check if event should exit (crash occurred)
                        if radar.handle_event(event):
                            running = False
                else:
                    # Pass all other events (including mouse events) to radar display
                    if radar.handle_event(event):
                        running = False
            
            # Update aircraft
            radar.update_aircraft(dt)
            
            # Render display
            radar.render()
        
        print("\nShutting down ATC Simulator...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main() 