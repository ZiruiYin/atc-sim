import sys
import pygame
from display.pygame_display import RadarDisplay

def main():
    pygame.init()
    airport_name = "egll"
    info = pygame.display.Info()
    screen_width = info.current_w
    screen_height = info.current_h
    radar = RadarDisplay(screen_width, screen_height, airport_name)
    clock = pygame.time.Clock()

    running = True
    UPDATE_INTERVAL = 0.25
    accumulator = 0.0
    while running:
        dt = clock.tick(60) / 1000.0
        accumulator += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            radar.handle_event(event)
        while accumulator >= UPDATE_INTERVAL:
            radar.update(UPDATE_INTERVAL)
            accumulator -= UPDATE_INTERVAL
        pygame.display.flip()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()