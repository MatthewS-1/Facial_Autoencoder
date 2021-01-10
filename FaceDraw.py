import pygame
import numpy as np

pygame.init()

rows = 128

black = (0, 0, 0)
gray = (180, 180, 180)
colors = {
    "black": black,
    "gray": gray
}


def create_grid(rows):
    grid = [["gray" for _ in range(rows)] for _ in range(rows)]
    return grid


def draw_grid(display, grid):
    width, height = (pygame.display.get_surface().get_size())
    width, height = width // rows, height // rows
    for y in range(rows):
        for x in range(rows):
            color = colors[grid[y][x]]
            pygame.draw.rect(display, color, (x * width, y * height, width, height), 0)


def int_grid(grid):
    for y in range(rows):
        for x in range(rows):
            grid[y][x] = colors[grid[y][x]][0]
    return grid


def main():
    grid = create_grid(rows=rows)
    display = pygame.display.set_mode((rows * 5, rows * 5))
    width, height = (pygame.display.get_surface().get_size())
    width, height = width // rows, height // rows

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return int_grid(grid)
                # quit()
            if pygame.mouse.get_pressed()[0]:
                position = pygame.mouse.get_pos()
                grid[position[1] // height][position[0] // width] = "black"

        draw_grid(display, grid)

        pygame.display.flip()


if __name__ == "__main__":
    grid = main()
    grid = np.array(grid)
    for y in grid:
        print(y)
    np.save("user_drawn_image", grid)
    loaded = np.load('user_drawn_image.npy')
    print(loaded.shape)
