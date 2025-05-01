import random
import pygame
import gymnasium as gym
from gymnasium import spaces

pygame.init()

# Define colors
WHITE = (255, 255, 255)
GREY = (20, 20, 20)
BLACK = (0, 0, 0)
RED = (255, 0, 0)    # Color for the start point
GREEN = (0, 255, 0)  # Color for the end point
BLUE = (0, 0, 255)   # Color for the agent

# Set up the display based on the maze size
def set_screen_size(rows, cols, cell_size):
    width = cols * cell_size
    height = rows * cell_size
    return width, height

# Define the maze dimensions
rows = 20  # Number of rows
cols = 20  # Number of columns
cell_size = 25  # Cell width/height

# Adjust the screen size based on the number of rows and columns
size = set_screen_size(rows, cols, cell_size)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Maze Generator")

stack = []
clock = pygame.time.Clock()

class Maze:
    def __init__(self, x, y):
        self.x = x * cell_size
        self.y = y * cell_size
        self.visited = False
        self.current = False
        self.walls = [True, True, True, True]  # top, right, bottom, left

    def draw(self):
        if self.current:
            pygame.draw.rect(screen, RED, (self.x, self.y, cell_size, cell_size))  # Draw current cell
        elif self.visited:
            pygame.draw.rect(screen, WHITE, (self.x, self.y, cell_size, cell_size))  # Draw visited cell
            if self.walls[0]:
                pygame.draw.line(screen, BLACK, (self.x, self.y), (self.x + cell_size, self.y), 1)  # Top
            if self.walls[1]:
                pygame.draw.line(screen, BLACK, (self.x + cell_size, self.y), (self.x + cell_size, self.y + cell_size), 1)  # Right
            if self.walls[2]:
                pygame.draw.line(screen, BLACK, (self.x + cell_size, self.y + cell_size), (self.x, self.y + cell_size), 1)  # Bottom
            if self.walls[3]:
                pygame.draw.line(screen, BLACK, (self.x, self.y + cell_size), (self.x, self.y), 1)  # Left

    def check_neighbors(self):
        neighbors = []
        
        # Check top neighbor
        if int(self.y / cell_size) - 1 >= 0:
            top = grid[int(self.y / cell_size) - 1][int(self.x / cell_size)]
            if not top.visited:
                neighbors.append(top)
                
        # Check right neighbor
        if int(self.x / cell_size) + 1 <= cols - 1:
            right = grid[int(self.y / cell_size)][int(self.x / cell_size) + 1]
            if not right.visited:
                neighbors.append(right)
                
        # Check bottom neighbor
        if int(self.y / cell_size) + 1 <= rows - 1:
            bottom = grid[int(self.y / cell_size) + 1][int(self.x / cell_size)]
            if not bottom.visited:
                neighbors.append(bottom)
                
        # Check left neighbor
        if int(self.x / cell_size) - 1 >= 0:
            left = grid[int(self.y / cell_size)][int(self.x / cell_size) - 1]
            if not left.visited:
                neighbors.append(left)

        if neighbors:
            return random.choice(neighbors)
        else:
            return None

def remove_walls(current, next):
    dx = int(current.x / cell_size) - int(next.x / cell_size)
    dy = int(current.y / cell_size) - int(next.y / cell_size)
    if dx == -1:  # next is to the right
        current.walls[1] = False
        next.walls[3] = False
    elif dx == 1:  # next is to the left
        current.walls[3] = False
        next.walls[1] = False
    elif dy == -1:  # next is below
        current.walls[2] = False
        next.walls[0] = False
    elif dy == 1:  # next is above
        current.walls[0] = False
        next.walls[2] = False

# Create the grid
grid = [[Maze(x, y) for x in range(cols)] for y in range(rows)]

# Start the maze generation
current_cell = grid[0][0]
current_cell.visited = True

while True:
    next_cell = current_cell.check_neighbors()
    if next_cell:
        stack.append(current_cell)
        remove_walls(current_cell, next_cell)
        current_cell.current = False
        current_cell = next_cell
        current_cell.visited = True
        current_cell.current = True
    elif stack:
        current_cell.current = False
        current_cell = stack.pop()
    else:
        break

# Define the agent class
class Player:
    def __init__(self, start_x, start_y):
        self.x = start_x
        self.y = start_y

    def move(self, direction):
        current_cell = grid[self.y][self.x]
        if direction == 'up' and not current_cell.walls[0] and self.y > 0:
            self.y -= 1
        elif direction == 'right' and not current_cell.walls[1] and self.x < cols - 1:
            self.x += 1
        elif direction == 'down' and not current_cell.walls[2] and self.y < rows - 1:
            self.y += 1
        elif direction == 'left' and not current_cell.walls[3] and self.x > 0:
            self.x -= 1

    def draw(self):
        pygame.draw.circle(screen, BLUE, (self.x * cell_size + cell_size // 2, self.y * cell_size + cell_size // 2), cell_size // 4)

# Initialize the agent
agent = Player(0, 0)

# Define the end point
end_x, end_y = cols - 1, rows - 1  # Bottom-right corner

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                agent.move('up')
            elif event.key == pygame.K_RIGHT:
                agent.move('right')
            elif event.key == pygame.K_DOWN:
                agent.move('down')
            elif event.key == pygame.K_LEFT:
                agent.move('left')

    # Check if the agent has reached the end point
    if agent.x == end_x and agent.y == end_y:
        print("Agent has reached the end point! Reward given.")
        running = False  # End the game

    # Draw the maze
    screen.fill(GREY)
    for row in grid:
        for cell in row:
            cell.draw()

    # Draw the start point (top-left corner)
    pygame.draw.rect(screen, RED, (0, 0, cell_size, cell_size))  # Start point

    # Draw the end point (bottom-right corner)
    pygame.draw.rect(screen, GREEN, ((cols - 1) * cell_size, (rows - 1) * cell_size, cell_size, cell_size))  # End point

    agent.draw()

    pygame.display.flip()
    pygame.time.delay(200)  # Adjust the speed of agent movement

pygame.quit()