import pygame
import numpy as np
import random
from gymnasium import spaces
import gymnasium as gym
from maze.envs.maze_model import MazeModel

from maze.envs.config import (WIDTH, SIZE, COLS, ROWS, AGENT_SIZE, 
START_END_SIZE, WHITE, GREY, BLACK, RED, GREEN, BLUE)

class MazeEnv(gym.Env):
    """Custom Maze Environment that follows gymnasium interface"""
    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 5}

    class MazeCell:
        """Class in which is supposed to represent each individual cell i.e., square
        within the maze grid; manages key information about each cell, their state and structure."""
        def __init__(self, x, y):  # Initializes each cell with its x and y position in the grid
            self.x = x * WIDTH  # The x-coordinate of the cell's position, scaled by WIDTH (cell size in pixels)
            self.y = y * WIDTH  # The y-coordinate of the cell's position, scaled by WIDTH
            
            # Maze generation attributes
            self.visited = False  # Tracks whether the cell has been visited during maze generation
            self.current = False  # Indicates if the cell is currently active in maze generation
            
            # Walls configuration
            self.walls = [True, True, True, True]  # top, right, bottom, left, `True` means the wall exists, while `False` means it has been removed

        def draw(self, screen):  # Method to draw the cell and its walls onto the Pygame screen
            if self.current:  # If the cell is the currently active cell during maze generation:
                pygame.draw.rect(screen, RED, (self.x, self.y, WIDTH, WIDTH))  # Draws the cell in red
            elif self.visited:  # If the cell has been visited but is not currently active:
                pygame.draw.rect(screen, WHITE, (self.x, self.y, WIDTH, WIDTH))  # Draws the cell in white
            
            # Draw walls if they exist
            if self.walls[0]:
                pygame.draw.line(screen, BLACK, (self.x, self.y), (self.x + WIDTH, self.y), 1)
            if self.walls[1]:
                pygame.draw.line(screen, BLACK, (self.x + WIDTH, self.y), (self.x + WIDTH, self.y + WIDTH), 1)
            if self.walls[2]:
                pygame.draw.line(screen, BLACK, (self.x + WIDTH, self.y + WIDTH), (self.x, self.y + WIDTH), 1)
            if self.walls[3]:
                pygame.draw.line(screen, BLACK, (self.x, self.y + WIDTH), (self.x, self.y), 1)

    def __init__(self, render_mode=None):
        # Initialize the base environment class
        super().__init__()
        
        # Define the action space with 4 discrete actions (0: up, 1: right, 2: down, 3: left)
        self.action_space = spaces.Discrete(4)

        # Define the observation space with agent's position and the maze structure
        self.observation_space = spaces.Dict({
            "agent_position": spaces.Box(
                low=0, 
                high=max(ROWS, COLS), 
                shape=(2,), 
                dtype=np.int32
            ),
            "maze_grid": spaces.Box(
                low=0, 
                high=1, 
                shape=(ROWS, COLS, 4), 
                dtype=np.int32
            )
        })

        # Set render mode and initialize pygame components if mode is set to "human"
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()  # Initialize all imported pygame modules
            self.screen = pygame.display.set_mode((SIZE[0], SIZE[1]))  # Set screen size for pygame display
            pygame.display.set_caption("Maze Environment")  # Set window caption
            self.clock = pygame.time.Clock()  # Initialize clock to control frame rate

        # Initialize additional attributes for rendering and environment management
        self.window = None  # Placeholder for window object if needed
        self.maze_model = MazeModel()  # Create an instance of MazeModel for maze handling logic
        
        # Initialize the grid as a 2D list of MazeCell objects
        self.grid = []
        for y in range(ROWS):
            row = []  # Create a new row for each y-coordinate
            for x in range(COLS):
                row.append(self.MazeCell(x, y))  # Append a MazeCell instance at (x, y) to the current row
            self.grid.append(row)  # Append the completed row to the grid
        
        # Set the starting position in the maze and initialize the stack for maze generation
        self.current_cell = self.grid[0][0]  # Starting cell is the top-left corner (0, 0)
        self.stack = []  # Stack to manage backtracking during maze generation

    def generate_maze(self):
        """Generate a new maze using depth-first search."""

        # Mark the current cell as visited to start the maze generation
        self.current_cell.visited = True

        # Continue until there are no more cells to visit
        while True:
            next_cell = self.check_neighbors(self.current_cell)  # Find an unvisited neighboring cell
            if next_cell:
                # If an unvisited neighbor is found:
                # Push the current cell onto the stack to return to it later
                self.stack.append(self.current_cell)

                # Remove walls between the current cell and the next cell to create a path
                self.remove_walls(self.current_cell, next_cell)

                # Set current cell's `current` attribute to False (no longer the active cell)
                self.current_cell.current = False
                
                self.current_cell = next_cell  # Move to the next cell and update its state
                self.current_cell.visited = True  # Mark the new cell as visited
                self.current_cell.current = True  # Mark it as the current cell for visualization
            elif self.stack:
                # If no unvisited neighbors but cells are left in the stack:
                # Set the current cell's `current` attribute to False (backtracking)
                self.current_cell.current = False

                # Pop the last cell from the stack and make it the current cell (backtrack)
                self.current_cell = self.stack.pop()
            else:
                # No unvisited neighbors and no cells in the stack means the maze is complete
                break

    def check_neighbors(self, cell):
        """Check for unvisited neighbors of the current cell."""
        neighbors = []  # List to store unvisited neighboring cells
        
        if cell.y > 0:  # Check the cell above the current cell (if within grid bounds)
            top = self.grid[cell.y // WIDTH - 1][cell.x // WIDTH]  # Calculate the position of the cell directly above
            # Add to neighbors if this cell hasn't been visited
            if not top.visited:
                neighbors.append(top)
                
        if cell.x < (COLS - 1) * WIDTH:
            right = self.grid[cell.y // WIDTH][cell.x // WIDTH + 1]
            if not right.visited:
                neighbors.append(right)
                
        if cell.y < (ROWS - 1) * WIDTH:
            bottom = self.grid[cell.y // WIDTH + 1][cell.x // WIDTH]
            if not bottom.visited:
                neighbors.append(bottom)
                
        if cell.x > 0:
            left = self.grid[cell.y // WIDTH][cell.x // WIDTH - 1]
            if not left.visited:
                neighbors.append(left)

        # Randomly select one of the unvisited neighbors if available, otherwise return None
        return random.choice(neighbors) if neighbors else None

    def remove_walls(self, current, next):
        """Remove walls between two adjacent cells in the maze."""

        # Calculate the difference in x and y positions between the current and next cells
        dx = current.x // WIDTH - next.x // WIDTH
        dy = current.y // WIDTH - next.y // WIDTH
        if dx == -1:  # If next cell is to the right of the current cell
            current.walls[1] = False  # Remove the wall to the right of the current cell
            next.walls[3] = False  # Remove the wall to the left of the next cell
        elif dx == 1:  # If next cell is to the left of the current cell
            current.walls[3] = False  # Remove the wall to the left of the current cell
            next.walls[1] = False  # Remove the wall to the right of the next cell
        elif dy == -1:  # If next cell is below the current cell
            current.walls[2] = False  # Remove the wall at the bottom of the current cell
            next.walls[0] = False  # Remove the wall at the top of the next cell
        elif dy == 1:  # If next cell is above the current cell
            current.walls[0] = False  # Remove the wall at the top of the current cell
            next.walls[2] = False  # Remove the wall at the bottom of the next cell

    def draw(self, screen):
        """Draw the maze, including walls, start/end points, and agent."""

        # Draw each cell and its walls in the grid
        for row in self.grid:
            for cell in row:
                cell.draw(screen)

        # Draw the start point as a small green square in the top-left corner
        pygame.draw.rect(screen, GREEN, (0 + WIDTH//4, 0 + WIDTH//4, START_END_SIZE, START_END_SIZE))

        # Draw the end point as a small red square in the bottom-right corner
        pygame.draw.rect(screen, RED, ((COLS-1)*WIDTH + WIDTH//4, (ROWS-1)*WIDTH + WIDTH//4, START_END_SIZE, START_END_SIZE))

    def _get_obs(self):
        """Get the current observation, including the agent's position and maze structure."""
        
        # Build a 3D numpy array representing the maze grid, with wall status for each cell
        maze_grid = np.array([[[cell.walls[w] for w in range(4)] 
                             for cell in row] 
                             for row in self.grid], dtype=np.int32)
        
        # Return a dictionary containing the agent's position and the maze grid structure
        return {
            "agent_position": np.array(self.maze_model.agent_position, dtype=np.int32),
            "maze_grid": maze_grid
        }

    def _get_info(self):
        """Get additional information about the environment."""

        # Return a dictionary with specific metrics and performance data from the maze model
        return {
            "total_moves": self.maze_model.total_moves,
            "dead_ends": self.maze_model.dead_ends,
            "loops": self.maze_model.loops,
            "performance": self.maze_model.performance_measure()
        }

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset maze model
        self.maze_model.reset()
        
        # Generate new maze
        self.grid = []
        for y in range(ROWS):
            row = []
            for x in range(COLS):
                row.append(self.MazeCell(x, y))
            self.grid.append(row)
        self.current_cell = self.grid[0][0]
        self.stack = []
        self.generate_maze()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """Execute one time step within the environment."""
        x, y = self.maze_model.agent_position
        valid_actions = self.maze_model.ACTIONS(self.maze_model.agent_position, 
                                              [[[cell.walls[w] for w in range(4)] for cell in row] for row in self.grid])
        
        new_pos = self.maze_model.agent_position
        moved = False

        try:
            action_idx = int(action)
            if action_idx == 0 and (x, y - 1) in valid_actions:  # Up
                new_pos = (x, y - 1)
                moved = True
            elif action_idx == 1 and (x + 1, y) in valid_actions:  # Right
                new_pos = (x + 1, y)
                moved = True
            elif action_idx == 2 and (x, y + 1) in valid_actions:  # Down
                new_pos = (x, y + 1)
                moved = True
            elif action_idx == 3 and (x - 1, y) in valid_actions:  # Left
                new_pos = (x - 1, y)
                moved = True
        except (TypeError, ValueError):
            pass

        if moved:
            self.maze_model.agent_position = new_pos
            self.maze_model.total_moves += 1

            walls = [[[cell.walls[w] for w in range(4)] for cell in row] for row in self.grid]
            if self.maze_model.check_if_dead_end(new_pos, walls):
                self.maze_model.dead_ends += 1
            if self.maze_model.is_loop(new_pos):
                self.maze_model.loops += 1

        terminated = self.maze_model.GOAL_TEST(self.maze_model.agent_position)
        reward = self.maze_model.performance_measure()
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render one frame of the environment."""
        if self.render_mode == "human":
            self.screen.fill(GREY)
            self.draw(self.screen)
            
            # Draw agent
            agent_x, agent_y = self.maze_model.agent_position
            pygame.draw.circle(self.screen, BLUE, 
                             (agent_x * WIDTH + WIDTH // 2, agent_y * WIDTH + WIDTH // 2), 
                             AGENT_SIZE // 2)
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """Clean up resources."""
        if self.render_mode == "human":
            pygame.quit()