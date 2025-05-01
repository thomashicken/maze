from typing import Tuple, List, Set
from maze.envs.config import COLS, ROWS

class MazeModel:
    def __init__(self):
        self.agent_position = (0, 0)
        self.total_moves = 0
        self.dead_ends = 0
        self.loops = 0
        self.visited_positions = set()
        
    def reset(self):
        """Reset the model state."""
        self.agent_position = (0, 0)
        self.total_moves = 0
        self.dead_ends = 0
        self.loops = 0
        self.visited_positions.clear()

    def ACTIONS(self, state, walls):
        """Return list of valid actions for the given state."""
        x, y = state
        actions = []
        
        # Check each direction if there's no wall and within bounds
        # Up
        if not walls[y][x][0] and y > 0:
            actions.append((x, y - 1))
            
        # Right    
        if not walls[y][x][1] and x < COLS - 1:
            actions.append((x + 1, y))
            
        # Down
        if not walls[y][x][2] and y < ROWS - 1:
            actions.append((x, y + 1))
            
        # Left
        if not walls[y][x][3] and x > 0:
            actions.append((x - 1, y))

        return actions
    
    def RESULT(self, state, action, walls):
        """Compute the result of taking an action in the given state."""
        possible_actions = self.ACTIONS(state, walls)
        
        if action in possible_actions:
            return action
            
        return state
    
    def GOAL_TEST(self, state):
        """Check if the given state is a goal state (bottom-right corner)."""
        goal_state = (COLS - 1, ROWS - 1)
        return state == goal_state

    def HEURISTIC(self, state):
        """Compute the Manhattan distance from current state to goal."""
        x, y = state
        goal_x = COLS - 1
        goal_y = ROWS - 1
        
        manhattan_distance = abs(goal_x - x) + abs(goal_y - y)
        return manhattan_distance

    def check_if_dead_end(self, position, walls):
        """Check if the current position is a dead-end (only one way out)."""
        possible_moves = self.ACTIONS(position, walls)
        return len(possible_moves) <= 1

    def is_loop(self, position):
        """Determine if the agent has returned to a previous position."""
        if position in self.visited_positions:
            return True
            
        self.visited_positions.add(position)
        return False

    def performance_measure(self):
        """Calculate performance score based on moves, dead ends, and loops."""
        penalty = (
            self.total_moves * 0.1  # Small penalty per move
        )
        reward = 0
        
        # Add a reward for reaching the goal
        if self.GOAL_TEST(self.agent_position):
            reward += 100  # Large reward for reaching the goal
        
        # Apply dead end and loop penalties if needed
        penalty += self.dead_ends * 2
        penalty += self.loops * 1

        # Final performance measure combines reward and penalty
        return int(reward - penalty)