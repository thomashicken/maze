from gymnasium.envs.registration import register
from maze.envs.maze_env import MazeEnv
from maze.envs.maze_model import MazeModel
import maze.envs.config as config

register(
    id="Maze-v0",
    entry_point="maze.envs.maze_env:MazeEnv",
)