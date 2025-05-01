# maze

This environment involves taking a blue dot, the agent, and navigating said blue dot to the bottom right of the screen.

Action Space: Discrete(4)
Observation Space: (ROWS,COLS)
import: gymnasium.make("Maze-v0")

## Description

The maze is generated randomly, the ROWS and COLS are defined by default to be 7x7 and the agent, the blue dot, always starts at the top left of the grid. The goal is to move the agent and navigate it through the randomly generated maze to the bottom left of the screen, the red square. The agent receives positive rewards for a completion of the maze and negative rewards based upon revisits of nodes, dead ends occured, and total steps taken to reach the red square.

## Action Space

The action shape is (0: up, 1: right, 2: down, 3: left), indicating each possible move the agent has available at any point in the navigation process.

## Observation Space

The number of discrete states is dependant on the current ROWS and COLS set in the config.py.

Start of Each Episode: [assuming ROWS and COLS are unchanged]
Agent Location: 'agent_position': array([1, 0], dtype=int32)
Destination: 'agent_position': array([6, 6], dtype=int32)

Note that if you wish to change the number of ROWS and COLS in the maze simply navigate to the config.py and change WIDTH of the display in the file.

## Starting State

The state the agent starts in is consistent, it will always start at the top left of the grid.

## Rewards

The reward is calculated and scored based on moves, dead ends, and loops. The following code showcases the calculations that are implemented in the model of the environment:

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

## Episode End

The episode ends if the following happens:
    Termination: The agent reaches the red square.
    Truncation: The agent did not reach the red square in the maximum number of steps allowed in a single episode.

## Arguments

import gymnasium as gym
gym.make("Maze-v0")

### Contributing
If you would like to contribute, follow these steps:
- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

PRs may require accompanying PRs in [the documentation repo](https://github.com/Farama-Foundation/Gymnasium/tree/main/docs).


## Installation

To install your new environment, run the following commands:

```{shell}
cd maze-environment
cd maze
make
```

