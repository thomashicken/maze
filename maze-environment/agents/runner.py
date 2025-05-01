import argparse
import logging
import sys
import gymnasium as gym
from agent_random import AgentRandom
from agent1 import Agent1
from maze import MazeModel

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Run Maze Agent')
    parser.add_argument("--episode-count", "-c", type=int, help="Number of episodes to run", default=1)
    parser.add_argument("--max-steps", "-m", type=int, help="Maximum number of steps per episode", default=500)
    parser.add_argument("--render-mode", "-r", type=str, help="Render mode", choices=("human", "none"), default="none")
    parser.add_argument("--logging-level", "-l", type=str, help="logging level: warn, info, debug", choices=("warn", "info", "debug"), default="warn")
    parser.add_argument("--agent", "-a", type=str, help="Agent to use", choices=("random", "agent1"), default="agent1")
    args = parser.parse_args(argv[1:])
    args.logging_level = getattr(logging, args.logging_level.upper())
    return args

def run_episode(env, agent, max_steps):
    """Run a single episode."""
    observation, info = env.reset()
    agent.reset()
    
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < max_steps:
        action = agent.agent_function(observation)  # Use agent_function for action selection
        if action is None:
            break
            
        observation, reward, terminated, truncated, info = env.step(action)
        agent.current_state = observation  # Update agent's state with new observation
        # print(observation)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
    return total_reward, steps, info

def run_episodes(env, episode_count, agent, max_steps):
    """Run multiple episodes and return statistics."""
    total_reward = 0
    total_steps = 0
    episode_rewards = []
    
    for episode in range(episode_count):
        reward, steps, info = run_episode(env, agent, max_steps)
        total_reward += reward
        total_steps += steps
        episode_rewards.append(reward)
        print(f"Episode {episode + 1}: Reward = {reward}, Steps = {steps}")
        
    avg_reward = total_reward / episode_count
    avg_steps = total_steps / episode_count
    return avg_reward, avg_steps, episode_rewards

def main(argv):
    args = parse_args(argv)
    logging.basicConfig(level=args.logging_level)

    # Create model and environment
    model = MazeModel()
    env = gym.make("Maze-v0", render_mode=args.render_mode)

    # Select agent based on argument
    if args.agent == "random":
        agent = AgentRandom(model, env)
    elif args.agent == "agent1":
        agent = Agent1(model, env)

    # Run episodes
    avg_reward, avg_steps, episode_rewards = run_episodes(
        env, 
        args.episode_count, 
        agent,
        args.max_steps
    )

    print(f"\nResults over {args.episode_count} episodes:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.2f}")

    env.close()

if __name__ == "__main__":
    main(sys.argv)