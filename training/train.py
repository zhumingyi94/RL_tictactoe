import numpy as np
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from agents.sarsa import SARSAAgent 
from envs.tic_tac_toe_env import TicTacToeEnv  

@hydra.main(config_name="config", config_path="config")
def main(cfg: DictConfig):
    # Create the environment (ensure the board size and action space size match the environment)
    env = TicTacToeEnv(board_size=cfg.environment.board_size, action_space_size=cfg.environment.action_space_size)
    
    # Create the SARSA agent
    agent = SARSAAgent(
        env=env,
        gamma=cfg.sarsa.gamma,
        alpha=cfg.sarsa.alpha,
        epsilon=cfg.sarsa.epsilon
    )

    # Train the agent
    agent.train(episodes=cfg.training.episodes)

    # Optionally, save the trained Q-table and policy
    np.save('trained_Q.npy', agent.Q)
    np.save('trained_policy.npy', agent.policy)

    print("Training complete!")

if __name__ == "__main__":
    main()
