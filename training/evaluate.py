import hydra
from omegaconf import DictConfig
from envs import TicTacToeEnv
import numpy as np
from agents.sarsa import SARSAAgent
from agents.policy_iteration import PolicyIterationAgent
from agents.q_learning import QLearningAgent
from agents.monte_carlo import MonteCarloAgent
from training.utils import evaluate_agent_performance
from envs.tic_tac_toe_env import TicTacToeEnv

@hydra.main(config_path="config", config_name="config")
def evaluate_agent(cfg: DictConfig):
    env = TicTacToeEnv(board_size=3)
    
    if "sarsa" in cfg.agent:
        agent = SARSAAgent(env, gamma=cfg.agent.gamma, alpha=cfg.agent.alpha, epsilon=cfg.agent.epsilon)
    elif "policy_iteration" in cfg.agent:
        agent = PolicyIterationAgent(env, gamma=cfg.agent.gamma, theta=cfg.agent.theta)
    elif "q_learning" in cfg.agent:
        agent = QLearningAgent(env, gamma=cfg.agent.gamma, alpha=cfg.agent.alpha, epsilon=cfg.agent.epsilon)
    elif "monte_carlo" in cfg.agent:
        agent = MonteCarloAgent(env, gamma=cfg.agent.gamma, epsilon=cfg.agent.epsilon)

    performance = evaluate_agent_performance(agent, env)
    print(f"Agent {agent.__class__.__name__} evaluation: {performance}")

if __name__ == "__main__":
    evaluate_agent()
