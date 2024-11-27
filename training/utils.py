import matplotlib.pyplot as plt
import numpy as np

def plot_results(rewards, agent_name):
    plt.plot(rewards)
    plt.title(f"Training Results - {agent_name}")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

def evaluate_agent_performance(agent, env, num_episodes=100):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            action = agent.policy[state]
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    return np.mean(total_rewards)
