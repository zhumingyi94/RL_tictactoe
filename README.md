# Reinforcement Learning Algorithms Repository

## Overview

This repository contains implementations of various reinforcement learning algorithms, providing a comprehensive exploration of different approaches to solving sequential decision-making problems.

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/reinforcement-learning-algorithms.git
   cd reinforcement-learning-algorithms
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Running Algorithms

To run a specific algorithm with a configuration:

```bash
python main.py +algorithm=<algorithm_name> +config=<config_file>
```

### Example

Run SARSA algorithm with a specific configuration:

```bash
python main.py +algorithm=SARSA +config=sarsa_config.yaml
```

## 📊 Experimental Results

**🔍 Important: Experimental Results and Analysis**

After running the algorithms, **navigate to the `/experiments` directory** to:
- View comprehensive Jupyter notebooks
- Analyze detailed performance metrics
- Visualize algorithm comparisons
- Understand algorithm behavior under different configurations

The notebooks provide in-depth insights that go beyond raw numerical results.

## 🧩 Supported Algorithms

| Algorithm | Status | Description |
|-----------|--------|-------------|
| Q-learning | ✅ Implemented | Classic model-free reinforcement learning algorithm |
| SARSA | ✅ Implemented | On-policy temporal-difference learning algorithm |
| Deep Q-Network (DQN) | ✅ Implemented | Deep learning-based value iteration method |
| Policy Iteration | ✅ Implemented | Dynamic programming approach for policy optimization |
| Value Iteration | ✅ Implemented | Dynamic programming method for value function estimation |
| Tic-Tac-Toe Environment | ✅ Implemented | Custom game environment for algorithm testing |

## 🛠 Built With

- [Gymnasium](https://gymnasium.farama.org/) - Reinforcement Learning Environment API
- [Hydra](https://hydra.cc/) - Flexible Configuration Management
- [NumPy](https://numpy.org/) - Numerical Computing Library

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📌 Notes

- Ensure you have the latest version of dependencies
- Experiment configurations are managed through Hydra
- Logging is automatically handled for each experiment
