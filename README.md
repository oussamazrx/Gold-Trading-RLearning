# **Deep Q-Learning Stock Trading Agents**

This repository contains implementations of two **Deep Q-Learning (DQN)** agents for stock trading. The project demonstrates how reinforcement learning techniques can be applied to make decisions about buying, selling, and holding stocks based on historical price data.

### **Project Overview**

This project uses **Deep Q-Learning** (DQN) to train agents that learn to trade stocks in an environment based on historical price data. The goal of the agent is to maximize its profit by learning an optimal trading strategy over time.

- **Agent 1** (`agent.py`) is implemented using **TensorFlow/Keras**.
- **Agent 2** (`agent2.py`) is implemented using **PyTorch**.

Both agents learn from historical stock data and take actions (buy, sell, hold) based on their learned policy. The agents are trained using a reinforcement learning paradigm, where they receive a reward (or penalty) based on their actions in the environment.

### **Key Features**
- **Deep Q-Learning (DQN)** for stock trading
- **Experience Replay** to store and sample past experiences for training
- **Epsilon-Greedy Exploration** for balancing exploration and exploitation
- Model training using either **TensorFlow/Keras** or **PyTorch**

### **Requirements**
- Python 3.x
- TensorFlow 2.x (for `agent.py`)
- PyTorch (for `agent2.py`)
- Numpy
- Pandas (optional, depending on data handling)
  
You can install the required packages using `pip`:

```bash
pip install tensorflow numpy torch
```
