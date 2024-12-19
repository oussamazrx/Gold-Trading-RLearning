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

## Desription
``agent.py`` (TensorFlow/Keras)
- This script implements the stock trading agent using TensorFlow/Keras.
- It defines a neural network using the Sequential API from Keras.
- The agent learns from stock price data by interacting with the environment and using experience replay to train the model.
- The model is trained using Adam optimizer and Mean Squared Error (MSE) loss function.
- The agent learns a policy by deciding whether to buy, sell, or hold based on the historical price data.
- 
``agent2.py`` (PyTorch)
- This script implements the stock trading agent using PyTorch.
- It defines the neural network model using the nn.Module class, offering more flexibility compared to the Keras model.
- The agent's training logic involves calculating the Q-values, performing gradient descent using the Adam optimizer, and updating the model using backpropagation.
- The agent also uses experience replay to train the model and makes decisions based on historical data.


## Usage
To run the stock trading simulation:

- Place your stock data in the data/ folder in CSV format. The CSV file should contain historical stock data with the closing prices (preferably in the 5th column).
- Modify the stock_name variable in both agent.py and agent2.py to the name of your stock data file (without the .csv extension).
- Adjust other parameters such as window_size and episode_count to control the length of the training process.
- Run either agent.py or agent2.py to start the simulation.
For example, running agent.py:

```bash
python agent.py
```
The agent will begin training using TensorFlow and will save the model at specified intervals (e.g., after every 10 episodes).
