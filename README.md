# federated_learning_basic_pipeline_autism

# Federated learning basics with Flower

A beginner-friendly federated pipeline using Flower and scikit-learn. Data stays on each client, and the server aggregates metrics.
Data is randomly splitted into clients, models run locally on clients and just weights and output metrics are shared.

This is a beginner guide to flower a federated learning framework and machine learning models using scikit-learn. The Dataset used is publicly available on Kaggle. 

## Quick start
- **Create a Python environment:**
  - Windows: open Command Prompt
  - Run:
    ```
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```
- **Run the server:**
- python flwr-server.py

- **Run 3 Clients on seperate terminal using below commands**
- python flwr_client.py -- client 0
- python flwr_client.py -- client 2
- python flwr_client.py -- client 2

- If you like and learn federated learning concepts from this with practical exposure, Thanks me Later !
- qamar.researcher@gmail.com for queries and doubts.
**
