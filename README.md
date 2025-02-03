## Prisoner's Dilemma Q-Learning with Argo Workflows
A containerized implementation of Q-learning for the Prisoner's Dilemma game, orchestrated using Argo Workflows.

### Overview
This project trains an AI agent to play the Prisoner's Dilemma using Q-learning. The agent can learn to play against various strategies:
* Tit-for-Tat
* Random (with configurable cooperation probability)
* Always Cooperate
* Always Defect

For more information please refer to the [Blog post on my website](https://danielbooth.cloud/building-a-q-learning-agent-for-game-theory-with-argo-workflows/ "blog post on my website")

### Prerequisites
* Kubernetes cluster
* Argo Workflows installed
* kubectl configured

### How It Works

Training Phase: The agent learns through Q-learning with:
* Epsilon-greedy exploration
* Configurable learning parameters
* State-action value updates
  
Evaluation Phase: Generates detailed performance reports including:
* Average score per round
* Cooperation rates
* Strategy analysis
* Improvement recommendations

### License
Apache License 2.0 
