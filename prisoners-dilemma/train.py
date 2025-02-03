import argparse
import pickle
import numpy as np

# Game parameters
PAYOFF_MATRIX = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1)
}

class UserStrategy:
    def __init__(self, strategy_type, cooperate_prob=0.5):
        self.strategy_type = strategy_type
        self.cooperate_prob = cooperate_prob
        self.history = []
    
    def move(self):
        if self.strategy_type == "random":
            return 'C' if np.random.random() < self.cooperate_prob else 'D'
        elif self.strategy_type == "titfortat":
            return self.history[-1][0] if self.history else 'C'
        elif self.strategy_type == "always_cooperate":
            return 'C'
        elif self.strategy_type == "always_defect":
            return 'D'
        return 'C'

def train_q_learning(strategy_type, cooperate_prob, episodes=5000):
    user = UserStrategy(strategy_type, cooperate_prob)
    q_table = {}
    alpha = 0.1  # Learning rate
    gamma = 0.95  # Discount factor
    epsilon = 0.1  # Exploration rate
    
    for episode in range(episodes):
        state = ('Start', 'Start')
        total_reward = 0
        
        for _ in range(20):  # 20 rounds per episode
            # Epsilon-greedy action selection
            if state not in q_table:
                q_table[state] = {'C': 0, 'D': 0}
                
            if np.random.random() < epsilon:
                ai_action = np.random.choice(['C', 'D'])  # Explore
            else:
                ai_action = max(q_table[state], key=q_table[state].get)  # Exploit
            
            user_action = user.move()
            reward = PAYOFF_MATRIX[(ai_action, user_action)][0]
            
            # Update Q-table
            next_state = (ai_action, user_action)
            if next_state not in q_table:
                q_table[next_state] = {'C': 0, 'D': 0}
                
            q_table[state][ai_action] += alpha * (
                reward + gamma * max(q_table[next_state].values()) - q_table[state][ai_action]
            )
            
            total_reward += reward
            state = next_state
            user.history.append((ai_action, user_action))
    
    return q_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy_type', required=True, help="User strategy: random, titfortat, always_cooperate, always_defect")
    parser.add_argument('--cooperate_prob', type=float, default=0.5, help="Probability of cooperation for random strategy")
    parser.add_argument('--output', default="qtable.pkl", help="Output file for Q-table")
    args = parser.parse_args()
    
    q_table = train_q_learning(args.strategy_type, args.cooperate_prob)
    
    with open(args.output, 'wb') as f:
        pickle.dump(q_table, f)
    
    print(f"Training complete. Q-table saved to {args.output}")