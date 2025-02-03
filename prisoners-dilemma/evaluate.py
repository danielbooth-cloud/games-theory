import argparse
import pickle
from collections import defaultdict

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

def evaluate_performance(q_table, strategy_type, cooperate_prob, rounds=1000):
    user = UserStrategy(strategy_type, cooperate_prob)
    total_score = 0
    state = ('Start', 'Start')
    stats = {
        'outcomes': defaultdict(int),
        'ai_actions': defaultdict(int),
        'user_actions': defaultdict(int),
        'score_progression': []
    }
    
    for i in range(rounds):
        try:
            ai_action = max(q_table[state], key=q_table[state].get)
        except KeyError:
            ai_action = 'D'
            
        user_action = user.move()
        payoff = PAYOFF_MATRIX[(ai_action, user_action)][0]
        
        # Record statistics
        stats['outcomes'][(ai_action, user_action)] += 1
        stats['ai_actions'][ai_action] += 1
        stats['user_actions'][user_action] += 1
        stats['score_progression'].append(payoff)
        total_score += payoff
        
        # Update state
        state = (ai_action, user_action)
        user.history.append((ai_action, user_action))
    
    avg_score = total_score / rounds
    cooperation_rate = stats['ai_actions']['C'] / rounds
    
    # Print detailed report
    print(f"\n{' Evaluation Report ':=^50}")
    print(f"▪ Against strategy: {strategy_type.capitalize()}")
    if strategy_type == 'random':
        print(f"▪ User cooperation probability: {cooperate_prob:.1%}")
    print(f"▪ Rounds played: {rounds}")
    
    print(f"\n{' Performance Metrics ':-^50}")
    print(f"▪ Average score per round: {avg_score:.2f}")
    print(f"▪ AI cooperation rate: {cooperation_rate:.1%}")
    print(f"▪ User cooperation rate: {stats['user_actions']['C']/rounds:.1%}")
    
    print(f"\n{' Outcome Distribution ':-^50}")
    for (ai, user), count in stats['outcomes'].items():
        print(f"{ai}/{user}: {count/rounds:.1%} ", end='')
    print()
    
    print(f"\n{' Strategy Analysis ':-^50}")
    if avg_score >= 4.5:
        print("▪ AI strategy: Ruthless exploiter (always defects when possible)")
    elif avg_score >= 3.0:
        print("▪ AI strategy: Balanced approach (mix of cooperation/defection)")
    elif avg_score >= 2.0:
        print("▪ AI strategy: Cooperative tendency")
    else:
        print("▪ AI strategy: Overly passive (needs improvement)")
    
    # Compare to theoretical baselines
    print(f"\n{' Theoretical Benchmarks ':-^50}")
    print(f"▪ Mutual cooperation: 3.00")
    print(f"▪ Optimal exploitation: 5.00")
    print(f"▪ Mutual defection: 1.00")
    print(f"▪ Random baseline (~0.5): ~2.50")
    
    # Display improvement suggestions
    print(f"\n{' Recommendations ':-^50}")
    if avg_score < 2.5:
        print("▪ Increase training episodes")
        print("▪ Reduce exploration rate (epsilon)")
        print("▪ Adjust learning rate parameters")
    elif avg_score < 3.5:
        print("▪ Moderate performance - try:")
        print("  ▪ Increase discount factor (gamma)")
        print("  ▪ Add reward shaping")
    else:
        print("▪ Excellent performance - consider:")
        print("  ▪ Testing against more complex strategies")
        print("  ▪ Implementing probabilistic strategies")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy_type', required=True)
    parser.add_argument('--cooperate_prob', type=float, default=0.5)
    parser.add_argument('--q_table', default="qtable.pkl")
    args = parser.parse_args()
    
    with open(args.q_table, 'rb') as f:
        q_table = pickle.load(f)
    
    evaluate_performance(q_table, args.strategy_type, args.cooperate_prob)