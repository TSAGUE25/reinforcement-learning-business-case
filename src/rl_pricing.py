"""
Reinforcement Learning — Dynamic Pricing with Q-Learning.

Environment: hotel/airline-style revenue management.
State: (hour_bucket, day_type, occupancy_bucket)
Actions: price level (0=low, 1=medium, 2=high, 3=premium)
Reward: revenue = price × demand(price, state)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PRICE_LEVELS = [80, 100, 120, 150]
N_ACTIONS    = len(PRICE_LEVELS)


class PricingEnvironment:
    """Simplified dynamic pricing environment."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.state = None
        self.reset()

    def reset(self):
        self.step = 0
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        hour    = self.step % 24
        day     = (self.step // 24) % 7
        # State = (hour_bucket 0-3, is_weekday, season_bucket 0-3)
        hour_bucket   = min(hour // 6, 3)
        is_weekday    = int(day < 5)
        season_bucket = int(2 + np.sin(2 * np.pi * self.step / (24 * 90)) * 1.5)
        return (hour_bucket, is_weekday, min(season_bucket, 3))

    def step_env(self, action):
        price = PRICE_LEVELS[action]
        hour  = self.step % 24
        day   = (self.step // 24) % 7
        peak  = (8 <= hour <= 10) or (12 <= hour <= 14) or (18 <= hour <= 21)
        season = 1 + 0.2 * np.sin(2 * np.pi * self.step / (24 * 365))

        base_demands = [120, 90, 60, 30]
        base_d = base_demands[action]
        mult = season * (1.3 if peak and day < 5 else 0.8 if day >= 5 else 1.0)
        demand = max(0, int(self.rng.poisson(base_d * mult)))

        reward = price * demand
        self.step += 1
        self.state = self._get_state()
        return self.state, reward, False


# ── Q-Learning Agent ─────────────────────────────────────────────────────────

class QLearningAgent:
    def __init__(self, n_states=(4, 2, 4), n_actions=N_ACTIONS,
                 lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.9995,
                 epsilon_min=0.05):
        self.q_table    = np.zeros(n_states + (n_actions,))
        self.lr         = lr
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.eps_decay  = epsilon_decay
        self.eps_min    = epsilon_min

    def choose_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def train(self, env, n_episodes=500, steps_per_episode=168):
        rewards_history = []
        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0
            for _ in range(steps_per_episode):
                action = self.choose_action(state)
                next_state, reward, _ = env.step_env(action)
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            rewards_history.append(total_reward)
        return rewards_history

    def evaluate(self, env, n_steps=1000):
        state = env.reset()
        total, revenues = 0, []
        actions_taken = []
        for _ in range(n_steps):
            action = self.choose_action(state, greedy=True)
            state, reward, _ = env.step_env(action)
            total += reward
            revenues.append(reward)
            actions_taken.append(action)
        return dict(total_revenue=total, avg_revenue=np.mean(revenues),
                    revenues=revenues, actions=actions_taken)


# ── Baseline: random policy ─────────────────────────────────────────────────

def random_policy_baseline(env, n_steps=1000, seed=42):
    rng = np.random.default_rng(seed)
    env.reset()
    total, revenues = 0, []
    for _ in range(n_steps):
        action = int(rng.integers(0, N_ACTIONS))
        _, reward, _ = env.step_env(action)
        total += reward
        revenues.append(reward)
    return dict(total_revenue=total, avg_revenue=np.mean(revenues), revenues=revenues)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curve(rewards_history, window=20):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rewards_history, alpha=0.4, color='#2196F3', label='Episode reward')
    ma = pd.Series(rewards_history).rolling(window).mean()
    ax.plot(ma.values, color='#F44336', lw=2, label=f'MA {window}')
    ax.set_xlabel('Episode'); ax.set_ylabel('Revenue total')
    ax.set_title('Convergence de l\'agent Q-Learning — Dynamic Pricing', fontweight='bold')
    ax.legend(); plt.tight_layout(); plt.show()


def plot_policy_comparison(ql_result, rand_result):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(ql_result['revenues'][:200], alpha=0.7, color='#4CAF50', label='Q-Learning')
    axes[0].plot(rand_result['revenues'][:200], alpha=0.7, color='#F44336', label='Aléatoire')
    axes[0].set_title('Revenue par période (200 premiers pas)')
    axes[0].set_xlabel('Pas de temps'); axes[0].legend()

    names = ['Aléatoire', 'Q-Learning']
    vals  = [rand_result['avg_revenue'], ql_result['avg_revenue']]
    colors_bar = ['#F44336', '#4CAF50']
    axes[1].bar(names, vals, color=colors_bar, edgecolor='white', width=0.5)
    pct = (ql_result['avg_revenue'] / rand_result['avg_revenue'] - 1) * 100
    axes[1].set_title(f'Revenue moyen par période\n(+{pct:.1f}% Q-Learning vs Aléatoire)')
    axes[1].set_ylabel('Revenue moyen (€/période)')

    plt.suptitle('RL Dynamic Pricing — Comparaison des politiques', fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_q_table_heatmap(agent):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, is_wday, label in zip(axes, [0, 1], ['Week-end', 'Semaine']):
        # Average Q-values over season_bucket dimension
        q_avg = agent.q_table[:, is_wday, :, :].mean(axis=1)  # (4 hours, 4 actions)
        im = ax.imshow(q_avg, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(N_ACTIONS))
        ax.set_xticklabels([f'{p}€' for p in PRICE_LEVELS])
        ax.set_yticks(range(4))
        ax.set_yticklabels(['0-6h', '6-12h', '12-18h', '18-24h'])
        ax.set_title(f'Q-values — {label}')
        ax.set_xlabel('Action (prix)'); ax.set_ylabel('Bucket horaire')
        plt.colorbar(im, ax=ax)
    plt.suptitle('Table Q apprise — Politique optimale', fontweight='bold')
    plt.tight_layout(); plt.show()
