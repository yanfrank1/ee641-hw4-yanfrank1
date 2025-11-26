"""
Training script for multi-agent DQN with communication.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
from typing import Tuple, Optional
from multi_agent_env import MultiAgentEnv
from models import DuelingDQN
from replay_buffer import PrioritizedReplayBuffer


def apply_observation_mask(obs: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply masking to observation based on ablation mode.

    Args:
        obs: 11-dimensional observation vector
        mode: One of 'independent', 'comm', 'full'

    Returns:
        Masked observation
    """
    # TODO: Implement masking logic
    # 'independent': Set elements 9 and 10 to zero
    # 'comm': Set element 10 to zero
    # 'full': No masking
    masked = obs.copy()
    if mode == 'independent':
        masked[9] = 0.0
        masked[10] = 0.0
    elif mode == 'comm':
        masked[10] = 0.0
    else:
        pass

    return masked


class MultiAgentTrainer:
    """
    Trainer for multi-agent DQN system.

    Handles training loop, exploration, and network updates.
    """

    def __init__(self, env: MultiAgentEnv, args):
        """
        Initialize trainer.

        Args:
            env: Multi-agent environment
            args: Training arguments
        """
        self.env = env
        self.args = args

        # Use CPU for small networks
        self.device = torch.device("cpu")

        # TODO: Initialize networks for both agents (remember to .to(self.device))
        # TODO: Initialize target networks (if using)
        # TODO: Initialize optimizers
        # TODO: Initialize replay buffer
        # TODO: Initialize epsilon for exploration
        input_dim = 11
        hidden_dim = args.hidden_dim
        num_actions = 5

        self.network_A = DuelingDQN(input_dim, hidden_dim, num_actions).to(self.device)
        self.network_B = DuelingDQN(input_dim, hidden_dim, num_actions).to(self.device)
        self.target_A = DuelingDQN(input_dim, hidden_dim, num_actions).to(self.device)
        self.target_B = DuelingDQN(input_dim, hidden_dim, num_actions).to(self.device)
        self.target_A.load_state_dict(self.network_A.state_dict())
        self.target_B.load_state_dict(self.network_B.state_dict())
        self.target_A.eval()
        self.target_B.eval()

        self.optimizer_A = optim.Adam(self.network_A.parameters(), lr=args.lr)
        self.optimizer_B = optim.Adam(self.network_B.parameters(), lr=args.lr)

        self.replay = PrioritizedReplayBuffer(capacity=10000, seed=args.seed)
        self.epsilon = args.epsilon_start
        self.train_steps = 0

    def select_action(self, state: np.ndarray, network: nn.Module,
                      epsilon: float) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Agent observation (11-dimensional, may need masking)
            network: Agent's DQN
            epsilon: Exploration probability

        Returns:
            action: Selected action
            comm_signal: Communication signal
        """
        # TODO: Apply observation masking based on self.args.mode
        #       masked_state = apply_observation_mask(state, self.args.mode)
        # TODO: With probability epsilon, select random action
        # TODO: Otherwise, select action with highest Q-value
        # TODO: Always get communication signal from network

        masked_state = apply_observation_mask(state, self.args.mode)
        state_tensor = torch.tensor(masked_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if np.random.rand() < epsilon:
            action = np.random.randint(0, 5)
            with torch.no_grad():
                _, comm_tensor = network(state_tensor)
                comm_signal = float(comm_tensor.squeeze().cpu().numpy())
        else:
            with torch.no_grad():
                q_values, comm_tensor = network(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
                comm_signal = float(comm_tensor.squeeze().cpu().numpy())

        comm_signal = float(np.clip(comm_signal, 0.0, 1.0))
        return (action, comm_signal)

    def update_networks(self, batch_size: int) -> float:
        """
        Sample batch and update both agent networks.

        Args:
            batch_size: Size of training batch

        Returns:
            loss: Combined loss value
        """
        # TODO: Sample batch from replay buffer
        # TODO: Convert to tensors and move to device
        # TODO: Compute Q-values for current states
        # TODO: Compute target Q-values using target networks
        # TODO: Calculate TD loss for both agents
        # TODO: Backpropagate and update networks
        # TODO: Return combined loss
        if len(self.replay) < batch_size:
            return 0.0
            
        (states_A, states_B,
         actions_A, actions_B,
         comms_A, comms_B,
         rewards,
         next_states_A, next_states_B,
         dones), weights, indices = self.replay.sample(batch_size)

        states_A_t = torch.tensor(states_A, dtype=torch.float32, device=self.device)
        states_B_t = torch.tensor(states_B, dtype=torch.float32, device=self.device)
        next_A_t = torch.tensor(next_states_A, dtype=torch.float32, device=self.device)
        next_B_t = torch.tensor(next_states_B, dtype=torch.float32, device=self.device)
        actions_A_t = torch.tensor(actions_A, dtype=torch.int64, device=self.device)
        actions_B_t = torch.tensor(actions_B, dtype=torch.int64, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        qA_values, _ = self.network_A(states_A_t)
        qB_values, _ = self.network_B(states_B_t)
        qA = qA_values.gather(1, actions_A_t.unsqueeze(1)).squeeze(1)
        qB = qB_values.gather(1, actions_B_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_qA_values, _ = self.target_A(next_A_t)
            next_qB_values, _ = self.target_B(next_B_t)
            max_next_qA, _ = next_qA_values.max(dim=1)
            max_next_qB, _ = next_qB_values.max(dim=1)
            target_A = rewards_t + self.args.gamma * (1.0 - dones_t) * max_next_qA
            target_B = rewards_t + self.args.gamma * (1.0 - dones_t) * max_next_qB

        error_A = target_A - qA
        error_B = target_B - qB
        error = (error_A.abs() + error_B.abs()) * 0.5
        loss_A = (weights * (error_A ** 2)).mean()
        loss_B = (weights * (error_B ** 2)).mean()
        loss = loss_A + loss_B
        self.optimizer_A.zero_grad()
        self.optimizer_B.zero_grad()
        loss.backward()
        self.optimizer_A.step()
        self.optimizer_B.step()
        self.replay.update_priorities(indices, error.detach().cpu().numpy())
        self.train_steps += 1
        if self.train_steps % self.args.target_update == 0:
            self.target_A.load_state_dict(self.network_A.state_dict())
            self.target_B.load_state_dict(self.network_B.state_dict())

        return float(loss.item())

    def train_episode(self) -> Tuple[float, bool]:
        """
        Run one training episode.

        Returns:
            episode_reward: Total reward for episode
            success: Whether agents reached target
        """
        # TODO: Reset environment
        # TODO: Initialize episode variables
        # TODO: Run episode until termination:
        #       - Select actions for both agents
        #       - Execute actions in environment
        #       - Store transition in replay buffer
        #       - Update networks if enough samples
        # TODO: Return episode reward and success flag
        obs_A, obs_B = self.env.reset()
        episode_reward = 0.0
        success = False
        done = False
        while not done:
            action_A, comm_A = self.select_action(obs_A, self.network_A, self.epsilon)
            action_B, comm_B = self.select_action(obs_B, self.network_B, self.epsilon)
            (next_A, next_B), reward, done = self.env.step(action_A, action_B, comm_A, comm_B)

            self.replay.push(
                obs_A, obs_B,
                action_A, action_B,
                comm_A, comm_B,
                reward,
                next_A, next_B,
                done
            )

            episode_reward += reward
            if done and reward == 10.0:
                success = True
            obs_A, obs_B = next_A, next_B
            self.update_networks(self.args.batch_size)

        return episode_reward, success

    def train(self) -> None:
        """
        Main training loop.
        """
        # TODO: Create results directories
        # TODO: Initialize logging
        # TODO: Main training loop:
        #       - Run episodes
        #       - Update epsilon
        #       - Update target networks periodically
        #       - Log progress
        #       - Save checkpoints
        # TODO: Save final models including TorchScript format:
        #       scripted_model = torch.jit.script(self.network_A)
        #       scripted_model.save("dqn_net.pt")
        results_dir = os.path.join("results", self.args.mode)
        os.makedirs(results_dir, exist_ok=True)

        rewards_log = []
        success_log = []
        epsilon_log = []

        for episode in range(1, self.args.num_episodes + 1):
            self.network_A.train()
            self.network_B.train()
            ep_reward, success = self.train_episode()
            rewards_log.append(ep_reward)
            success_log.append(1.0 if success else 0.0)
            epsilon_log.append(self.epsilon)
            self.epsilon = max(self.args.epsilon_end, self.epsilon * self.args.epsilon_decay)

            if episode % 10 == 0:
                avg_r = np.mean(rewards_log[-10:])
                avg_s = np.mean(success_log[-10:])
                print(f"Episode {episode}/{self.args.num_episodes} "
                      f"| Reward (last 10 avg): {avg_r:.2f} "
                      f"| Success: {avg_s:.2f} "
                      f"| Epsilon: {self.epsilon:.3f}")

            if episode % self.args.save_freq == 0:
                ckpt_path = os.path.join(results_dir, f"checkpoint_ep{episode}.pth")
                torch.save({
                    "network_A": self.network_A.state_dict(),
                    "network_B": self.network_B.state_dict(),
                    "epsilon": self.epsilon,
                    "episode": episode
                }, ckpt_path)

        log_data = {
            "rewards": rewards_log,
            "successes": success_log,
            "epsilon": epsilon_log
        }
        with open(os.path.join(results_dir, "training_log.json"), "w") as f:
            json.dump(log_data, f)
            
        self.network_A.eval()
        self.network_B.eval()
        scripted_A = torch.jit.script(self.network_A)
        scripted_B = torch.jit.script(self.network_B)
        scripted_A.save(os.path.join(results_dir, "dqn_agent_A.pt"))
        scripted_B.save(os.path.join(results_dir, "dqn_agent_B.pt"))

    def evaluate(self, num_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate current policy.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            mean_reward: Average reward
            success_rate: Fraction of successful episodes
        """
        # TODO: Set networks to evaluation mode
        # TODO: Run episodes without exploration
        # TODO: Track rewards and successes
        # TODO: Return statistics
        self.network_A.eval()
        self.network_B.eval()

        total_reward = 0.0
        success_count = 0
        eval_epsilon = 0.0

        for _ in range(num_episodes):
            obs_A, obs_B = self.env.reset()
            done = False
            ep_reward = 0.0
            last_reward = 0.0

            while not done:
                action_A, comm_A = self.select_action(obs_A, self.network_A, eval_epsilon)
                action_B, comm_B = self.select_action(obs_B, self.network_B, eval_epsilon)
                (next_A, next_B), reward, done = self.env.step(action_A, action_B, comm_A, comm_B)
                ep_reward += reward
                last_reward = reward
                obs_A, obs_B = next_A, next_B

            if last_reward >= 10.0 and done:
                success_count += 1

            total_reward += ep_reward

        mean_reward = total_reward / num_episodes
        success_rate = success_count / num_episodes
        return mean_reward, success_rate


def main():
    """
    Parse arguments and run training.
    """
    parser = argparse.ArgumentParser(description='Train Multi-Agent DQN')

    # Environment parameters
    parser.add_argument('--grid_size', type=int, nargs=2, default=[10, 10],
                       help='Grid dimensions')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='Maximum steps per episode')

    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')

    # Exploration parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                       help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.999,
                       help='Epsilon decay rate')

    # Network parameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer size')
    parser.add_argument('--target_update', type=int, default=100,
                       help='Target network update frequency')

    # Ablation study mode
    parser.add_argument('--mode', type=str, default='full',
                       choices=['independent', 'comm', 'full'],
                       help='Information mode: independent (mask comm+dist), '
                            'comm (mask dist only), full (no masking)')

    # Other parameters
    parser.add_argument('--seed', type=int, default=641,
                       help='Random seed')
    parser.add_argument('--save_freq', type=int, default=500,
                       help='Model save frequency')

    args = parser.parse_args()

    # TODO: Set random seeds
    # TODO: Create environment
    # TODO: Create trainer
    # TODO: Run training
    # TODO: Final evaluation
    np.random.seed(args.seed)
    env = MultiAgentEnv(grid_size=tuple(args.grid_size), max_steps=args.max_steps, seed=args.seed)
    trainer = MultiAgentTrainer(env, args)
    trainer.train()
    mean_reward, success_rate = trainer.evaluate(num_episodes=20)
    print(f"Final evaluation over 20 episodes: mean reward = {mean_reward:.2f}, success rate = {success_rate:.2f}")

if __name__ == '__main__':
    main()