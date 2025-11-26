"""
Training script for Value Iteration and Q-Iteration.
"""

import numpy as np
import argparse
import json
import os
from environment import GridWorldEnv
from value_iteration import ValueIteration
from q_iteration import QIteration


def main():
    """
    Run both algorithms and save results.
    """
    parser = argparse.ArgumentParser(description='Train RL algorithms on GridWorld')
    parser.add_argument('--seed', type=int, default=641, help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Convergence threshold')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations')
    args = parser.parse_args()

    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)

    # TODO: Initialize environment with seed
    # TODO: Run Value Iteration
    #       - Create ValueIteration solver
    #       - Solve for optimal values
    #       - Extract policy
    #       - Save results
    # TODO: Run Q-Iteration
    #       - Create QIteration solver
    #       - Solve for optimal Q-values
    #       - Extract policy and values
    #       - Save results
    # TODO: Compare algorithms
    #       - Print convergence statistics
    #       - Check if policies match
    #       - Save comparison results
    env = GridWorldEnv(seed=args.seed)
    vi_solver = ValueIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    vi_values, vi_iters = vi_solver.solve(max_iterations=args.max_iter)
    vi_history = []
    V_old = np.zeros_like(vi_values)
    for i in range(vi_iters):
        V_new = np.zeros_like(V_old)
        for s in range(vi_solver.n_states):
            V_new[s] = vi_solver.bellman_backup(s, V_old)
        err = np.max(np.abs(V_new - V_old))
        vi_history.append(err)
        V_old = V_new.copy()

    vi_policy = vi_solver.extract_policy(vi_values)
    np.save("results/vi_values.npy", vi_values)
    np.save("results/vi_policy.npy", vi_policy)
    np.save("results/vi_bellman_history.npy", np.array(vi_history))
    print(f"Value Iteration converged in {vi_iters} iterations.")
    
    qi_solver = QIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    qi_qvalues, qi_iters = qi_solver.solve(max_iterations=args.max_iter)
    qi_values = qi_solver.extract_values(qi_qvalues)
    qi_policy = qi_solver.extract_policy(qi_qvalues)
    qi_history = []
    Q_old = np.zeros_like(qi_qvalues)

    for i in range(qi_iters):
        Q_new = np.zeros_like(Q_old)
        for s in range(qi_solver.n_states):
            for a in range(qi_solver.n_actions):
                Q_new[s, a] = qi_solver.bellman_update(s, a, Q_old)
        err = np.max(np.abs(Q_new - Q_old))
        qi_history.append(err)
        Q_old = Q_new.copy()
        
    np.save("results/qi_qvalues.npy", qi_qvalues)
    np.save("results/qi_values.npy", qi_values)
    np.save("results/qi_policy.npy", qi_policy)
    np.save("results/qi_bellman_history.npy", np.array(qi_history))
    print(f"Q-Iteration converged in {qi_iters} iterations.")
    
    policy_match = np.sum(vi_policy == qi_policy)
    policy_total = len(vi_policy)
    summary = {
        "value_iteration_iterations": int(vi_iters),
        "q_iteration_iterations": int(qi_iters),
        "policy_match": int(policy_match),
        "policy_total_states": int(policy_total),
        "policy_difference": int(policy_total - policy_match),
        "vi_value_range": [float(vi_values.min()), float(vi_values.max())],
        "qi_value_range": [float(qi_values.min()), float(qi_values.max())],
    }
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    print("Training completed successfully.")


if __name__ == '__main__':
    main()