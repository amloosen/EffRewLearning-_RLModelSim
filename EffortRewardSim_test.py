import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Change the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

class RewardLearner:
    def __init__(self, alpha_r=0.2, init_reward=3.5):
        self.alpha_r = alpha_r
        self.R_hat = init_reward

    def update(self, received_reward):
        self.R_hat += self.alpha_r * (received_reward - self.R_hat)
        print(f"Updated R_hat: {self.R_hat}")

class EffortLearner:
    def __init__(self, alpha_e=0.1, init_effort=50.0):
        self.alpha_e = alpha_e
        self.E_hat = init_effort

    def update(self, threshold):
        self.E_hat += self.alpha_e * (threshold - self.E_hat)
        print(f"Updated E_hat: {self.E_hat}")

class EffortDiscounter:
    def __init__(self, kappa=0.5, sigma=7.0, beta=5.0, policy='argmax'):
        """
        kappa : effort sensitivity
        sigma : noise/uncertainty for success probability
        beta  : inverse temperature for softmax
        policy: 'argmax' or 'softmax'
        """
        self.kappa = kappa
        self.sigma = sigma
        self.beta = beta
        self.policy = policy

    def choose_effort(self, R_hat, E_hat):
        """
        We enumerate e in [0..100], compute:
          p_success(e) = norm.cdf( (e - E_hat)/sigma )
          SV(e)        = R_hat - kappa* e
          EU(e)        = SV(e)* p_success(e)
        
        If policy == 'argmax', pick e that maximizes EU(e).
        If policy == 'softmax', sample e proportionally to exp(beta*EU(e)).
        """
        e_values = np.arange(101)
        EU_values = []

        # 1) Compute expected utility for each candidate effort e
        for e in e_values:
            p_success = norm.cdf((e - E_hat) / self.sigma)
            SV = R_hat - self.kappa * e
            EU = SV * p_success
            EU_values.append(EU)

        # 2) Decide
        EU_values = np.array(EU_values)
        if self.policy == 'argmax':
            chosen_e = e_values[np.argmax(EU_values)]
        else:
            # softmax sampling
            # subtract max for numerical stability
            max_eu = np.max(EU_values)
            exp_vals = np.exp(self.beta * (EU_values - max_eu))
            probs = exp_vals / np.sum(exp_vals)
            chosen_e = np.random.choice(e_values, p=probs)

        print(f"Chosen effort: {chosen_e}")
        return chosen_e

def read_csv_files(directory):
    csv_files = []
    print(f"Checking files in directory: {directory}")
    for filename in os.listdir(directory):
        if filename.startswith("sub") and filename.endswith(".csv"):
            print(f"Reading CSV file: {filename}")
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            csv_files.append(df)
        else:
            print(f"Skipping file: {filename}")
    return csv_files

def simulate_experiment(num_trials=50,
                        reward_learner_params=None,
                        effort_learner_params=None,
                        discounter_params=None,
                        data=None,
                        ignore_zero_reward=True):
    if reward_learner_params is None:
        reward_learner_params = {}
    if effort_learner_params is None:
        effort_learner_params = {}
    if discounter_params is None:
        discounter_params = {}

    RL = RewardLearner(**reward_learner_params)
    EL = EffortLearner(**effort_learner_params)
    discounter = EffortDiscounter(**discounter_params)

    results = {
        'trial': [],
        'threshold': [],
        'reward': [],
        'chosen_effort': [],
        'outcome': [],
        'R_hat': [],
        'E_hat': []
    }

    for t in range(1, num_trials+1):
        if data is not None:
            threshold = data.iloc[t-1]['threshold']
            reward = data.iloc[t-1]['reward']
        else:
            threshold = np.random.uniform(30, 100)
            reward = np.random.uniform(1, 7)

        if ignore_zero_reward and reward == 0:
            print(f"Trial {t}: Skipping zero-reward trial")
            continue

        R_hat_t = RL.R_hat
        E_hat_t = EL.E_hat

        chosen_e = discounter.choose_effort(R_hat_t, E_hat_t)

        p_success = norm.cdf((chosen_e - E_hat_t) / discounter.sigma)
        outcome = np.random.binomial(1, p_success)

        received_reward = reward if outcome == 1 else 0.0
        RL.update(received_reward)
        EL.update(threshold)

        # Debugging prints
        print(f"Trial {t}: R_hat = {RL.R_hat}, E_hat = {EL.E_hat}, chosen_e = {chosen_e}")

        results['trial'].append(t)
        results['threshold'].append(threshold)
        results['reward'].append(reward)
        results['chosen_effort'].append(chosen_e)
        results['outcome'].append(outcome)
        results['R_hat'].append(RL.R_hat)
        results['E_hat'].append(EL.E_hat)

    return results

def main():
    # Get data
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')

    # Load the task data only once
    task_data_path = os.path.join(data_directory, 'task_data_nobehav.csv')
    
    if os.path.exists(task_data_path):
        task_data = pd.read_csv(task_data_path)
        print(task_data.head())
    else:
        print(f"Task data file not found: {task_data_path}")
        return
    
    task_data.rename(columns={'Points': 'reward', 'effLevel': 'threshold'}, inplace=True)
    task_data = task_data[['reward', 'threshold']]
    print(task_data.head())

    # Example usage: simulate 50 trials with random threshold & reward
    # purely deterministic policy (argmax)
    discounter_params = {'kappa':0.2, 'sigma':7.0, 'beta':5.0, 'policy':'argmax'}
    
    sim_results = simulate_experiment(
        num_trials=len(task_data),
        reward_learner_params={'alpha_r':0.2, 'init_reward':3.5},
        effort_learner_params={'alpha_e':0.8, 'init_effort':50.0},
        discounter_params=discounter_params,
        data=task_data,  # Ensure task_data is defined and passed correctly
        ignore_zero_reward=True  # Set to False if you don't want to ignore zero-reward trials
    )
    
    # Let's do a quick plot: chosen_effort & threshold
    trials = sim_results['trial']
    chosen_effort = sim_results['chosen_effort']
    threshold     = sim_results['threshold']
    
    plt.figure(figsize=(10,6))
    plt.plot(trials, chosen_effort, label='Chosen Effort')
    plt.plot(trials, threshold, label='True Threshold', alpha=0.7)
    plt.xlabel('Trial')
    plt.ylabel('Effort')
    plt.title('Chosen Effort vs. True Threshold')
    plt.legend()
    plt.show()
    
    # If we had pilot data, we would do something like:
    # data = load_my_pilot_data()  # a list of dicts or something
    # results_with_data = simulate_experiment(
    #     num_trials=len(data),
    #     reward_learner_params=...,
    #     effort_learner_params=...,
    #     discounter_params=...,
    #     data=data
    # )
    # Then do the same plotting but using real thresholds & rewards.

# Ensure to call the main function
if __name__ == "__main__":
    main()