import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

def simulate_gbm(S0: float, mu: float, sigma: float, T: float, dt: float, n_sims: int) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths.

    Args:
        S0: Initial stock price
        mu: Drift (annualized)
        sigma: Volatility (annualized)
        T: Total time (years)
        dt: Time step
        n_sims: Number of simulations

    Returns:
        np.ndarray of shape (n_steps + 1, n_sims) containing price paths.
    """
    n_steps = int(T / dt)
    # Generate standard normal random variables
    Z = np.random.standard_normal((n_steps, n_sims))

    # Calculate log returns
    # d(ln S) = (mu - 0.5 * sigma^2) dt + sigma * dW
    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_term = sigma * np.sqrt(dt) * Z

    log_returns = drift_term + diffusion_term

    # Prepend zeros for the initial state (t=0)
    log_returns = np.vstack([np.zeros(n_sims), log_returns])

    # Cumulative sum to get log prices
    cumulative_log_returns = np.cumsum(log_returns, axis=0)

    # Exponentiate to get prices
    price_paths = S0 * np.exp(cumulative_log_returns)
    return price_paths

def evaluate_strategy(
    price_paths: np.ndarray,
    n_splits: int,
    total_investment: float,
    deployment_horizon: float,
    performance_horizon: float,
    dt: float,
    cash_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate an N-split investment strategy.

    Args:
        price_paths: Array of simulated prices, shape (n_steps + 1, n_sims).
        n_splits: Number of equal parts to split the investment into.
        total_investment: Total initial cash.
        deployment_horizon: Time over which to deploy capital (e.g., 0.5 years).
        performance_horizon: Time at which to evaluate final value (e.g., 1.0 years).
        dt: Time step of the simulation.
        cash_rate: Risk-free rate for uninvested cash (annualized, continuous).

    Returns:
        Tuple of (shares_acquired, final_portfolio_values).
        Both are 1D arrays of length n_sims.
    """
    n_sims = price_paths.shape[1]

    if n_splits == 1:
        # Lump sum at t=0
        shares_acquired = total_investment / price_paths[0, :]
        final_prices = price_paths[-1, :]
        final_values = shares_acquired * final_prices
        return shares_acquired, final_values

    # Calculate deployment times
    # Example: if 2 splits over 0.5 years, deploy at t=0 and t=0.5
    # If 3 splits over 0.5 years, deploy at t=0, t=0.25, t=0.5
    deployment_times = np.linspace(0, deployment_horizon, n_splits)
    deployment_indices = np.round(deployment_times / dt).astype(int)

    investment_per_split = total_investment / n_splits
    total_shares = np.zeros(n_sims)

    for t_idx, t_time in zip(deployment_indices, deployment_times):
        # Calculate how much the cash has grown up to this point
        # Cash compounding: C(t) = C(0) * exp(r * t)
        cash_available = investment_per_split * np.exp(cash_rate * t_time)

        # Buy shares at the current price
        current_prices = price_paths[t_idx, :]
        shares_bought = cash_available / current_prices
        total_shares += shares_bought

    final_prices = price_paths[-1, :]
    final_values = total_shares * final_prices

    return total_shares, final_values

def run_simulation_grid():
    # Parameters
    S0 = 100.0
    total_investment = 10000.0
    deployment_horizon = 0.5  # 6 months
    performance_horizon = 1.0 # 1 year
    dt = 1/252.0  # Daily steps
    n_sims = 10000
    cash_rate = 0.03

    # Ranges
    # Drift from -10% to +30%
    mu_range = np.linspace(-0.10, 0.30, 41)
    # Volatility from 5% to 50%
    sigma_range = np.linspace(0.05, 0.50, 46)

    # Strategies: 1 (Lump Sum), 2, 3, 4, 6 (monthly over 6m)
    splits_to_test = [1, 2, 3, 4, 6]

    results_median = np.zeros((len(mu_range), len(sigma_range)))
    results_shares = np.zeros((len(mu_range), len(sigma_range)))

    print("Starting simulation grid...")
    for i, mu in enumerate(mu_range):
        for j, sigma in enumerate(sigma_range):
            # Simulate paths for this (mu, sigma) pair
            np.random.seed(42) # For reproducibility and fair comparison across splits
            price_paths = simulate_gbm(S0, mu, sigma, performance_horizon, dt, n_sims)

            best_median_split = None
            max_median_value = -np.inf

            best_shares_split = None
            max_expected_shares = -np.inf

            for n_splits in splits_to_test:
                shares, final_values = evaluate_strategy(
                    price_paths, n_splits, total_investment,
                    deployment_horizon, performance_horizon, dt, cash_rate
                )

                # Metric 1: Median Final Value
                median_value = np.median(final_values)
                if median_value > max_median_value:
                    max_median_value = median_value
                    best_median_split = n_splits

                # Metric 2: Expected (Mean) Number of Shares
                expected_shares = np.mean(shares)
                if expected_shares > max_expected_shares:
                    max_expected_shares = expected_shares
                    best_shares_split = n_splits

            results_median[i, j] = best_median_split
            results_shares[i, j] = best_shares_split

    print("Simulation complete. Generating heatmaps...")

    # Generate Heatmaps
    mu_labels = [f"{m:.0%}" for m in mu_range]
    sigma_labels = [f"{s:.0%}" for s in sigma_range]

    # Ensure consistent formatting for labels to avoid clutter
    # Subsample ticks for cleaner axis
    mu_ticks = np.arange(0, len(mu_range), 5)
    sigma_ticks = np.arange(0, len(sigma_range), 5)

    # 1. Heatmap for Median Final Value
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(results_median, cmap="YlGnBu",
                     xticklabels=sigma_labels, yticklabels=mu_labels)
    ax.invert_yaxis() # Put lower drifts at the bottom

    # Adjust ticks
    ax.set_xticks(sigma_ticks + 0.5)
    ax.set_xticklabels([sigma_labels[i] for i in sigma_ticks])
    ax.set_yticks(mu_ticks + 0.5)
    ax.set_yticklabels([mu_labels[i] for i in mu_ticks])

    plt.title("Optimal Number of Splits to Maximize Median Final Value (1 Yr)")
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Drift (μ)")
    plt.savefig("heatmap_median_value.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Heatmap for Expected Shares
    plt.figure(figsize=(10, 8))
    ax2 = sns.heatmap(results_shares, cmap="YlOrRd",
                      xticklabels=sigma_labels, yticklabels=mu_labels)
    ax2.invert_yaxis()

    # Adjust ticks
    ax2.set_xticks(sigma_ticks + 0.5)
    ax2.set_xticklabels([sigma_labels[i] for i in sigma_ticks])
    ax2.set_yticks(mu_ticks + 0.5)
    ax2.set_yticklabels([mu_labels[i] for i in mu_ticks])

    plt.title("Optimal Number of Splits to Maximize Expected Shares Acquired")
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Drift (μ)")
    plt.savefig("heatmap_expected_shares.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Heatmaps saved to 'heatmap_median_value.png' and 'heatmap_expected_shares.png'.")

if __name__ == "__main__":
    run_simulation_grid()
