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
    cash_rate: float,
    sigma: float,
    dip_multiplier: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate an N-split investment strategy with a price-contingent trigger,
    dynamic allocation sizing (dip multiplier), and a final cash sweep.

    Args:
        price_paths: Array of simulated prices, shape (n_steps + 1, n_sims).
        n_splits: Number of equal parts to split the investment into.
        total_investment: Total initial cash.
        deployment_horizon: Time over which to deploy capital (e.g., 0.5 years).
        performance_horizon: Time at which to evaluate final value (e.g., 1.0 years).
        dt: Time step of the simulation.
        cash_rate: Risk-free rate for uninvested cash (annualized, continuous).
        sigma: Volatility used to calculate the price trigger margin.
        dip_multiplier: Multiplier for the base allocation when hitting deeper triggers.

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

    # Sizing logic: Calculate the maximum possible units if all triggers hit
    # U_max = 1 (t=0) + sum_{k=1}^{n_splits-1} (1 + k * dip_multiplier)
    u_max = 1.0
    for k in range(1, n_splits):
        u_max += (1.0 + k * dip_multiplier)

    base_allocation = total_investment / u_max

    # Calculate target dates for the time-based fallback
    deployment_times = np.linspace(0, deployment_horizon, n_splits)
    target_indices = np.round(deployment_times / dt).astype(int)

    total_shares = np.zeros(n_sims)
    remaining_cash = np.full(n_sims, total_investment)

    # Track which index in the simulation each path is currently at
    current_indices = np.zeros(n_sims, dtype=int)

    # 1. First Allocation at t=0 (Always 1 base unit)
    deploy_cash = np.minimum(base_allocation, remaining_cash)
    current_prices = price_paths[0, :]
    total_shares += deploy_cash / current_prices
    remaining_cash -= deploy_cash

    S0 = price_paths[0, 0] # Initial stock price is the same for all paths

    # Process remaining allocations 1 through N-1
    for k in range(1, n_splits):
        target_idx = target_indices[k]

        # Cascading trigger price: drop by k * 0.25 * sigma
        trigger_margin = k * 0.25 * sigma
        trigger_price = S0 * max(0.0, 1.0 - trigger_margin) # Prevent negative prices

        # Vectorized approach to find the first time price drops below trigger_price
        time_steps = np.arange(target_idx + 1)[:, None]
        valid_search_mask = (time_steps > current_indices) & (time_steps <= target_idx)

        trigger_mask = (price_paths[:target_idx + 1, :] <= trigger_price) & valid_search_mask

        first_hit_indices = np.argmax(trigger_mask, axis=0)
        hit_trigger = trigger_mask[first_hit_indices, np.arange(n_sims)]

        deploy_indices = np.where(hit_trigger, first_hit_indices, target_idx)
        deploy_times_array = deploy_indices * dt

        execution_prices = price_paths[deploy_indices, np.arange(n_sims)]

        # Grow the remaining cash from the PREVIOUS execution time to THIS execution time
        prev_deploy_times = current_indices * dt
        time_diffs = deploy_times_array - prev_deploy_times
        remaining_cash = remaining_cash * np.exp(cash_rate * time_diffs)

        # Determine how much cash to deploy
        # If trigger hit: Base * (1 + k * dip_multiplier)
        # If fallback: Base
        allocation_multiplier = np.where(hit_trigger, 1.0 + k * dip_multiplier, 1.0)
        desired_deploy_cash = base_allocation * allocation_multiplier

        # We can't deploy more cash than we have
        actual_deploy_cash = np.minimum(desired_deploy_cash, remaining_cash)

        total_shares += actual_deploy_cash / execution_prices
        remaining_cash -= actual_deploy_cash

        current_indices = deploy_indices

    # The Sweep: Deploy all remaining cash at the end of the deployment horizon
    final_target_idx = target_indices[-1]

    # Grow remaining cash from the LAST execution time to the FINAL deployment horizon time
    final_time_diffs = (final_target_idx * dt) - (current_indices * dt)
    # Some paths might already be at the final_target_idx, where time_diffs is 0
    remaining_cash = remaining_cash * np.exp(cash_rate * final_time_diffs)

    final_deployment_prices = price_paths[final_target_idx, :]

    # Sweep
    total_shares += remaining_cash / final_deployment_prices
    remaining_cash = 0.0 # All cash deployed

    # Calculate final value at year 1
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
    dip_multiplier = 1.0 # E.g., double the base allocation on the first drop

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
                    deployment_horizon, performance_horizon, dt, cash_rate, sigma, dip_multiplier
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
