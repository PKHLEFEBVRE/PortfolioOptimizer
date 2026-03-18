import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

def simulate_gbm(S0: float, mu: float, sigma: float, T: float, dt: float, n_sims: int) -> np.ndarray:
    n_steps = int(T / dt)
    Z = np.random.standard_normal((n_steps, n_sims))
    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_term = sigma * np.sqrt(dt) * Z
    log_returns = drift_term + diffusion_term
    log_returns = np.vstack([np.zeros(n_sims), log_returns])
    cumulative_log_returns = np.cumsum(log_returns, axis=0)
    price_paths = S0 * np.exp(cumulative_log_returns)
    return price_paths

def evaluate_strategy_before(
    price_paths: np.ndarray,
    n_splits: int,
    total_investment: float,
    deployment_horizon: float,
    performance_horizon: float,
    dt: float,
    cash_rate: float,
    sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    n_sims = price_paths.shape[1]

    if n_splits == 1:
        shares_acquired = total_investment / price_paths[0, :]
        final_prices = price_paths[-1, :]
        final_values = shares_acquired * final_prices
        return shares_acquired, final_values

    deployment_times = np.linspace(0, deployment_horizon, n_splits)
    target_indices = np.round(deployment_times / dt).astype(int)

    investment_per_split = total_investment / n_splits
    total_shares = np.zeros(n_sims)
    current_indices = np.zeros(n_sims, dtype=int)

    cash_available = investment_per_split * np.exp(cash_rate * 0.0)
    current_prices = price_paths[0, :]
    total_shares += cash_available / current_prices

    S0 = price_paths[0, 0]

    for k in range(1, n_splits):
        target_idx = target_indices[k]
        trigger_margin = k * 0.25 * sigma
        trigger_price = S0 * max(0.0, 1.0 - trigger_margin)

        time_steps = np.arange(target_idx + 1)[:, None]
        valid_search_mask = (time_steps > current_indices) & (time_steps <= target_idx)
        trigger_mask = (price_paths[:target_idx + 1, :] <= trigger_price) & valid_search_mask

        first_hit_indices = np.argmax(trigger_mask, axis=0)
        hit_trigger = trigger_mask[first_hit_indices, np.arange(n_sims)]
        deploy_indices = np.where(hit_trigger, first_hit_indices, target_idx)
        deploy_times_array = deploy_indices * dt

        cash_available = investment_per_split * np.exp(cash_rate * deploy_times_array)
        execution_prices = price_paths[deploy_indices, np.arange(n_sims)]

        total_shares += cash_available / execution_prices
        current_indices = deploy_indices

    final_prices = price_paths[-1, :]
    final_values = total_shares * final_prices

    return total_shares, final_values

def run_simulation_grid_before():
    S0 = 100.0
    total_investment = 10000.0
    deployment_horizon = 0.5
    performance_horizon = 1.0
    dt = 1/252.0
    n_sims = 10000
    cash_rate = 0.03

    mu_range = np.linspace(-0.10, 0.30, 41)
    sigma_range = np.linspace(0.05, 0.50, 46)
    splits_to_test = [1, 2, 3, 4, 6]

    results_median = np.zeros((len(mu_range), len(sigma_range)))
    results_shares = np.zeros((len(mu_range), len(sigma_range)))

    print("Starting BEFORE simulation grid...")
    for i, mu in enumerate(mu_range):
        for j, sigma in enumerate(sigma_range):
            np.random.seed(42)
            price_paths = simulate_gbm(S0, mu, sigma, performance_horizon, dt, n_sims)

            best_median_split = None
            max_median_value = -np.inf
            best_shares_split = None
            max_expected_shares = -np.inf

            for n_splits in splits_to_test:
                shares, final_values = evaluate_strategy_before(
                    price_paths, n_splits, total_investment,
                    deployment_horizon, performance_horizon, dt, cash_rate, sigma
                )

                median_value = np.median(final_values)
                if median_value > max_median_value:
                    max_median_value = median_value
                    best_median_split = n_splits

                expected_shares = np.mean(shares)
                if expected_shares > max_expected_shares:
                    max_expected_shares = expected_shares
                    best_shares_split = n_splits

            results_median[i, j] = best_median_split
            results_shares[i, j] = best_shares_split

    print("Generating BEFORE heatmaps...")
    mu_labels = [f"{m:.0%}" for m in mu_range]
    sigma_labels = [f"{s:.0%}" for s in sigma_range]
    mu_ticks = np.arange(0, len(mu_range), 5)
    sigma_ticks = np.arange(0, len(sigma_range), 5)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(results_median, cmap="YlGnBu", xticklabels=sigma_labels, yticklabels=mu_labels)
    ax.invert_yaxis()
    ax.set_xticks(sigma_ticks + 0.5)
    ax.set_xticklabels([sigma_labels[i] for i in sigma_ticks])
    ax.set_yticks(mu_ticks + 0.5)
    ax.set_yticklabels([mu_labels[i] for i in mu_ticks])
    plt.title("[BEFORE] Optimal Number of Splits (Median Final Value)")
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Drift (μ)")
    plt.savefig("heatmap_median_value_BEFORE.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    ax2 = sns.heatmap(results_shares, cmap="YlOrRd", xticklabels=sigma_labels, yticklabels=mu_labels)
    ax2.invert_yaxis()
    ax2.set_xticks(sigma_ticks + 0.5)
    ax2.set_xticklabels([sigma_labels[i] for i in sigma_ticks])
    ax2.set_yticks(mu_ticks + 0.5)
    ax2.set_yticklabels([mu_labels[i] for i in mu_ticks])
    plt.title("[BEFORE] Optimal Number of Splits (Expected Shares)")
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Drift (μ)")
    plt.savefig("heatmap_expected_shares_BEFORE.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_simulation_grid_before()