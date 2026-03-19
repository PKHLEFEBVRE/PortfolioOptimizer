import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
import scipy.stats as stats
import warnings
from typing import Tuple

# Suppress warnings from optimization
warnings.filterwarnings('ignore')

class SP500Estimator:
    """Estimates empirical Merton Jump-Diffusion parameters from S&P 500 history."""

    def __init__(self, start_date="2000-01-01", end_date="2025-01-01", ticker="^GSPC"):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.dt = 1 / 252.0

    def download_data(self) -> np.ndarray:
        print(f"Downloading historical data for {self.ticker} ({self.start_date} to {self.end_date})...")
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            prices = data['Close']

        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        df = pd.DataFrame({'Price': prices})
        df['LogReturn'] = np.log(df['Price'] / df['Price'].shift(1))
        df = df.dropna()
        return df['LogReturn'].values

    def estimate_threshold(self, returns: np.ndarray, threshold_sigma: float = 3.0):
        overall_std = np.std(returns)
        upper_bound = threshold_sigma * overall_std
        lower_bound = -threshold_sigma * overall_std

        is_jump = (returns > upper_bound) | (returns < lower_bound)
        jump_returns = returns[is_jump]
        normal_returns = returns[~is_jump]

        sigma_cont = np.std(normal_returns) / np.sqrt(self.dt)
        mu_cont = (np.mean(normal_returns) / self.dt) + 0.5 * sigma_cont**2

        n_years = len(returns) * self.dt
        lambda_jump = len(jump_returns) / n_years

        if len(jump_returns) > 0:
            mu_jump = np.mean(jump_returns)
            sigma_jump = np.std(jump_returns)
        else:
            mu_jump = 0.0
            sigma_jump = 0.0

        return mu_cont, sigma_cont, lambda_jump, mu_jump, sigma_jump

    @staticmethod
    def _merton_log_likelihood(params, returns, dt):
        mu, sigma, lam, mu_j, sigma_j = params
        if sigma <= 0 or lam < 0 or sigma_j <= 0:
            return 1e10

        max_jumps = 10
        n = len(returns)
        pdf = np.zeros(n)

        for j in range(max_jumps):
            poisson_prob = stats.poisson.pmf(j, lam * dt)
            mean_j = (mu - 0.5 * sigma**2) * dt + j * mu_j
            var_j = sigma**2 * dt + j * sigma_j**2
            std_j = np.sqrt(var_j)
            norm_pdf = stats.norm.pdf(returns, loc=mean_j, scale=std_j)
            pdf += poisson_prob * norm_pdf

        pdf = np.maximum(pdf, 1e-10)
        return -np.sum(np.log(pdf))

    def estimate_mle(self, returns: np.ndarray, initial_guess: tuple):
        bounds = (
            (None, None),
            (1e-4, 2.0),
            (0.0, 50.0),
            (-1.0, 1.0),
            (1e-4, 1.0)
        )
        result = minimize(
            self._merton_log_likelihood,
            initial_guess,
            args=(returns, self.dt),
            method='L-BFGS-B',
            bounds=bounds
        )
        return result.x

    def get_empirical_ratios(self) -> Tuple[float, float, float]:
        """Returns averaged (lambda_jump, mu_jump_ratio, sigma_jump_ratio) from Threshold & MLE."""
        returns = self.download_data()

        # 1. Threshold Method
        mu_a, sig_a, lam_a, mu_j_a, sig_j_a = self.estimate_threshold(returns)
        ratio_mu_a = mu_j_a / sig_a
        ratio_sig_a = sig_j_a / sig_a

        # 2. MLE Method
        initial_guess = (mu_a, sig_a, lam_a, mu_j_a, max(sig_j_a, 0.01))
        mu_b, sig_b, lam_b, mu_j_b, sig_j_b = self.estimate_mle(returns, initial_guess)
        ratio_mu_b = mu_j_b / sig_b
        ratio_sig_b = sig_j_b / sig_b

        # 3. Average the methods
        avg_lam = (lam_a + lam_b) / 2.0
        avg_ratio_mu = (ratio_mu_a + ratio_mu_b) / 2.0
        avg_ratio_sig = (ratio_sig_a + ratio_sig_b) / 2.0

        print("\n--- S&P 500 Empirical Crash Calibration (2000-2025) ---")
        print(f"Threshold Ratios : μ_j/σ = {ratio_mu_a:.3f}, σ_j/σ = {ratio_sig_a:.3f}")
        print(f"MLE Ratios       : μ_j/σ = {ratio_mu_b:.3f}, σ_j/σ = {ratio_sig_b:.3f}")
        print(f"AVERAGED RATIOS  : μ_j/σ = {avg_ratio_mu:.3f}, σ_j/σ = {avg_ratio_sig:.3f}")
        print(f"AVERAGE FREQ (λ) : {avg_lam:.2f} jumps/year")
        print("------------------------------------------------------\n")

        return avg_lam, avg_ratio_mu, avg_ratio_sig


class MarketDynamics:
    """Simulates asset price paths using Merton Jump-Diffusion."""

    @staticmethod
    def simulate_paths(
        S0: float, mu: float, sigma: float,
        lambda_jump: float, ratio_mu: float, ratio_sig: float,
        T: float, dt: float, n_sims: int
    ) -> np.ndarray:

        n_steps = int(T / dt)

        # Dynamic jump scaling based on empirical S&P 500 ratios
        mu_jump = ratio_mu * sigma
        sigma_jump = ratio_sig * sigma

        # Continuous GBM
        Z = np.random.standard_normal((n_steps, n_sims))
        drift_term = (mu - 0.5 * sigma**2) * dt
        diffusion_term = sigma * np.sqrt(dt) * Z

        # Jumps
        N_jumps = np.random.poisson(lam=lambda_jump * dt, size=(n_steps, n_sims))
        Z_jump = np.random.standard_normal((n_steps, n_sims))
        jump_term = N_jumps * mu_jump + np.sqrt(N_jumps) * sigma_jump * Z_jump

        log_returns = drift_term + diffusion_term + jump_term
        log_returns = np.vstack([np.zeros(n_sims), log_returns])

        cumulative_log_returns = np.cumsum(log_returns, axis=0)
        return S0 * np.exp(cumulative_log_returns)


class ExecutionStrategy:
    """Evaluates the 'Buy the Dip' cascading trigger and sweep strategy."""

    @staticmethod
    def evaluate(
        price_paths: np.ndarray,
        n_splits: int,
        total_investment: float,
        deployment_horizon: float,
        dt: float,
        cash_rate: float,
        sigma: float,
        dip_multiplier: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:

        n_sims = price_paths.shape[1]

        if n_splits == 1:
            shares = total_investment / price_paths[0, :]
            final_vals = shares * price_paths[-1, :]
            return shares, final_vals

        u_max = 1.0
        for k in range(1, n_splits):
            u_max += (1.0 + k * dip_multiplier)

        base_allocation = total_investment / u_max
        deployment_times = np.linspace(0, deployment_horizon, n_splits)
        target_indices = np.round(deployment_times / dt).astype(int)

        total_shares = np.zeros(n_sims)
        remaining_cash = np.full(n_sims, total_investment)
        current_indices = np.zeros(n_sims, dtype=int)

        # t=0 Allocation
        deploy_cash = np.minimum(base_allocation, remaining_cash)
        total_shares += deploy_cash / price_paths[0, :]
        remaining_cash -= deploy_cash
        S0 = price_paths[0, 0]

        # Cascading allocations
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
            execution_prices = price_paths[deploy_indices, np.arange(n_sims)]

            time_diffs = deploy_times_array - (current_indices * dt)
            remaining_cash = remaining_cash * np.exp(cash_rate * time_diffs)

            allocation_multiplier = np.where(hit_trigger, 1.0 + k * dip_multiplier, 1.0)
            actual_deploy_cash = np.minimum(base_allocation * allocation_multiplier, remaining_cash)

            total_shares += actual_deploy_cash / execution_prices
            remaining_cash -= actual_deploy_cash
            current_indices = deploy_indices

        # Final Sweep
        final_idx = target_indices[-1]
        final_time_diffs = (final_idx * dt) - (current_indices * dt)
        remaining_cash = remaining_cash * np.exp(cash_rate * final_time_diffs)

        total_shares += remaining_cash / price_paths[final_idx, :]

        final_values = total_shares * price_paths[-1, :]
        return total_shares, final_values


class GridSimulator:
    """Orchestrates the grid simulation and generates heatmaps."""

    def __init__(self):
        self.S0 = 100.0
        self.total_investment = 10000.0
        self.deployment_horizon = 0.5
        self.performance_horizon = 1.0
        self.dt = 1/252.0
        self.n_sims = 10000
        self.cash_rate = 0.03
        self.dip_multiplier = 1.0
        self.splits_to_test = [1, 2, 3, 4, 6]

        self.mu_range = np.linspace(-0.10, 0.30, 41)
        self.sigma_range = np.linspace(0.05, 0.50, 46)

    def run_and_plot(self):
        # 1. Get Empirical Parameters
        estimator = SP500Estimator()
        emp_lam, emp_ratio_mu, emp_ratio_sig = estimator.get_empirical_ratios()

        results_median = np.zeros((len(self.mu_range), len(self.sigma_range)))
        results_shares = np.zeros((len(self.mu_range), len(self.sigma_range)))

        print("Starting OOP Grid Simulation...")
        for i, mu in enumerate(self.mu_range):
            for j, sigma in enumerate(self.sigma_range):
                np.random.seed(42)

                # Generate paths using the empirical ratios
                price_paths = MarketDynamics.simulate_paths(
                    self.S0, mu, sigma,
                    emp_lam, emp_ratio_mu, emp_ratio_sig,
                    self.performance_horizon, self.dt, self.n_sims
                )

                best_median_split = None
                max_median = -np.inf
                best_shares_split = None
                max_shares = -np.inf

                for n_splits in self.splits_to_test:
                    shares, final_vals = ExecutionStrategy.evaluate(
                        price_paths, n_splits, self.total_investment,
                        self.deployment_horizon, self.dt, self.cash_rate,
                        sigma, self.dip_multiplier
                    )

                    median_val = np.median(final_vals)
                    if median_val > max_median:
                        max_median = median_val
                        best_median_split = n_splits

                    mean_shares = np.mean(shares)
                    if mean_shares > max_shares:
                        max_shares = mean_shares
                        best_shares_split = n_splits

                results_median[i, j] = best_median_split
                results_shares[i, j] = best_shares_split

        print("Generating Heatmaps...")
        self._plot_heatmap(results_median, "heatmap_median_value.png",
                           f"Optimal Splits (Median Value)\nEmpirical S&P500 Jumps: μ_j={emp_ratio_mu:.2f}σ, σ_j={emp_ratio_sig:.2f}σ")

        self._plot_heatmap(results_shares, "heatmap_expected_shares.png",
                           f"Optimal Splits (Expected Shares)\nEmpirical S&P500 Jumps: μ_j={emp_ratio_mu:.2f}σ, σ_j={emp_ratio_sig:.2f}σ")

    def _plot_heatmap(self, data: np.ndarray, filename: str, title: str):
        mu_labels = [f"{m:.0%}" for m in self.mu_range]
        sigma_labels = [f"{s:.0%}" for s in self.sigma_range]
        mu_ticks = np.arange(0, len(self.mu_range), 5)
        sigma_ticks = np.arange(0, len(self.sigma_range), 5)

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(data, cmap="YlGnBu" if "Median" in title else "YlOrRd",
                         xticklabels=sigma_labels, yticklabels=mu_labels)
        ax.invert_yaxis()
        ax.set_xticks(sigma_ticks + 0.5)
        ax.set_xticklabels([sigma_labels[i] for i in sigma_ticks])
        ax.set_yticks(mu_ticks + 0.5)
        ax.set_yticklabels([mu_labels[i] for i in mu_ticks])

        plt.title(title)
        plt.xlabel("Continuous Volatility (σ)")
        plt.ylabel("Continuous Drift (μ)")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    simulator = GridSimulator()
    simulator.run_and_plot()
