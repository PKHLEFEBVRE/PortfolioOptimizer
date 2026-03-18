import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats
import warnings

# Suppress warnings from optimization
warnings.filterwarnings('ignore')

def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Downloading historical data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    # Ensure we use Close prices
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']

    # yf.download sometimes returns MultiIndex columns. Flatten if necessary
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()

    df = pd.DataFrame({'Price': prices})
    df['LogReturn'] = np.log(df['Price'] / df['Price'].shift(1))
    return df.dropna()

def estimate_threshold_method(returns: np.ndarray, dt: float, threshold_sigma: float = 3.0):
    """
    Method A: The Threshold Method
    Classifies jumps based on returns exceeding a certain number of standard deviations.
    """
    print("\n--- Running Method A: Threshold Method ---")

    # Initial estimate of overall volatility
    overall_std = np.std(returns)

    # Identify jumps
    upper_bound = threshold_sigma * overall_std
    lower_bound = -threshold_sigma * overall_std

    is_jump = (returns > upper_bound) | (returns < lower_bound)
    jump_returns = returns[is_jump]
    normal_returns = returns[~is_jump]

    # 1. Estimate continuous parameters (from normal returns)
    # E[normal_returns] = (mu - 0.5 * sigma^2) * dt
    # Var(normal_returns) = sigma^2 * dt
    sigma_cont = np.std(normal_returns) / np.sqrt(dt)
    mu_cont = (np.mean(normal_returns) / dt) + 0.5 * sigma_cont**2

    # 2. Estimate jump parameters
    n_days = len(returns)
    n_years = n_days * dt
    n_jumps = len(jump_returns)

    lambda_jump = n_jumps / n_years

    if n_jumps > 0:
        mu_jump = np.mean(jump_returns)
        sigma_jump = np.std(jump_returns)
    else:
        mu_jump = 0.0
        sigma_jump = 0.0

    print(f"Identified {n_jumps} jumps out of {n_days} days using a {threshold_sigma}σ threshold.")

    return mu_cont, sigma_cont, lambda_jump, mu_jump, sigma_jump, is_jump

def merton_log_likelihood(params, returns, dt):
    """
    Negative log-likelihood function for Merton Jump-Diffusion.
    We truncate the infinite sum of the Poisson distribution to a small number of jumps (max_j=10).
    """
    mu, sigma, lam, mu_j, sigma_j = params

    # Constraints/Bounds enforcement via penalties
    if sigma <= 0 or lam < 0 or sigma_j <= 0:
        return 1e10

    max_jumps = 10
    n = len(returns)

    pdf = np.zeros(n)
    for j in range(max_jumps):
        # Probability of exactly j jumps in dt
        # P(N=j) = exp(-lam*dt) * (lam*dt)^j / j!
        poisson_prob = stats.poisson.pmf(j, lam * dt)

        # Mean and variance of the return given j jumps
        # mean_j = (mu - 0.5*sigma^2)*dt + j*mu_j
        # var_j = sigma^2*dt + j*sigma_j^2
        mean_j = (mu - 0.5 * sigma**2) * dt + j * mu_j
        var_j = sigma**2 * dt + j * sigma_j**2
        std_j = np.sqrt(var_j)

        # Normal PDF for returns given j jumps
        norm_pdf = stats.norm.pdf(returns, loc=mean_j, scale=std_j)

        pdf += poisson_prob * norm_pdf

    # Prevent log(0)
    pdf = np.maximum(pdf, 1e-10)

    # Negative log-likelihood
    nll = -np.sum(np.log(pdf))
    return nll

def estimate_mle_method(returns: np.ndarray, dt: float, initial_guess: tuple):
    """
    Method B: Maximum Likelihood Estimation
    """
    print("\n--- Running Method B: Maximum Likelihood Estimation ---")

    # Bounds for parameters: (mu, sigma, lambda, mu_j, sigma_j)
    bounds = (
        (None, None),   # mu
        (1e-4, 2.0),    # sigma (positive)
        (0.0, 50.0),    # lam (positive, realistically < 50 jumps/year)
        (-1.0, 1.0),    # mu_j
        (1e-4, 1.0)     # sigma_j (positive)
    )

    # Use L-BFGS-B which handles bounds
    result = minimize(
        merton_log_likelihood,
        initial_guess,
        args=(returns, dt),
        method='L-BFGS-B',
        bounds=bounds
    )

    if result.success:
        print("MLE Optimization successful.")
    else:
        print(f"MLE Optimization failed: {result.message}")

    mu_mle, sigma_mle, lam_mle, mu_j_mle, sigma_j_mle = result.x
    return mu_mle, sigma_mle, lam_mle, mu_j_mle, sigma_j_mle

def main():
    ticker = "TSLA"
    start_date = "2018-01-01"
    end_date = "2023-12-31"
    dt = 1 / 252.0  # Daily data

    df = download_data(ticker, start_date, end_date)
    returns = df['LogReturn'].values

    # --- METHOD A ---
    res_a = estimate_threshold_method(returns, dt, threshold_sigma=3.0)
    mu_a, sig_a, lam_a, mu_j_a, sig_j_a, is_jump = res_a

    # --- METHOD B ---
    # Use Method A results as a smart initial guess for MLE
    initial_guess = (mu_a, sig_a, lam_a, mu_j_a, max(sig_j_a, 0.01))
    mu_b, sig_b, lam_b, mu_j_b, sig_j_b = estimate_mle_method(returns, dt, initial_guess)

    # --- PRINT COMPARISON ---
    print("\n=======================================================")
    print(f"      PARAMETER ESTIMATION RESULTS FOR {ticker}")
    print("=======================================================")
    print(f"{'Parameter':<15} | {'Method A (Threshold)':<20} | {'Method B (MLE)'}")
    print("-" * 55)
    print(f"{'Drift (μ)':<15} | {mu_a:>18.4%} | {mu_b:>18.4%}")
    print(f"{'Volatility (σ)':<15} | {sig_a:>18.4%} | {sig_b:>18.4%}")
    print(f"{'Jump Freq (λ)':<15} | {lam_a:>18.2f} jumps/yr | {lam_b:>18.2f} jumps/yr")
    print(f"{'Jump Mean (μ_j)':<15} | {mu_j_a:>18.4%} | {mu_j_b:>18.4%}")
    print(f"{'Jump Vol (σ_j)':<15} | {sig_j_a:>18.4%} | {sig_j_b:>18.4%}")
    print("=======================================================\n")

    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['LogReturn'], color='blue', alpha=0.5, label='Normal Returns')

    # Scatter the identified jumps from Method A
    jump_dates = df.index[is_jump]
    jump_vals = df['LogReturn'][is_jump]
    plt.scatter(jump_dates, jump_vals, color='red', label=f'Jumps (Threshold > 3σ)', zorder=5)

    plt.title(f"{ticker} Daily Log Returns and Identified Jumps (2018-2023)")
    plt.xlabel("Date")
    plt.ylabel("Daily Log Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("tsla_jumps_estimation.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved plot to 'tsla_jumps_estimation.png'")

if __name__ == "__main__":
    main()
