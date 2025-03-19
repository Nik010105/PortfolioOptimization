import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import scipy.optimize as sco
from datetime import datetime, timedelta
from fredapi import Fred
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Fetch FRED API key
fred_api_key = os.getenv("FRED_API_KEY")
if not fred_api_key:
    st.error("FRED API Key not found. Please set it in the .env file.")
    st.stop()

fred = Fred(api_key=fred_api_key)

# Streamlit UI Setup
st.set_page_config(page_title="Portfolio Optimization", layout="wide")

st.title("📈 Portfolio Optimization Dashboard")
st.sidebar.header("Portfolio Configuration")

# Use only the tickers from your original code
available_tickers = ["AAPL", "MSFT", "GOOGL", "JNJ", "JPM", "SPY", "GLD", "BND", "VTI"]

# User selects tickers from dropdown
tickers = st.sidebar.multiselect("Select Tickers for Portfolio", available_tickers, 
                                 default=available_tickers)  # Default selects all

# Date Selection
end_date = st.sidebar.date_input("Select End Date", datetime.today())
start_date = st.sidebar.date_input("Select Start Date", datetime.today() - timedelta(days=365 * 10))

# Fetch Stock Data
st.sidebar.write(f"Fetching data from {start_date} to {end_date}...")
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
adj_close_df = data.ffill().dropna()

# Log Returns & Covariance Matrix
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
cov_matrix = log_returns.cov() * 252

# Fetch risk-free rate from FRED API
ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
risk_free_rate = ten_year_treasury_rate.iloc[-1]

# Portfolio Optimization (Sharpe Ratio Maximization)
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.2) if ticker in ["AAPL", "MSFT", "GOOGL", "JNJ", "JPM"] 
          else (0, 0.6) for ticker in tickers]
initial_weights = np.array([1/len(tickers)] * len(tickers))

# Optimize for Maximum Sharpe Ratio
optimized_results = sco.minimize(
    lambda w: -((np.sum(log_returns.mean() * w) * 252 - risk_free_rate) / 
                np.sqrt(w.T @ cov_matrix @ w)),  
    initial_weights, method='SLSQP', constraints=constraints, bounds=bounds)

optimal_weights = optimized_results.x
optimal_portfolio_return = np.sum(log_returns.mean() * optimal_weights) * 252
optimal_portfolio_volatility = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
optimal_sharpe_ratio = (optimal_portfolio_return - risk_free_rate) / optimal_portfolio_volatility

# Monte Carlo Simulation (100,000 Portfolios)
st.subheader("🎲 Monte Carlo Simulation (100,000 Portfolios)")
num_portfolios = 100000
all_weights = np.random.dirichlet(np.ones(len(tickers)), size=num_portfolios)
ret_arr = np.sum(all_weights * log_returns.mean().values * 252, axis=1)
vol_arr = np.sqrt(np.einsum('ij,jk,ik->i', all_weights, cov_matrix.values, all_weights))
sharpe_arr = (ret_arr - risk_free_rate) / vol_arr

# Find the best Sharpe Ratio portfolio from Monte Carlo
max_sharpe_idx = np.argmax(sharpe_arr)
mc_best_sharpe_return = ret_arr[max_sharpe_idx]
mc_best_sharpe_volatility = vol_arr[max_sharpe_idx]
mc_best_sharpe_ratio = sharpe_arr[max_sharpe_idx]
mc_best_weights = all_weights[max_sharpe_idx]

# Monte Carlo Scatter Plot
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.5)
ax.set_xlabel("Volatility (Risk)")
ax.set_ylabel("Expected Return")
ax.set_title("Monte Carlo Simulation: Portfolio Risk vs. Return")
fig.colorbar(scatter, label="Sharpe Ratio")

# Highlight the optimized portfolio from the optimizer
ax.scatter(optimal_portfolio_volatility, optimal_portfolio_return, 
           c='red', marker='*', s=200, label="Optimized Portfolio (Formula)")

# Highlight the best Sharpe Ratio portfolio found in Monte Carlo
ax.scatter(mc_best_sharpe_volatility, mc_best_sharpe_return, 
           c='blue', marker='*', s=200, label="Best Monte Carlo Portfolio")

ax.legend()
st.pyplot(fig)

# Portfolio Weights for Optimized Portfolio (SLSQP Formula)
st.subheader("📊 Optimized Portfolio Allocation (Formula)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(tickers, optimal_weights, color='skyblue')
ax.set_xlabel('Assets')
ax.set_ylabel('Optimal Weights')
ax.set_title('Optimal Portfolio Weights (Formula)')

# Add percentage labels
for i, weight in enumerate(optimal_weights):
    ax.text(i, weight, f'{weight:.2%}', ha='center', va='bottom', fontsize=12)

st.pyplot(fig)

# Portfolio Weights for Best Sharpe Ratio in Monte Carlo
st.subheader("📊 Best Portfolio Allocation from Monte Carlo")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(tickers, mc_best_weights, color='orange')
ax.set_xlabel('Assets')
ax.set_ylabel('Optimal Weights')
ax.set_title('Best Portfolio Weights from Monte Carlo')

# Add percentage labels
for i, weight in enumerate(mc_best_weights):
    ax.text(i, weight, f'{weight:.2%}', ha='center', va='bottom', fontsize=12)

st.pyplot(fig)
