import numpy as np
import pandas as pd
from typing import Tuple

def run_strategy_simulation(df: pd.DataFrame, eligibility_col: str, strategy_prefix: str) -> Tuple[pd.Series, pd.Series]:
    """
    Simulates an equal-weight portfolio strategy based on an eligibility column.

    Args:
        df (pd.DataFrame): The backtest dataframe with 'date', 'code', and 'log_return' columns.
        eligibility_col (str): The name of the boolean column indicating eligibility.
        strategy_prefix (str): A prefix for the new columns (e.g., 'base', 'enhanced').

    Returns:
        tuple: A tuple containing:
            - pd.Series: Daily portfolio log returns.
            - pd.Series: Daily count of eligible codes.
    """
    # Calculate the number of eligible codes each day
    daily_eligible_count = df[df[eligibility_col]].groupby("date")["code"].count()

    # Map the count back to the main dataframe
    count_col = f"daily_eligible_{strategy_prefix}_count"
    df[count_col] = df["date"].map(daily_eligible_count).fillna(0)

    # Calculate weights: equal weight for eligible codes
    weight_col = f"{strategy_prefix}_weight"
    df[weight_col] = 0.0
    
    mask = (df[eligibility_col]) & (df[count_col] > 0)
    df.loc[mask, weight_col] = 1.0 / df.loc[mask, count_col]

    # Calculate the daily portfolio return for the strategy
    return_col = f"{strategy_prefix}_portfolio_return"
    df[return_col] = df[weight_col] * df["log_return"]
    
    daily_returns = df.groupby("date")[return_col].sum()
    
    return daily_returns, daily_eligible_count


def calculate_kpis(returns_series):
    """Calculates key performance indicators for a returns series."""
    kpis = {}
    annualization_factor = 252

    if len(returns_series) == 0 or returns_series.sum() == 0:
        return {
            "Total Cumulative Return": 0.0, "Annualized Return": 0.0,
            "Annualized Volatility": 0.0, "Annualized Sharpe Ratio": 0.0,
            "Maximum Drawdown": 0.0,
        }

    cumulative_returns = np.exp(returns_series.cumsum())
    kpis["Total Cumulative Return"] = (cumulative_returns.iloc[-1] - 1) * 100

    n_days = len(returns_series)
    annualized_return = (cumulative_returns.iloc[-1] ** (annualization_factor / n_days)) - 1
    kpis["Annualized Return"] = annualized_return * 100

    kpis["Annualized Volatility"] = (returns_series.std() * np.sqrt(annualization_factor) * 100)

    if returns_series.std() > 0:
        daily_sharpe = returns_series.mean() / returns_series.std()
        kpis["Annualized Sharpe Ratio"] = daily_sharpe * np.sqrt(annualization_factor)
    else:
        kpis["Annualized Sharpe Ratio"] = 0.0

    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    kpis["Maximum Drawdown"] = drawdown.min() * 100

    return kpis


def calculate_annualized_turnover(daily_weights_df):
    """Calculates the annualized turnover for a portfolio from a daily weights dataframe."""
    daily_turnover = daily_weights_df.diff().abs().sum(axis=1) / 2
    annualized_turnover = daily_turnover.mean() * 252
    return annualized_turnover