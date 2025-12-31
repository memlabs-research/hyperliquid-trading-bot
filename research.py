"""
Research Module for Trading Strategy Development

This module provides tools for backtesting and evaluating trading strategies
using historical data. It focuses on autoregressive (AR) models that predict
future price movements based on lagged returns.

Key functionalities:
- Download and preprocess historical OHLC data
- Create lagged features for time series analysis
- Train and evaluate linear regression models
- Calculate trading performance metrics
- Visualize cumulative returns

Typical workflow:
1. Create AR dataframe with lagged features
2. Train linear regression model on historical data
3. Evaluate performance using directional trading signals
4. Analyze win rate and cumulative returns

Example usage:
    >>> # Create dataset with 5 lags
    >>> df = create_ar_df('BTC', '1h', '2024-01-01', '2024-12-31', no_lags=5)
    >>>
    >>> # Evaluate using first lag as feature
    >>> features = ['close_log_return_lag_1']
    >>> results = eval_linreg(df, features, 'close_log_return')
    >>>
    >>> # Results include model coefficients, win rate, and total return
"""

import hl
import numpy as np
from sklearn.linear_model import LinearRegression


def create_ar_df(sym: str, interval: str, start: datetime, end: datetime, no_lags: int):
    """
    Create an autoregressive (AR) dataframe with lagged price features.

    Downloads historical OHLC data and engineers features for time series
    prediction. Creates both log return lags (for model input) and price
    lags (for analysis). Log returns are used because they have better
    statistical properties than raw price changes.

    Args:
        sym: Trading symbol (e.g., 'BTC', 'BTCUSD')
        interval: Time interval for candles (e.g., '1h', '15m', '1d')
        start: Start date in ISO format (e.g., '2024-01-01')
        end: End date in ISO format (e.g., '2024-12-31')
        no_lags: Number of lagged features to create

    Returns:
        DataFrame with columns:
        - Original OHLC columns (o, h, l, c, v, t)
        - close_log_return: Log return of closing price
        - close_log_return_lag_1 to lag_N: Lagged log returns
        - close_lag_1 to lag_N: Lagged closing prices

    Example:
        >>> df = create_ar_df('BTC', '1h', datetime(2024,01,01), datetime(2024,06,01), no_lags=3)
        >>> print(df.columns)
        ['o', 'h', 'l', 'c', 'v', 't', 'close_log_return',
         'close_log_return_lag_1', 'close_log_return_lag_2', 'close_log_return_lag_3',
         'close_lag_1', 'close_lag_2', 'close_lag_3']
        >>>
        >>> # First row will have NaN for lags (not enough history)
        >>> print(df.head())

    Note:
        - First (no_lags + 1) rows will contain NaN values due to insufficient history
        - Log returns are calculated as: ln(price_t / price_{t-1})
        - These NaN rows are typically dropped before model training
    """
    # Download historical OHLC data from exchange
    df = hl.dl_ohlc_df(sym, interval, start, end)

    # Calculate log returns: ln(current_price / previous_price)
    # Log returns are additive and have better statistical properties
    df['close_log_return'] = np.log(df['c'] / df['c'].shift(1))

    # Create lagged features for autoregressive modeling
    for i in range(1, no_lags + 1):
        # Lagged log returns (main features for prediction)
        df[f'close_log_return_lag_{i}'] = df['close_log_return'].shift(i)

        # Lagged prices (useful for analysis and visualization)
        df[f'close_lag_{i}'] = df['c'].shift(i)

    return df


def eval_linreg(df, features: list, target: str, train_size: float = 0.3):
    """
    Train and evaluate a linear regression trading strategy.

    This function implements a complete backtesting pipeline:
    1. Splits data into train and test sets based on train_size
    2. Trains a linear regression model on training data
    3. Generates predictions and trading signals
    4. Calculates performance metrics (win rate, total return)
    5. Visualizes cumulative returns

    Trading logic:
    - Positive prediction → Buy (long position)
    - Negative prediction → Sell (short position)
    - Returns are multiplied by signal direction

    Args:
        df: DataFrame with features and target (from create_ar_df)
        features: List of column names to use as features
                 (e.g., ['close_log_return_lag_1', 'close_log_return_lag_2'])
        target: Column name of prediction target (typically 'close_log_return')
        train_size: Proportion of data to use for training (default 0.3 = 30%)

    Returns:
        DataFrame with added columns:
        - y_hat: Model predictions (forecasted log returns)
        - dir_signal: Trading direction (+1 for buy, -1 for sell)
        - trade_log_return: Realized log return from the trade
        - cum_trade_log_return: Cumulative log returns over time
        - trade_won: Boolean indicating if trade was profitable

    Side effects:
        - Prints model coefficients and intercept
        - Prints win rate (percentage of profitable trades)
        - Prints total return (as a percentage)
        - Displays plot of cumulative log returns

    Example:
        >>> df = create_ar_df('BTC', '1h', '2024-01-01', '2024-12-31', no_lags=5)
        >>>
        >>> # Test using first lag only (default 30% train, 70% test)
        >>> features = ['close_log_return_lag_1']
        >>> results = eval_linreg(df, features, 'close_log_return')
        [-0.0001] + -0.0000002
        0.512  # 51.2% win rate
        0.087  # 8.7% total return

        >>> # Test using multiple lags with custom train/test split
        >>> features = ['close_log_return_lag_1', 'close_log_return_lag_2', 'close_log_return_lag_3']
        >>> results = eval_linreg(df, features, 'close_log_return', train_size=0.5)  # 50/50 split

        >>> # Analyze the results
        >>> print(results[['y_hat', 'dir_signal', 'trade_log_return']].tail())
        >>> print(f"Sharpe Ratio: {results['trade_log_return'].mean() / results['trade_log_return'].std()}")

    Performance metrics explained:
        - Win rate: Proportion of trades where trade_log_return > 0
        - Total return: exp(sum of log returns) - 1, converted to percentage
        - Cumulative log return: Running sum of trade log returns

    Note:
        - Rows with NaN values are dropped before training
        - Default uses 30% for training, 70% for testing (adjustable via train_size)
        - Model coefficients printed can be used in production (params['model'])
        - Positive coefficient = momentum strategy, negative = mean reversion
    """
    # Remove rows with missing values (from lagged features)
    df.dropna(inplace=True)

    # Split into train and test sets based on train_size parameter
    # Using earlier data for training helps avoid lookahead bias
    i = int(len(df) * train_size)
    df_train, df_test = df[:i].copy(), df[i:].copy()

    # Separate features (X) and target (y) for train and test sets
    X_train, X_test = df_train[features], df_test[features]
    y_train, y_test = df_train[target], df_test[target]

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Generate predictions on test set
    y_hat = model.predict(X_test)

    # === Calculate test set performance ===
    df_test['y_hat'] = y_hat
    df_test['dir_signal'] = np.sign(y_hat)  # +1 for buy, -1 for sell
    df_test['trade_log_return'] = df_test['close_log_return'] * df_test['dir_signal']
    df_test['cum_trade_log_return'] = df_test['trade_log_return'].cumsum()
    df_test['trade_won'] = df_test['trade_log_return'] > 0

    # === Calculate performance on full dataset (for visualization) ===
    df['y_hat'] = model.predict(df[features])
    df['dir_signal'] = np.sign(df['y_hat'])
    df['trade_log_return'] = df['dir_signal'] * df['close_log_return']
    df['cum_trade_log_return'] = df['trade_log_return'].cumsum()
    df['trade_won'] = df['trade_log_return'] > 0

    # === Print performance metrics ===
    # Model parameters (can be copied to params['model'] for production)
    print(f"Model coefficients: {model.coef_} + {model.intercept_}")

    # Win rate: percentage of profitable trades
    print(f"Win rate: {df['trade_won'].mean():.3f}")

    # Total return: convert sum of log returns to percentage return
    # Formula: exp(sum of log returns) - 1
    total_return = np.exp(df['trade_log_return'].sum()) - 1
    print(f"Total return: {total_return:.3f}")

    # Plot cumulative log returns over time
    # Shows the equity curve of the strategy
    df['cum_trade_log_return'].plot()

    return df