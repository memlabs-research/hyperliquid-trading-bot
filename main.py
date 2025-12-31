"""
Main Trading Application

This is the entry point for a cryptocurrency trading bot that:
1. Connects to live market data via WebSocket
2. Monitors price updates in real-time
3. Executes trades at fixed intervals using a predictive model
4. Handles connection failures with automatic reconnection

The bot operates on a time-based schedule (e.g., every 1 hour) and uses
a linear regression model to predict future price movements and generate
trading signals.

Architecture:
- Async WebSocket connection for real-time price feeds
- Periodic trading execution (synchronized to interval boundaries)
- Historical data initialization for feature calculation
- Automatic reconnection with exponential backoff

Configuration:
- All parameters are defined in the 'params' dictionary
- Model weights and bias are loaded from params['model']
- Trading symbol and interval from params['sym'] and params['interval']

Required Environment Variables:
- HL_SECRET: API secret key for the exchange
- HL_WALLET: Wallet address for trading
"""

import asyncio
import os
import hl
import websockets
import json
import models
import strategy

from typing import List, Tuple
from datetime import datetime, timezone
from stream import LogReturn

# Global variable to store the most recent price from WebSocket
# Used by the periodic trading task to access latest market data
last_price = None

# Configuration parameters for the trading strategy
# All strategy settings are centralized here for easy modification
params = {
    'sym': 'BTC',  # Trading symbol
    'interval': '1m',  # Trading frequency
    'model': {
        'weight': -0.0001,  # Model coefficient (negative = mean reversion)
        'bias': -0.0000002  # Model intercept
    }
}


def interval_mins(interval: str) -> int:
    """
    Convert a time interval string to minutes.

    Supports standard trading intervals:
    - 'Xm' for minutes (e.g., '15m' = 15 minutes)
    - 'Xh' for hours (e.g., '1h' = 60 minutes)
    - 'Xd' for days (e.g., '1d' = 1440 minutes)

    Args:
        interval: Time interval string (e.g., '1h', '15m', '1d')

    Returns:
        Number of minutes in the interval

    Raises:
        ValueError: If interval format is invalid

    Example:
        >>> interval_mins('1h')
        60
        >>> interval_mins('15m')
        15
        >>> interval_mins('1d')
        1440
    """
    # Extract the time unit (last character)
    dur = interval[-1]

    if dur == "h":
        # Hours: extract number and convert to minutes
        no_hours = int(interval[:-1])
        return no_hours * 60
    elif dur == "m":
        # Minutes: extract number directly
        no_minutes = int(interval[:-1])
        return no_minutes
    elif dur == 'd':
        # Days: extract number and convert to minutes
        no_days = int(interval[:-1])
        return no_days * 60 * 24

    # Invalid format
    raise ValueError(f"Invalid interval: {interval}")


def dl_prices_ts(coin: str, interval: str) -> List[Tuple[int, str]]:
    """
    Download historical price data for a trading pair.

    Fetches candlestick data from the exchange API and converts it to
    a simple list of (timestamp, price) tuples. This data is used to
    initialize the lag calculator with historical context before live trading.

    Args:
        coin: Cryptocurrency symbol (e.g., 'BTCUSD')
        interval: Time interval (e.g., '1h')

    Returns:
        List of (timestamp, price) tuples where:
        - timestamp: Unix timestamp as float (seconds since epoch)
        - price: Closing price as string

    Example:
        >>> prices = dl_prices_ts('BTC', '1h')
        >>> # Returns: [(1640000000.0, '50000.0'), ...]
        >>> for ts, price in prices[-5:]:
        ...     dt = datetime.fromtimestamp(ts)
        ...     print(f"{dt}: ${price}")
    """
    # Download candle data from the exchange API
    candles = hl.dl_last_candles(coin, interval)

    # Convert the downloaded candle data to a list of tuples:
    #   - timestamp converted from milliseconds to seconds
    #   - closing price as a string
    prices = [(candle['t'], candle['c']) for candle in candles]

    # Return the list of price tuples
    return prices


async def trade_periodically(interval: str, strat) -> None:
    """
    Execute trades at fixed time intervals, synchronized to interval boundaries.

    This async task runs continuously and executes trades exactly at the start
    of each interval period (e.g., at :00 for 1h intervals, at :00, :15, :30, :45
    for 15m intervals). It uses the most recent price from the WebSocket feed.

    Timing example for '1h' interval:
    - Current time: 10:37:45
    - Next execution: 11:00:00
    - Wait time: 22 minutes, 15 seconds

    Args:
        interval: Trading interval string (e.g., '1h', '15m')
        strat: Strategy instance with on_tick() method

    Global variables used:
        last_price: Most recent price from WebSocket feed

    Example output:
        --- [Sync Every 1h] 11:00:00 | Price: 50123.45 | Lag: 0.0024 | Forecast: 0.0001 ---
    """
    global last_price

    # Convert interval to minutes for timing calculations
    no_mins = interval_mins(interval)
    period_mins = max(1, no_mins)  # Ensure at least 1 minute

    while True:
        now = datetime.now(timezone.utc)

        # Step 1: Calculate how many minutes past the last interval boundary
        # Example: If it's 10:07 and period is 5 mins, mins_past = 2
        mins_past = now.minute % period_mins

        # Step 2: Calculate seconds until the NEXT interval boundary
        # Formula: (remaining minutes in period * 60) - current seconds - microseconds
        seconds_until_next = (
                (period_mins - mins_past) * 60
                - now.second
                - (now.microsecond / 1_000_000.0)
        )

        # Step 3: Wait until the next interval (with tiny buffer to ensure we cross)
        await asyncio.sleep(seconds_until_next + 0.001)

        # Step 4: Execute trade at the interval boundary
        execution_time = datetime.now(timezone.utc)

        if last_price:
            # Process the price tick through the strategy
            tick = strat.on_tick(last_price)
            if tick:
                print(
                    f"--- [Sync Every {interval}] {execution_time.strftime('%H:%M:%S')} | "
                    f"Price: {last_price} | Lag: {tick.lag} | Forecast: {tick.y_hat} ---"
                )
        else:
            # No price data available yet
            print(
                f"--- [Sync Every {interval}] {execution_time.strftime('%H:%M:%S')} | "
                f"Price: {last_price} ---"
            )


async def connect_and_listen(interval: str, strat) -> None:
    """
    Connect to WebSocket feed and listen for real-time price updates.

    This function:
    1. Establishes WebSocket connection to the exchange
    2. Subscribes to trade updates for the specified coin
    3. Updates global last_price variable with each trade
    4. Runs the periodic trading task in parallel
    5. Handles graceful shutdown and task cleanup

    Args:
        interval: Trading interval for periodic execution
        strat: Strategy instance containing coin symbol and trading logic

    Global variables modified:
        last_price: Updated with each incoming trade

    Connection details:
        - Uses ping interval of 20 seconds to keep connection alive
        - Subscribes to 'trades' channel for real-time updates
        - Automatically cancels periodic task on disconnect

    Raises:
        websockets.ConnectionClosed: When connection drops
        OSError: On network errors
    """
    global last_price

    # Start the background timer task for periodic trading
    timer_task = asyncio.create_task(trade_periodically(interval, strat))

    try:
        # Connect to WebSocket with keepalive ping
        async with websockets.connect(hl.URL, ping_interval=20) as ws:
            print(f"Connected to {strat.coin} stream")

            # Subscribe to trade updates for the specified coin
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": strat.coin}
            }))

            # Listen for incoming messages
            async for message in ws:
                data = json.loads(message)
                trade_data = data.get("data")

                # Extract the most recent trade price
                if isinstance(trade_data, list):
                    last_trade = trade_data[-1]
                    last_price = float(last_trade['px'])
                    # Price is stored; periodic task will use it for trading

    finally:
        # Clean up: cancel the periodic trading task when WebSocket disconnects
        # This prevents multiple timer tasks from accumulating on reconnects
        timer_task.cancel()


def create_model() -> models.LinReg:
    """
    Create and initialize the prediction model from params configuration.

    Loads model parameters from the global params dictionary and constructs
    a linear regression model. The parameters should be pre-trained values
    obtained from backtesting or model training.

    Returns:
        Initialized LinReg model ready for prediction

    Configuration used:
        params['model']['weight']: Model coefficient
        params['model']['bias']: Model intercept

    Note:
        The negative weight in default params suggests a mean-reversion strategy
        (predicting prices will move opposite to recent returns).

    Example:
        >>> model = create_model()
        >>> prediction = model.predict(0.001)  # Predict with log return of 0.001
    """
    # Extract model parameters from global configuration
    model_params = params['model']
    weight = model_params['weight']
    bias = model_params['bias']

    return models.LinReg(weight, bias)


def create_strategy(exchange) -> strategy.BasicTakerStrat:
    """
    Create and initialize the trading strategy from params configuration.

    This function:
    1. Extracts configuration from global params dictionary
    2. Creates the prediction model
    3. Initializes the lag calculator (LogReturn)
    4. Downloads historical price data
    5. Warms up the lag calculator with historical prices
    6. Constructs the complete strategy instance

    Args:
        exchange: Exchange client for executing trades

    Returns:
        Fully initialized BasicTakerStrat ready for live trading

    Configuration used:
        params['sym']: Trading symbol (e.g., 'BTCUSD')
        params['interval']: Time interval for historical data

    Example:
        >>> exchange = hl.init(secret, wallet)[2]
        >>> strat = create_strategy(exchange)
        >>> # Strategy is now ready to process live ticks
    """
    # Extract configuration from params
    coin = params['sym']
    interval = params['interval']

    # Create the prediction model
    model = create_model()

    # Define position size per trade (0.0002 BTC)
    trade_sz = 0.0002

    # Create the lag/feature calculator
    lag = LogReturn()

    # Download historical prices to warm up the lag calculator
    # This ensures the first live trade has proper context
    prices = dl_prices_ts(coin, interval)
    for _, price in prices:
        lag.on_tick(float(price))

    # Construct the complete strategy with all components
    return strategy.BasicTakerStrat(exchange, coin, model, trade_sz, lag)


async def main() -> None:
    """
    Main application entry point.

    This function:
    1. Loads credentials from environment variables
    2. Initializes the exchange connection
    3. Creates the trading strategy (using params configuration)
    4. Runs the WebSocket listener with automatic reconnection
    5. Handles connection failures with exponential backoff

    The application runs indefinitely, reconnecting automatically if
    the WebSocket connection drops for any reason.

    Configuration:
        Uses params['interval'] for trading frequency and validation

    Environment variables required:
        HL_SECRET: Exchange API secret key
        HL_WALLET: Wallet address for trading

    Reconnection strategy:
        - Initial backoff: 1 second
        - Max backoff: 30 seconds
        - Doubles on each failure (exponential backoff)
        - Resets to 1 second on successful connection

    Example:
        $ export HL_SECRET="your_secret_key"
        $ export HL_WALLET="your_wallet_address"
        $ python main.py
    """
    backoff = 1  # Initial reconnection delay in seconds
    interval = params['interval']  # Get trading interval from params

    # Load credentials from environment variables
    secret_key = os.environ["HL_SECRET"]
    wallet = os.environ["HL_WALLET"]

    # Initialize exchange connection and get client
    address, info, exchange = hl.init(secret_key, wallet)

    # Create the trading strategy using params configuration
    strat = create_strategy(exchange)

    # Validate interval is supported by the exchange
    if interval not in hl.TIME_INTERVALS:
        raise Exception(f"Invalid time interval: {interval}")

    # Main connection loop with automatic reconnection
    while True:
        try:
            # Connect to WebSocket and start trading
            await connect_and_listen(interval, strat)
            backoff = 1  # Reset backoff on clean exit

        except (websockets.ConnectionClosed, OSError) as e:
            # Connection failed - wait before reconnecting
            print(f"Disconnected: {e}. Reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)

            # Increase backoff for next failure (exponential backoff)
            backoff = min(backoff * 2, 30)  # Cap at 30 seconds


# Application entry point
# Note: URL constant for WebSocket connection should be defined
# (missing from provided code - likely defined in hl module)
asyncio.run(main())