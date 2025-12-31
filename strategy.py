"""
Trading Strategy Module

This module implements a basic algorithmic trading strategy that uses machine learning
predictions to execute trades. It operates on streaming market data and makes real-time
trading decisions based on predicted price movements.

The strategy follows a simple workflow:
1. Receive new price tick
2. Calculate technical indicators (e.g., log returns)
3. Generate prediction using ML model
4. Determine trading action (buy/sell)
5. Execute trade on exchange
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from stream import Tick


@dataclass
class Order:
    """
    Represents a trading order to be executed.

    Attributes:
        coin: The cryptocurrency or asset symbol (e.g., 'BTC', 'ETH')
        sz: Size of the order (quantity to trade)
        is_buy: True for buy orders, False for sell orders
    """
    coin: str
    sz: float
    is_buy: bool


@dataclass
class TickReplay:
    """
    Records all information about a single trading tick for analysis and backtesting.

    This class captures the complete state of a trading decision, including
    the input data, model prediction, and resulting order. Useful for
    post-trade analysis, strategy evaluation, and debugging.

    Attributes:
        coin: The asset being traded
        sz: Size of the executed order
        is_buy: Whether the order was a buy (True) or sell (False)
        y_hat: The model's prediction (forecasted return)
        last_price: The market price when the decision was made
        lag: The calculated feature (e.g., log return) used for prediction
    """
    coin: str
    sz: float
    is_buy: bool
    y_hat: float
    last_price: float
    lag: float


class BasicTakerStrat(Tick):
    """
    A basic market-taking trading strategy using predictive models.

    This strategy operates as follows:
    - Uses streaming price data to calculate features (e.g., log returns)
    - Feeds features into a machine learning model to predict future returns
    - Generates buy/sell signals based on predictions
    - Executes market orders to enter positions
    - Closes previous positions before opening new ones

    The strategy is "taker" because it uses market orders that immediately
    execute at current market prices, paying the taker fee.

    Example:
        >>>
        >>> strategy = BasicTakerStrat(
        ...     exchange=exchange,
        ...     coin='BTC',
        ...     model=model,
        ...     sz=0.1,  # Trade 0.1 BTC
        ...     lag=lag_calculator,
        ...     leverage=2.0
        ... )
        >>>
        >>> # Feed streaming prices to the strategy
        >>> strategy.on_tick(50000.0)
        >>> strategy.on_tick(50100.0)
    """

    def __init__(
            self,
            exchange,
            coin: str,
            model,
            sz: float,
            lag,
            leverage: float = 1.0
    ):
        """
        Initialize the trading strategy.

        Args:
            exchange: Exchange client with market_open() and market_close() methods
            coin: The asset symbol to trade (e.g., 'BTC', 'ETH')
            model: Trained ML model with a predict() method
            sz: Position size for each trade
            lag: Feature calculator (e.g., LogReturn instance) with on_tick() method
            leverage: Leverage multiplier (default 1.0 = no leverage)
        """
        self.exchange = exchange
        self.coin = coin
        self.model = model
        self.lag = lag
        self.sz = sz
        self.leverage = leverage

    def predict(self, px) -> float:
        """
        Generate a prediction using the ML model.

        Args:
            px: Price or feature input for the model

        Returns:
            Model prediction (typically forecasted return)
        """
        return self.model.predict(px)

    def strategy(self, y_hat: float) -> Order:
        """
        Convert a model prediction into a trading order.

        This implements a simple directional strategy:
        - Positive prediction → Buy order
        - Negative prediction → Sell order

        Args:
            y_hat: Model prediction (forecasted return)

        Returns:
            Order object specifying the trade to execute
        """
        # Determine direction: positive prediction means buy, negative means sell
        is_buy = np.sign(y_hat) == 1

        return Order(self.coin, self.sz, is_buy)

    def execute(self, order: Order) -> None:
        """
        Execute a trade on the exchange.

        This method follows a two-step process:
        1. Close any existing position in the asset
        2. Open a new position according to the order

        This ensures we're always in a clean state and avoids accumulating
        unintended position sizes.

        Args:
            order: The order to execute

        Note:
            Errors are caught and logged rather than raised to prevent
            strategy crashes during live trading.
        """
        # Step 1: Close any existing position
        try:
            r = self.exchange.market_close(self.coin)
            print(f"Position closed: {r}")
        except Exception as e:
            print(f'Error closing position: {e}')

        # Step 2: Open new position with the specified direction and size
        try:
            r = self.exchange.market_open(
                self.coin,
                bool(order.is_buy),
                float(order.sz)
            )
            print(f'Order opened: {r}')
        except Exception as e:
            print(f'Error opening position: {e}')

    def on_tick(self, px: float) -> Optional[TickReplay]:
        """
        Process a new price tick and execute the full trading pipeline.

        This is the main entry point for streaming data. Each price tick triggers:
        1. Feature calculation (e.g., log return)
        2. Model prediction
        3. Order generation
        4. Trade execution
        5. Recording of the tick for analysis

        Args:
            px: The current market price

        Returns:
            TickReplay object containing all information about this trading decision,
            or None if feature calculation hasn't produced a value yet

        Example:
            >>> replay = strategy.on_tick(50123.45)
            >>> print(f"Predicted return: {replay.y_hat}")
            >>> print(f"Order direction: {'BUY' if replay.is_buy else 'SELL'}")
        """
        print(f'On tick: {px}')

        # Calculate feature (e.g., log return) from the new price
        # This might return None if we don't have enough data yet
        lag = self.lag.on_tick(px)
        print(f'Calculated log return: {lag}')

        # Generate prediction from the model
        # Note: This will use the lag value; if lag is None, model should handle it
        y_hat = self.model.predict(lag)
        print(f'Forecast future log return: {y_hat}')

        # Convert prediction into a trading order
        order = self.strategy(y_hat)
        print(f'Order: {order}')

        # Execute the order on the exchange
        self.execute(order)

        # Record all information about this tick for later analysis
        return TickReplay(
            coin=self.coin,
            sz=order.sz,
            is_buy=order.is_buy,
            y_hat=y_hat,
            last_price=px,
            lag=lag
        )