"""
Time Series Window Processing Module

This module provides a set of classes for processing streaming time series data
using sliding windows. It's designed for real-time calculations like log returns
and lagged values, commonly used in financial data analysis.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import numpy as np


class Tick(ABC):
    """
    Abstract base class for streaming data processors.

    All tick-based processors should inherit from this class and implement
    the on_tick method to handle incoming data points.
    """

    @abstractmethod
    def on_tick(self, x):
        """
        Process a single incoming data point.

        Args:
            x: The incoming data point (type depends on implementation)

        Returns:
            Implementation-specific return value
        """
        pass


class Window(Tick):
    """
    A fixed-size sliding window for streaming data.

    This class maintains a queue of the most recent data points up to a
    specified size. When new data arrives and the window is full, the oldest
    element is removed to make room for the new one.

    Example:
        >>> window = Window(size=3)
        >>> window.on_tick(10)  # Returns None (window not full yet)
        >>> window.on_tick(20)  # Returns None
        >>> window.on_tick(30)  # Returns None
        >>> window.on_tick(40)  # Returns 10 (oldest element evicted)
    """

    def __init__(self, size: int):
        """
        Initialize a sliding window with a fixed size.

        Args:
            size: Maximum number of elements to store in the window
        """
        # Using deque for O(1) append and popleft operations
        self.data = deque(maxlen=size)

    def on_tick(self, elem):
        """
        Add a new element to the window.

        When the window is full, the oldest element is removed before
        adding the new one.

        Args:
            elem: The new element to add to the window

        Returns:
            The evicted element if window was full, None otherwise
        """
        old_elem = None

        # If window is at capacity, capture the element about to be removed
        if self.is_full():
            old_elem = self.data.popleft()

        # Add the new element (deque maxlen handles overflow automatically)
        self.data.append(elem)

        return old_elem

    def is_full(self) -> bool:
        """
        Check if the window has reached its maximum capacity.

        Returns:
            True if window is full, False otherwise
        """
        return len(self.data) == self.data.maxlen


class LogReturn(Tick):
    """
    Calculate logarithmic returns from streaming price data.

    Log returns are calculated as ln(P_t / P_{t-1}), which represents
    the continuously compounded rate of return between consecutive prices.
    This is commonly used in financial analysis because log returns are
    time-additive and more suitable for statistical analysis.

    Example:
        >>> log_ret = LogReturn()
        >>> log_ret.on_tick(100.0)  # Returns None (need 2 prices)
        >>> log_ret.on_tick(105.0)  # Returns ~0.0488 (log return)
    """

    def __init__(self):
        """Initialize the log return calculator with a 2-element price window."""
        # We only need to store the last 2 prices to calculate returns
        self.prices = Window(2)

    def on_tick(self, px) -> Optional[float]:
        """
        Process a new price and calculate log return if possible.

        Args:
            px: The new price (must be positive)

        Returns:
            The log return if we have 2+ prices, None otherwise
        """
        # Add the new price to our window
        self.prices.on_tick(px)

        # We need at least 2 prices to calculate a return
        if self.prices.is_full():
            # Calculate log return: ln(current_price / previous_price)
            # prices.data[0] is the older price, prices.data[1] is the newer one
            return np.log(self.prices.data[1] / self.prices.data[0])

        return None


class Lags:
    """
    Store and retrieve lagged values from streaming data.

    This class maintains a window of historical values and provides convenient
    access to lagged observations, useful for time series analysis and creating
    features for prediction models.

    Example:
        >>> lags = Lags(no_lags=5)
        >>> for i in range(10):
        ...     lags.on_tick(i)
        >>> lags.lag(0)  # Most recent value: 9
        >>> lags.lag(1)  # One lag back: 8
        >>> lags.lags([0, 1, 2])  # Multiple lags: [9, 8, 7]
    """

    def __init__(self, no_lags: int):
        """
        Initialize the lag storage.

        Args:
            no_lags: Number of historical values to store
        """
        self.window = Window(no_lags)

    def on_tick(self, x):
        """
        Add a new observation to the lag window.

        Args:
            x: The new observation to store
        """
        self.window.on_tick(x)

    def lag(self, i: int):
        """
        Retrieve a specific lagged value.

        Args:
            i: The lag index (0 = most recent, 1 = one step back, etc.)

        Returns:
            The value at the specified lag

        Note:
            Indexing starts from the oldest value in the window.
            For chronological access (newest first), use negative indices
            or access from the end of the window.
        """
        return self.window.data[i]

    def lags(self, ixs: list) -> list:
        """
        Retrieve multiple lagged values at once.

        Args:
            ixs: List of lag indices to retrieve

        Returns:
            List of values at the specified lags

        Example:
            >>> lags.lags([0, 2, 4])  # Get lags 0, 2, and 4
        """
        return [self.window.data[i] for i in ixs]