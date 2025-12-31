#Hyperliquid exchange imports
from typing import Any, List, Dict

import eth_account
import requests
from datetime import datetime, timedelta, timezone

from eth_account.signers.local import LocalAccount
from hyperliquid.utils import constants
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
import pandas as pd

# The URL for streaming data from the hyperliquid exchange
URL = "wss://api.hyperliquid.xyz/ws"

# The accepted time intervals for the hyperliquid exchange
TIME_INTERVALS = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d", "1w", "1M")

# Maps the accepted time intervals to their respective time delta
_interval_map = {
    '1m': lambda: timedelta(minutes = 1),
    '3m': lambda: timedelta(minutes = 3),
    '5m': lambda: timedelta(minutes = 5),
    '15m': lambda: timedelta(minutes = 15),
    '30m': lambda: timedelta(minutes = 30),
    '1h': lambda: timedelta(hours = 1),
    '2h': lambda: timedelta(hours = 2),
    '4h': lambda: timedelta(hours = 4),
    '8h': lambda: timedelta(hours = 8),
    '12h': lambda: timedelta(hours = 12),
    '1d': lambda: timedelta(days = 1),
    '3d': lambda: timedelta(days = 3),
    '1w': lambda: timedelta(days = 7),
}


def dl_ohlc(sym: str, interval: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
    print(f'dl_ohlc({sym},{interval}, {start}, {end})')
    """
    Download OHLC (candlestick) data for a given symbol and time range from Hyperliquid.

    Args:
        sym (str): Trading symbol / coin name (e.g. "BTC", "ETH").
        interval (str): Candle interval (e.g. "1m", "5m", "1h").
        start (datetime): Start time for the data range (UTC).
        end (datetime): End time for the data range (UTC).

    Returns:
        List[Dict[str, Any]]: A list of candlestick data dictionaries as returned
        by the Hyperliquid API. Each dictionary typically includes open, high,
        low, close, volume, and timestamp fields.

    Raises:
        requests.RequestException: If the HTTP request fails.
        ValueError: If the response cannot be decoded as JSON.
    """
    end_time_ms = int(end.timestamp() * 1000)
    start_time_ms = int(start.timestamp() * 1000)

    resp = requests.post(
        "https://api.hyperliquid.xyz/info",
        headers={"Content-Type": "application/json"},
        json={
            "type": "candleSnapshot",
            "req": {
                "coin": sym,
                "interval": interval,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
            }
        }
    )

    return resp.json()



def dl_ohlc_df(sym, interval, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Download OHLC candlestick data and return it as a pandas DataFrame.

    This function wraps `dl_ohlc`, converting the raw JSON candle data into
    a pandas DataFrame with properly typed timestamp and price columns.

    Args:
        sym (str): Trading symbol / coin name (e.g. "BTC", "ETH").
        interval (str): Candle interval (e.g. "1m", "5m", "1h").
        start (datetime): Start time for the data range (UTC).
        end (datetime): End time for the data range (UTC).

    Returns:
        pd.DataFrame: A DataFrame containing OHLC candlestick data with:
            - 't' (datetime64[ns]): Candle open time
            - 'T' (datetime64[ns]): Candle close time
            - 'o' (float): Open price
            - 'h' (float): High price
            - 'l' (float): Low price
            - 'c' (float): Close price
            - Additional fields provided by the API (e.g. volume)

    Raises:
        ValueError: If price columns cannot be converted to float.
        KeyError: If expected columns are missing from the API response.
    """
    df = pd.DataFrame(dl_ohlc(sym, interval, start, end))
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df['T'] = pd.to_datetime(df['T'], unit='ms')
    for col in ['o','h','l','c']:
        df[col] = df[col].astype(float)
    return df


def dl_last_candles(sym: str, interval: str, no_lags: int = 5):
    """
    Download the most recent candlestick data for a given symbol.

    This function calculates a time window ending at the current UTC time
    and retrieves the last `no_lags` candles for the specified interval.

    Args:
        sym (str): Trading symbol / coin name (e.g. "BTC", "ETH").
        interval (str): Candle interval (e.g. "1m", "5m", "1h").
        no_lags (int, optional): Number of most recent candles to fetch.
            Defaults to 5.

    Returns:
        List[Dict[str, Any]]: A list of candlestick data dictionaries as
        returned by the Hyperliquid API.

    Raises:
        KeyError: If `interval` is not present in `_interval_map`.
    """
    now = datetime.now(timezone.utc)
    delta = _interval_map[interval]
    start = now - delta() * no_lags

    return dl_ohlc(sym, interval, start, now)



def init(
    secret_key: str,
    address: str,
    main_net: bool = False,
    skip_ws: bool = False,
    perp_dexs=None,
):
    """
    Initialize a connection to Hyperliquid and validate account state.

    This function:
    - Creates a local Ethereum account from the provided secret key
    - Selects the Hyperliquid testnet or mainnet API
    - Initializes the Info and Exchange clients
    - Fetches and validates the user's spot and perp account state
    - Ensures the account has equity before proceeding

    Args:
        secret_key (str): Private key used to sign transactions.
        address (str): On-chain account address. If empty or None,
            the address derived from `secret_key` is used.
        main_net (bool, optional): If True, connect to Hyperliquid mainnet.
            Defaults to False (testnet).
        skip_ws (bool, optional): If True, skip initializing WebSocket
            connections. Defaults to False.
        perp_dexs (optional): Optional list of perp DEX identifiers to
            enable. Defaults to None.

    Returns:
        Tuple[str, Info, Exchange]:
            - local_address (str): Resolved on-chain account address
            - info (Info): Initialized Hyperliquid Info client
            - exchange (Exchange): Initialized Hyperliquid Exchange client

    Raises:
        Exception: If the account has no spot balances and zero margin
            account value, indicating no usable equity.
    """
    account: LocalAccount = eth_account.Account.from_key(secret_key)
    local_address = address if address else account.address
    print("Running with account address:", local_address)

    base_url = constants.MAINNET_API_URL if main_net else constants.TESTNET_API_URL
    info = Info(base_url, skip_ws, perp_dexs=perp_dexs)
    user_state = info.user_state(local_address)
    spot_user_state = info.spot_user_state(local_address)
    margin_summary = user_state["marginSummary"]
    print(spot_user_state)
    print(margin_summary)
    if float(margin_summary["accountValue"]) == 0 and len(spot_user_state["balances"]) == 0:
        print("Not running because the provided account has no equity.")
        url = info.base_url.split(".", 1)[1]
        error_string = (
            f"No accountValue:\n"
            f"If you think this is a mistake, make sure that {local_address} "
            f"has a balance on {url}.\n"
            f"If address shown is your API wallet address, update the config "
            f"to specify the address of your account, not the address of the API wallet."
        )
        raise Exception(error_string)

    exchange = Exchange(
        account,
        base_url,
        account_address=local_address,
        perp_dexs=perp_dexs,
    )
    return local_address, info, exchange


