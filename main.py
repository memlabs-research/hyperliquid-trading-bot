import asyncio
import os
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional

import numpy as np

import hl
import websockets
from datetime import datetime, timedelta, timezone
import json

import strategy
from strategy import LogReturn

URL = "wss://api.hyperliquid.xyz/ws"

last_price = None


def interval_mins(interval: str) -> int:
    dur = interval[-1]
    if dur == "h":
        no_hours = int(interval[:-1])
        return no_hours * 60
    elif dur == "m":
        no_minutes = int(interval[:-1])
        return no_minutes
    elif dur == 'd':
        no_days = int(interval[:-1])
        return no_days * 60 * 24
    raise ValueError(f"Invalid interval: {interval}")


def save_prices_ts(sym: str, interval: str, prices: list):
    """
    Save a list of price dictionaries to a JSONL file.

    :param sym: Symbol name, e.g. 'BTCUSD'
    :param interval: Time interval, e.g. '1h'
    :param prices: List of dicts containing price data
    """
    file_path = f"{sym}-{interval}-prices.jsonl"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for price in prices:
                line = json.dumps(price)
                f.write(line + "\n")
        print(f"Saved {len(prices)} prices to {file_path}")
    except Exception as e:
        print(f"Error saving prices: {e}")


def load_prices_ts(sym: str, interval: str):
    """
    Load price data from a JSONL file for a given symbol and interval.

    :param sym: Symbol name, e.g., 'BTCUSD'
    :param interval: Time interval, e.g., '1h'
    :return: List of price dictionaries; empty list if file not found or error occurs
    """
    # Construct the file name based on symbol and interval
    file_path = f"{sym}-{interval}-prices.jsonl"

    # Initialize an empty list to hold the time series data
    ts = []

    try:
        # Open the JSONL file for reading with UTF-8 encoding
        with open(file_path, "r", encoding="utf-8") as f:
            # Read the file line by line
            for line in f:
                line = line.strip()  # Remove leading/trailing whitespace and newlines
                if not line:  # Skip empty lines
                    continue
                obj = json.loads(line)  # Parse the JSON object from the line
                ts.append(obj)  # Add the parsed object to the time series list
    except Exception as e:
        # If any error occurs (file missing, invalid JSON, etc.), print the error
        print(f"Error loading prices: {e}")
        return []  # Return an empty list to indicate failure

    # Return the loaded list of price data
    return ts


def dl_prices_ts(coin: str, interval: str) -> List[Tuple[int, str]]:
    """
    Load price time series for a given coin and interval.
    If no local data exists, download it using hl.dl_candles().

    :param coin: Cryptocurrency symbol, e.g., 'BTCUSD'
    :param interval: Time interval, e.g., '1h'
    :return: List of tuples: (timestamp as datetime, closing price as string)
    """
    # Try to load local price data from a JSONL file
    prices = load_prices_ts(coin, interval)

    # If no local price data is available
    if len(prices) == 0:
        print("No price data")

        # Download candle data using some external function hl.dl_candles()
        candles = hl.dl_last_candles(coin, interval)

        # Convert the downloaded candle data to a list of tuples:
        #   - timestamp converted from milliseconds to a datetime object
        #   - closing price as a string
        prices = [(candle['t'] / 1000, candle['c']) for candle in candles]

        save_prices_ts(coin, interval, prices)

    # Return the list of price tuples
    return prices


async def trade_periodically(interval, strategy):
    global last_price

    no_mins = interval_mins(interval)
    period_mins = max(1, no_mins)

    while True:
        now = datetime.now(timezone.utc)

        # 1. Calculate how many minutes past the last "round" interval we are
        # e.g., if it's 10:07 and interval is 5, mins_past = 2
        mins_past = now.minute % period_mins

        # 2. Calculate seconds until the NEXT interval
        # (Total interval seconds) - (seconds already passed in this interval)
        seconds_until_next = (period_mins - mins_past) * 60 - now.second - (now.microsecond / 1_000_000.0)

        # 3. Wait
        await asyncio.sleep(seconds_until_next + 0.001)  # Tiny buffer to ensure we cross the line

        # 4. Action
        execution_time = datetime.now(timezone.utc)

        print(f"--- [Sync Every {interval}] {execution_time.strftime('%H:%M:%S')} | Price: {last_price}  ---")

        if last_price:
            print('got last price')
            tick = strategy.on_tick(last_price)
            print(tick)
        #     if tick:
        #         print(f"--- [Sync Every {interval}] {execution_time.strftime('%H:%M:%S')} | Price: {last_price} | Lag: {tick.lag} | Forecast: {tick.y_hat} ---")

async def connect_and_listen(interval: str, strategy):
    global last_price

    # Start the background timer task
    timer_task = asyncio.create_task(trade_periodically(interval, strategy))

    try:
        async with websockets.connect(URL, ping_interval=20) as ws:
            print(f"Connected to {strategy.coin} stream")
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": strategy.coin}
            }))

            async for message in ws:
                data = json.loads(message)
                trade_data = data.get("data")

                if isinstance(trade_data, list):
                    last_trade = trade_data[-1]
                    last_price = float(last_trade['px'])
                    # We don't need to print here; let the timer handle the reporting

    finally:
        # If the websocket disconnects, stop the timer so we don't
        # have multiple timers running when the while loop restarts it.
        timer_task.cancel()


def create_model():
    weight = -0.0001
    bias = -0.0000001
    return strategy.LinReg(weight, bias)

def create_strategy(exchange, interval):
    coin = "BTC"
    model = create_model()
    trade_sz = 0.0002

    lag = LogReturn()
    prices = dl_prices_ts(coin, interval)
    print(f"prices = {prices}")
    for _, price in prices:
        lag.on_tick(float(price))

    return strategy.BasicTakerStrat(exchange, coin, model, trade_sz, lag)

async def main():
    backoff = 1
    interval = "1h"

    secret_key = os.environ["HL_SECRET"]
    wallet = os.environ["HL_WALLET"]

    address, info, exchange = hl.hl_init(secret_key, wallet)
    strat = create_strategy(exchange, interval)

    if interval not in hl.TIME_INTERVALS:
        raise Exception(f"Invalid time interval: {interval}")

    while True:
        try:
            await connect_and_listen(interval, strat)
            backoff = 1  # reset if clean exit

        except (websockets.ConnectionClosed, OSError) as e:
            print(f"Disconnected: {e}. Reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)  # exponential backoff

asyncio.run(main())
