from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Order:
    coin: str
    sz: float
    is_buy: bool

@dataclass
class TickReplay:
    coin: str
    sz: float
    is_buy: bool
    y_hat: float
    last_price: float
    lag: float

class Window:
    def __init__(self, size: int):
        self.data = deque(maxlen=size)

    def on_tick(self, elem):
        old_elem = None
        if self.is_full():
            old_elem = self.data.popleft()
        self.data.append(elem)
        return old_elem

    def is_full(self) -> bool:
        return len(self.data) == self.data.maxlen

class LogReturn:
    def __init__(self):
        self.prices = Window(2)

    def on_tick(self, px) -> Optional[float]:
        self.prices.on_tick(px)
        if self.prices.is_full():
            return np.log(self.prices.data[1] / self.prices.data[0])
        return None

class LinReg:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias


class BasicTakerStrat:
    def __init__(self, exchange, coin: str, model, sz: float, lag, leverage = 1.0):
        self.exchange = exchange
        self.coin = coin
        self.model = model
        self.lag = lag
        self.sz = sz
        self.leverage = leverage

    def predict(self, px):
        return self.model.predict(px)

    def strategy(self, y_hat):
        is_buy = np.sign(y_hat) == 1
        return Order(self.coin, self.sz, is_buy)

    def execute(self, order):
        try:
            r = self.exchange.market_close(self.coin)
            print(f"position closed: {r}")
        except Exception as e:
            print(f'error closing position: {e}')
        print('sending order')
        try:
            r = self.exchange.market_open(self.coin, bool(order.is_buy), float(order.sz))
            print(f'order opened = {r}')
        except Exception as e:
            print(f'error closing position: {e}')



    def on_tick(self, px) -> Optional[TickReplay]:
        print(f'on tick: {px}')
        lag = self.lag.on_tick(px)
        print(f'calculated log return: {lag}')
        y_hat = self.model.predict(lag)
        print(f'forecast future log return: {y_hat}')
        order = self.strategy(y_hat)
        print(f'order: {order}')
        self.execute(order)

        return TickReplay(self.coin, order.sz, order.is_buy, y_hat, px, lag)



