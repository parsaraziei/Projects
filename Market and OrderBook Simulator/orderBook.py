import numpy as np

class OrderBook:
    def __init__(self, startPrice=100.0, defaultSpread=0.0):
        self.startPrice = startPrice
        self.defaultSpread = defaultSpread
        self.bids = []
        self.asks = []
        self.price_history = []
        self.average_price = 0
        self.current_price = 0
        self.geometric_mean = 1
        self.traders = []
        self.mostrecentask = startPrice
        self.mostrecentbid = startPrice
        self.counter = 0
        self.pnl = 0

        # New attributes for candle data:
        self.candles = []           # List of completed candles
        self.current_candle = None  # Candle being formed

    def midPrice(self) -> float:
        #print(self.lowestAsk(), self.highestBid())
        # If no bids and no asks, return startPrice; if one is missing, use the other.
        return (self.lowestAsk() + self.highestBid()) / 2

    def highestBid(self):
        if len(self.bids) == 0:
            return self.mostrecentbid
        bestBidPrice, _, _, _ = max(self.bids)
        return bestBidPrice

    def lowestAsk(self):
        if len(self.asks) == 0:
            return self.mostrecentask
        bestAskPrice, _, _, _ = min(self.asks)
        return bestAskPrice

    def update_stats(self):
        new_price = self.midPrice()
        self.current_price = new_price

        # Append the new price to history.
        self.price_history.append(new_price)

        # Candle update logic:
        if self.current_candle is None:
            self.current_candle = {
                "open": new_price,
                "high": new_price,
                "low": new_price,
                "close": new_price
            }
        else:
            self.current_candle["high"] = max(self.current_candle["high"], new_price)
            self.current_candle["low"] = min(self.current_candle["low"], new_price)
            self.current_candle["close"] = new_price

        # Finalize the candle every N ticks (e.g., every 5 ticks).
        if len(self.price_history) % 5 == 0:
            self.candles.append(self.current_candle.copy())
            self.current_candle = None

        # Optionally update pnl using the previous price if available.
        if len(self.price_history) > 1:
            self.pnl = self.current_price - self.price_history[-2]

        self.update_average()
        self.update_geometric()
        self.remove_unfilled_trades()

    def limitOrder(self, trader, volume: float, limit: float, buyOrSell: str):
        trader.prevTradeVol = 0
        if buyOrSell == "buy":  # we hit ask
            while volume > 0 and len(self.asks) > 0:
                idx, (bestAskPrice, bestAskVolume, bestAskTrader, timestep) = min(
                    enumerate(self.asks), key=lambda x: x[1][0]
                )
                if bestAskPrice <= limit:  # lower ask price than limit is good
                    self.mostrecentask = bestAskPrice
                    if volume >= bestAskVolume:
                        volume -= bestAskVolume
                        tradeValue = bestAskPrice * bestAskVolume
                        bestAskTrader.balance += tradeValue
                        trader.balance -= tradeValue
                        trader.prevTradeVol += bestAskVolume
                        del self.asks[idx]
                    else:
                        tradeValue = bestAskPrice * volume
                        bestAskTrader.balance += tradeValue
                        trader.balance -= tradeValue
                        trader.prevTradeVol += volume
                        self.asks[idx] = (bestAskPrice, bestAskVolume - volume, bestAskTrader, timestep)
                        volume = 0
                else:
                    self.bids.append((limit, volume, trader, self.counter))
                    volume = 0
            if volume > 0:
                self.bids.append((limit, volume, trader, self.counter))
                volume = 0
        elif buyOrSell == "sell":  # we hit bid
            while volume > 0 and len(self.bids) > 0:
                idx, (bestBidPrice, bestBidVolume, bestBidTrader, timestep) = min(
                    enumerate(self.bids), key=lambda x: x[1][0]
                )
                if bestBidPrice >= limit:  # higher bid price than limit is good
                    self.mostrecentbid = bestBidPrice
                    if volume >= bestBidVolume:
                        volume -= bestBidVolume
                        tradeValue = bestBidPrice * bestBidVolume
                        bestBidTrader.balance += tradeValue
                        trader.balance -= tradeValue
                        trader.prevTradeVol -= bestBidVolume
                        del self.bids[idx]
                    else:
                        tradeValue = bestBidPrice * volume
                        bestBidTrader.balance += tradeValue
                        trader.balance -= tradeValue
                        trader.prevTradeVol -= volume
                        self.bids[idx] = (bestBidPrice, bestBidVolume - volume, bestBidTrader, timestep)
                        volume = 0
                else:
                    self.asks.append((limit, volume, trader, self.counter))
                    volume = 0
            if volume > 0:
                self.asks.append((limit, volume, trader, self.counter))
                volume = 0

    def marketOrder(self, trader, volume: float, buyOrSell: str):
        trader.prevTradeVol = 0
        if buyOrSell == "buy":
            while volume > 0 and len(self.asks) > 0:
                idx, (bestAskPrice, bestAskVolume, bestAskTrader, timestep) = min(
                    enumerate(self.asks), key=lambda x: x[1][0]
                )
                self.mostrecentask = bestAskPrice
                if volume >= bestAskVolume:
                    volume -= bestAskVolume
                    tradeValue = bestAskPrice * bestAskVolume
                    bestAskTrader.balance += tradeValue
                    trader.balance -= tradeValue
                    trader.prevTradeVol += bestAskVolume
                    del self.asks[idx]
                else:
                    tradeValue = bestAskPrice * volume
                    bestAskTrader.balance += tradeValue
                    trader.balance -= tradeValue
                    trader.prevTradeVol += volume
                    self.asks[idx] = (bestAskPrice, bestAskVolume - volume, bestAskTrader, timestep)
                    volume = 0
        elif buyOrSell == "sell":
            while volume > 0 and len(self.bids) > 0:
                idx, (bestBidPrice, bestBidVolume, bestBidTrader, timestep) = min(
                    enumerate(self.bids), key=lambda x: x[1][0]
                )
                self.mostrecentbid = bestBidPrice
                if volume >= bestBidVolume:
                    volume -= bestBidVolume
                    tradeValue = bestBidPrice * bestBidVolume
                    bestBidTrader.balance += tradeValue
                    trader.balance -= tradeValue
                    trader.prevTradeVol -= bestBidVolume
                    del self.bids[idx]
                else:
                    tradeValue = bestBidPrice * volume
                    bestBidTrader.balance += tradeValue
                    trader.balance -= tradeValue
                    trader.prevTradeVol -= volume
                    self.bids[idx] = (bestBidPrice, bestBidVolume - volume, bestBidTrader, timestep)
                    volume = 0

    def update_geometric(self):
        if len(self.price_history) == 0:
            self.geometric_mean = 1
        else:
            last_prices = self.price_history[-20:]
            self.geometric_mean = np.power(last_prices[-1] / last_prices[0], 1 / len(last_prices))

    def update_average(self):
        size = len(self.price_history)
        self.average_price = (self.average_price * (size - 1) + self.current_price) / size

    def remove_unfilled_trades(self):
        self.counter += 1
        self.asks = [x for x in self.asks if x[3] > self.counter - 3]
        self.bids = [x for x in self.bids if x[3] > self.counter - 3]
