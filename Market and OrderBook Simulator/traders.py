from orderBook import OrderBook
from numpy import random
import numpy as np
from typing import Optional

try:
    from main import volatility_slider_global
except ImportError:
    volatility_slider_global = None


class Trader:
    fillColor: Optional[str] = None
    idcounter = 0
    
    def __init__(self, balance: float, riskAversion: float, orderBook: OrderBook, canvas, alpha, volatility=1):
        self.balance = balance
        self.prevBalance = balance
        self.prevTradeVol = 0 # Shows how much was traded in the previous timestep, nothing done with this yet
        self.riskAversion = riskAversion
        self.alpha = alpha  # Sensitivity parameter from slider
        self.volatility = volatility
        self.buyOrSell = None
        self.orderBook = orderBook # reference to the same orderBook object for each trader
        self.trader_type = "default"
        self.canvas = canvas
        self.canvas.update_idletasks()
        self.xcor = random.randint(0, self.canvas.winfo_width())
        self.ycor = random.randint(0, self.canvas.winfo_height())
        self.trader = self.canvas.create_oval(self.xcor, self.ycor, self.xcor+20, self.ycor+20, fill="blue")
        self.id = Trader.idcounter
        Trader.idcounter += 1
        
    def wasSuccessful(self):
        if self.orderBook.pnl == 0: return True
        return (self.orderBook.pnl > 0) == (self.buyOrSell == "buy")

    def adjustRiskAversion(self):
        """  profit_loss = (self.balance - self.prevBalance) / self.prevBalance

        if profit_loss < 0:
            adjustment = 1.0 + abs(profit_loss) * self.alpha * 2
        else:
            adjustment = np.exp(-self.alpha * profit_loss)

        self.riskAversion *= adjustment
        self.riskAversion = max(0.5, min(self.riskAversion, 5.0)) """


        """ profit_loss = self.balance - self.prevBalance
        #sigmoid func (bounded)
        delta = -self.alpha * profit_loss / (1+abs(profit_loss))
        self.riskAversion += delta
        self.riskAversion = max(0.5, min(self.riskAversion, 5.0))
        self.prevBalance = self.balance """

        #new sigmoid lower bounds
        profit_loss = self.balance - self.prevBalance
        self.prevBalance = self.balance

        # normalise p&l relative to current balance
        if self.prevBalance != 0:
            pct_change = profit_loss / self.prevBalance
        else:
            pct_change = 0

        delta = -self.alpha * pct_change

        self.riskAversion = max(0.5, min(self.riskAversion + delta, 2))

    def trade(self): # Default - should override
        self.buyOrSell = random.choice(["buy", "sell"])
        self.submit_trade(self.buyOrSell)
           
    def submit_trade(self, buyOrsell, orderType="limit"):
        # Get volatility factor
        volatility = volatility_slider_global.get() if volatility_slider_global else 1.0

        # Price influenced by volatility and risk aversion
        if buyOrsell == "buy":
            price = self.orderBook.midPrice() + random.uniform(-11, 0) * self.riskAversion * volatility
        else:
            price = self.orderBook.midPrice() + random.uniform(0, 11) * self.riskAversion * volatility

        volume = random.randint(10)

        if orderType == "limit":
            self.orderBook.limitOrder(self, volume, price, self.buyOrSell)
        elif orderType == "market":
            self.orderBook.marketOrder(self, volume, self.buyOrSell)

        self.adjustRiskAversion()


    def move(self):
        xborder = self.canvas.winfo_width()
        yborder = self.canvas.winfo_height()

        directions = random.randint(-30,30,size=2)
        self.xcor = (self.xcor + directions[0]) % xborder
        self.ycor = (self.ycor + directions[1]) % yborder
        
        if self.fillColor is not None: # actually referencing static variable
            self.canvas.itemconfig(self.trader, fill=self.fillColor)
        else:
            match self.buyOrSell:
                case "buy":
                    self.canvas.itemconfig(self.trader, fill="green")
                case "sell":
                    self.canvas.itemconfig(self.trader, fill="red")
                case _:
                    self.canvas.itemconfig(self.trader, fill="gray") # for error
        
        self.canvas.coords(self.trader, self.xcor-10, self.ycor-10, self.xcor + 10, self.ycor + 10)
        
class FundamentalTrader(Trader):
    def trade(self):
        self.trader_type = "Fundamental"
        self.fundamental_value = 100 + random.uniform(-2, 2)
        self.fillColor = "blue"

   
        current_price = self.orderBook.midPrice()

        # Only trade if price deviates significantly from perceived value
        threshold = 0.5  # can make this dynamic if needed

        if current_price < self.fundamental_value - threshold:
            self.buyOrSell = "buy"
        elif current_price > self.fundamental_value + threshold:
            self.buyOrSell = "sell"
        else:
            return  # No trade

        self.submit_trade(self.buyOrSell, orderType="market")  # or "market" if you want aggression


class MeanReversionTrader(Trader):
    #fillColor: Optional[str] = "yellow"

    def trade(self):
        if(self.orderBook.current_price <= self.orderBook.average_price):
            self.buyOrSell = "buy"
        else:
            self.buyOrSell = "sell"
        #print("average price :"+str(self.orderBook.average_price)+ "  current price :"+str(self.orderBook.current_price)+ " "+str(self.buyOrSell))
        self.submit_trade(self.buyOrSell)
        self.fillColor = "green"
        self.trader_type = "Mean Reversion"
       
class MomentumTrader(Trader):
    #fillColor: Optional[str] = "red"

    def trade(self):
        if(self.orderBook.geometric_mean>1):
            self.buyOrSell = "buy"   
        else:
            self.buyOrSell = "sell"
        #print("geometric_mean :"+str(self.orderBook.geometric_mean)+ " Therefore I "+str(self.buyOrSell))
        self.submit_trade(self.buyOrSell)
        self.fillColor = "red"
        self.trader_type = "Momentum"
    
        
class HerdTrader(Trader):
    #fillColor: Optional[str] = "orange"

    def __init__(self, balance: float, riskAversion: float, orderBook: OrderBook, canvas, alpha, ringRadius = 50):
        super().__init__(balance, riskAversion, orderBook, canvas, alpha)
        self.strategy = ""
        self.target_trader = None
        self.ringRadius = ringRadius
        self.ring = self.canvas.create_oval(self.xcor - self.ringRadius, self.ycor - self.ringRadius, self.xcor + self.ringRadius, self.ycor + self.ringRadius, outline="black", width=2)
        self.strategy_lifetime = 3
        self.fillColor = "black"
        self.trader_type = "Herd"
        
    def trade(self):
        successful_proximity_traders = []

        if len(self.orderBook.price_history) >= 2:
            if not self.wasSuccessful() and self.strategy_lifetime <= 0:
                self.strategy_lifetime = 3
                proximity_traders = [
                    trader for trader in self.orderBook.traders
                    if np.sqrt((self.xcor - trader.xcor) ** 2 + (self.ycor - trader.ycor) ** 2) < self.ringRadius
                    and trader.id != self.id
                ]
                successful_proximity_traders = [
                    trader for trader in proximity_traders if trader.wasSuccessful()
                ]

                if len(successful_proximity_traders) > 0:
                    idol = random.choice(successful_proximity_traders)
                    self.strategy = type(idol)

                    if self.strategy == HerdTrader:
                        self.strategy = idol.strategy

                    self.target_trader = idol
                    self.fillColor = getattr(idol, 'fillColor', "gray")  # inherit their color
                else:
                    self.strategy = random.choice([
                        subcls for subcls in Trader.__subclasses__() if subcls != HerdTrader
                    ])
                    self.target_trader = None
                    self.fillColor = "orange"
        else:
            self.strategy = random.choice([
                subcls for subcls in Trader.__subclasses__() if subcls != HerdTrader
            ])
            self.target_trader = None
            self.fillColor = "orange"

        self.strategy_lifetime -= 1

        # Use the chosen strategy’s trade function, passing self as the agent
        self.strategy.trade(self)


    def move(self):
        super().move()
        self.canvas.coords(self.ring, self.xcor - self.ringRadius, self.ycor - self.ringRadius, self.xcor + self.ringRadius, self.ycor + self.ringRadius)
