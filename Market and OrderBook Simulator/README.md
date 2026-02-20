# Agent-Based Market Simulator (ABM)

A dynamic financial market simulation tool that models the interaction between various trading strategies and a centralized Limit Order Book (LOB). The project visualizes how micro-level agent behaviors (herd mentality, fundamental analysis, etc.) emerge as macro-level price patterns and volatility.

## 🧠 Trading Strategies
The simulation features several distinct agent types:
* **Fundamental Traders**: Trade based on the perceived "intrinsic value" of the asset.
* **Momentum Traders**: Follow price trends, buying when prices rise and selling when they fall.
* **Mean Reversion Traders**: Bet that the price will eventually return to its historical average.
* **Herd Traders**: A social agent type that mimics the strategies of successful nearby traders on the visual canvas.

## 🚀 Key Features
* **Dynamic Order Book**: A custom `OrderBook` implementation that handles bids, asks, and trade matching.
* **Visual Simulation**: A Tkinter-based GUI that shows agents moving in a 2D space, representing social proximity and influence.
* **Real-time Analytics**: 
    * Live price charting (Mid-price).
    * Candlestick (OHLC) chart generation.
    * Strategy distribution tracking (visualizing which strategies are currently dominating the market).
* **Configurable Parameters**: Adjust market volatility, agent sensitivity (Alpha), and the ratio of trader types via a sidebar control panel.

## 🛠 Tech Stack
* **Python 3.x**
* **Tkinter**: For the graphical user interface and agent animation.
* **Matplotlib / mplfinance**: For financial charting and live data visualization.
* **NumPy**: For stochastic modeling and agent decision logic.
* **Pandas**: For data structuring and candle generation.

## 📂 File Structure
* `main.py`: The control center. Manages the GUI lifecycle, animation loops, and real-time plotting.
* `traders.py`: Contains the logic for the `Trader` base class and all specific strategy subclasses.
* `orderBook.py`: Manages the bid/ask queues, price history, and trade execution logic.

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/market-simulator.git](https://github.com/your-username/market-simulator.git)
   cd market-simulator
