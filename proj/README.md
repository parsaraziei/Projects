# Crypto Option Pricing & Monte Carlo Simulation Suite

A professional-grade quantitative finance dashboard built in Python for simulating asset price paths and pricing European options for cryptocurrencies. This tool integrates advanced stochastic volatility models with live market data to provide accurate derivatives analysis.

## 🚀 Features

* **Multi-Model Engine**: Supports four distinct pricing frameworks:
    * **Black-Scholes (BLS)**: Standard log-normal simulation.
    * **Merton’s Jump Diffusion (JUD)**: Captures market "shocks" using Poisson processes.
    * **Heston Model (HES)**: Models mean-reverting stochastic volatility and the leverage effect.
    * **Rough Heston (RH)**: Implements Fractional Brownian Motion to model modern "rough" volatility clusters.
* **Live Data Integration**:
    * **Market Prices**: Real-time spot prices via **Binance API** (using `ccxt`).
    * **Implied Volatility**: Fetches current option chain data from **Deribit**.
    * **Macro Rates**: Automatically retrieves risk-free rates from **US Treasury Yields** (via `fredapi`) or DeFi lending rates.
* **Performance Optimized**:
    * Utilizes **Numba (JIT)** for high-speed numerical execution.
    * Employs **Antithetic Variates (AV)** for variance reduction in Monte Carlo simulations.
    * Multi-threaded API calls for seamless data fetching.
* **Interactive GUI**: A feature-rich Tkinter interface including:
    * Dynamic Matplotlib charting for price paths and distributions.
    * 3D Volatility Surface mapping.
    * Comprehensive Greek calculations (Delta, Gamma, Theta, Rho, Vega).

## 🛠 Tech Stack

* **Language**: Python 3.x
* **Math & Stats**: `NumPy`, `SciPy`, `pandas`, `fbm`, `statsmodels`.
* **APIs**: `ccxt` (Binance), `fredapi` (Federal Reserve), Deribit API.
* **Visualization**: `Matplotlib` (2D & 3D), `Pillow`.
* **UI**: `Tkinter` with `tkcalendar`.

## 📂 Project Structure

* `new.py`: The main entry point. Handles the GUI lifecycle, threading, and plotting.
* `models.py`: The core simulation engine containing the mathematical classes for all pricing models.
* `utils.py`: Utility layer for API interactions, historical data processing, and closed-form Greek calculations.

## ⚙️ Installation

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/crypto-pricing-suite.git](https://github.com/your-username/crypto-pricing-suite.git)
    cd crypto-pricing-suite
    ```

2.  **Install Dependencies**:
    ```bash
    pip install numpy pandas scipy matplotlib ccxt fredapi tkcalendar fbm numba pillow statsmodels
    ```

3.  **API Configuration**:
    * To enable live Treasury rates, ensure you have a [FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html).

## 📊 Usage

1.  Launch the application:
    ```bash
    python new.py
    ```
2.  **Select Asset**: Choose a ticker (e.g., BTC/USDT) and define the option expiry dates.
3.  **Configure Parameters**: Use the tabs to set model-specific inputs (e.g., Rho for Heston or Jump Mean for Merton).
4.  **Run Simulation**: Click **Submit** to generate price paths and calculate Call/Put premiums.

## 🌡 Environment Note
All hardware monitoring or computational temperature references are displayed in **Celsius** as per user preference.

## ⚖️ Disclaimer
This software is for educational and research purposes only. It does not constitute financial advice. Use at your own risk.
