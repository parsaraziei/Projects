# High-Frequency Trading Strategy: Directional Changes & Backtesting

This repository contains a quantitative trading framework based on the **Directional Changes (DC)** framework. Unlike time-series analysis, this strategy uses event-based data to identify market extremes and trend reversals in high-frequency tick data (specifically for the AUS/USD forex pair).

## 📈 Project Overview

The project is split into two primary phases:

1.  **Exploratory Data Analysis (Part 1)**: A Jupyter Notebook dedicated to processing raw tick data, detecting local price extremes, and visualizing "Directional Changes" and "Intrinsic Events."
2.  **Backtesting Engine (Part 2)**: A Python-based simulation environment that executes long/short positions based on threshold-driven event triggers, managing risk via dynamic Stop Loss (SL) and Take Profit (TP) levels.

## ✨ Features

* **Event-Based Analysis**: Moves away from traditional candle charts to focus on price-action events (Directional Changes).
* **Dynamic Thresholds**: Identifies "Extremes" and "Directional Changes" based on a configurable percentage threshold.
* **Automated Backtesting**: 
    * Simulates trades on historical tick data.
    * Implements **Long** and **Short** position logic.
    * Calculates performance metrics: Win-Ratio, Net Profit, and Total Successful vs. Failed trades.
* **Data Visualization**: Generates clear plots of price movement overlayed with detected market extremes.
* **Detailed Reporting**: Exports position history and strategy overviews to Excel (`positions1.xlsx`) for deep-dive analysis.

## 🛠 Tech Stack

* **Python 3.x**
* **Pandas**: For high-speed tick data manipulation.
* **NumPy**: For numerical calculations.
* **Matplotlib**: For visualizing DC events and price paths.
* **Jupyter Notebook**: For research and documentation of the model logic.

## 📂 File Structure

* `PART 1 - Jupyter Notebook.ipynb`: Documentation of the environment setup, data loading, and the logic for detecting directional changes.
* `PART 2 - Backtest.py`: The execution script that runs the backtest across multiple months of data and outputs the financial results.

## ⚙️ How to Run

### 1. Prerequisites
Ensure you have the tick data stored in the expected folder structure (e.g., `exchange_data/AUS-USD`).

### 2. Installation
```bash
pip install pandas numpy matplotlib openpyxl
