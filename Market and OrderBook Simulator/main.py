from traders import Trader, FundamentalTrader, HerdTrader, MeanReversionTrader, MomentumTrader
from orderBook import OrderBook
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk

import pandas as pd
import mplfinance as mpf

from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

stop_updates = False  # Stop flag

# Global simulation control variables
simulation_after_id = None
simulation_running = False
stop_updates = False

mean_reversion_ratio = []
momuentum_ratio = []
fundamental_ratio = []


def plot_candle_chart():
    global candle_axes, candle_canvas_global

    # If no candle data, just display a message
    if len(orderBook.candles) == 0:
        candle_axes.clear()
        candle_axes.text(0.5, 0.5, "No Candle Data", ha='center', va='center')
        candle_canvas_global.draw()
        return

    # Build your DataFrame
    df = pd.DataFrame(orderBook.candles)
    df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='T')
    df.index.name = "Date"
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})

    # --- Clear the axes before plotting ---
    candle_axes.clear()

    # --- Plot onto the existing axis with volume disabled ---
    mpf.plot(df, type='candle', ax=candle_axes, volume=False)

    # Draw the updated canvas
    candle_canvas_global.draw()

def simulate_step_wrapper():
    global simulation_after_id
    if stop_updates:
        return
    simulate_step(0)  # Passing a dummy parameter.
    simulation_after_id = root.after(50, simulate_step_wrapper)

def pause_simulation():
    global simulation_after_id, simulation_running, stop_updates
    stop_updates = True  # Halt simulation updates in simulate_step and scheduled loops
    if simulation_after_id is not None:
        root.after_cancel(simulation_after_id)
        simulation_after_id = None
    simulation_running = False
    print("Simulation paused.")

def resume_simulation():
    global stop_updates, simulation_after_id, simulation_running, root
    print("resume_simulation called")
    if simulation_running:
        return  # Prevent multiple schedules.
    stop_updates = False
    update_order_book_window()
    update_agent_stats_window()
    simulation_after_id = root.after(5000, simulate_step_wrapper)
    simulation_running = True
    print("Simulation resumed.")

def reset_simulation():
    global simulation_after_id, simulation_running, stop_updates, orderBook
    if simulation_running and simulation_after_id is not None:
        root.after_cancel(simulation_after_id)
        simulation_after_id = None
    simulation_running = False
    stop_updates = False  # Ensure updates will run

    # Clear graphs
    ax1.clear()
    ratio_axes.clear()
    candle_axes.clear()
    fundamental_ratio.clear()
    mean_reversion_ratio.clear() #clear the herd list as well
    momuentum_ratio.clear()
    graph_canvas.draw()
    traders_canvas.delete("all")

    Trader.idcounter = 0  # Reset agent IDs



    # Reinitialize simulation state
    orderBook = OrderBook()  # New instance clears bids, asks, price history, etc.
    orderBook.traders = initialize_traders()  # Create traders based on current slider values

    update_order_book_window()
    update_agent_stats_window()
    
    simulation_after_id = root.after(5000, simulate_step_wrapper)
    simulation_running = True

def update_order_book_window():
    global order_book_tree
    if stop_updates:
        return

    for item in order_book_tree.get_children():
        order_book_tree.delete(item)

    bid_orders = {}
    ask_orders = {}

    for bid in orderBook.bids:
        price = round(bid[0], 2)
        volume = bid[1]
        bid_orders[price] = bid_orders.get(price, 0) + volume

    for ask in orderBook.asks:
        price = round(ask[0], 2)
        volume = ask[1]
        ask_orders[price] = ask_orders.get(price, 0) + volume

    sorted_bids = sorted(bid_orders.items(), key=lambda x: -x[0])
    sorted_asks = sorted(ask_orders.items(), key=lambda x: x[0])

    max_rows = max(len(sorted_bids), len(sorted_asks))
    for i in range(max_rows):
        bid_price, bid_size = sorted_bids[i] if i < len(sorted_bids) else ("", "")
        ask_price, ask_size = sorted_asks[i] if i < len(sorted_asks) else ("", "")
        order_book_tree.insert("", "end", values=(bid_price, bid_size, ask_price, ask_size))

    order_book_frame.update_idletasks()
    order_book_frame.update()
    root.after(1000, update_order_book_window)

def update_agent_stats_window():
    global agent_stats_tree
    if stop_updates:
        return

    current_scroll_position = agent_stats_tree.yview()
    for item in agent_stats_tree.get_children():
        agent_stats_tree.delete(item)

    sorted_traders = sorted(orderBook.traders, key=lambda t: t.id, reverse=True)

    for i, trader in enumerate(sorted_traders):
    # Format risk aversion and balance to 2 decimals
        balance = f"{trader.balance:.2f}"
        risk = f"{trader.riskAversion:.2f}"

    # Handle herd strategy display
        if isinstance(trader, HerdTrader):
            if hasattr(trader, "strategy") and trader.strategy:
                current_strategy = trader.strategy.__name__
            else:
                current_strategy = "Unknown"
            trader_type = f"Herd → {current_strategy}"
        else:
            trader_type = trader.trader_type  # e.g., "Fundamental"

        #agent_stats_tree.insert("", "end", values=(i + 1, balance, risk, trader_type))
        agent_stats_tree.insert("", "end", values=(trader.id + 1, balance, risk, trader_type))



    agent_stats_tree.yview_moveto(current_scroll_position[0])
    agent_stats_frame.update_idletasks()
    agent_stats_frame.update()
    root.after(1000, update_agent_stats_window)

def create_main_window():
    global order_book_tree, agent_stats_tree, fundamental_slider, momentum_slider
    global mean_reversion_slider, herd_slider, total_agents_label, alpha_slider_global
    global graph_axes, ratio_axes, candle_axes
    global price_canvas_global, ratio_canvas_global, candle_canvas_global, traders_canvas_global
    global order_book_frame, agent_stats_frame

    root = tk.Tk()
    root.title("Trading Simulation")
    root.geometry("1280x950")  # Fixed window size
    #root.geometry("1600x1000")

    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=0)
    root.columnconfigure(0, weight=1)

    # Main Content Frame
    main_frame = ttk.Frame(root)
    main_frame.grid(row=0, column=0, sticky="nsew")
    main_frame.rowconfigure(0, weight=0)   # Order book & agent stats
    main_frame.rowconfigure(1, weight=1)   # Notebook (graphs)
    main_frame.rowconfigure(2, weight=1)   # Traders canvas
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)

    # Control Frame for Buttons
    control_frame = ttk.Frame(root)
    control_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

    # Row 0 in Main Frame: Order Book and Agent Stats
    order_book_frame = ttk.Frame(main_frame)
    order_book_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    agent_stats_frame = ttk.Frame(main_frame)
    agent_stats_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

    order_book_tree = ttk.Treeview(order_book_frame, columns=("Bid Price", "Bid Volume", "Ask Price", "Ask Volume"), show="headings")
    for col in ("Bid Price", "Bid Volume", "Ask Price", "Ask Volume"):
        order_book_tree.heading(col, text=col)
        order_book_tree.column(col, width=100, anchor="center")
    order_book_tree.pack(expand=True, fill="both")

    agent_stats_tree = ttk.Treeview(agent_stats_frame, columns=("Agent", "Balance", "Risk Aversion", "Trader Type"), show="headings")
    for col in ("Agent", "Balance", "Risk Aversion", "Trader Type"):
        agent_stats_tree.heading(col, text=col)
        agent_stats_tree.column(col, width=100, anchor="center")
    agent_stats_tree.pack(expand=True, fill="both")

    # Row 1 in Main Frame: Notebook for Graphs (now with 3 tabs)
    notebook = ttk.Notebook(main_frame)
    notebook.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

    # Tab 1: Price Graph
    price_tab = ttk.Frame(notebook)
    notebook.add(price_tab, text="Price Graph")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    price_canvas = FigureCanvasTkAgg(fig1, master=price_tab)
    price_canvas.get_tk_widget().pack(expand=True, fill="both")

    # Tab 2: Herd Trader Ratios
    ratio_tab = ttk.Frame(notebook)
    notebook.add(ratio_tab, text="Herd Trader Ratios")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ratio_canvas = FigureCanvasTkAgg(fig2, master=ratio_tab)
    ratio_canvas.get_tk_widget().pack(expand=True, fill="both")

    # Tab 3: Candlestick Chart (new tab)
    candle_tab = ttk.Frame(notebook)
    notebook.add(candle_tab, text="Candlestick Chart")
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    candle_canvas = FigureCanvasTkAgg(fig3, master=candle_tab)
    candle_canvas.get_tk_widget().pack(expand=True, fill="both")

    # Save canvases and axes globally
    graph_axes = ax1
    ratio_axes = ax2
    candle_axes = ax3
    price_canvas_global = price_canvas
    ratio_canvas_global = ratio_canvas
    candle_canvas_global = candle_canvas

    # Row 2 in Main Frame: Traders Movement Canvas & Legend
    canvas_frame = ttk.Frame(main_frame)
    canvas_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    # Traders Canvas (row 0)
    traders_canvas = tk.Canvas(canvas_frame, width=500, height=500, bg="white", bd=2, relief="solid")
    traders_canvas.grid(row=0, column=0, sticky="nsew")
    traders_canvas_global = traders_canvas

    # Legend Frame (row 1, below canvas)
    legend_frame = ttk.Frame(canvas_frame)
    legend_frame.grid(row=1, column=0, pady=10)

    # Agent type color mapping
    agent_colors = {
        "Fundamental": "blue",
        "Momentum": "red",
        "Mean Reversion": "green",
        "Herd": "white"
    }

    # Horizontal legend: all items on the same row
    legend_row = ttk.Frame(legend_frame)
    legend_row.pack(anchor="center")

    for label, color in agent_colors.items():
        entry = ttk.Frame(legend_row)
        entry.pack(side="left", padx=10)

        dot = tk.Canvas(entry, width=20, height=20, highlightthickness=0, bg="white")

        # Inner agent dot
        dot.create_oval(6, 6, 14, 14, fill=color, outline="black")

        # Outer ring for Herd only
        if label == "Herd":
            dot.create_oval(2, 2, 18, 18, outline="black", width=2)

        dot.pack(side="left")

        lbl = ttk.Label(entry, text=label)
        lbl.pack(side="left", padx=5)


    # Optional: ensure the canvas_frame allocates space correctly
    canvas_frame.rowconfigure(0, weight=1)
    canvas_frame.rowconfigure(1, weight=0)
    canvas_frame.columnconfigure(0, weight=1)

    
    # Sliders Frame
    slider_frame = ttk.Frame(main_frame)
    slider_frame.grid(row=0, column=2, rowspan=3, padx=10, pady=10, sticky="n")

    # Sliders Frame
    slider_frame = ttk.Frame(main_frame)
    slider_frame.grid(row=0, column=2, rowspan=3, padx=10, pady=10, sticky="n")

    # --- Section 1: Agent Composition ---
    ttk.Label(slider_frame, text="Agent Composition", font=("Arial", 10, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(10, 5)
    )

    ttk.Label(slider_frame, text="Fundamental Traders").grid(row=1, column=0, padx=5, pady=2, sticky="w")
    fundamental_slider = tk.Scale(slider_frame, from_=0, to=50, orient=tk.HORIZONTAL, command=update_total_agents)
    fundamental_slider.set(15)
    fundamental_slider.grid(row=1, column=1, padx=5, pady=2)

    ttk.Label(slider_frame, text="Momentum Traders").grid(row=2, column=0, padx=5, pady=2, sticky="w")
    momentum_slider = tk.Scale(slider_frame, from_=0, to=50, orient=tk.HORIZONTAL, command=update_total_agents)
    momentum_slider.set(20)
    momentum_slider.grid(row=2, column=1, padx=5, pady=2)

    ttk.Label(slider_frame, text="Mean Reversion Traders").grid(row=3, column=0, padx=5, pady=2, sticky="w")
    mean_reversion_slider = tk.Scale(slider_frame, from_=0, to=50, orient=tk.HORIZONTAL, command=update_total_agents)
    mean_reversion_slider.set(10)
    mean_reversion_slider.grid(row=3, column=1, padx=5, pady=2)

    ttk.Label(slider_frame, text="Herd Traders").grid(row=4, column=0, padx=5, pady=2, sticky="w")
    herd_slider = tk.Scale(slider_frame, from_=0, to=50, orient=tk.HORIZONTAL, command=update_total_agents)
    herd_slider.set(20)
    herd_slider.grid(row=4, column=1, padx=5, pady=2)

    ttk.Label(slider_frame, text="Total Agents").grid(row=5, column=0, padx=5, pady=(8, 2), sticky="w")
    total_agents_label = ttk.Label(slider_frame, text="60")
    total_agents_label.grid(row=5, column=1, padx=5, pady=(8, 2))

    # --- Section 2: Simulation Parameters ---
    ttk.Label(slider_frame, text="Simulation Parameters", font=("Arial", 10, "bold")).grid(row=6, column=0, columnspan=2, pady=(10, 5))

    ttk.Label(slider_frame, text="Risk Aversion Sensitivity").grid(row=7, column=0, padx=5, pady=2, sticky="w")
    alpha_slider = tk.Scale(slider_frame, from_=0.0001, to=0.01, resolution=0.0001, orient=tk.HORIZONTAL)
    alpha_slider.set(0.003)
    alpha_slider.grid(row=7, column=1, padx=5, pady=2)
    alpha_slider_global = alpha_slider

    ttk.Label(slider_frame, text="Volatility").grid(row=8, column=0, padx=5, pady=2, sticky="w")
    volatility_slider = tk.Scale(slider_frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL)
    volatility_slider.set(1.3)
    volatility_slider.grid(row=8, column=1, padx=5, pady=2)
    volatility_slider_global = volatility_slider

    # Control Buttons in the Control Frame
    resume_button = ttk.Button(control_frame, text="Start", command=resume_simulation)
    resume_button.pack(side="right", padx=5)
    pause_button = ttk.Button(control_frame, text="Stop", command=pause_simulation)
    pause_button.pack(side="right", padx=5)
    reset_button = ttk.Button(control_frame, text="Reset", command=reset_simulation)
    reset_button.pack(side="right", padx=5)


    return root, fig1, ax1, ratio_axes, candle_axes, price_canvas, traders_canvas

def update_total_agents(event=None):
    total_agents = fundamental_slider.get() + momentum_slider.get() + mean_reversion_slider.get() + herd_slider.get()
    total_agents_label.config(text=str(total_agents))

def simulate_step(t):
    print("New simulation step")
    random.shuffle(orderBook.traders)
    for trader in orderBook.traders:
        trader.trade()
        trader.move()

    # --- Update Price Graph (Tab 1) ---
    ax1.clear()
    ax1.set_title("Real-Time Stock Price")
    ax1.set_xlabel("Ticks")
    ax1.set_ylabel("Price")
    orderBook.update_stats()
    ax1.plot(orderBook.price_history, color="blue")
    ax1.plot([orderBook.highestBid()] * len(orderBook.price_history), color="green")
    ax1.plot([orderBook.lowestAsk()] * len(orderBook.price_history), color="orange")
    price_canvas_global.draw()

    # --- Update Herd Trader Ratios (Tab 2) ---
    showratio()

    # --- Update Candlestick Chart (Tab 3) ---
    plot_candle_chart()

def showratio():
    traderTypes = [subcls.__name__ for subcls in Trader.__subclasses__() if subcls != HerdTrader]
    herdTraders = list(filter(lambda x: isinstance(x, HerdTrader), orderBook.traders))
    tot = len(herdTraders)
    #print("-------------------------------")
    #print("HerdTrader Ratios")
    ratios = []
    for subcls in Trader.__subclasses__():
        if subcls == HerdTrader:
            continue
        ratio = len(list(filter(lambda x: x.strategy == subcls, herdTraders))) / tot if tot != 0 else 0
        ratios.append(ratio)
        #print(f"Ratio of {subcls.__name__}: {ratio}")

    if ratios:
        fundamental_ratio.append(ratios[0])
        mean_reversion_ratio.append(ratios[1])
        momuentum_ratio.append(ratios[2])
    ratio_axes.clear()
    ratio_axes.set_title("HerdTrader Ratios")
    ratio_axes.set_xlabel("Trader Type")
    ratio_axes.set_ylabel("Ratio")
    ratio_axes.plot(fundamental_ratio, color="green", label="Fundamental")
    ratio_axes.plot(mean_reversion_ratio, color="red", label="Mean Reversion")
    ratio_axes.plot(momuentum_ratio, color="blue", label="Momentum")
    ratio_axes.legend(loc="upper left")
    ratio_canvas_global.draw()

# --- Rest of your functions (update_geometric, initialize_traders, on_closing) remain unchanged ---

def initialize_traders():
    print("Initializing traders...")  # Debug print
    num_fundamental_traders = fundamental_slider.get()
    num_momentum_traders = momentum_slider.get()
    num_mean_reversion_traders = mean_reversion_slider.get()
    num_herd_traders = herd_slider.get()

    initialBalances = [1000] * (num_fundamental_traders + num_momentum_traders + num_mean_reversion_traders + num_herd_traders)
    # riskAversions = random.random(len(initialBalances))
    alpha_value = alpha_slider_global.get()
    riskAversions = random.random(len(initialBalances))


    traders = []
    for i in range(num_fundamental_traders):
        traders.append(FundamentalTrader(initialBalances[i], riskAversions[i], orderBook, traders_canvas, alpha=alpha_value))
    for i in range(num_momentum_traders):
        traders.append(MomentumTrader(initialBalances[num_fundamental_traders + i], riskAversions[num_fundamental_traders + i], orderBook, traders_canvas, alpha=alpha_value))
    for i in range(num_mean_reversion_traders):
        traders.append(MeanReversionTrader(initialBalances[num_fundamental_traders + num_momentum_traders + i], riskAversions[num_fundamental_traders + num_momentum_traders + i], orderBook, traders_canvas, alpha=alpha_value))
    for i in range(num_herd_traders):
        traders.append(HerdTrader(initialBalances[num_fundamental_traders + num_momentum_traders + num_mean_reversion_traders + i], riskAversions[num_fundamental_traders + num_momentum_traders + num_mean_reversion_traders + i], orderBook, traders_canvas, alpha=alpha_value))
    return traders

def on_closing():
    global stop_updates
    stop_updates = True
    root.quit()
    root.destroy()

if __name__ == "__main__":
    print("Starting simulation...")  # Debug print
    orderBook = OrderBook()  # Start price defaults to 100
    root, fig1, ax1, ratio_axes, candle_axes, graph_canvas, traders_canvas = create_main_window()
    orderBook.traders = initialize_traders()
    root.after(10000, update_order_book_window)
    root.after(10000, update_agent_stats_window)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    print("Entering main loop...")  # Debug print
    root.mainloop()
