
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkcalendar import DateEntry
import numpy as np
import numpy.random as rand
import cmath
import math
import multiprocessing as mp
from itertools import repeat
from multiprocessing import Pool, cpu_count
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
from deribit_api import RestClient
import requests
from fredapi import Fred
import time as time
import datetime as datetime
import scipy.stats as si
from scipy.stats import norm
from scipy.optimize import brentq, minimize, differential_evolution
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from statsmodels.robust import mad
import math
from scipy.integrate import quad
from functools import partial,lru_cache
import statsmodels.api as sm
from concurrent.futures import ThreadPoolExecutor  # For parallel API calls



# get all the current option prices for ETH and Bitcoin
def get_all_options_prices(currency):
    # Get all active option instruments for the currency
    url_instruments = "https://www.deribit.com/api/v2/public/get_instruments"
    params = {"currency": currency, "kind": "option", "option_type":"call"}
    resp = requests.get(url_instruments, params=params).json()
    instruments = pd.DataFrame(resp['result'])
    results = []
    instruments = instruments[instruments["option_type"] == "call"]
    # Choose the top 7 soonest-expiring instruments
    instruments = instruments.sort_values(by="expiration_timestamp")
    unique_expiries = instruments["expiration_timestamp"].drop_duplicates().iloc[1:9]
    filtered_instruments = instruments[instruments["expiration_timestamp"].isin(unique_expiries)]
    
    # For each instrument, get current price data
    url_orderbook = "https://www.deribit.com/api/v2/public/get_order_book"
    for _, inst in filtered_instruments.iterrows():
        print(inst["instrument_name"])
        params = {"instrument_name": inst["instrument_name"]}
        r = requests.get(url_orderbook, params=params).json()
        data = r.get("result", {})
        if data and data.get("mark_price") is not None :
            results.append({
                "instrument_name": inst["instrument_name"],
                "last_price": data["last_price"],
                "bid_price": data.get("best_bid_price"),
                "ask_price": data.get("best_ask_price"),
                "mark_price": data.get("mark_price"),
                "expiration_timestamp": inst["expiration_timestamp"],
                "strike": inst["strike"],
                "option_type": inst["option_type"],
            })
    df = pd.DataFrame(results)[["mark_price","strike","expiration_timestamp","option_type"]]
    df["expiration_timestamp"] = pd.to_datetime(df["expiration_timestamp"], unit='ms')
    df_sorted = df.sort_values(by=["expiration_timestamp", "strike"], ascending=[True, True])
    print(df_sorted)
    return df_sorted

#def sol_xrp_option_prices(currency):
    

#fred API key db7c8153dabd9ce6f9f38256d61f6cb1 
def option_duration(start_time, end_time):
    x = str(pd.to_datetime(end_time)-pd.to_datetime(start_time))
    return int(x[:x.find('days')])+1

    
#treasury rates for r
def get_risk_free_treasury_rate(start_date,end_date):
    dur = option_duration(start_date,end_date)
    treasury_type = None
    
    if(dur <= 30.5): treasury_type = "DGS1MO" 
    elif(dur > 30.5 and dur <= 91.5): treasury_type = "DGS3MO"
    elif(dur > 91.5 and dur <= 182.5): treasury_type = "DGS6MO"
    elif(dur > 182.5 and dur <= 365): treasury_type = "DGS1"
    elif(dur > 365.5 and dur <= 730.5): treasury_type = "DGS2"
    elif(dur > 730.5 and dur <= 1825.5): treasury_type = "DGS5"
    else:  treasury_type = "DGS10"
    
    fred = Fred(api_key='db7c8153dabd9ce6f9f38256d61f6cb1')
    data = []
    s_timestamp = str(start_date)
    
    while(len(data) == 0):
        data = fred.get_series(treasury_type, observation_start = s_timestamp , observation_end = s_timestamp)
        s_timestamp = str(pd.to_datetime(s_timestamp) - pd.Timedelta(days=1))
        
    return data.iloc[0]/100
    
    
#lend rate for r in the alternative case
def get_usdc_lend_rate(date):
    
    _date = int(pd.Timestamp(str(date)).timestamp())
    url = "https://aave-api-v2.aave.com/data/rates-history"
    params = {
        "reserveId" : "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb480xB53C1a33016B2DC2fF3653530bfF1848a515c8c5",
        "from" : _date,
        "resolutionInHours" : 24
    }
    response = requests.get(url,params = params)

    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return None

    data = response.json()
    if not data:
        print(f"No USDC lending rate was returned for this date {date}")
        prev_date = pd.to_datetime(date) - pd.Timedelta(days=1)
        return get_usdc_lend_rate(prev_date)
    
    
    df = pd.DataFrame(data)
    print(f"Lending Rate was obtained for {date} at {df['liquidityRate_avg'].iloc[0]}")
    return df["liquidityRate_avg"].iloc[0]


def get_historical_prices(currency,start_time, end_time, duration):
    period = 0
    today = pd.Timestamp.now().normalize().date()
    if((pd.to_datetime(end_time)).date() > today):
        period = option_duration(start_time,today)
    else: 
        period = duration
    
    if(period > 0 ):
        binance = ccxt.binance()
        since = pd.to_datetime(start_time)
        since = binance.parse8601(str(since))
        y = binance.fetch_ohlcv(f"{currency}/USDT", timeframe='1d',since = since, limit = period)
        x = pd.DataFrame(y, columns = ["time", "open", "high", "low", "close", "volume"])
        x.index = pd.to_datetime(x["time"],unit = 'ms')
        return x["close"], period
    else :
        return None,0
    

#gets the implied volatility for that specific date however if there is no data on the implied volatility
#it would return 0 due to it being outdated or there not being no currency data.
def get_implied_vol(currency,start_time):
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    start_time = int(datetime.datetime.strptime(str(start_time), "%m/%d/%y").timestamp()*1000)
    params = {
        "currency":currency,
        "start_timestamp": start_time,
        "end_timestamp":start_time,
        "resolution": "1D"
    }
    response = requests.get(url,params)
    if response.status_code != 200:
        print("data was not found")
        return None
    df = pd.DataFrame(response.json()['result']['data'], columns = ["date","open","high","low","close"])
    df.index = pd.to_datetime(df["date"] ,unit='ms')
    df.drop(columns = ["date","open","high","low"],inplace = True)
    if(len(df)!=0):
        return (df.iloc[0,0])/100
    else:
        print("data was not found")
        return None
    
#to calculate the vol surface:
#blackscholes call option pricing.
# def get_historical_vol(currency,start_time):
#     binance = ccxt.binance()
#     since = pd.to_datetime(start_time)-pd.Timedelta(days=90)
#     limit = 90
#     since = binance.parse8601(str(since))
#     y = binance.fetch_ohlcv(f"{currency}/USDT", timeframe='1d',since=since, limit = limit)
#     x = pd.DataFrame(y, columns = ["time", "open", "high", "low", "close", "volume"])
#     x.index = pd.to_datetime(x["time"],unit = 'ms')
#     x.drop(["time","open","high","low","volume"],axis = 1,inplace = True)
#     z = (np.log(x)).diff().dropna().std()
#     return z.close*np.sqrt(365)

def get_realized_vol(currency,date_):
    if pd.to_datetime(date_).date() == pd.Timestamp.today().date():
        since = pd.Timestamp.now().replace(second=0, microsecond=0)-pd.Timedelta(hours=25)
    else:
        since = pd.to_datetime(date_)
    
    binance = ccxt.binance()
    since = binance.parse8601(str(since))
    y1 = binance.fetch_ohlcv(f'{currency}/USDT', timeframe='1m',since=since, limit = 1000)
    since = binance.parse8601(str(pd.to_datetime(y1[-1][0], unit='ms') + pd.Timedelta(minutes=1)))
    y2 = binance.fetch_ohlcv(f'{currency}/USDT', timeframe='1m',since=since, limit = 440)
    y_all = y1 + y2
    x = pd.DataFrame(y_all, columns = ["time", "open", "high", "low", "close", "volume"])
    #y = binance.
    #x = x[x.index %2==0]
    x.index = pd.to_datetime(x["time"],unit = 'ms')
    x.drop(["time","open","high","low","volume"],axis = 1,inplace = True)
    z = (np.sum(np.log(x).diff()**2,axis=0)["close"])*365
    return np.sqrt(z)



def get_historical_mean(currency,start_time):
    binance = ccxt.binance()
    since = pd.to_datetime(start_time)-pd.Timedelta(days=90)
    limit = 90
    since = binance.parse8601(str(since))
    y = binance.fetch_ohlcv(f"{currency}/USD", timeframe='1d',since=since, limit = limit)
    x = pd.DataFrame(y, columns = ["time", "open", "high", "low", "close", "volume"])
    x.index = pd.to_datetime(x["time"],unit = 'ms')
    x.drop(["time","open","high","low","volume"],axis = 1,inplace = True)
    z = (np.log(x)).diff().dropna()
    return np.mean(z.close)


#to calculate the vol surface:

#blackscholes put option pricing.
def BLS_put_option(S,K,sig,r,ta):
    print(S,K,sig,r,ta)
    d1 = ((np.log(S/K)+((r+0.5*(sig**2)))*ta))/(sig*np.sqrt(ta))
    d2 = ((np.log(S/K)+((r-0.5*(sig**2)))*ta))/(sig*np.sqrt(ta))
    put =  -S*(si.norm.cdf(-d1)) + K * np.exp(-r*ta) * (si.norm.cdf(-d2))
    return put


#blackscholes call option pricing.
def BLS_call_option(S,K,sig,r,ta):
    print(S,K,sig,r,ta)
    d1 = ((np.log(S/K)+((r+0.5*(sig**2)))*ta))/(sig*np.sqrt(ta))
    d2 = ((np.log(S/K)+((r-0.5*(sig**2)))*ta))/(sig*np.sqrt(ta))
    call = S*(si.norm.cdf(d1)) - K * np.exp(-r*ta) * (si.norm.cdf(d2))
    put =  -S*(si.norm.cdf(-d1)) + K * np.exp(-r*ta) * (si.norm.cdf(-d2))
    return call


def BLS_vega(S, K, sigma, r, T):
    """Vega of the Black-Scholes call/put option."""
    if sigma <= 0 or T <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

# def extract_IV_call(price, S, K, T, r=0.0, tol=1e-8, max_iter=100):
#     """
#     Implied volatility for a European call option using Brent's method.
#     Falls back to damped Newton-Raphson if Brent fails.
#     """
#     VOL_MIN, VOL_MAX = 1e-6, 5.0

#     # Quick arbitrage sanity checks
#     intrinsic = max(S - K * np.exp(-r * T), 0)
#     if price < intrinsic - tol:
#         return None
#     elif abs(price - intrinsic) < tol:
#         return 0.0
#     elif price > S + tol:
#         return None

#     def objective(sigma):
#         return BLS_call_option(S, K, sigma, r, T) - price

#     # --- Try Brent's method ---
#     try:
#         iv = brentq(objective, VOL_MIN, VOL_MAX, xtol=tol, maxiter=max_iter)
#         return iv
#     except ValueError:
#         pass  # Fall back to Newton-Raphson

#     # --- Newton-Raphson fallback ---
#     sig_guess = 0.2
#     for _ in range(max_iter):
#         diff = objective(sig_guess)
#         if abs(diff) < tol:
#             return np.clip(sig_guess, VOL_MIN, VOL_MAX)

#         vega = BLS_vega(S, K, sig_guess, r, T)
#         if vega < 1e-6:
#             break  # Avoid division by near-zero vega

#         # Damped Newton-Raphson step
#         sig_guess -= 0.5 * diff / vega
#         sig_guess = np.clip(sig_guess, VOL_MIN, VOL_MAX)

#     return None  # All methods failed
 
 
 
# get the jump diffusion components.
def get_jump_diff_comp(currency,start_time):
    binance = ccxt.binance()
    num = 730
    limit = 200
    data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    end_time = pd.to_datetime(start_time)

    while num > 0:
        # to see if I should take a value = 200 or smaller
        lim = min(limit, num)
        num -= lim
        
        since = end_time - pd.Timedelta(days=lim)
        since_ms = binance.parse8601(str(since))

        y = binance.fetch_ohlcv(f"{currency}/USDT", timeframe='1d', since=since_ms, limit=lim)
        
        x = pd.DataFrame(y, columns=["time", "open", "high", "low", "close", "volume"])
        x.index = pd.to_datetime(x["time"], unit='ms')
        x.drop(columns=["time"], inplace=True)
        
        if(len(data) == 0):
            data = x
        else: 
            data = pd.concat([data, x])
            
        # Update the new end_time for the next chunk
        end_time = since
    data.drop(columns=["open","high","low","volume"], inplace=True)
    data = data.dropna().sort_index()
    detect_jumps(data)
    return detect_jumps(data)


def detect_jumps(data):
    window = 60; alpha=0.99
    log_returns = np.log(data['close']).diff().dropna()
    rolling_vol = mad(log_returns, center=0)
    volatility_percentile = np.percentile(np.abs(log_returns), 75)  # 75th percentile of absolute returns
    min_jump = max(rolling_vol * 2.8, 0.008, 0.6 * volatility_percentile)
    
    bipower_var = (np.abs(log_returns) * np.abs(log_returns.shift(1))).rolling(window).mean()
    bipower_var = bipower_var.dropna()  # Add this
    
    critical_value = norm.ppf(alpha)
    threshold = critical_value * np.sqrt(bipower_var / window)
    
    # Use aligned returns
    jumps = log_returns[bipower_var.index][
        (np.abs(log_returns[bipower_var.index]) > threshold) &
        (np.abs(log_returns[bipower_var.index]) > min_jump) & 
        (np.abs(log_returns[bipower_var.index]) < 0.4)
    ]
    
    if len(jumps) < 3:
        return 0, 0, 0
        
    possion_rate = len(jumps) / (len(log_returns) / 365)
    std_jump = np.std(jumps)
    mean_jump = np.mean(jumps)
    print(f"the poisson is {possion_rate} the std is {std_jump} and the mean jump is {mean_jump}")
    return possion_rate, std_jump, mean_jump


#extract implied volatility from put
def extract_IV_put(price, S, K, ta, r=0, tol=1e-10, max_iter=200):
    # Objective function: difference between BS price and market price
    def objective(sig):
        return BLS_put_option(S,K,sig,r,ta) - price

    # Brent's method to find root in volatility range [1e-6, 5]
    try:
        implied_vol = brentq(objective, 1e-10, 20, xtol=tol, maxiter=max_iter)
        return implied_vol
    except ValueError:
        print("fail")
        return None


def jump_diffusion_call(S,K,sig,r,ta,lamb,mu_j,sig_j,N=40):
    k = np.exp(mu_j + 0.5 * sig_j**2) - 1
    sum_arr = 0
    for i in range(N+1):
        r_jump = r-lamb*k+(i*np.log(1+mu_j)/ta)
        sig_jump = np.sqrt(sig**2 + (i * sig_j**2) / ta)
        coef = ((np.exp(-lamb*ta)*((lamb*ta)**i))/(math.factorial(i)))
        sum_arr += coef * BLS_call_option(S,K,sig_jump,r_jump,ta)
    return sum_arr


def extract_jump_components(currency):
    options = get_all_options_prices
    options = get_all_options_prices(currency)
    implied_vols = []
    tensor = []
    vol = None
    binance = ccxt.binance()
    ticker = binance.fetch_ticker(f'{currency}/USDT')
    S = ticker['last']
    
    for ind in range(len(options)):
        price = options.loc[ind,"mark_price"]
        K = options.loc[ind,"strike"]
        expiry = options.loc[ind,"expiration_timestamp"].date()
        today = pd.Timestamp.today().date()
        ta = option_duration(today,expiry)/365
        #r =  get_risk_free_treasury_rate(today,expiry)
        r = 0 # since it is deribits data
        vol = extract_IV_call(price, S, K, ta, r)
        print(f"at the spot price {S} for strike : {K} and the time to expiry of {ta*365} the IV is {vol} has the price {price}")
        tensor.append(ta)
        implied_vols.append(vol)
    

def plot_surface_vol(currency):
    options = get_all_options_prices(currency)
    implied_vols = []
    tensor = []
    vol = None
    binance = ccxt.binance()
    ticker = binance.fetch_ticker(f'{currency}/USDT')
    S = ticker['last']
    
    for ind in range(len(options)):
        price = options.loc[ind,"mark_price"]
        K = options.loc[ind,"strike"]
        expiry = options.loc[ind,"expiration_timestamp"].date()
        today = pd.Timestamp.today().date()
        ta = option_duration(today,expiry)/365
        #r =  get_risk_free_treasury_rate(today,expiry)
        r = 0 # since it is deribits data
        vol = extract_IV_call(price, S, K, ta, r)
        print(f"at the spot price {S} for strike : {K} and the time to expiry of {ta*365} the IV is {vol} has the price {price}")
        tensor.append(ta)
        implied_vols.append(vol)
    
    # --- 1. Initial Cleaning ---
    options = options[~options.duplicated(subset=['expiration_timestamp', 'strike'], keep=False)]
    options["time-to-expiry (days)"] = np.array(tensor) * 365
    options["implied_vols"] = implied_vols
    options = options[options["implied_vols"] >= 0.001]
    options.dropna(subset=["implied_vols"], inplace=True)
    options.drop(columns=["mark_price", "option_type"], inplace=True)
    
    # --- 2. Rename Columns for Simplicity ---
    df = options.copy()
    df.columns = ["strike", "expiration", "time_to_expiry (days)", "implied_vols"]
    
    # --- 3. OPTIONAL: Transform to Moneyness ---
    spot_price = S  # Replace this with your actual spot price
    df["moneyness"] = df["strike"] / spot_price
    df["log-moneyness"] = np.log(df["moneyness"])
    X = df["moneyness"]
    
    
    # --- 4. Filter Outliers in Implied Vols ---
    df = df[(df["implied_vols"] >= 0.05) & (df["implied_vols"] <= 1.0)]
    X = df["strike"]
    
    Y = df["time_to_expiry (days)"]
    Z = df["implied_vols"]
    
    # --- 5. Create Meshgrid for Interpolation ---
    strike_grid, expiry_grid = np.meshgrid(
        np.unique(X),
        np.unique(Y)
    )
    
    # --- 6. Interpolation + Smoothing ---
    z_grid = griddata((X, Y), Z, (strike_grid, expiry_grid), method='linear')
    z_grid = np.nan_to_num(z_grid, nan=0.0)  # Fill gaps with 0s (or could use 'nearest')
    z_grid = gaussian_filter(z_grid, sigma=1.2)  # Smooth out sharp spikes
    
    # --- 7. Plotting the Surface ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(strike_grid, expiry_grid, z_grid, cmap='viridis', edgecolor='none')
    
    # Optional: Rotate the view for better readability
    ax.view_init(elev=25, azim=135)
    
    # Axis Labels
    ax.set_title(f"Smoothed Implied Volatility Surface of {currency}")
    ax.set_xlabel("strike")
    #ax.set_xlabel("log-moneyness")
    ax.set_ylabel("Time to Expiry (days)")
    ax.set_zlabel("Implied Volatility")
    
    # Format Z-axis nicely
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Color bar for scale
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()
    return df


def get_initial_value(currency,date):
    binance = ccxt.binance()
    strt = binance.parse8601(pd.to_datetime(date).strftime('%Y-%m-%dT%H:%M:%S.000Z'))
    y = binance.fetch_ohlcv(f'{currency}/USDT', timeframe='1d',since = strt, limit = 1)
    x = pd.DataFrame(y, columns = ["time", "open", "high", "low", "close", "volume"])
    x.index = pd.to_datetime(x["time"],unit = 'ms')
    x.drop(["time","open","high","low","volume"],axis = 1,inplace = True)
    return x.iloc[0,0]


def jump_diffusion_call(S, K, sig, r, ta, lamb, mu_j, sig_j):
    """Your original pricing function with critical fixes"""
    if mu_j <= -1 + 1e-6:
        return 1e10
    
    k = np.exp(mu_j + 0.5 * sig_j**2) - 1
    sum_arr = 0
    N = min(200, max(50, int(4 * lamb * ta)))  # Your original N calculation
    
    # Precompute log(1+mu_j) once
    log_mu = math.log(1 + mu_j) if mu_j > -1 else 0
    
    for i in range(N+1):
        # Stable Poisson probability calculation
        log_p = -lamb*ta + i*math.log(lamb*ta) - math.lgamma(i+1)
        
        r_jump = r - lamb*k + (i*log_mu)/ta
        sig_jump = math.sqrt(sig**2 + (i * sig_j**2) / ta)
        
        # Use YOUR original BLS_call_option (critical for correctness)
        price = BLS_call_option(S, K, sig_jump, r_jump, ta)
        
        if np.isnan(price) or np.isinf(price):
            return 1e10
        
        sum_arr += math.exp(log_p) * price
        
    return sum_arr

import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def calibrate_jump_fixed_sigma(S, K_list, T_list, r, option_prices, sigma, diagnostics=False):
    
    K_list = np.asarray(K_list)
    T_list = np.asarray(T_list)
    option_prices = np.asarray(option_prices)
    price_scale = np.mean(option_prices)

    def sse(params):
        l, mj, sj = params
        # Early rejection of clearly invalid regions
        if mj <= -0.99:
            return 1e20

        try:
            prices = [
                jump_diffusion_call(S, K, sigma, r, T, l, mj, sj)
                for K, T in zip(K_list, T_list)
            ]

            rel_error = (np.array(prices) - option_prices) / price_scale
            error = np.sum(rel_error**2)

            penalty = (
                1e4 * np.tanh(max(0, l - 100) / 10) +
                1e4 * np.tanh(max(0, sj - 2.5) / 10) +
                1e4 * np.tanh(max(0, -0.9 - mj) / 10)
            )
            return error + penalty
        except Exception:
            return 1e20

    bounds = [(1.0, 50.0), (-0.5, 0.5), (0.2, 2.0)]

    # ---------- Global stage: DE ----------
    t0_global = time.perf_counter()
    global_res = differential_evolution(
        sse,
        bounds=bounds,
        strategy='best1bin',
        maxiter=30,
        popsize=10,
        tol=1e-4,
        polish=False
    )
    t_global = time.perf_counter() - t0_global

    # ---------- Local stage: SLSQP ----------
    t0_local = time.perf_counter()
    local_res = minimize(
        sse,
        x0=global_res.x,
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 100, 'ftol': 1e-6}
    )
    t_local = time.perf_counter() - t0_local

    # Decide which solution to trust (fallback logic)
    use_local = local_res.success and (local_res.fun <= global_res.fun)
    final_x = local_res.x if use_local else global_res.x
    final_fun = local_res.fun if use_local else global_res.fun

    # ---------- Diagnostics printout ----------
    print("\n[JD calibration] fixed-sigma routine")
    print(f"  DE time       : {t_global:.3f} s   | obj = {global_res.fun:.4e}")
    print(f"  Local time    : {t_local:.3f} s   | obj = {local_res.fun:.4e} "
          f"(success={local_res.success})")
    print(f"  Used solution : {'LOCAL (refined)' if use_local else 'GLOBAL (DE only)'}")
    print(f"  λ  (lambda)   : {final_x[0]:.4f}")
    print(f"  μ_J (mean)    : {final_x[1]:+.4f}")
    print(f"  σ_J (std dev) : {final_x[2]:.4f}")
    print("-" * 60)

    result = {
        'lambda':  final_x[0],
        'mu_j':    final_x[1],
        'sigma_j': final_x[2],
        'success': bool(use_local or global_res.success)
    }

    if diagnostics:
        diag = {
            'de_time': t_global,
            'local_time': t_local,
            'de_obj': float(global_res.fun),
            'local_obj': float(local_res.fun),
            'de_params': global_res.x.tolist(),
            'local_params': local_res.x.tolist(),
            'used_local': bool(use_local)
        }
        return result, diag

    return result
    
def get_heston_params(currency, date_):
    end = pd.to_datetime(date_)
    num_days = 180
    start = end - pd.Timedelta(days=num_days)
    date_range = pd.date_range(start=start, end=end, freq='D')

    binance = ccxt.binance()
    since = binance.parse8601(str(start))
    y = binance.fetch_ohlcv(f'{currency}/USDT', timeframe='1d', since=since, limit=num_days+1)
    x = pd.DataFrame(y, columns=["time", "open", "high", "low", "close", "volume"])
    x.index = pd.to_datetime(x["time"], unit='ms')
    log_rets = np.log(x["close"]).diff().dropna()

    RVs = pd.DataFrame(index=date_range, columns=["RV"])
    
    def fetch_rv(date):
        return get_realized_vol(currency, date.strftime('%Y-%m-%d'))
    
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust workers based on API rate limits
        results = list(executor.map(fetch_rv, date_range))
    RVs["RV"] = results

    RVs = RVs.dropna().astype(float)
    #hurst 
    hist_realized_vol = ((RVs["RV"])/np.sqrt(num_days)).copy()
    hist_realized_vol
    mean_lst=[]
    for i in range(1, 21): 
        diffs = np.abs(hist_realized_vol.diff(i).dropna())
        mean_lst.append(np.mean(diffs))
        
    vals = pd.Series(mean_lst, index=range(1, 21))
    x=np.log(vals.index)
    y=np.log(vals.values)
    
    X = sm.add_constant(x)  # Add intercept
    model = sm.OLS(y, X).fit()
    hurst = model.params[1]
    print("Slope:", model.params[1])
        
    # theta: mean variance
    Vars = RVs["RV"] ** 2  # Compute once
    theta = np.mean(Vars)
    
    # vol_of_vol (vov): Std of daily variance changes
    vol_of_vol_value = Vars.diff().std(ddof=1)  

    # rho: Correlation between variance changes and log returns
    realized_variance_diff = Vars.diff().dropna()
    aligned_log_rets = log_rets.loc[realized_variance_diff.index]  # Align indices
    rho = realized_variance_diff.corr(aligned_log_rets)  # OPTIMIZATION: Direct pandas corr()

    # kappa: Mean reversion speed (via OLS)
    Y = Vars.diff().dropna()
    X = sm.add_constant(Vars.shift(1).dropna()) 
    model = sm.OLS(Y, X).fit()
    kappa = -model.params[1]  # Slope coefficient
    theta_estimated = model.params[0] / kappa  # Intercept-to-slope ratio

    return vol_of_vol_value, theta, rho, kappa, hurst


def prepare_vol_surface_data(currency, IR=0, file_path=None, initial_val=None):
    # Get and prepare data
    if file_path is None:
        options = get_all_options_prices(currency)
    else:
        options = pd.read_csv(file_path, 
                            dtype={"mark_price": float, "strike": float},
                            parse_dates=["expiration_timestamp"])
    
    # Get spot price
    if initial_val is None:
        binance = ccxt.binance()
        ticker = binance.fetch_ticker(f'{currency}/USDT')
        S = ticker['last']
    else:
        S = initial_val
    
    today = pd.Timestamp.today().date()
    
    # Calculate time to expiration and filter
    options["TTE"] = (options["expiration_timestamp"].dt.date - today).apply(lambda x: x.days / 365)
    options = options[
        (options["TTE"] > 0.01) & 
        (options["TTE"] < 0.6) & 
        (options["strike"] > 0.8 * S) & 
        (options["strike"] < 1.4 * S)
    ].copy()
    
    # Enhanced IV extraction with put-call parity for ITM options
    def extract_iv_with_smile(row):
        try:
            price = row['mark_price']
            K = row['strike']
            T = (row["expiration_timestamp"].date() - today).days / 365
            r = IR

            # ITM options → synthetic parity price
            if (row['option_type'] == 'call' and K < S) or (row['option_type'] == 'put' and K > S):

                # Call ITM -> Synthetic put
                if row['option_type'] == 'call' and (K / S > 0.85):
                    synthetic_put_price = price - S + K * np.exp(-r*T)
                    if synthetic_put_price > 0:
                        iv = extract_IV_call(synthetic_put_price, S, K, T, r)
                        print(f"[IV extracted] Call ITM (synthetic) | K={K}, T={T:.4f}, price={price:.4f} → IV={iv}")
                        return iv

                # Put ITM -> Synthetic call
                elif row['option_type'] == 'put' and (K / S < 1.4):
                    synthetic_call_price = price + S - K * np.exp(-r*T)
                    if synthetic_call_price > 0:
                        iv = extract_IV_call(synthetic_call_price, S, K, T, r)
                        print(f"[IV extracted] Put ITM (synthetic) | K={K}, T={T:.4f}, price={price:.4f} → IV={iv}")
                        return iv

            # Direct IV extraction (OTM options)
            if row['option_type'] == 'call':
                iv = extract_IV_call(price, S, K, T, r)
                print(f"[IV extracted] Call OTM | K={K}, T={T:.4f}, price={price:.4f} → IV={iv}")
                return iv
            else:
                iv = extract_IV_put(price, S, K, T, r)
                print(f"[IV extracted] Put OTM | K={K}, T={T:.4f}, price={price:.4f} → IV={iv}")
                return iv

        except Exception as e:
            print(f"[IV extraction failed] K={row['strike']}, T={T if 'T' in locals() else 'N/A'}, Error: {e}")
            return np.nan

    # Apply IV extraction across all rows

    options['implied_vol'] = options.apply(extract_iv_with_smile, axis=1)
    
    # Convert to days and calculate moneyness
    options["time_to_expiry"] = options["TTE"] * 365
    options['moneyness'] = options['strike'] / S
    options['log-moneyness'] = np.log(options['moneyness'])
    
    # Clean data
    options = options[options['implied_vol'].notna()]
    options = options[options['implied_vol'] > 0.001]
    
    # Symmetry Enforcement for Missing Data
    all_strikes = options['strike'].unique()
    
    for expiry in options['time_to_expiry'].unique():
        expiry_data = options[options['time_to_expiry'] == expiry]
        existing_strikes = expiry_data['strike'].unique()

        # Missing puts
        missing_puts = [K for K in existing_strikes 
                       if K < S and (2*S - K) not in existing_strikes 
                       and (2*S - K)/S <= 1.4]

        # Missing calls
        missing_calls = [K for K in existing_strikes 
                        if K > S and (2*S - K) not in existing_strikes 
                        and (2*S - K)/S >= 0.85]

        synthetic_rows = []

        for K in missing_puts:
            mirror_K = 2*S - K
            mirror_iv = expiry_data[expiry_data['strike'] == K]['implied_vol'].values[0]
            synthetic_rows.append({
                'strike': mirror_K,
                'implied_vol': mirror_iv,
                'time_to_expiry': expiry,
                'moneyness': mirror_K/S,
                'log-moneyness': np.log(mirror_K/S),
                'option_type': 'call'
            })
        
        for K in missing_calls:
            mirror_K = 2*S - K
            mirror_iv = expiry_data[expiry_data['strike'] == K]['implied_vol'].values[0]
            synthetic_rows.append({
                'strike': mirror_K,
                'implied_vol': mirror_iv,
                'time_to_expiry': expiry,
                'moneyness': mirror_K/S,
                'log-moneyness': np.log(mirror_K/S),
                'option_type': 'put'
            })
        
        if synthetic_rows:
            options = pd.concat([options, pd.DataFrame(synthetic_rows)], ignore_index=True)
    
    # Prepare final DataFrame
    df = options[['strike', 'expiration_timestamp', 'time_to_expiry', 'implied_vol', 'moneyness', 'log-moneyness']].copy()
    df.columns = ['strike', 'expiration', 'time_to_expiry (days)', 'implied_vol', 'moneyness', 'log-moneyness']
    
    return df, S


def characteristic_function(S0, ta, r, sigma, kappa, theta, phi, rho, v0):
    i = 1j
    a = kappa * theta
    b = kappa
    d = np.sqrt((rho * sigma * phi * i - b)**2 - sigma**2 * (2 * i * phi - phi**2) + 1e-16j)
    g = (b - rho * sigma * phi * i + d) / (b - rho * sigma * phi * i - d + 1e-16)
    exp_dt = np.exp(d * ta)
    C = r * phi * i * ta + (a / sigma**2) * ((b - rho * sigma * phi * i + d) * ta - 2 * np.log((1 - g * exp_dt) / (1 - g + 1e-16)))
    D = ((b - rho * sigma * phi * i + d) / sigma**2) * (1 - exp_dt) / (1 - g * exp_dt + 1e-16)
    return np.exp(C + D * v0 + i * phi * np.log(S0))

def integrand(phi, S0, K, r, ta, kappa, theta, sigma, rho, v0, j):
    i = 1j
    phi = np.array(phi, dtype=complex)
    numer = np.exp(-i * phi * np.log(K))
    denom = i * phi

    if j == 1:
        cf = characteristic_function(S0, ta, r, sigma, kappa, theta, phi - i, rho, v0)
        denom *= S0 * np.exp(r * ta)
    else:
        cf = characteristic_function(S0, ta, r, sigma, kappa, theta, phi, rho, v0)

    return (cf * numer / denom).real

def heston_call_option(S0, K, ta, r, kappa, theta, sigma, rho, v0, N=128):
    upper = 100
    phi_vals = np.linspace(1e-6, upper, N)
    dphi = phi_vals[1] - phi_vals[0]

    int1 = integrand(phi_vals, S0, K, r, ta, kappa, theta, sigma, rho, v0, 1)
    int2 = integrand(phi_vals, S0, K, r, ta, kappa, theta, sigma, rho, v0, 2)

    P1 = 0.5 + (1 / np.pi) * np.trapz(int1, dx=dphi)
    P2 = 0.5 + (1 / np.pi) * np.trapz(int2, dx=dphi)

    return max(0, S0 * P1 - K * math.exp(-r * ta) * P2)

def heston_calibration_imp(market_prices, strikes, expiries, S0, r, v0, diagnostics=False):
    market_prices = np.asarray(market_prices)
    strikes = np.asarray(strikes)
    expiries = np.asarray(expiries)

    price_scale = np.mean(market_prices) if np.mean(market_prices) > 0 else 1.0
    eval_counter = {"count": 0}

    def objective(params):
        kappa, theta, sigma, rho = params

        # Feller penalty
        if 2.0 * kappa * theta < sigma**2:
            return 1e9

        eval_counter["count"] += 1

        try:
            model_prices = [
                heston_call_option(S0, K, T, r, kappa, theta, sigma, rho, v0)
                for K, T in zip(strikes, expiries)
            ]
        except Exception:
            return 1e9

        diff = (np.array(model_prices) - market_prices) / price_scale
        return float(np.mean(diff**2))

    bounds = [
        (0.1, 10.0),
        (1e-4, 1.0),
        (0.01, 2.0),
        (-0.999, 0.999)
    ]

    print("\n--- HESTON CALIBRATION STARTED ---")
    print(f"Initial variance v0: {v0}")

    start = time.time()
    result_global = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=80,
        popsize=10,
        polish=False
    )
    mid = time.time()

    print("\n[GLOBAL SEARCH COMPLETED]")
    print(f"Global params: {result_global.x}")
    print(f"Global objective: {result_global.fun:.6f}")
    print(f"Global duration: {(mid - start):.3f}s")

    result_local = minimize(
        objective,
        x0=result_global.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-6}
    )
    end = time.time()

    print("\n[LOCAL REFINEMENT COMPLETED]")
    print(f"Local params: {result_local.x}")
    print(f"Local objective: {result_local.fun:.6f}")
    print(f"Local success: {result_local.success}")
    print(f"Local duration: {(end - mid):.3f}s")

    # choose best solution
    if result_local.success and result_local.fun <= result_global.fun:
        best_x = result_local.x
        best_fun = float(result_local.fun)
        used = "local"
        print("\n→ Local optimiser selected (better fit).")
    else:
        best_x = result_global.x
        best_fun = float(result_global.fun)
        used = "global"
        print("\n→ Global optimiser selected (local failed or worse).")

    kappa, theta, sigma, rho = best_x
    feller_ok = 2.0 * kappa * theta > sigma**2

    print("\n[FINAL RESULT]")
    print(f"kappa={kappa:.4f}, theta={theta:.4f}, sigma={sigma:.4f}, rho={rho:.4f}")
    print(f"Feller condition satisfied: {feller_ok}")
    print(f"Final objective: {best_fun:.6f}")
    print(f"Total calibration time: {(end - start):.3f}s")
    print(f"Objective evaluations: {eval_counter['count']}")
    print(f"Selected stage: {used}")
    print("--- HESTON CALIBRATION END ---\n")

    params_dict = {
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rho": rho,
        "v0": v0,
        "calibration_error": best_fun,
        "success": bool(result_global.success or result_local.success),
        "feller_condition": feller_ok,
        "duration": end - start,
        "selected_stage": used,
    }

    # Optional: return structured diagnostics
    if diagnostics:
        diag = {
            "global": {
                "x": result_global.x.tolist(),
                "fun": float(result_global.fun),
                "success": bool(result_global.success),
                "time": mid - start,
            },
            "local": {
                "x": result_local.x.tolist(),
                "fun": float(result_local.fun),
                "success": bool(result_local.success),
                "time": end - mid,
            },
            "total_time": end - start,
            "objective_evaluations": eval_counter["count"],
            "used_solution": used,
        }
        return params_dict, diag

    return params_dict


def extract_IV_call(market_price, S, K, T, r=0.0, tol=1e-8, max_iter=100):

    # Quick arbitrage checks
    intrinsic = max(S - K * np.exp(-r * T), 0)
    if market_price < intrinsic - tol:
        return np.nan
    if market_price <= intrinsic + tol:
        return 0.0
    if market_price > S + tol:
        return np.nan
    
    def call_obj(sigma):
        return BLS_call_option(S, K, sigma, r, T) - market_price
    
    try:
        # Brent's method (most reliable)
        iv = brentq(call_obj, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
        return iv
    except ValueError:
        # Newton-Raphson fallback
        sigma = 0.3  # Initial guess
        for _ in range(max_iter):
            price_diff = call_obj(sigma)
            if abs(price_diff) < tol:
                return sigma
                
            # Calculate vega numerically
            h = 0.001  # Small perturbation
            vega = (BLS_call_option(S, K, sigma + h, r, T) - 
                   BLS_call_option(S, K, sigma - h, r, T)) / (2 * h)
            
            if abs(vega) < 1e-10:  # Avoid division by zero
                break
                
            sigma -= price_diff / vega
            sigma = max(1e-6, min(sigma, 10.0))  # Keep within bounds
    
    return np.nan

def extract_IV_put(market_price, S, K, T, r=0.0, tol=1e-8, max_iter=100):

    # Quick arbitrage checks
    intrinsic = max(K * np.exp(-r * T) - S, 0)
    if market_price < intrinsic - tol:
        return np.nan
    if market_price <= intrinsic + tol:
        return 0.0
    if market_price > K * np.exp(-r * T) + tol:
        return np.nan
    
    def put_obj(sigma):
        return BLS_put_option(S, K, sigma, r, T) - market_price
    
    try:
        # Brent's method (most reliable)
        iv = brentq(put_obj, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
        return iv
    except ValueError:
        # Newton-Raphson fallback
        sigma = 0.3  # Initial guess
        for _ in range(max_iter):
            price_diff = put_obj(sigma)
            if abs(price_diff) < tol:
                return sigma
                
            # Calculate vega numerically
            h = 0.001  # Small perturbation
            vega = (BLS_put_option(S, K, sigma + h, r, T) - 
                   BLS_put_option(S, K, sigma - h, r, T)) / (2 * h)
            
            if abs(vega) < 1e-10:  # Avoid division by zero
                break
                
            sigma -= price_diff / vega
            sigma = max(1e-6, min(sigma, 10.0))  # Keep within bounds
    
    return np.nan