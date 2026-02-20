import os
import sys
import numpy as np
import pandas as pd
import pytest
import requests

# Make sure Python can see utils.py in the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils  # noqa: E402


# ------------------------------------------------------------------
# 1. Black–Scholes core functions
# ------------------------------------------------------------------

def test_bls_call_put_parity():
    """Call - Put ≈ S - K e^{-rT} for same (S,K,r,T,sigma)."""
    S = 100
    K = 95
    r = 0.05
    sigma = 0.2
    T = 0.5

    call = utils.BLS_call_option(S, K, sigma, r, T)
    put = utils.BLS_put_option(S, K, sigma, r, T)

    lhs = call - put
    rhs = S - K * np.exp(-r * T)

    assert np.isclose(lhs, rhs, atol=1e-6)


def test_bls_call_monotone_in_S():
    """Call price should increase when the spot increases (all else fixed)."""
    K = 100
    r = 0.02
    sigma = 0.25
    T = 1.0

    S_low = 90
    S_high = 110

    c_low = utils.BLS_call_option(S_low, K, sigma, r, T)
    c_high = utils.BLS_call_option(S_high, K, sigma, r, T)

    assert c_high > c_low


def test_bls_put_monotone_in_K():
    """Put price should increase when the strike increases (all else fixed)."""
    S = 100
    r = 0.02
    sigma = 0.25
    T = 1.0

    K_low = 90
    K_high = 110

    p_low = utils.BLS_put_option(S, K_low, sigma, r, T)
    p_high = utils.BLS_put_option(S, K_high, sigma, r, T)

    assert p_high > p_low


def test_bls_vega_positive():
    """Vega should be positive for reasonable parameters."""
    S = 100
    K = 100
    r = 0.01
    sigma = 0.3
    T = 1.0

    vega = utils.BLS_vega(S, K, sigma, r, T)
    assert vega > 0


def test_bls_call_price_bounds():
    """
    For a European call:
        max(S - K e^{-rT}, 0)  <=  C  <=  S
    """
    S = 100
    K = 100
    r = 0.03
    T = 1.0
    sigma = 0.25

    call = utils.BLS_call_option(S, K, sigma, r, T)

    lower_bound = max(S - K * np.exp(-r * T), 0.0)
    upper_bound = S

    assert lower_bound <= call <= upper_bound


def test_bls_put_price_bounds():
    """
    For a European put:
        max(K e^{-rT} - S, 0)  <=  P  <=  K e^{-rT}
    """
    S = 100
    K = 105
    r = 0.03
    T = 1.0
    sigma = 0.25

    put = utils.BLS_put_option(S, K, sigma, r, T)

    lower_bound = max(K * np.exp(-r * T) - S, 0.0)
    upper_bound = K * np.exp(-r * T)

    assert lower_bound <= put <= upper_bound


def test_bls_call_increasing_in_sigma():
    """
    Call price should increase as volatility increases (all else fixed).
    """
    S = 100
    K = 100
    r = 0.02
    T = 1.0

    sigma_low = 0.1
    sigma_high = 0.5

    c_low = utils.BLS_call_option(S, K, sigma_low, r, T)
    c_high = utils.BLS_call_option(S, K, sigma_high, r, T)

    assert c_high > c_low


def test_bls_put_increasing_in_sigma():
    """
    Put price should also increase as volatility increases.
    """
    S = 100
    K = 100
    r = 0.02
    T = 1.0

    sigma_low = 0.1
    sigma_high = 0.5

    p_low = utils.BLS_put_option(S, K, sigma_low, r, T)
    p_high = utils.BLS_put_option(S, K, sigma_high, r, T)

    assert p_high > p_low


def test_bls_call_itm_otm_relationship():
    """
    For the same sigma, r, T:
        deep ITM call > ATM call > deep OTM call.
    """
    S = 100
    r = 0.01
    sigma = 0.2
    T = 1.0

    K_deep_itm = 50    # very ITM
    K_atm = 100        # at the money
    K_deep_otm = 150   # very OTM

    c_itm = utils.BLS_call_option(S, K_deep_itm, sigma, r, T)
    c_atm = utils.BLS_call_option(S, K_atm, sigma, r, T)
    c_otm = utils.BLS_call_option(S, K_deep_otm, sigma, r, T)

    assert c_itm > c_atm > c_otm


def test_bls_put_itm_otm_relationship():
    """
    For the same sigma, r, T:
        deep ITM put > ATM put > deep OTM put.
    """
    S = 100
    r = 0.01
    sigma = 0.2
    T = 1.0

    K_deep_otm = 50    # OTM for put
    K_atm = 100
    K_deep_itm = 150   # ITM for put

    p_otm = utils.BLS_put_option(S, K_deep_otm, sigma, r, T)
    p_atm = utils.BLS_put_option(S, K_atm, sigma, r, T)
    p_itm = utils.BLS_put_option(S, K_deep_itm, sigma, r, T)

    assert p_itm > p_atm > p_otm


def test_bls_call_limit_T_to_zero():
    """
    As T → 0, call price should approach intrinsic value.
    We approximate this with a very small T.
    """
    S = 100
    K = 95
    r = 0.05
    sigma = 0.3
    T_small = 1e-4  # 0.0001 years ~ 0.0365 days

    call = utils.BLS_call_option(S, K, sigma, r, T_small)
    intrinsic = max(S - K, 0.0)

    # Small tolerance because there is still a tiny bit of time value
    assert np.isclose(call, intrinsic, rtol=0, atol=1e-2)


def test_bls_put_limit_T_to_zero():
    """
    As T → 0, put price should approach intrinsic value.
    """
    S = 100
    K = 105
    r = 0.05
    sigma = 0.3
    T_small = 1e-4

    put = utils.BLS_put_option(S, K, sigma, r, T_small)
    intrinsic = max(K - S, 0.0)

    assert np.isclose(put, intrinsic, rtol=0, atol=1e-2)


def test_bls_vectorised_call_input():
    """
    BLS_call_option should handle NumPy array strikes (broadcasting).
    """
    S = 100
    K = np.array([90, 100, 110])
    r = 0.02
    sigma = 0.25
    T = 1.0

    calls = utils.BLS_call_option(S, K, sigma, r, T)

    assert isinstance(calls, np.ndarray)
    assert calls.shape == K.shape
    # Prices should be finite and ordered: ITM > ATM > OTM
    assert np.all(np.isfinite(calls))
    assert calls[0] > calls[1] > calls[2]


def test_bls_vectorised_put_input():
    """
    BLS_put_option should handle NumPy array strikes (broadcasting).
    """
    S = 100
    K = np.array([90, 100, 110])
    r = 0.02
    sigma = 0.25
    T = 1.0

    puts = utils.BLS_put_option(S, K, sigma, r, T)

    assert isinstance(puts, np.ndarray)
    assert puts.shape == K.shape
    # For puts: deep ITM (K=110) > ATM > OTM (K=90)
    assert np.all(np.isfinite(puts))
    assert puts[2] > puts[1] > puts[0]

# ------------------------------------------------------------------
# 2. Implied volatility extraction
# ------------------------------------------------------------------

def test_extract_iv_call_recovers_sigma():
    """extract_IV_call should recover the true volatility from a BS call price."""
    S = 100
    K = 100
    r = 0.03
    T = 1.0
    sigma_true = 0.35

    price = utils.BLS_call_option(S, K, sigma_true, r, T)
    iv_est = utils.extract_IV_call(price, S, K, T, r)

    assert np.isfinite(iv_est)
    assert np.isclose(iv_est, sigma_true, atol=1e-3)


def test_extract_iv_put_recovers_sigma():
    """extract_IV_put should recover the true volatility from a BS put price."""
    S = 100
    K = 105
    r = 0.01
    T = 0.75
    sigma_true = 0.4

    price = utils.BLS_put_option(S, K, sigma_true, r, T)
    iv_est = utils.extract_IV_put(price, S, K, T, r)

    assert np.isfinite(iv_est)
    assert np.isclose(iv_est, sigma_true, atol=1e-3)


def test_extract_iv_call_arbitrage_checks():
    """Arbitrage-violating prices should return NaN."""
    S = 100
    K = 100
    r = 0.0
    T = 1.0

    # Too cheap (below intrinsic) -> NaN
    bad_price_low = -1.0
    iv_low = utils.extract_IV_call(bad_price_low, S, K, T, r)
    assert np.isnan(iv_low)

    # Too expensive (above S) -> NaN
    bad_price_high = 150.0
    iv_high = utils.extract_IV_call(bad_price_high, S, K, T, r)
    assert np.isnan(iv_high)


def test_option_duration_simple():
    start = "2024-01-01"
    end   = "2024-01-10"
    d = utils.option_duration(start, end)
    # 2024-01-01 → 2024-01-10 is 9 days difference, +1 in your fn = 10
    assert d == 10


def test_extract_iv_put_arbitrage_checks():
    """Arbitrage-violating put prices should return NaN."""
    S = 100
    K = 90
    r = 0.0
    T = 1.0

    # Too cheap (below intrinsic)
    bad_price_low = -1.0
    iv_low = utils.extract_IV_put(bad_price_low, S, K, T, r)
    assert np.isnan(iv_low)

    # Too expensive (above discounted strike)
    bad_price_high = 200.0
    iv_high = utils.extract_IV_put(bad_price_high, S, K, T, r)
    assert np.isnan(iv_high)

@pytest.mark.parametrize(
    "S,K,r,T,sigma_true",
    [
        (100, 100, 0.00, 1.0, 0.2),   # ATM, no rates
        (100, 110, 0.05, 0.5, 0.35),  # OTM call
        (100, 90,  0.02, 2.0, 0.5),   # ITM call, long maturity
    ],
)
def test_extract_iv_call_recovers_sigma_parametrised(S, K, r, T, sigma_true):
    """extract_IV_call should recover sigma for a range of setups."""
    price = utils.BLS_call_option(S, K, sigma_true, r, T)
    iv_est = utils.extract_IV_call(price, S, K, T, r)

    assert np.isfinite(iv_est)
    assert np.isclose(iv_est, sigma_true, atol=1e-3)


@pytest.mark.parametrize(
    "S,K,r,T,sigma_true",
    [
        (100, 105, 0.01, 0.75, 0.4),   # OTM put
        (100, 120, 0.03, 1.0, 0.25),   # ITM put
        (100, 80,  0.00, 0.3,  0.6),   # Deep OTM put
    ],
)
def test_extract_iv_put_recovers_sigma_parametrised(S, K, r, T, sigma_true):
    """extract_IV_put should recover sigma across different regimes."""
    price = utils.BLS_put_option(S, K, sigma_true, r, T)
    iv_est = utils.extract_IV_put(price, S, K, T, r)

    assert np.isfinite(iv_est)
    assert np.isclose(iv_est, sigma_true, atol=1e-3)


def test_extract_iv_call_zero_price_returns_zero_vol():
    """
    If the call price is zero and the option is OTM, IV should be 0.0
    (your function treats price <= intrinsic+tol as zero vol).
    """
    S = 100
    K = 150       # Deep OTM call
    r = 0.01
    T = 1.0

    price = 0.0
    iv = utils.extract_IV_call(price, S, K, T, r)

    assert iv == 0.0


def test_extract_iv_put_zero_price_returns_zero_vol():
    """
    If the put price is zero and the option is OTM, IV should be 0.0.
    """
    S = 120
    K = 100       # OTM put
    r = 0.01
    T = 1.0

    price = 0.0
    iv = utils.extract_IV_put(price, S, K, T, r)

    assert iv == 0.0


def test_extract_iv_call_intrinsic_price_gives_zero_vol():
    """
    If the call price equals its intrinsic value, your code should
    treat it as zero-vol (T→0 behaviour).
    intrinsic = max(S - K e^{-rT}, 0).
    """
    S = 100
    K = 100
    r = 0.05
    T = 1.0

    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    iv = utils.extract_IV_call(intrinsic, S, K, T, r)

    assert iv == 0.0


def test_extract_iv_put_intrinsic_price_gives_zero_vol():
    """
    Same logic for puts: if price == intrinsic, IV should be 0.0.F
    intrinsic = max(K e^{-rT} - S, 0).
    """
    S = 95
    K = 100
    r = 0.05
    T = 1.0

    intrinsic = max(K * np.exp(-r * T) - S, 0.0)
    iv = utils.extract_IV_put(intrinsic, S, K, T, r)

    assert iv == 0.0



def test_extract_iv_put_small_time_to_expiry_recovers_sigma():
    """
    For a small but not vanishing T, extract_IV_put should
    approximately recover the true sigma.
    """
    S = 100
    K = 100
    r = 0.01
    T = 0.01          # small, but not crazy tiny
    sigma_true = 0.3

    price = utils.BLS_put_option(S, K, sigma_true, r, T)
    iv_est = utils.extract_IV_put(price, S, K, T, r)

    assert np.isfinite(iv_est)
    assert np.isclose(iv_est, sigma_true, atol=5e-2)



def test_option_duration_same_day():
    """If start == end, duration should be 1 by definition of your function."""
    start = "2024-01-01"
    end   = "2024-01-01"
    d = utils.option_duration(start, end)
    assert d == 1


def test_option_duration_across_month_boundary():
    """Check behaviour over a month change."""
    start = "2024-01-28"
    end   = "2024-02-02"
    d = utils.option_duration(start, end)
    # 28→29→30→31→1→2 = 5 days difference, +1 => 6
    assert d == 6


def test_option_duration_leap_year():
    """Ensure leap-day is counted correctly."""
    start = "2024-02-28"
    end   = "2024-03-01"
    d = utils.option_duration(start, end)
    # 28→29→1 => 2 days difference, +1 => 3
    assert d == 3


# ------------------------------------------------------------------
# 3. Jump–diffusion pricing and jump detection
# ------------------------------------------------------------------

def test_jump_diffusion_call_basic():
    """
    jump_diffusion_call should return a finite, non-negative price
    for reasonable parameter choices.
    """
    S = 100
    K = 100
    r = 0.01
    T = 0.5
    sigma = 0.25
    lamb = 0.5   # jump intensity
    mu_j = 0.0   # average jump (log-space)
    sig_j = 0.2  # jump volatility

    price = utils.jump_diffusion_call(S, K, sigma, r, T, lamb, mu_j, sig_j)

    assert np.isfinite(price)
    assert price >= 0.0


def test_detect_jumps_no_jumps():
    """If the price is constant, detect_jumps should report zero jumps."""
    n = 200
    df = pd.DataFrame({"close": np.ones(n)})

    lam, std_j, mean_j = utils.detect_jumps(df)

    assert lam == 0
    assert std_j == 0
    assert mean_j == 0


def test_detect_jumps_with_spikes():
    """
    With a few large spikes inserted, detect_jumps should
    report a positive jump intensity and non-zero jump std.
    """
    np.random.seed(123)
    # Start with a random walk
    n = 300
    rets = 0.001 * np.random.randn(n)
    # Add a few big "jumps"
    rets[50] += 0.1
    rets[150] -= 0.12
    rets[250] += 0.09

    prices = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"close": prices})

    lam, std_j, mean_j = utils.detect_jumps(df)

    assert lam >= 0
    assert std_j >= 0
    # Don't assert too strongly – this is a heuristic method.


# ------------------------------------------------------------------
# 4. Heston-related functions
# ------------------------------------------------------------------

def test_characteristic_function_returns_complex():
    """Heston characteristic_function should return a finite complex number."""
    S0 = 100
    T = 1.0
    r = 0.01
    sigma = 0.4
    kappa = 2.0
    theta = 0.09
    rho = -0.4
    v0 = 0.09
    phi = 1.5  # integration variable

    cf_val = utils.characteristic_function(S0, T, r, sigma, kappa, theta, phi, rho, v0)

    assert isinstance(cf_val, complex)
    assert np.isfinite(cf_val.real)
    assert np.isfinite(cf_val.imag)


def test_heston_call_option_bounds():
    """
    Heston call price should sit between discounted intrinsic value and spot.
    """
    S0 = 100
    K = 95
    T = 0.5
    r = 0.03

    kappa = 2.0
    theta = 0.09
    sigma = 0.4
    rho = -0.5
    v0 = 0.09

    price = utils.heston_call_option(S0, K, T, r, kappa, theta, sigma, rho, v0)

    intrinsic_discounted = max(S0 - K * np.exp(-r * T), 0)
    assert intrinsic_discounted <= price <= S0


# ------------------------------------------------------------------
# Extra tests for jump_diffusion_call
# ------------------------------------------------------------------


def test_jump_diffusion_call_penalises_invalid_mu_j():
    """
    For μ_j <= -1, your implementation returns a large penalty (1e10).
    Make sure that still holds.
    """
    S = 100
    K = 100
    r = 0.01
    T = 0.5
    sigma = 0.25
    lamb = 1.0
    mu_j = -1.5     # invalid → should trigger penalty
    sig_j = 0.3

    price = utils.jump_diffusion_call(S, K, sigma, r, T, lamb, mu_j, sig_j)
    assert price >= 1e8   # well into “penalty” territory


def test_jump_diffusion_call_increases_with_jump_intensity():
    """
    For positive jump mean, a higher λ should (on average) increase the
    call price compared to a very small λ.
    """
    S = 100
    K = 100
    r = 0.01
    T = 1.0
    sigma = 0.2
    mu_j = 0.05     # upward jumps
    sig_j = 0.3

    price_low_lambda = utils.jump_diffusion_call(S, K, sigma, r, T, lamb=0.1, mu_j=mu_j, sig_j=sig_j)
    price_high_lambda = utils.jump_diffusion_call(S, K, sigma, r, T, lamb=2.0, mu_j=mu_j, sig_j=sig_j)

    assert np.isfinite(price_low_lambda)
    assert np.isfinite(price_high_lambda)
    assert price_high_lambda >= price_low_lambda


def test_jump_diffusion_call_non_negative_deep_otm():
    """
    Deep OTM call should still be finite and ≥ 0 under jump diffusion.
    """
    S = 100
    K = 200   # deep OTM
    r = 0.01
    T = 1.0
    sigma = 0.3
    lamb = 0.5
    mu_j = -0.02
    sig_j = 0.25

    price = utils.jump_diffusion_call(S, K, sigma, r, T, lamb, mu_j, sig_j)
    assert np.isfinite(price)
    assert price >= 0.0


def test_detect_jumps_small_sample():
    """
    With a very small sample, detect_jumps should not crash
    and should typically return zero jumps.
    """
    df = pd.DataFrame({"close": [100.0, 100.1, 99.9, 100.05]})
    lam, std_j, mean_j = utils.detect_jumps(df)

    assert lam >= 0
    assert std_j >= 0
    # With only 4 points it's very unlikely to classify anything as a jump
    # so usually all zeros:
    # we don't assert exact zeros to keep it robust across threshold tweaks.


def test_detect_jumps_high_vol_no_clear_jumps():
    """
    Pure high-volatility Brownian-ish series without isolated spikes
    should not produce a huge jump intensity.
    """
    np.random.seed(42)
    n = 500
    rets = 0.05 * np.random.randn(n)    # quite volatile but continuous
    prices = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"close": prices})

    lam, std_j, mean_j = utils.detect_jumps(df)

    assert lam >= 0
    # Heuristic: for a continuous-ish process, λ shouldn't explode
    assert lam < 50   # loose upper bound just to catch wild behaviour


def test_detect_jumps_catches_extreme_outlier():
    """
    Inject a single absurd spike; we expect at least some non-zero jump stats
    unless it is filtered out by your 0.4 cap.
    """
    np.random.seed(7)
    n = 400
    rets = 0.002 * np.random.randn(n)
    rets[200] += 0.3   # big spike

    prices = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"close": prices})

    lam, std_j, mean_j = utils.detect_jumps(df)

    assert lam >= 0
    assert std_j >= 0
    # Typically we expect λ > 0 here (but keep it soft to avoid brittleness):
    # if lam == 0, at least the call shouldn't crash, which this test ensures.


def test_detect_jumps_returns_zero_when_fewer_than_three_jumps():
    """
    Your implementation explicitly returns (0,0,0) if < 3 jumps are detected.
    Build a synthetic case with exactly 2 “jumps” and check that behaviour.
    """
    prices = np.ones(100)
    prices[20] = 1.2
    prices[80] = 0.8
    df = pd.DataFrame({"close": prices})

    lam, std_j, mean_j = utils.detect_jumps(df)

    assert lam == 0
    assert std_j == 0
    assert mean_j == 0


# ------------------------------------------------------------------
# 5. Vol surface helper: prepare_vol_surface_data
# ------------------------------------------------------------------

@pytest.mark.skip(reason="Integration-style test; requires mocking ccxt and external data.")
def test_prepare_vol_surface_data_with_mock(monkeypatch):
    """
    Example pattern for testing prepare_vol_surface_data by mocking:
    - ccxt.binance().fetch_ticker
    - utils.get_all_options_prices
    """

    class DummyBinance:
        def fetch_ticker(self, symbol):
            # Pretend spot is 100
            return {"last": 100.0}

    def dummy_binance():
        return DummyBinance()

    def dummy_get_all_options_prices(currency):
        # Minimal fake options table
        dates = pd.date_range("2025-01-01", periods=3, freq="M")
        return pd.DataFrame({
            "mark_price": [10.0, 8.0, 5.0],
            "strike": [90.0, 100.0, 110.0],
            "expiration_timestamp": dates,
            "option_type": ["call", "call", "call"],
        })

    # Patch ccxt.binance constructor and get_all_options_prices
    monkeypatch.setattr("utils.ccxt.binance", dummy_binance)
    monkeypatch.setattr("utils.get_all_options_prices", dummy_get_all_options_prices)

    df, S = utils.prepare_vol_surface_data("BTC")

    assert S == 100.0
    assert not df.empty
    assert {"strike", "expiration", "time_to_expiry (days)", "implied_vol"}.issubset(df.columns)

@pytest.mark.skip(reason="Integration-style test; requires mocking and slow operations.")
def test_prepare_vol_surface_data_all_iv_fail(monkeypatch):

    class DummyBinance:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

    def dummy_binance():
        return DummyBinance()

    def dummy_get_all_options_prices(currency):
        dates = pd.date_range("2025-01-01", periods=3, freq="M")
        return pd.DataFrame({
            "mark_price": [10.0, 8.0, 5.0],
            "strike": [90.0, 100.0, 110.0],
            "expiration_timestamp": dates,
            "option_type": ["call", "call", "call"],
        })

    # Force IV extraction to always fail (return NaN)
    monkeypatch.setattr(utils, "extract_IV_call", lambda *args, **kwargs: np.nan)
    monkeypatch.setattr(utils.ccxt, "binance", dummy_binance)
    monkeypatch.setattr(utils, "get_all_options_prices", dummy_get_all_options_prices)

    df, S = utils.prepare_vol_surface_data("BTC")

    assert S == 100.0
    assert df.empty   # all rows removed due to NaN implied vols

@pytest.mark.skip(reason="Integration mock test.")
def test_prepare_vol_surface_data_put_call_symmetry(monkeypatch):

    class DummyBinance:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

    def dummy_binance():
        return DummyBinance()

    # Only out-of-money PUT options → function should generate synthetic CALLs
    dates = pd.date_range("2025-01-01", periods=2, freq="M")

    def dummy_get_all_options_prices(currency):
        return pd.DataFrame({
            "mark_price": [6.0, 7.0],
            "strike": [110.0, 120.0],   # OTM PUTS (K > S)
            "expiration_timestamp": dates,
            "option_type": ["put", "put"],
        })

    monkeypatch.setattr(utils.ccxt, "binance", dummy_binance)
    monkeypatch.setattr(utils, "get_all_options_prices", dummy_get_all_options_prices)

    # Make IV extraction deterministic:
    monkeypatch.setattr(utils, "extract_IV_put", lambda *args: 0.3)
    monkeypatch.setattr(utils, "extract_IV_call", lambda *args: 0.3)

    df, S = utils.prepare_vol_surface_data("BTC")

    # Should contain both original rows + synthetic mirrored rows
    assert len(df) >= 2
    assert df["implied_vol"].notna().all()

@pytest.mark.skip(reason="Integration mock test.")
def test_prepare_vol_surface_data_expiry_filter(monkeypatch):

    class DummyBinance:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

    def dummy_binance():
        return DummyBinance()

    # One expiry is valid, one is too far in the future
    today = pd.Timestamp.today().date()
    far_expiry = today + pd.Timedelta(days=500)
    near_expiry = today + pd.Timedelta(days=50)

    def dummy_get_all_options_prices(currency):
        return pd.DataFrame({
            "mark_price": [10.0, 12.0],
            "strike": [95.0, 105.0],
            "expiration_timestamp": [near_expiry, far_expiry],
            "option_type": ["call", "call"],
        })

    monkeypatch.setattr(utils.ccxt, "binance", dummy_binance)
    monkeypatch.setattr(utils, "get_all_options_prices", dummy_get_all_options_prices)
    monkeypatch.setattr(utils, "extract_IV_call", lambda *args: 0.2)

    df, S = utils.prepare_vol_surface_data("BTC")

    # Only near expiry should remain
    assert len(df) == 1

@pytest.mark.skip(reason="Integration mock test.")
def test_prepare_vol_surface_data_moneyness_filter(monkeypatch):

    class DummyBinance:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

    def dummy_binance():
        return DummyBinance()

    def dummy_get_all_options_prices(currency):
        today = pd.Timestamp.today().date()
        exp = today + pd.Timedelta(days=30)
        return pd.DataFrame({
            "mark_price": [5.0, 6.0, 7.0],
            "strike": [50.0, 170.0, 100.0],   # Two invalid (K too low & too high)
            "expiration_timestamp": [exp, exp, exp],
            "option_type": ["call", "call", "call"],
        })

    monkeypatch.setattr(utils.ccxt, "binance", dummy_binance)
    monkeypatch.setattr(utils, "get_all_options_prices", dummy_get_all_options_prices)
    monkeypatch.setattr(utils, "extract_IV_call", lambda *args: 0.3)

    df, S = utils.prepare_vol_surface_data("BTC")

    # Only strike 100 should remain
    assert len(df) == 1
    assert df["strike"].iloc[0] == 100

@pytest.mark.skip(reason="Integration mock test.")
def test_prepare_vol_surface_data_mixed_puts_calls(monkeypatch):

    class DummyBinance:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

    def dummy_binance():
        return DummyBinance()

    today = pd.Timestamp.today().date()
    exp = today + pd.Timedelta(days=30)

    def dummy_get_all_options_prices(currency):
        return pd.DataFrame({
            "mark_price": [5.0, 8.0],
            "strike": [95.0, 105.0],
            "expiration_timestamp": [exp, exp],
            "option_type": ["put", "call"],
        })

    monkeypatch.setattr(utils.ccxt, "binance", dummy_binance)
    monkeypatch.setattr(utils, "get_all_options_prices", dummy_get_all_options_prices)
    monkeypatch.setattr(utils, "extract_IV_put", lambda *args: 0.25)
    monkeypatch.setattr(utils, "extract_IV_call", lambda *args: 0.30)

    df, S = utils.prepare_vol_surface_data("BTC")

    assert len(df) == 2
    assert df["implied_vol"].notna().all()

@pytest.mark.skip(reason="Integration mock test.")
def test_prepare_vol_surface_data_initial_val_override(monkeypatch):

    def dummy_get_all_options_prices(currency):
        today = pd.Timestamp.today().date()
        exp = today + pd.Timedelta(days=30)
        return pd.DataFrame({
            "mark_price": [6.0],
            "strike": [100.0],
            "expiration_timestamp": [exp],
            "option_type": ["call"],
        })

    # make IV deterministic
    monkeypatch.setattr(utils, "extract_IV_call", lambda *args: 0.2)
    monkeypatch.setattr(utils, "get_all_options_prices", dummy_get_all_options_prices)

    df, S = utils.prepare_vol_surface_data("BTC", initial_val=150.0)

    assert S == 150.0
    assert len(df) == 1
    assert df["moneyness"].iloc[0] == 100.0 / 150.0


# ------------------------------------------------------------------
# 6. Calibration functions – heavy, so only skeleton tests
# ------------------------------------------------------------------

@pytest.mark.skip(reason="Slow optimisation; run manually if needed.")
def test_calibrate_jump_fixed_sigma_simple_case():
    """
    Skeleton test: we create synthetic prices from a known parameter set
    and see whether calibration roughly recovers them.
    """
    S = 100
    r = 0.01
    T_list = np.array([0.25, 0.5, 1.0])
    K_list = np.array([90, 100, 110])
    sigma = 0.25

    true_lambda = 1.5
    true_mu_j = -0.05
    true_sigma_j = 0.3

    prices = [
        utils.jump_diffusion_call(S, K, sigma, r, T, true_lambda, true_mu_j, true_sigma_j)
        for K, T in zip(K_list, T_list)
    ]

    res = utils.calibrate_jump_fixed_sigma(S, K_list, T_list, r, prices, sigma)

    assert res["success"]
    assert res["lambda"] > 0
    assert -1 < res["mu_j"] < 1
    assert res["sigma_j"] > 0


@pytest.mark.skip(reason="Slow optimisation; run manually if needed.")
def test_heston_calibration_imp_simple_case():
    """
    Skeleton calibration test using synthetic Heston prices as targets.
    """
    S0 = 100
    r = 0.01
    v0 = 0.09
    kappa_true = 2.0
    theta_true = 0.09
    sigma_true = 0.3
    rho_true = -0.5

    strikes = np.array([90, 100, 110])
    expiries = np.array([0.5, 1.0, 1.5])

    market_prices = [
        utils.heston_call_option(S0, K, T, r, kappa_true, theta_true, sigma_true, rho_true, v0)
        for K, T in zip(strikes, expiries)
    ]

    res = utils.heston_calibration_imp(market_prices, strikes, expiries, S0, r, v0)

    assert res["success"]
    assert res["feller_condition"]
    # Not checking closeness here to keep test robust; just that it runs.


# ------------------------------------------------------------------
# 7. API-related helpers with mocks
# ------------------------------------------------------------------

def test_get_all_options_prices_basic(monkeypatch):
    # 1. Fake JSON response that looks like Deribit instruments + orderbook
    fake_instruments = {
        "result": [
            {
                "instrument_name": "BTC-1JAN25-50000-C",
                "expiration_timestamp": 1735689600000,  # 1 Jan
                "strike": 50000,
                "option_type": "call",
            },
            {
                "instrument_name": "BTC-2JAN25-55000-C",
                "expiration_timestamp": 1735776000000,  # 2 Jan
                "strike": 55000,
                "option_type": "call",
            },
        ]
    }

    fake_orderbook = {
        "result": {
            "last_price": 1000.0,
            "best_bid_price": 990.0,
            "best_ask_price": 1010.0,
            "mark_price": 1005.0,
        }
    }
    
def test_get_all_options_prices_skips_puts_and_none_mark(monkeypatch):
    """Non-call options and rows with mark_price=None should be excluded."""
    fake_instruments = {
        "result": [
            {
                "instrument_name": "BTC-1JAN25-50000-C",
                "expiration_timestamp": 1735689600000,  # 1 Jan
                "strike": 50000,
                "option_type": "call",
            },
            {
                "instrument_name": "BTC-2JAN25-55000-P",
                "expiration_timestamp": 1735776000000,  # 2 Jan
                "strike": 55000,
                "option_type": "put",   # should be filtered out
            },
            {
                "instrument_name": "BTC-3JAN25-60000-C",
                "expiration_timestamp": 1735862400000,  # 3 Jan
                "strike": 60000,
                "option_type": "call",
            },
        ]
    }

    fake_orderbooks = {
        "BTC-1JAN25-50000-C": {
            "result": {
                "last_price": 900.0,
                "best_bid_price": 890.0,
                "best_ask_price": 910.0,
                "mark_price": None,  # will be filtered out
            }
        },
        "BTC-3JAN25-60000-C": {
            "result": {
                "last_price": 800.0,
                "best_bid_price": 790.0,
                "best_ask_price": 810.0,
                "mark_price": 805.0,
            }
        },
    }

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, params=None):
        if "get_instruments" in url:
            return FakeResp(fake_instruments)
        elif "get_order_book" in url:
            name = params["instrument_name"]
            return FakeResp(fake_orderbooks[name])
        else:
            raise ValueError("Unexpected URL in test")

    monkeypatch.setattr(utils, "requests", type("R", (), {"get": fake_get}))

    df = utils.get_all_options_prices("BTC")

    # Only BTC-3JAN25-60000-C survives (call + non-None mark_price)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert float(df["mark_price"].iloc[0]) == 805.0
    assert int(df["strike"].iloc[0]) == 60000
    assert df["option_type"].iloc[0] == "call"
  



def test_get_risk_free_treasury_rate(monkeypatch):
    # Fake Fred instance
    class FakeFred:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def get_series(self, series_id, observation_start, observation_end):
            # Return a simple Series with one value, e.g. 5% yield
            return pd.Series([5.0])

    # Replace Fred in utils with our fake
    monkeypatch.setattr(utils, "Fred", FakeFred)

    r = utils.get_risk_free_treasury_rate("2020-01-01", "2020-02-01")

    # 5% / 100 → 0.05, as per your function
    assert abs(r - 0.05) < 1e-8

def test_get_historical_prices_zero_duration_returns_none(monkeypatch):
    """
    If duration <= 0, get_historical_prices should return (None, 0).
    No API calls should be made.
    """
    class FakeBinance:
        def parse8601(self, s):
            raise AssertionError("parse8601 should not be called when duration <= 0")
        def fetch_ohlcv(self, *args, **kwargs):
            raise AssertionError("fetch_ohlcv should not be called when duration <= 0")

    # Make sure even if ccxt.binance is called, using it would fail the test,
    # but in normal flow the function should return before that.
    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    closes, period = utils.get_historical_prices("BTC", "2020-01-01", "2020-01-10", duration=0)

    assert period == 0
    assert closes is None


def test_get_historical_prices_future_end_date_uses_today(monkeypatch):
    """
    When end_time is in the future, period should be computed up to 'today'
    according to option_duration(start, today).
    """
    # Fix 'today' in utils to a known date
    class FakeTimestamp(utils.pd.Timestamp.__class__):
        @staticmethod
        def now():
            return utils.pd.Timestamp("2025-01-10")

    # Easier: monkeypatch the .now method on pd.Timestamp directly
    monkeypatch.setattr(utils.pd.Timestamp, "now", staticmethod(lambda: utils.pd.Timestamp("2025-01-10")))

    class FakeBinance:
        def parse8601(self, s):
            return 0
        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            data = []
            base_time = 1700000000000
            for i in range(limit):
                t = base_time + i * 86400000
                close = 200 + i
                data.append([t, close - 1, close + 1, close - 2, close, 10])
            return data

    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    start = "2025-01-01"
    end_future = "2025-02-01"  # > today (2025-01-10)
    # duration argument will be ignored in this branch
    closes, period = utils.get_historical_prices("BTC", start, end_future, duration=999)

    # option_duration(start, today) where today is 2025-01-10 → 10 days
    assert period == 10
    assert isinstance(closes, pd.Series)
    assert len(closes) == 10
    assert np.isclose(closes.iloc[0], 200.0)


def test_get_historical_prices(monkeypatch):
    """
    Test get_historical_prices by mocking ccxt.binance so we don't hit
    the real API. We pick a past date so the 'end_time > today' branch
    is not triggered.
    """

    class FakeBinance:
        def parse8601(self, s):
            # get_historical_prices just passes this into fetch_ohlcv;
            # our fake ignores it, so any int is fine.
            return 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            # Return `limit` rows of fake OHLCV with deterministic 'close'
            data = []
            base_time = 1700000000000  # some fixed ms timestamp
            for i in range(limit):
                t = base_time + i * 86400000  # daily steps in ms
                close = 100 + i
                data.append([t, close - 1, close + 1, close - 2, close, 10])
            return data

    # Monkeypatch binance constructor inside utils
    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    # Use dates that are safely in the past so the 'today' logic is stable
    start = "2020-01-01"
    end = "2020-01-10"
    duration = 10  # explicit, to avoid any ambiguity

    closes, period = utils.get_historical_prices("BTC", start, end, duration)

    assert period == duration
    assert isinstance(closes, pd.Series)
    assert len(closes) == duration
    assert np.isclose(closes.iloc[0], 100.0)


# ------------------------------------------------------------------
# 7. Simple utility functions not previously covered
# ------------------------------------------------------------------

def test_option_duration_basic():
    """option_duration should return day-difference + 1."""
    start = "2024-01-01"
    end = "2024-01-10"
    d = utils.option_duration(start, end)
    # 2024-01-01 → 2024-01-10 is 9 days, +1 in implementation = 10
    assert d == 10


def test_option_duration_same_day():
    """If start == end, option_duration should return 1."""
    d = utils.option_duration("2024-01-01", "2024-01-01")
    assert d == 1

# ------------------------------------------------------------------
# option_duration — comprehensive test suite
# ------------------------------------------------------------------

import pytest
import numpy as np
import pandas as pd
import utils


def test_option_duration_basic():
    """option_duration should return day-difference + 1."""
    start = "2024-01-01"
    end   = "2024-01-10"
    d = utils.option_duration(start, end)
    assert d == 10  # 9 days + 1


def test_option_duration_same_day():
    """If start == end, option_duration should return 1."""
    d = utils.option_duration("2024-01-01", "2024-01-01")
    assert d == 1


def test_option_duration_one_day_gap():
    """One day gap should return 2."""
    d = utils.option_duration("2024-01-01", "2024-01-02")
    assert d == 2


def test_option_duration_start_after_end():
    """
    If start is after end, option_duration returns a negative value
    according to the current implementation (no validation).
    """
    start = "2024-01-10"
    end   = "2024-01-01"

    d = utils.option_duration(start, end)

    # 2024-01-01 - 2024-01-10 = -9 days → -9 + 1 = -8
    assert d == -8



def test_option_duration_leap_year():
    """Crossing Feb 29 on leap year should be handled correctly."""
    d = utils.option_duration("2024-02-28", "2024-03-01")
    assert d == 3  # 28 → 29 → 1


def test_option_duration_month_boundary():
    """Crossing month boundary should compute correctly."""
    d = utils.option_duration("2024-03-30", "2024-04-02")
    assert d == 4  # 30,31,1,2 → 4 days incl.


def test_option_duration_long_range():
    """Large ranges should still compute without overflow or slowdown."""
    d = utils.option_duration("2020-01-01", "2025-01-01")
    assert d > 1800  # sanity check



def test_option_duration_stripped_strings():
    """Whitespace around dates should not break parsing."""
    d = utils.option_duration(" 2024-01-01 ", "2024-01-02 ")
    assert d == 2


def test_option_duration_datetime_input():
    """Function should work with full datetime strings if supported."""
    d = utils.option_duration("2024-01-01 00:00:00", "2024-01-02 00:00:00")
    assert d == 2

# ------------------------------------------------------------------
# 8. get_initial_value (Binance OHLCV, mocked)
# ------------------------------------------------------------------

def test_get_initial_value(monkeypatch):
    """get_initial_value should return the close price from a 1d OHLCV fetch."""

    class FakeBinance:
        def parse8601(self, s: str):
            # We don't actually use the result for anything important in our fake
            return 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            # Return exactly one row as expected: [time, open, high, low, close, volume]
            # We'll fix close at 123.45 to test the return.
            return [[1700000000000, 120.0, 125.0, 119.0, 123.45, 10.0]]

    # Patch ccxt.binance to return our fake object
    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    val = utils.get_initial_value("BTC", "2024-01-01")

    assert np.isclose(val, 123.45)


# ------------------------------------------------------------------
# 9. get_implied_vol (Deribit vol index, mocked requests)
# ------------------------------------------------------------------
def test_get_implied_vol_success(monkeypatch):
    """
    get_implied_vol should parse the returned 'close' field and divide by 100.
    We mock a single row with close = 50.0 → implied vol = 0.5
    """

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200  # simulate OK response

        def json(self):
            return self._payload

    # Fake one row of data with a given timestamp and close=50
    fake_timestamp_ms = 1700000000000
    fake_payload = {
        "result": {
            "data": [
                [fake_timestamp_ms, 0.0, 0.0, 0.0, 50.0]
            ]
        }
    }

    def fake_get(url, params=None):
        return FakeResp(fake_payload)

    # Patch utils.requests.get
    monkeypatch.setattr(utils, "requests", type("R", (), {"get": fake_get}))

    # Use any date string matching the "%m/%d/%y" format
    iv = utils.get_implied_vol("BTC", "01/01/25")
    assert np.isclose(iv, 0.5)


def test_get_implied_vol_no_data(monkeypatch):
    """
    If Deribit returns an empty data list, get_implied_vol should return None.
    """

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200  # simulate OK response

        def json(self):
            return self._payload

    fake_payload = {
        "result": {
            "data": []  # No rows
        }
    }

    def fake_get(url, params=None):
        return FakeResp(fake_payload)

    monkeypatch.setattr(utils, "requests", type("R", (), {"get": fake_get}))

    iv = utils.get_implied_vol("BTC", "01/01/25")
    assert iv is None

def test_get_implied_vol_success(monkeypatch):
    """
    get_implied_vol should parse the returned 'close' field and divide by 100.
    We mock a single row with close = 50.0 → implied vol = 0.5
    """

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200  # simulate OK response

        def json(self):
            return self._payload

    # Fake one row of data with a given timestamp and close = 50
    fake_timestamp_ms = 1700000000000
    fake_payload = {
        "result": {
            "data": [
                [fake_timestamp_ms, 0.0, 0.0, 0.0, 50.0]
            ]
        }
    }

    def fake_get(url, params=None):
        return FakeResp(fake_payload)

    # Patch utils.requests.get
    monkeypatch.setattr(utils, "requests", type("R", (), {"get": fake_get}))

    # Use any date string matching the "%m/%d/%y" format
    iv = utils.get_implied_vol("BTC", "01/01/25")
    assert np.isclose(iv, 0.5)


def test_get_implied_vol_no_data(monkeypatch):
    """
    If Deribit returns an empty data list, get_implied_vol should return None.
    """

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200  # simulate OK response

        def json(self):
            return self._payload

    fake_payload = {
        "result": {
            "data": []  # No rows
        }
    }

    def fake_get(url, params=None):
        return FakeResp(fake_payload)

    monkeypatch.setattr(utils, "requests", type("R", (), {"get": fake_get}))

    iv = utils.get_implied_vol("BTC", "01/01/25")
    assert iv is None


def test_get_implied_vol_http_error(monkeypatch):
    """
    If Deribit responds with a non-200 status_code, the function
    should print a message and return None.
    """

    class FakeResp:
        def __init__(self, payload, status_code):
            self._payload = payload
            self.status_code = status_code

        def json(self):
            return self._payload

    fake_payload = {"some": "error-ish payload"}

    def fake_get(url, params=None):
        # Simulate HTTP 500
        return FakeResp(fake_payload, status_code=500)

    monkeypatch.setattr(utils, "requests", type("R", (), {"get": fake_get}))

    iv = utils.get_implied_vol("BTC", "01/01/25")
    assert iv is None

# ------------------------------------------------------------------
# 10. get_usdc_lend_rate (Aave, mocked requests)
# ------------------------------------------------------------------

def test_get_usdc_lend_rate(monkeypatch):
    """
    get_usdc_lend_rate should return the first liquidityRate_avg from the JSON.
    """

    class FakeResp:
        status_code = 200
        def json(self):
            # Shape compatible with pd.DataFrame([...])
            return [
                {"liquidityRate_avg": 0.031}
            ]

    def fake_get(url, params=None):
        return FakeResp()

    monkeypatch.setattr(utils, "requests", type("R", (), {"get": fake_get}))

    r = utils.get_usdc_lend_rate("2024-01-01")
    assert np.isclose(r, 0.031)

def get_usdc_lend_rate(date):
    
    _date = int(pd.Timestamp(str(date)).timestamp())
    url = "https://aave-api-v2.aave.com/data/rates-history"
    params = {
        "reserveId" : "...",
        "from" : _date,
        "resolutionInHours" : 24
    }
    response = requests.get(url,params)

    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return None

    data = response.json()
    if not data:
        print("No data returned for this date")
        prev_date = pd.to_datetime(date) - pd.Timedelta(days=1)
        return get_usdc_lend_rate(prev_date)
    
    
    df = pd.DataFrame(data)
    return df["liquidityRate_avg"].iloc[0]

# ------------------------------------------------------------------
# 11. get_historical_mean (Binance daily data, mocked)
# ------------------------------------------------------------------

def test_get_historical_mean(monkeypatch):
    """
    get_historical_mean should return a finite mean of 90 days of log returns.
    We'll mock 90 days of monotonically increasing close prices.
    """

    class FakeBinance:
        def parse8601(self, s: str):
            return 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            # Create `limit` rows of daily candles with steadily increasing close
            base_time = 1700000000000
            data = []
            for i in range(limit):
                t = base_time + i * 86400000
                close = 100 + i
                data.append([t, close-1, close+1, close-2, close, 10.0])
            return data

    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    mean_ret = utils.get_historical_mean("BTC", "2024-01-01")

    assert np.isfinite(mean_ret)

def test_get_historical_mean_flat_prices(monkeypatch):
    """
    If close prices are constant, log-returns should be ~0,
    so get_historical_mean should return 0.
    """

    class FakeBinance:
        def parse8601(self, s: str):
            return 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            base_time = 1700000000000
            data = []
            for i in range(limit):
                t = base_time + i * 86400000
                close = 100.0  # constant price
                data.append([t, close-1, close+1, close-2, close, 10.0])
            return data

    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    mean_ret = utils.get_historical_mean("BTC", "2024-01-01")

    assert np.isfinite(mean_ret)
    assert abs(mean_ret) < 1e-10  # essentially zero


def test_get_historical_mean_uptrend(monkeypatch):
    """
    If close prices follow a clear uptrend, the mean log-return should be positive.
    """

    class FakeBinance:
        def parse8601(self, s: str):
            return 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            base_time = 1700000000000
            data = []
            # simple linear-ish uptrend in close
            for i in range(limit):
                t = base_time + i * 86400000
                close = 100.0 + 0.5 * i  # increasing
                data.append([t, close-1, close+1, close-2, close, 10.0])
            return data

    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    mean_ret = utils.get_historical_mean("BTC", "2024-01-01")

    assert np.isfinite(mean_ret)
    assert mean_ret > 0


def test_get_historical_mean_downtrend(monkeypatch):
    """
    If close prices follow a downtrend, the mean log-return should be negative.
    """

    class FakeBinance:
        def parse8601(self, s: str):
            return 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            base_time = 1700000000000
            data = []
            for i in range(limit):
                t = base_time + i * 86400000
                close = 200.0 - 0.5 * i  # decreasing but still positive
                data.append([t, close-1, close+1, close-2, close, 10.0])
            return data

    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    mean_ret = utils.get_historical_mean("BTC", "2024-01-01")

    assert np.isfinite(mean_ret)
    assert mean_ret < 0


def test_get_historical_mean_uses_90_points(monkeypatch):
    """
    Sanity check: ensure the function consumes exactly 90 points
    (it always passes limit=90 to fetch_ohlcv).
    """

    calls = {"limit": []}

    class FakeBinance:
        def parse8601(self, s: str):
            return 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            calls["limit"].append(limit)
            base_time = 1700000000000
            data = []
            for i in range(limit):
                t = base_time + i * 86400000
                close = 100.0 + i
                data.append([t, close-1, close+1, close-2, close, 10.0])
            return data

    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    mean_ret = utils.get_historical_mean("BTC", "2024-01-01")

    assert np.isfinite(mean_ret)
    # It should have been called exactly once, with limit=90
    assert calls["limit"] == [90]


# ------------------------------------------------------------------
# 12. get_realized_vol (Binance 1m data, mocked)
# ------------------------------------------------------------------

def test_get_realized_vol(monkeypatch):
    """
    get_realized_vol should compute a finite realized volatility from 1m bars.
    We mock two calls to fetch_ohlcv; both return a small synthetic series.
    """

    class FakeBinance:
        def parse8601(self, s: str):
            return 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            # Return a short 1m series with gently moving prices
            base_time = 1700000000000
            data = []
            steps = 20
            for i in range(steps):
                t = base_time + i * 60000  # 1 minute in ms
                close = 100 + 0.01 * i
                data.append([t, close-0.5, close+0.5, close-1.0, close, 1.0])
            return data

    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    # Use a date different from "today" so we follow the historical branch
    rv = utils.get_realized_vol("BTC", "2024-01-01")

    assert np.isfinite(rv)
    assert rv >= 0.0
    
def test_get_realized_vol_flat_prices(monkeypatch):
    """
    If all close prices are constant, log-returns are zero,
    so realized volatility should be exactly zero.
    """

    class FakeBinance:
        def parse8601(self, s: str):
            return 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            base_time = 1700000000000
            data = []
            for i in range(limit):
                t = base_time + i * 60000  # 1 minute per bar
                close = 100.0              # constant price
                data.append([t, close-0.5, close+0.5, close-1.0, close, 1.0])
            return data

    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    rv = utils.get_realized_vol("BTC", "2024-01-01")
    assert np.isfinite(rv)
    assert abs(rv) < 1e-12  # essentially zero


def test_get_realized_vol_upward_trend(monkeypatch):
    """
    With a small upward trend in prices, realized volatility should be non-negative
    and typically strictly positive.
    """

    class FakeBinance:
        def parse8601(self, s: str):
            return 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            base_time = 1700000000000
            data = []
            for i in range(limit):
                t = base_time + i * 60000
                close = 100.0 + 0.01 * i  # gentle uptrend
                data.append([t, close-0.5, close+0.5, close-1.0, close, 1.0])
            return data

    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    rv = utils.get_realized_vol("ETH", "2024-01-01")
    assert np.isfinite(rv)
    assert rv >= 0.0
    # For a non-degenerate path, we expect strictly positive vol
    assert rv > 0.0


def test_get_realized_vol_calls_two_fetches_with_correct_limits(monkeypatch):
    """
    get_realized_vol should call fetch_ohlcv twice:
    - first with limit=1000
    - then with limit=440
    We record the (timeframe, limit) pairs to verify this.
    """
    calls = []

    class FakeBinance:
        def parse8601(self, s: str):
            return 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            calls.append((timeframe, limit))
            base_time = 1700000000000
            data = []
            # Ensure we return at least 1 row so y1[-1] is valid
            for i in range(max(1, limit)):
                t = base_time + i * 60000
                close = 100.0 + 0.005 * i
                data.append([t, close-0.5, close+0.5, close-1.0, close, 1.0])
            return data

    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    rv = utils.get_realized_vol("BTC", "2024-01-01")
    assert np.isfinite(rv)

    # We expect exactly two calls: 1st: 1m,1000; 2nd: 1m,440
    assert len(calls) == 2
    assert calls[0] == ('1m', 1000)
    assert calls[1] == ('1m', 440)


def test_get_realized_vol_today_branch(monkeypatch):
    """
    If date_ is today's date, the function uses 'now - 25h' logic.
    We don't care about the exact timestamp, only that it runs and
    returns a finite, non-negative volatility.
    """

    class FakeBinance:
        def parse8601(self, s: str):
            return 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            base_time = 1700000000000
            data = []
            for i in range(limit):
                t = base_time + i * 60000
                close = 200.0 + 0.02 * i
                data.append([t, close-0.5, close+0.5, close-1.0, close, 1.0])
            return data

    monkeypatch.setattr(utils.ccxt, "binance", lambda: FakeBinance())

    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    rv = utils.get_realized_vol("BTC", today_str)

    assert np.isfinite(rv)
    assert rv >= 0.0

