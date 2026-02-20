# test_folder/test_fuzz_pricers.py
import math
from hypothesis import given, settings
import hypothesis.strategies as st

from models import models as Model  # your class


# ---- Strategies for parameters ----

S0_strat = st.floats(min_value=1.0, max_value=200_000.0)
K_strat  = st.floats(min_value=1.0, max_value=200_000.0)
T_days_strat = st.integers(min_value=1, max_value=365)      # maturity in days
r_strat  = st.floats(min_value=-0.05, max_value=0.25)
sigma_strat = st.floats(min_value=1e-4, max_value=3.0)


def _finite(x: float) -> bool:
    return math.isfinite(x)


def _make_bls_model(S0, K, T_days, r, sigma, num_sims=1000):
    """
    Helper: create a models instance for BLS MC and run the path simulation.
    """
    m = Model(
        S0=S0,
        K=K,
        IR=r,
        ta=T_days,
        sig=sigma,
        sym="BTC-USD",              # dummy symbol just to satisfy ctor
        beg_date="2024-01-01",      # dummy dates
        fin_date="2024-12-31",
        _seed=123,
        num_sims=num_sims,
        _AV=False,
    )

    # Run GBM steps up to maturity
    for i in range(T_days):
        m.step_lognormal(i)

    return m


@settings(max_examples=40, deadline=None)
@given(
    S0=S0_strat,
    K=K_strat,
    T_days=T_days_strat,
    r=r_strat,
    sigma=sigma_strat,
)
def test_bls_mc_call_price_finite_and_sane(S0, K, T_days, r, sigma):
    """
    Fuzz the Monte Carlo BLS call pricer via the `models` class.
    We only require:
    - finite call price
    - non-negative
    - not absurdly large vs S0 (very loose cap)
    """
    m = _make_bls_model(S0, K, T_days, r, sigma)
    call, put = m.price_option("BLS")

    assert _finite(call)
    assert call >= -1e-6
    assert call <= S0 * 10.0  # stupidly loose cap just to catch explosions


@settings(max_examples=40, deadline=None)
@given(
    S0=S0_strat,
    K=K_strat,
    T_days=T_days_strat,
    r=r_strat,
    sigma=sigma_strat,
)
def test_bls_mc_put_price_finite_and_non_negative(S0, K, T_days, r, sigma):
    """
    Same idea for MC BLS put prices.
    """
    m = _make_bls_model(S0, K, T_days, r, sigma)
    call, put = m.price_option("BLS")

    assert _finite(put)
    assert put >= -1e-6
    assert put <= K * 10.0
