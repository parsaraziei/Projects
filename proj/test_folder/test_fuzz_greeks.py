# test_folder/test_fuzz_greeks.py
import math
from hypothesis import given, settings
import hypothesis.strategies as st

from models import models as Model


S0_strat = st.floats(min_value=1.0, max_value=200_000.0)
K_strat  = st.floats(min_value=1.0, max_value=200_000.0)
T_days_strat = st.integers(min_value=1, max_value=365)
r_strat  = st.floats(min_value=-0.05, max_value=0.25)
sigma_strat = st.floats(min_value=1e-4, max_value=3.0)


def _finite(x: float) -> bool:
    return math.isfinite(x)


def _make_bls_model_with_paths(S0, K, T_days, r, sigma, num_sims=2000):
    m = Model(
        S0=S0,
        K=K,
        IR=r,
        ta=T_days,
        sig=sigma,
        sym="BTC-USD",
        beg_date="2024-01-01",
        fin_date="2024-12-31",
        _seed=789,
        num_sims=num_sims,
        _AV=False,
    )
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
def test_pw_delta_finite_and_reasonable(S0, K, T_days, r, sigma):
    m = _make_bls_model_with_paths(S0, K, T_days, r, sigma)
    delta_call, delta_put = m.PW_delta()

    assert _finite(delta_call)
    assert _finite(delta_put)

    # Theoretical ranges with slack
    assert -0.1 <= delta_call <= 1.1
    assert -1.1 <= delta_put <= 0.1



@settings(max_examples=40, deadline=None)
@given(
    S0=S0_strat,
    K=K_strat,
    T_days=T_days_strat,
    r=r_strat,
    sigma=sigma_strat,
)
def test_pw_vega_finite(S0, K, T_days, r, sigma):
    """
    Fuzz pathwise Vega; just require finite outputs.
    """
    m = _make_bls_model_with_paths(S0, K, T_days, r, sigma)
    vega_call, vega_put = m.PW_vega()

    assert _finite(vega_call)
    assert _finite(vega_put)
