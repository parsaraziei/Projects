# test_folder/test_fuzz_mc_paths.py
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st

from models import models as Model


S0_strat = st.floats(min_value=1.0, max_value=200_000.0)
T_days_strat = st.integers(min_value=1, max_value=365)
r_strat  = st.floats(min_value=-0.05, max_value=0.25)
sigma_strat = st.floats(min_value=1e-4, max_value=3.0)
n_paths_strat = st.integers(min_value=10, max_value=3000)


def _all_finite(arr: np.ndarray) -> bool:
    return np.isfinite(arr).all()


@settings(max_examples=30, deadline=None)
@given(
    S0=S0_strat,
    T_days=T_days_strat,
    r=r_strat,
    sigma=sigma_strat,
    n_paths=n_paths_strat,
)
def test_mc_terminal_prices_finite_and_positive(S0, T_days, r, sigma, n_paths):
    """
    Fuzz GBM MC terminal prices via step_lognormal:
    - prev_vals must be finite and strictly positive.
    """
    m = Model(
        S0=S0,
        K=S0,                        # strike irrelevant here
        IR=r,
        ta=T_days,
        sig=sigma,
        sym="BTC-USD",
        beg_date="2024-01-01",
        fin_date="2024-12-31",
        _seed=456,
        num_sims=n_paths,
        _AV=False,
    )

    for i in range(T_days):
        m.step_lognormal(i)

    # After final step, prev_vals is terminal distribution
    assert hasattr(m, "prev_vals")
    assert _all_finite(m.prev_vals)
    assert (m.prev_vals > 0).all()
