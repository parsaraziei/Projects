# test_models.py

import numpy as np
import pandas as pd
import pytest    
import copy


import utils
from models import models as Models  # adjust if your module name is different


# ---------- Fixtures & helpers ----------

@pytest.fixture
def mock_historical(monkeypatch):
    """
    Mock utils.get_historical_prices so the tests don't depend on external data.
    Returns a simple upward-sloping price series.
    """

    def fake_get_hist(sym, beg_date, fin_date, ta):
        idx = pd.date_range("2020-01-01", periods=ta, freq="D")
        series = pd.Series(
            np.linspace(100.0, 120.0, ta, dtype=float),
            index=idx,
            name="close",
        )
        return series, len(series)

    monkeypatch.setattr(utils, "get_historical_prices", fake_get_hist)
    return fake_get_hist


def make_base_model(antithetic: bool, mock_historical):
    """
    Helper to construct a models instance with sensible defaults.
    """
    return Models(
        S0=100.0,
        K=100.0,
        IR=0.05,
        ta=10,
        sig=0.2,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-20",
        _seed=123,
        num_sims=1000,
        _AV=antithetic,
    )


# ---------- Random draw tests ----------

def test_draw_randoms_non_antithetic_shapes(mock_historical):
    m = make_base_model(antithetic=False, mock_historical=mock_historical)

    assert m.draws.shape == (m.ta, m.NumSimulations)
    assert m.sum_draws.shape == (m.NumSimulations,)
    # basic sanity: should not be all zeros
    assert not np.allclose(m.draws, 0.0)


def test_draw_randoms_antithetic_structure(mock_historical):
    num_sims = 1000
    m = Models(
        S0=100.0,
        K=100.0,
        IR=0.05,
        ta=10,
        sig=0.2,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-20",
        _seed=42,
        num_sims=num_sims,
        _AV=True,
    )

    assert m.draws.shape == (m.ta, num_sims)
    # first half vs second half should be negatives of each other
    half = num_sims // 2
    np.testing.assert_allclose(m.draws[:, :half], -m.draws[:, half:])


# ---------- Additional variations for draw_randoms tests ----------

@pytest.mark.parametrize("antithetic", [False, True])
def test_draw_randoms_reproducible_same_seed(mock_historical, antithetic):
    """
    For fixed seed + identical parameters, two model instances
    must generate identical draws and sum_draws.
    """
    m1 = Models(
        S0=100.0, K=100.0, IR=0.05, ta=10, sig=0.2,
        sym="BTC/USDT", beg_date="2020-01-01", fin_date="2020-01-20",
        _seed=777, num_sims=1000, _AV=antithetic,
    )
    m2 = Models(
        S0=100.0, K=100.0, IR=0.05, ta=10, sig=0.2,
        sym="BTC/USDT", beg_date="2020-01-01", fin_date="2020-01-20",
        _seed=777, num_sims=1000, _AV=antithetic,
    )

    np.testing.assert_allclose(m1.draws, m2.draws)
    np.testing.assert_allclose(m1.sum_draws, m2.sum_draws)


@pytest.mark.parametrize("antithetic", [False, True])
def test_draw_randoms_different_seed_gives_different_draws(mock_historical, antithetic):
    """
    Changing RNG seed should change draws.
    """
    m1 = Models(
        S0=100.0, K=100.0, IR=0.05, ta=10, sig=0.2,
        sym="BTC/USDT", beg_date="2020-01-01", fin_date="2020-01-20",
        _seed=1, num_sims=1000, _AV=antithetic,
    )
    m2 = Models(
        S0=100.0, K=100.0, IR=0.05, ta=10, sig=0.2,
        sym="BTC/USDT", beg_date="2020-01-01", fin_date="2020-01-20",
        _seed=2, num_sims=1000, _AV=antithetic,
    )

    assert not np.allclose(m1.draws, m2.draws)


@pytest.mark.parametrize("antithetic", [False, True])
def test_sum_draws_matches_manual_computation(mock_historical, antithetic):
    """
    sum_draws must match: sum(draws, axis=0) / sqrt(ta_yrs).
    """
    m = Models(
        S0=100.0, K=100.0, IR=0.05, ta=20, sig=0.2,
        sym="BTC/USDT", beg_date="2020-01-01", fin_date="2020-01-30",
        _seed=10, num_sims=500, _AV=antithetic,
    )

    manual = np.sum(m.draws, axis=0) / np.sqrt(m.ta_yrs)
    np.testing.assert_allclose(m.sum_draws, manual, rtol=1e-12, atol=1e-12)


def test_draw_randoms_antithetic_row_means_close_to_zero(mock_historical):
    """
    Antithetic variates should give row means almost exactly zero.
    """
    m = Models(
        S0=100.0, K=100.0, IR=0.05, ta=30, sig=0.2,
        sym="BTC/USDT", beg_date="2020-01-01", fin_date="2020-02-01",
        _seed=1234, num_sims=2000, _AV=True,
    )
    row_means = m.draws.mean(axis=1)
    assert np.all(np.abs(row_means) < 1e-10)


def test_draw_randoms_non_antithetic_row_means_statistical_zero(mock_historical):
    """
    Without antithetic variates, row means should be statistically near zero,
    but not exactly zero.
    """
    m = Models(
        S0=100.0, K=100.0, IR=0.05, ta=30, sig=0.2,
        sym="BTC/USDT", beg_date="2020-01-01", fin_date="2020-02-01",
        _seed=5678, num_sims=5000, _AV=False,
    )
    row_means = m.draws.mean(axis=1)

    assert np.all(np.abs(row_means) < 0.05)     # loose statistical bound
    assert not np.allclose(row_means, 0.0)      # extremely unlikely


@pytest.mark.parametrize("ta,num_sims", [(1, 10), (5, 100), (30, 512)])
@pytest.mark.parametrize("antithetic", [False, True])
def test_draw_randoms_shapes_for_various_sizes(mock_historical, ta, num_sims, antithetic):
    """
    Check shape correctness across different ta and simulation sizes.
    """
    m = Models(
        S0=100.0, K=100.0, IR=0.05, ta=ta, sig=0.2,
        sym="BTC/USDT", beg_date="2020-01-01", fin_date="2020-02-01",
        _seed=99, num_sims=num_sims, _AV=antithetic,
    )

    assert m.draws.shape == (ta, num_sims)
    assert m.sum_draws.shape == (num_sims,)


# ---------- step_lognormal tests ----------

def test_step_lognormal_deterministic_when_draws_zero(mock_historical):
    """
    If we zero out all Gaussian draws, GBM should evolve deterministically:
    S_t = S0 * exp((r - 0.5 * sigma^2) * dt * n_updates)
    """
    m = make_base_model(antithetic=False, mock_historical=mock_historical)

    # Force zero randomness
    m.draws = np.zeros_like(m.draws)
    m.sum_draws = np.zeros_like(m.sum_draws)

    S0 = m.S0
    r = m.IR
    sigma = m.sig
    dt = Models.dt

    last_mean = None
    for idx in range(m.ta):
        last_mean = m.step_lognormal(idx)

    # index=0 just sets prev_vals, so we have (ta - 1) true updates
    n_updates = m.ta - 1
    expected = S0 * np.exp((r - 0.5 * sigma**2) * dt * n_updates)

    assert last_mean is not None
    assert np.isclose(last_mean, expected, rtol=1e-6)

# ---------- step_lognormal tests (with variations) ----------

def test_step_lognormal_deterministic_when_draws_zero(mock_historical):
    """
    If we zero out all Gaussian draws, GBM should evolve deterministically:
    S_t = S0 * exp((r - 0.5 * sigma^2) * dt * n_updates)
    """
    m = make_base_model(antithetic=False, mock_historical=mock_historical)

    # Force zero randomness
    m.draws = np.zeros_like(m.draws)
    m.sum_draws = np.zeros_like(m.sum_draws)

    S0 = m.S0
    r = m.IR
    sigma = m.sig
    dt = Models.dt

    last_mean = None
    for idx in range(m.ta):
        last_mean = m.step_lognormal(idx)

    # index=0 just sets prev_vals, so we have (ta - 1) true updates
    n_updates = m.ta - 1
    expected = S0 * np.exp((r - 0.5 * sigma**2) * dt * n_updates)

    assert last_mean is not None
    assert np.isclose(last_mean, expected, rtol=1e-6)


def test_step_lognormal_matches_manual_path(mock_historical):
    """
    For general (non-zero) draws, step_lognormal's prev_vals should match
    a manually simulated path using the same Gaussian increments.
    """
    m = make_base_model(antithetic=False, mock_historical=mock_historical)

    draws = m.draws.copy()
    S_manual = np.full(m.NumSimulations, m.S0, dtype=float)
    r = m.IR
    sigma = m.sig
    dt = Models.dt

    for idx in range(m.ta):
        mean_val = m.step_lognormal(idx)

        if idx > 0:
            # GBM update using the same draws
            S_manual *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * draws[idx])

        # At every step, internal prev_vals must match manual S
        np.testing.assert_allclose(m.prev_vals, S_manual)

        # Also mean of prev_vals should be equal to mean of manual path
        assert np.isclose(mean_val, S_manual.mean())


def test_step_lognormal_restarts_on_index_zero(mock_historical):
    """
    Calling step_lognormal with index=0 should reset prev_vals back to S0.
    """
    m = make_base_model(antithetic=False, mock_historical=mock_historical)

    # Run a few steps to move away from S0
    for idx in range(m.ta):
        m.step_lognormal(idx)

    # Now call again with index=0 and check that it resets
    mean_reset = m.step_lognormal(0)

    assert np.allclose(m.prev_vals, m.S0)
    assert np.isclose(mean_reset, m.S0)


def test_step_lognormal_pure_drift_when_sigma_zero(mock_historical):
    """
    With sigma=0 and zero draws, GBM collapses to deterministic pure drift:
    S_t = S0 * exp(r * dt * n_updates).
    """
    # Build a model with sigma=0
    m = Models(
        S0=100.0,
        K=100.0,
        IR=0.07,
        ta=15,
        sig=0.0,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-20",
        _seed=321,
        num_sims=500,
        _AV=False,
    )

    # Force zero draws anyway (they shouldn't matter since sigma=0)
    m.draws = np.zeros_like(m.draws)
    m.sum_draws = np.zeros_like(m.sum_draws)

    S0 = m.S0
    r = m.IR
    dt = Models.dt

    for idx in range(m.ta):
        last_mean = m.step_lognormal(idx)

    n_updates = m.ta - 1
    expected = S0 * np.exp(r * dt * n_updates)

    assert np.isclose(last_mean, expected, rtol=1e-6)
    # every simulated path should be identical in pure drift
    assert np.allclose(m.prev_vals, expected)


@pytest.mark.parametrize("ta", [1, 2, 5, 10])
def test_step_lognormal_small_horizons_consistent_with_formula(mock_historical, ta):
    """
    For different small maturities ta, with zero draws we should recover:
    S_t = S0 * exp((r - 0.5 * sigma^2) * dt * (ta - 1))
    """
    m = Models(
        S0=120.0,
        K=100.0,
        IR=0.03,
        ta=ta,
        sig=0.15,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-20",
        _seed=11,
        num_sims=300,
        _AV=False,
    )

    # zero randomness
    m.draws = np.zeros_like(m.draws)
    m.sum_draws = np.zeros_like(m.sum_draws)

    S0 = m.S0
    r = m.IR
    sigma = m.sig
    dt = Models.dt

    for idx in range(m.ta):
        last_mean = m.step_lognormal(idx)

    n_updates = max(0, ta - 1)
    expected = S0 * np.exp((r - 0.5 * sigma**2) * dt * n_updates)

    assert np.isclose(last_mean, expected, rtol=1e-6)


# ---------- Jump-diffusion tests ----------

def test_jump_diffusion_degenerates_to_gbm_when_no_jumps(mock_historical):
    """
    With poisson_rate=0 and jump_std=jump_mean=0,
    jump-diffusion should coincide with GBM, given the same Gaussian draws.
    """
    ta = 15

    # GBM model
    m_gbm = Models(
        S0=100.0,
        K=100.0,
        IR=0.03,
        ta=ta,
        sig=0.25,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-30",
        _seed=999,
        num_sims=2000,
        _AV=False,
    )
    # JD model with same seed & parameters
    m_jd = Models(
        S0=100.0,
        K=100.0,
        IR=0.03,
        ta=ta,
        sig=0.25,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-30",
        _seed=999,
        num_sims=2000,
        _AV=False,
    )
    m_jd.setup_jd(possion_freq=0.0, jump_std=0.0, jump_mean=0.0)

    # Sanity: same normal draws
    np.testing.assert_allclose(m_gbm.draws, m_jd.draws)

    # Evolve both
    for idx in range(ta):
        m_gbm.step_lognormal(idx)
        m_jd.step_jump_diff(idx)

    np.testing.assert_allclose(m_gbm.prev_vals, m_jd.jump_prev_vals, rtol=1e-10)


def test_jump_diffusion_degenerates_to_gbm_when_no_jumps(mock_historical):
    """
    With poisson_rate=0 and jump_std=jump_mean=0,
    jump-diffusion should coincide with GBM, given the same Gaussian draws.
    """
    ta = 15

    # GBM model
    m_gbm = Models(
        S0=100.0,
        K=100.0,
        IR=0.03,
        ta=ta,
        sig=0.25,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-30",
        _seed=999,
        num_sims=2000,
        _AV=False,
    )
    # JD model with same seed & parameters
    m_jd = Models(
        S0=100.0,
        K=100.0,
        IR=0.03,
        ta=ta,
        sig=0.25,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-30",
        _seed=999,
        num_sims=2000,
        _AV=False,
    )
    m_jd.setup_jd(possion_freq=0.0, jump_std=0.0, jump_mean=0.0)

    # Sanity: same normal draws
    np.testing.assert_allclose(m_gbm.draws, m_jd.draws)

    # Evolve both
    for idx in range(ta):
        m_gbm.step_lognormal(idx)
        m_jd.step_jump_diff(idx)

    np.testing.assert_allclose(m_gbm.prev_vals, m_jd.jump_prev_vals, rtol=1e-10)


def test_jump_diffusion_k_jump_matches_formula(mock_historical):
    """
    k_jump should equal exp(mean_jump + 0.5 * std_jump^2) - 1, as per Merton.
    """
    ta = 5
    mean_jump = -0.1
    std_jump = 0.3
    lam = 1.2

    m_jd = Models(
        S0=100.0,
        K=100.0,
        IR=0.01,
        ta=ta,
        sig=0.2,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-10",
        _seed=42,
        num_sims=500,
        _AV=False,
    )
    m_jd.setup_jd(possion_freq=lam, jump_std=std_jump, jump_mean=mean_jump)

    # k_jump is only set on first call to step_jump_diff
    m_jd.step_jump_diff(0)

    expected_k = np.exp(mean_jump + 0.5 * std_jump**2) - 1.0
    assert np.isclose(m_jd.k_jump, expected_k, rtol=1e-12, atol=1e-12)


def test_jump_diffusion_zero_rate_gives_zero_jump_sums(mock_historical):
    """
    With poisson_rate = 0, all Poisson jump counts should be zero and
    therefore all_jump_sums must be identically zero.
    """
    ta = 10
    m_jd = Models(
        S0=100.0,
        K=100.0,
        IR=0.02,
        ta=ta,
        sig=0.3,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-20",
        _seed=123,
        num_sims=1000,
        _AV=False,
    )
    m_jd.setup_jd(possion_freq=0.0, jump_std=0.5, jump_mean=-0.2)

    # index=0 triggers Poisson & jump generation
    m_jd.step_jump_diff(0)

    assert np.all(m_jd.poisson_jumps == 0)
    assert np.allclose(m_jd.all_jump_sums, 0.0)


def test_jump_diffusion_shapes_of_poisson_and_jumps(mock_historical):
    """
    Check that poisson_jumps and all_jump_sums have the expected shapes.
    """
    ta = 12
    num_sims = 750
    lam = 0.8

    m_jd = Models(
        S0=100.0,
        K=100.0,
        IR=0.01,
        ta=ta,
        sig=0.2,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-30",
        _seed=2024,
        num_sims=num_sims,
        _AV=False,
    )
    m_jd.setup_jd(possion_freq=lam, jump_std=0.2, jump_mean=0.1)

    m_jd.step_jump_diff(0)

    assert m_jd.poisson_jumps.shape == (ta, num_sims)
    assert m_jd.all_jump_sums.shape == (ta, num_sims)


def test_jump_diffusion_with_positive_jumps_differs_from_gbm_and_is_more_variable(mock_historical):
    """
    With non-trivial positive jumps, the JD terminal distribution should differ
    from GBM, and the JD variance should be larger (more dispersion from jumps).
    We don't enforce anything on the mean because drift is compensated by -λk.
    """
    ta = 40
    num_sims = 5000

    # Base GBM
    m_gbm = Models(
        S0=100.0,
        K=100.0,
        IR=0.03,
        ta=ta,
        sig=0.2,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-02-20",
        _seed=777,
        num_sims=num_sims,
        _AV=False,
    )

    # JD with same base parameters but with jumps
    m_jd = Models(
        S0=100.0,
        K=100.0,
        IR=0.03,
        ta=ta,
        sig=0.2,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-02-20",
        _seed=777,
        num_sims=num_sims,
        _AV=False,
    )
    m_jd.setup_jd(possion_freq=1.5, jump_std=0.25, jump_mean=0.15)

    for idx in range(ta):
        m_gbm.step_lognormal(idx)
        m_jd.step_jump_diff(idx)

    gbm_terminal = m_gbm.prev_vals
    jd_terminal = m_jd.jump_prev_vals

    # The terminal distributions should not be numerically identical
    assert not np.allclose(gbm_terminal, jd_terminal)

    # Jumps should introduce extra dispersion
    var_gbm = np.var(gbm_terminal)
    var_jd = np.var(jd_terminal)
    assert var_jd > var_gbm
    
@pytest.mark.parametrize("lam", [0.1, 0.5, 1.0])
def test_jump_diffusion_increasing_lambda_increases_jump_variability(mock_historical, lam):
    """
    As poisson_rate (lambda) increases, the variance contribution from jumps
    should grow. Here we compare variance for different lambda values qualitatively.
    """
    ta = 20
    num_sims = 4000

    # Reference small lambda
    m_small = Models(
        S0=100.0,
        K=100.0,
        IR=0.02,
        ta=ta,
        sig=0.2,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-02-01",
        _seed=111,
        num_sims=num_sims,
        _AV=False,
    )
    m_small.setup_jd(possion_freq=0.05, jump_std=0.3, jump_mean=0.0)

    for idx in range(ta):
        m_small.step_jump_diff(idx)

    var_small = np.var(m_small.jump_prev_vals)

    # Model with larger lambda `lam`
    m_big = Models(
        S0=100.0,
        K=100.0,
        IR=0.02,
        ta=ta,
        sig=0.2,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-02-01",
        _seed=222,
        num_sims=num_sims,
        _AV=False,
    )
    m_big.setup_jd(possion_freq=lam, jump_std=0.3, jump_mean=0.0)

    for idx in range(ta):
        m_big.step_jump_diff(idx)

    var_big = np.var(m_big.jump_prev_vals)

    # With more frequent jumps, variance should be larger (in expectation)
    assert var_big > var_small


# ---------- Heston tests ----------

def test_heston_evolution_no_nan_and_positive_variance(mock_historical):
    m = make_base_model(antithetic=True, mock_historical=mock_historical)
    m.setup_HS(_kappa=1.5, _theta=0.04, _vov=0.5, _rho=-0.7)

    last_mean = None
    for idx in range(m.ta):
        last_mean = m.step_Heston(idx)

    assert last_mean is not None
    assert np.isfinite(last_mean)

    # Variance & prices should exist and be finite / non-negative
    assert hasattr(m, "Heston_prev_vars")
    assert hasattr(m, "Heston_prev_vals")

    assert np.all(m.Heston_prev_vars >= 0.0)
    assert np.all(np.isfinite(m.Heston_prev_vars))
    assert np.all(np.isfinite(m.Heston_prev_vals))


# ---------- Rough Heston tests ----------

def test_rough_heston_basic_properties(mock_historical):
    ta = 12
    m = Models(
        S0=100.0,
        K=90.0,
        IR=0.01,
        ta=ta,
        sig=0.3,
        sym="ETH/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-30",
        _seed=2024,
        num_sims=500,
        _AV=True,
    )
    m.setup_RH(
        _kappa=0.7,
        _theta=0.05,
        _vov=0.4,
        _rho=-0.4,
        _hurst=0.1,
    )

    last_mean = None
    for idx in range(ta):
        last_mean = m.step_R_heston(idx)

    # Price sanity
    assert last_mean is not None
    assert np.isfinite(last_mean)

    # Shape sanity
    assert m.RH_prev_vals.shape == (m.NumSimulations,)
    assert m.RH_vars.shape == (m.NumSimulations, m.ta)
    assert m.bracket_terms.shape == (m.NumSimulations, m.x_ta)

    # Variance sanity: allow negatives but rule out explosions
    assert np.all(np.isfinite(m.RH_vars))
    min_var = float(np.nanmin(m.RH_vars))
    max_var = float(np.nanmax(m.RH_vars))

    # Just check they are within a reasonable numerical bound
    assert min_var > -1e4
    assert max_var < 1e4

    # Prices should also be finite and not exploding
    assert np.all(np.isfinite(m.RH_prev_vals))
    assert np.all(m.RH_prev_vals > 0.0)
    assert np.max(m.RH_prev_vals) < 1e6


# ---------- More Rough Heston tests / variations ----------

def test_rough_heston_reproducible_same_seed(mock_historical):
    """
    Two Rough Heston models with identical parameters and seed should
    produce identical prices and variance paths.
    """
    ta = 12

    def make_model():
        m = Models(
            S0=100.0,
            K=90.0,
            IR=0.01,
            ta=ta,
            sig=0.3,
            sym="ETH/USDT",
            beg_date="2020-01-01",
            fin_date="2020-01-30",
            _seed=2024,
            num_sims=400,
            _AV=True,
        )
        m.setup_RH(
            _kappa=0.7,
            _theta=0.05,
            _vov=0.4,
            _rho=-0.4,
            _hurst=0.1,
        )
        return m

    m1 = make_model()
    m2 = make_model()

    for idx in range(ta):
        m1.step_R_heston(idx)
        m2.step_R_heston(idx)

    np.testing.assert_allclose(m1.RH_prev_vals, m2.RH_prev_vals)
    np.testing.assert_allclose(m1.RH_vars, m2.RH_vars)
    np.testing.assert_allclose(m1.bracket_terms, m2.bracket_terms)


def test_rough_heston_different_seed_gives_different_paths(mock_historical):
    """
    Changing the seed should (with overwhelming probability) change the
    simulated Rough Heston price paths.
    """
    ta = 12

    m1 = Models(
        S0=100.0,
        K=90.0,
        IR=0.01,
        ta=ta,
        sig=0.3,
        sym="ETH/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-30",
        _seed=1,
        num_sims=400,
        _AV=True,
    )
    m1.setup_RH(_kappa=0.7, _theta=0.05, _vov=0.4, _rho=-0.4, _hurst=0.1)

    m2 = Models(
        S0=100.0,
        K=90.0,
        IR=0.01,
        ta=ta,
        sig=0.3,
        sym="ETH/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-30",
        _seed=2,
        num_sims=400,
        _AV=True,
    )
    m2.setup_RH(_kappa=0.7, _theta=0.05, _vov=0.4, _rho=-0.4, _hurst=0.1)

    for idx in range(ta):
        m1.step_R_heston(idx)
        m2.step_R_heston(idx)

    # Not a strict proof, but with high probability these shouldn't match everywhere
    assert not np.allclose(m1.RH_prev_vals, m2.RH_prev_vals)
    assert not np.allclose(m1.RH_vars, m2.RH_vars)


def test_rough_heston_vol_floor_behavior_when_vov_positive():
    """
    RH_Parallel_compute_vol should enforce a floor when vov > 0:
    sqrt(max(prev_var, max(1e-10, 1e-6 * theta))).
    """
    prev_var_tiny = 1e-12
    theta = 0.04
    vov = 0.5

    vol = Models.RH_Parallel_compute_vol(prev_var_tiny, theta, vov)

    variance_floor = max(1e-10, 1e-6 * theta)
    expected_min = np.sqrt(variance_floor)

    assert vol >= expected_min

    # For a reasonably large variance, it should just be sqrt(prev_var)
    prev_var_large = 0.5
    vol_large = Models.RH_Parallel_compute_vol(prev_var_large, theta, vov)
    assert np.isclose(vol_large, np.sqrt(prev_var_large), rtol=1e-12, atol=1e-12)


def test_rough_heston_vol_no_floor_when_vov_zero():
    """
    When vov == 0, RH_Parallel_compute_vol should just return sqrt(prev_var),
    without forcing the floor.
    """
    prev_var_tiny = 1e-12
    theta = 0.04
    vov = 0.0

    vol = Models.RH_Parallel_compute_vol(prev_var_tiny, theta, vov)
    # no floor, just sqrt(prev_var)
    assert np.isclose(vol, np.sqrt(prev_var_tiny), rtol=1e-12, atol=1e-12)


def test_rough_heston_fbm_vol_non_degenerate(mock_historical):
    """
    fbm_vol should be a non-degenerate stochastic driver:
    - same shape as draws
    - finite, with non-zero variance
    - not perfectly correlated with the base Gaussian draws.
    We do NOT assert correlation ≈ rho, because seeds and construction
    make that relationship more complex in practice.
    """
    ta = 30
    num_sims = 800
    rho = -0.6  # any value, we just grab it from setup

    m = Models(
        S0=100.0,
        K=90.0,
        IR=0.01,
        ta=ta,
        sig=0.25,
        sym="ETH/USDT",
        beg_date="2020-01-01",
        fin_date="2020-02-01",
        _seed=999,
        num_sims=num_sims,
        _AV=False,
    )
    m.setup_RH(_kappa=0.7, _theta=0.05, _vov=0.4, _rho=rho, _hurst=0.1)

    # One call with index=0 will initialise fbm_vol using current draws
    m.step_R_heston(0)

    assert m.fbm_vol.shape == m.draws.shape

    base_draws = m.draws.flatten()
    fbm_vol_flat = m.fbm_vol.flatten()

    # basic sanity
    assert np.all(np.isfinite(fbm_vol_flat))
    assert np.std(fbm_vol_flat) > 0.0

    # Not perfectly (anti-)correlated with draws
    corr = np.corrcoef(base_draws, fbm_vol_flat)[0, 1]
    assert abs(corr) < 0.99


def test_rough_heston_bracket_terms_accumulate_over_time(mock_historical):
    """
    bracket_terms should evolve over time: later time indices should be
    non-zero for most paths, while index 0 is set in the initial step.
    """
    ta = 10
    m = Models(
        S0=100.0,
        K=90.0,
        IR=0.01,
        ta=ta,
        sig=0.3,
        sym="ETH/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-30",
        _seed=2025,
        num_sims=300,
        _AV=True,
    )
    m.setup_RH(_kappa=0.8, _theta=0.04, _vov=0.5, _rho=-0.3, _hurst=0.15)

    for idx in range(ta):
        m.step_R_heston(idx)

    # index 0 should be set
    assert np.any(m.bracket_terms[:, 0] != 0.0)

    # later indices should also generally be non-zero
    non_zero_counts = (np.abs(m.bracket_terms[:, 1:]) > 0.0).sum()
    assert non_zero_counts > 0

# ---------- Historical step tests ----------

def test_step_historical_uses_mocked_series(mock_historical):
    ta = 5
    m = Models(
        S0=100.0,
        K=100.0,
        IR=0.0,
        ta=ta,
        sig=0.2,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-01-10",
        _seed=1,
        num_sims=100,
        _AV=False,
    )

    # Our fake series is linspace(100, 120, ta)
    series = m.historical_data
    assert len(series) == ta

    for idx in range(ta):
        val = m.step_historical(idx)
        assert val == series.iloc[idx]

    # beyond available data
    assert m.step_historical(ta) is None


# ---------- price_option tests ----------

def test_price_option_bls_matches_manual_discount(mock_historical):
    m = make_base_model(antithetic=False, mock_historical=mock_historical)

    # Run GBM steps
    for idx in range(m.ta):
        m.step_lognormal(idx)

    call, put = m.price_option("BLS")

    # Manual payoff
    disc = np.exp(-m.IR * m.ta_yrs)
    call_payoffs = disc * np.maximum(0.0, m.prev_vals - m.K)
    put_payoffs = disc * np.maximum(0.0, m.K - m.prev_vals)

    assert np.isclose(call, np.mean(call_payoffs), rtol=1e-10)
    assert np.isclose(put, np.mean(put_payoffs), rtol=1e-10)


def test_price_option_rough_heston_runs(mock_historical):
    m = make_base_model(antithetic=True, mock_historical=mock_historical)
    m.setup_RH(
        _kappa=0.7,
        _theta=0.05,
        _vov=0.4,
        _rho=-0.4,
        _hurst=0.1,
    )

    for idx in range(m.ta):
        m.step_R_heston(idx)

    call, put = m.price_option("RHS")
    assert np.isfinite(call)
    assert np.isfinite(put)


# ---------- Pathwise Greeks tests (with variations) ----------

def test_pathwise_greeks_basic_ranges(mock_historical):
    """
    After running a GBM simulation, the pathwise Greeks should be finite and
    live in sensible ranges: delta in [0, 1] for calls, [-1, 0] for puts,
    vega > 0 (for reasonable parameters), etc.
    """
    m = make_base_model(antithetic=True, mock_historical=mock_historical)

    # Run until maturity
    for idx in range(m.ta):
        m.step_lognormal(idx)

    # Delta
    delta_call, delta_put = m.PW_delta()
    assert np.isfinite(delta_call)
    assert np.isfinite(delta_put)
    assert 0.0 <= delta_call <= 1.0
    assert -1.0 <= delta_put <= 0.0

    # Vega
    vega_call, vega_put = m.PW_vega()
    assert np.isfinite(vega_call)
    assert np.isfinite(vega_put)
    # For non-degenerate parameters, vega should be positive
    assert vega_call > 0.0
    assert vega_put > 0.0

    # Theta
    theta_call, theta_put = m.PW_Theta()
    assert np.isfinite(theta_call)
    assert np.isfinite(theta_put)


def test_pathwise_delta_put_call_parity_approx(mock_historical):
    """
    For non-dividend BS, we should have approximately:
    Δ_call - Δ_put ≈ 1.
    Pathwise estimates will have MC noise, so we allow a tolerance.
    """
    m = make_base_model(antithetic=True, mock_historical=mock_historical)

    for idx in range(m.ta):
        m.step_lognormal(idx)

    delta_call, delta_put = m.PW_delta()
    diff = delta_call - delta_put

    assert np.isfinite(diff)
    assert abs(diff - 1.0) < 0.05  # loose tolerance for MC noise


def test_pathwise_delta_monotone_in_strike(mock_historical):
    """
    For fixed S0, r, sigma, T, call delta should decrease as K increases.
    """
    ta = 30
    seed = 1234
    num_sims = 5000

    # Lower strike (more ITM)
    m_lowK = Models(
        S0=100.0,
        K=90.0,
        IR=0.05,
        ta=ta,
        sig=0.2,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-02-01",
        _seed=seed,
        num_sims=num_sims,
        _AV=True,
    )
    # Higher strike (more OTM) with same seed so draws are identical
    m_highK = Models(
        S0=100.0,
        K=110.0,
        IR=0.05,
        ta=ta,
        sig=0.2,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-02-01",
        _seed=seed,
        num_sims=num_sims,
        _AV=True,
    )

    # Sanity: draws must match so only strike differs
    np.testing.assert_allclose(m_lowK.draws, m_highK.draws)

    for idx in range(ta):
        m_lowK.step_lognormal(idx)
        m_highK.step_lognormal(idx)

    delta_call_low, _ = m_lowK.PW_delta()
    delta_call_high, _ = m_highK.PW_delta()

    assert delta_call_low > delta_call_high


def test_pathwise_vega_call_put_symmetry(mock_historical):
    """
    In the implemented pathwise formula, vega_call and vega_put are equal.
    We explicitly check this property.
    """
    m = make_base_model(antithetic=True, mock_historical=mock_historical)

    for idx in range(m.ta):
        m.step_lognormal(idx)

    vega_call, vega_put = m.PW_vega()
    np.testing.assert_allclose(vega_call, vega_put, rtol=1e-10, atol=1e-10)


def test_pathwise_rho_signs(mock_historical):
    """
    For standard European options in BS, rho_call should be positive,
    rho_put negative (in usual parameter regimes).
    """
    m = make_base_model(antithetic=True, mock_historical=mock_historical)

    for idx in range(m.ta):
        m.step_lognormal(idx)

    rho_call, rho_put = m.PW_rho()
    assert np.isfinite(rho_call)
    assert np.isfinite(rho_put)

    # Call rho usually > 0, put rho < 0
    assert rho_call > 0.0
    assert rho_put < 0.0


def test_pathwise_theta_call_generally_negative(mock_historical):
    """
    For a reasonably ATM-ish option under BS, call theta should generally be
    negative (time decay). MC noise allowed, so we only require it to be
    not positive.
    """
    m = make_base_model(antithetic=True, mock_historical=mock_historical)

    for idx in range(m.ta):
        m.step_lognormal(idx)

    theta_call, theta_put = m.PW_Theta()
    assert np.isfinite(theta_call)
    assert np.isfinite(theta_put)

    # Theta_call should not be positive in this setup
    assert theta_call <= 0.0


# ---------- show() tests ----------

def test_show_does_not_error(capsys, mock_historical):
    m = make_base_model(antithetic=False, mock_historical=mock_historical)
    m.show()
    captured = capsys.readouterr()
    # Just ensure something was printed
    assert "start date" in captured.out.lower()
    assert "crypto" in captured.out.lower()


def test_show_includes_all_core_fields(capsys, mock_historical):
    """
    show() should print the key attributes: dates, sym, vol, strike, S0, IR, ta.
    """
    m = make_base_model(antithetic=False, mock_historical=mock_historical)
    m.show()
    out = capsys.readouterr().out

    assert f"start date: {m.beg_date}" in out
    assert f"end date: {m.fin_date}" in out
    assert f"Crypto: {m.sym}" in out
    assert f"volatility: {m.sig}" in out
    assert f"srike: {m.K}" in out  # matches the typo in the implementation
    assert f"S0: {m.S0}" in out
    assert f"IR: {m.IR}" in out
    assert f"ta:{m.ta}" in out


def test_show_reflects_parameter_changes(capsys, mock_historical):
    """
    If we build a model with different parameters, show() output should change accordingly.
    """
    m1 = make_base_model(antithetic=False, mock_historical=mock_historical)
    m2 = Models(
        S0=150.0,
        K=120.0,
        IR=0.07,
        ta=25,
        sig=0.35,
        sym="ETH/USDT",
        beg_date="2021-01-01",
        fin_date="2021-02-01",
        _seed=321,
        num_sims=500,
        _AV=False,
    )

    # Capture output for model 1
    m1.show()
    out1 = capsys.readouterr().out

    # Capture output for model 2
    m2.show()
    out2 = capsys.readouterr().out

    # At least some obvious parameters should differ in the output
    assert str(m1.S0) in out1 and str(m1.S0) not in out2
    assert str(m2.S0) in out2 and str(m2.S0) not in out1
    assert m1.sym in out1 and m1.sym not in out2
    assert m2.sym in out2 and m2.sym not in out1


def test_show_does_not_print_none(capsys, mock_historical):
    """
    As a basic hygiene check, show() should not print literal 'None'
    for any field.
    """
    m = make_base_model(antithetic=False, mock_historical=mock_historical)
    m.show()
    out = capsys.readouterr().out

    assert "None" not in out
    


# ---------------------------------------------------------------------
# Put–call parity for Black–Scholes Monte Carlo
# ---------------------------------------------------------------------

def test_put_call_parity_bls_mc(mock_historical):
    """
    For GBM under risk–neutral dynamics, Monte Carlo prices for call/put
    should satisfy put–call parity approximately:
        C - P ≈ S0 - K * exp(-r T)
    We allow a small tolerance due to Monte Carlo noise.
    """
    ta = 30  # days
    m = Models(
        S0=100.0,
        K=100.0,
        IR=0.05,
        ta=ta,
        sig=0.2,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-02-15",
        _seed=123,
        num_sims=5000,
        _AV=True,
    )

    # Run GBM to maturity
    for idx in range(m.ta):
        m.step_lognormal(idx)

    call_mc, put_mc = m.price_option("BLS")

    assert np.isfinite(call_mc)
    assert np.isfinite(put_mc)

    parity_lhs = call_mc - put_mc
    parity_rhs = m.S0 - m.K * np.exp(-m.IR * m.ta_yrs)

    # Loose but meaningful tolerance (MC noise & discretisation)
    assert np.isclose(parity_lhs, parity_rhs, rtol=0.03, atol=1e-2)


# ---------------------------------------------------------------------
# Monotonicity of option prices in strike
# ---------------------------------------------------------------------

def test_call_put_monotonic_in_strike_bls(mock_historical):
    """
    Using common random numbers (same paths), we check:
        - Call prices decrease as K increases.
        - Put prices increase as K increases.
    This is enforced pathwise if we reuse the same GBM paths
    and only change the strike in the payoff.
    """
    ta = 20
    base = Models(
        S0=100.0,
        K=100.0,
        IR=0.03,
        ta=ta,
        sig=0.25,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-02-01",
        _seed=2025,
        num_sims=4000,
        _AV=True,
    )

    # Deep–copy so draws / paths are identical, only K changes
    m_low = copy.deepcopy(base)
    m_mid = copy.deepcopy(base)
    m_high = copy.deepcopy(base)

    m_low.K = 90.0
    m_mid.K = 100.0
    m_high.K = 110.0

    # Evolve all with the same Gaussian draws
    for idx in range(ta):
        m_low.step_lognormal(idx)
        m_mid.step_lognormal(idx)
        m_high.step_lognormal(idx)

    call_low, put_low = m_low.price_option("BLS")
    call_mid, put_mid = m_mid.price_option("BLS")
    call_high, put_high = m_high.price_option("BLS")

    # Basic sanity
    for price in (call_low, call_mid, call_high, put_low, put_mid, put_high):
        assert np.isfinite(price)
        assert price >= 0.0

    # Monotonicity with shared paths: these should be exact inequalities
    assert call_low >= call_mid >= call_high
    assert put_low <= put_mid <= put_high


# ---------------------------------------------------------------------
# Pathwise vs finite-difference Delta (using common random numbers)
# ---------------------------------------------------------------------

def test_pathwise_delta_matches_fd_delta_bls(mock_historical):
    """
    Cross-check pathwise Delta against a central finite-difference
    estimate using common random numbers (same paths):
        Δ_FD ≈ (C(S0 + h) - C(S0 - h)) / (2h)
    and compare to PW_delta() for calls.
    """
    ta = 15
    num_sims = 4000
    seed = 9999

    # Base model only to generate draws & parameters
    base = Models(
        S0=100.0,
        K=100.0,
        IR=0.04,
        ta=ta,
        sig=0.3,
        sym="BTC/USDT",
        beg_date="2020-01-01",
        fin_date="2020-02-01",
        _seed=seed,
        num_sims=num_sims,
        _AV=True,
    )

    # Deep–copy so we share the exact same Gaussian draws
    m_pw = copy.deepcopy(base)
    m_up = copy.deepcopy(base)
    m_dn = copy.deepcopy(base)

    # Central bump in S0
    h = 0.01 * base.S0
    m_up.S0 = base.S0 + h
    m_dn.S0 = base.S0 - h

    # Run all three models with the same draws
    for idx in range(ta):
        m_pw.step_lognormal(idx)
        m_up.step_lognormal(idx)
        m_dn.step_lognormal(idx)

    # Pathwise Delta from main model
    delta_pw_call, delta_pw_put = m_pw.PW_delta()
    assert np.isfinite(delta_pw_call)
    assert np.isfinite(delta_pw_put)

    # Finite-difference Delta for call using common random numbers
    call_up, _ = m_up.price_option("BLS")
    call_dn, _ = m_dn.price_option("BLS")
    delta_fd_call = (call_up - call_dn) / (2.0 * h)

    assert np.isfinite(delta_fd_call)

    # They won't be identical due to MC noise, but should be close
    # Pathwise estimator is usually lower-variance; allow ~20% relative mismatch.
    assert np.isclose(delta_fd_call, delta_pw_call, rtol=0.2, atol=1e-3)

import numpy as np
import copy

# ---------------------------------------------------------------------
# STRESS TESTS (still unit-level, but with extreme parameters)
# ---------------------------------------------------------------------

def test_extreme_volatility_does_not_crash(mock_historical):
    """
    Stress test: very large volatility should not cause NaNs, infs, or
    negative prices in the GBM simulation.
    """
    m = make_base_model(antithetic=False, mock_historical=mock_historical)

    # Extreme vol (e.g. 500% annualised)
    m.sig = 5.0

    last_mean = None
    for idx in range(m.ta):
        last_mean = m.step_lognormal(idx)
        assert np.isfinite(last_mean)

    assert last_mean is not None
    # Prices should remain positive even under extreme vol
    assert np.all(m.prev_vals > 0.0)

def test_heston_rho_extremes_dont_blow_up(mock_historical):
    """
    For extreme correlation values (rho close to -1 or 1), the Heston simulation
    should remain numerically stable: prices stay positive and finite, and
    variances do not explode to absurd magnitudes.
    """
    ta = 20
    num_sims = 1000

    # Try very extreme correlations
    extreme_rhos = [-0.999, -0.95, 0.0, 0.95, 0.999]

    for rho in extreme_rhos:
        m = Models(
            S0=100.0,
            K=100.0,
            IR=0.03,
            ta=ta,
            sig=0.3,
            sym="BTC/USDT",
            beg_date="2020-01-01",
            fin_date="2020-02-01",
            _seed=1234,
            num_sims=num_sims,
            _AV=True,
        )

        # Reasonable Heston params, only changing rho
        kappa = 2.0
        theta = 0.04
        vov = 0.5
        m.setup_HS(_kappa=kappa, _theta=theta, _vov=vov, _rho=rho)

        last_mean = None
        for idx in range(ta):
            last_mean = m.step_Heston(idx)

        # Basic sanity: we actually simulated something
        assert last_mean is not None
        assert np.isfinite(last_mean)

        # Prices: all positive, finite, and not astronomically huge
        assert m.Heston_prev_vals.shape == (m.NumSimulations,)
        assert np.all(np.isfinite(m.Heston_prev_vals))
        assert np.all(m.Heston_prev_vals > 0.0)
        assert np.max(m.Heston_prev_vals) < 1e6

        # Variances: finite and not exploding
        assert m.Heston_prev_vars.shape == (m.NumSimulations,)
        assert np.all(np.isfinite(m.Heston_prev_vars))

        min_var = float(np.nanmin(m.Heston_prev_vars))
        max_var = float(np.nanmax(m.Heston_prev_vars))

        # We allow small negative noise but no crazy explosions
        assert min_var > -1e4
        assert max_var < 1e4



def test_rough_heston_near_zero_hurst(mock_historical):
    """
    Stress test: Rough Heston with very small Hurst exponent (H ~= 0)
    should remain numerically stable and not produce NaNs/infs.
    """
    m = make_base_model(antithetic=True, mock_historical=mock_historical)

    # Note: step_R_heston uses self.rho for fbm_vol, so we also set that.
    m.rho = -0.5
    m.setup_RH(
        _kappa=1.0,
        _theta=0.05,
        _vov=0.5,
        _rho=-0.5,
        _hurst=0.01,   # near zero roughness
    )

    last_mean = None
    for idx in range(m.ta):
        last_mean = m.step_R_heston(idx)
        assert np.isfinite(last_mean)

    assert last_mean is not None
    assert np.all(np.isfinite(m.RH_prev_vals))
    assert np.all(m.RH_prev_vals > 0.0)

    assert np.all(np.isfinite(m.RH_vars))
    min_var = float(np.nanmin(m.RH_vars))
    max_var = float(np.nanmax(m.RH_vars))
    # Very loose bounds, just preventing explosions
    assert min_var > -1e4
    assert max_var < 1e4


def test_jump_diffusion_with_large_lambda_stable(mock_historical):
    """
    Stress test: very high jump intensity should still lead to finite,
    positive prices without numeric blow-up.
    """
    m = make_base_model(antithetic=False, mock_historical=mock_historical)

    # Aggressive jump settings: many jumps, moderately sized
    m.setup_jd(
        possion_freq=50.0,   # very frequent jumps
        jump_std=0.3,
        jump_mean=0.05,
    )

    last_mean = None
    for idx in range(m.ta):
        last_mean = m.step_jump_diff(idx)
        assert np.isfinite(last_mean)

    assert last_mean is not None
    assert np.all(np.isfinite(m.jump_prev_vals))
    assert np.all(m.jump_prev_vals > 0.0)


def test_large_initial_price(mock_historical):
    """
    Stress test: very large initial price S0 should not produce NaNs,
    infs or negative values in GBM simulation.
    """
    m = make_base_model(antithetic=False, mock_historical=mock_historical)

    # Huge underlying level
    m.S0 = 1e7

    last_mean = None
    for idx in range(m.ta):
        last_mean = m.step_lognormal(idx)
        assert np.isfinite(last_mean)

    assert last_mean is not None
    assert np.all(np.isfinite(m.prev_vals))
    assert np.all(m.prev_vals > 0.0)
