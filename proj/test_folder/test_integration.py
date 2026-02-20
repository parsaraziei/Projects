# test_folder/test_integration.py

import math
import threading

import numpy as np
import pandas as pd
import pytest

from models import models as Model
import utils


# =====================================================
# Global mocking of external API / historical prices
# =====================================================

@pytest.fixture(autouse=True)
def mock_get_historical_prices(monkeypatch):
    """
    Automatically mock utils.get_historical_prices for all integration tests.

    This prevents live Binance/ccxt calls and makes the tests deterministic.
    """
    def fake_get_historical_prices(sym, beg_date, fin_date, ta):
        # Simple deterministic series of length `ta`
        if ta <= 0:
            ta = 1
        prices = pd.Series(
            np.linspace(100.0, 100.0 + ta - 1, ta),  # 100, 101, ...
            name="close"
        )
        return prices, len(prices)

    monkeypatch.setattr(utils, "get_historical_prices", fake_get_historical_prices)


# =====================================================
# 1. Pricing Pipeline Integration
# =====================================================

def test_bls_pricing_pipeline_basic():
    """
    Full BLS pipeline:
    parameters -> Model -> GBM simulation -> price_option("BLS").
    """
    m = Model(
        S0=20000,
        K=21000,
        IR=0.05,
        ta=30,                 # 30 days
        sig=0.6,
        sym="BTCUSDT",
        beg_date="2024-01-01",
        fin_date="2024-12-31",
        _seed=42,
        num_sims=5000,
        _AV=False,
    )

    # Run GBM simulation over ta days
    for i in range(m.ta):
        m.step_lognormal(i)

    call, put = m.price_option("BLS")

    # Integration-level checks: pipeline works & outputs are sane
    assert math.isfinite(call)
    assert math.isfinite(put)
    assert call >= 0
    assert put >= 0


# =====================================================
# 2. Model-Switching Integration
# =====================================================

def test_switching_between_models_resets_state_and_changes_price():
    """
    Check that switching from BLS to Jump Diffusion:
    - runs end-to-end
    - produces a different (but finite) call price.
    """
    m = Model(
        S0=20000,
        K=21000,
        IR=0.03,
        ta=30,
        sig=0.5,
        sym="BTCUSDT",
        beg_date="2024-01-01",
        fin_date="2024-12-31",
        _seed=123,
        num_sims=3000,
        _AV=False,
    )

    # BLS simulation
    for i in range(m.ta):
        m.step_lognormal(i)
    call_bls, _ = m.price_option("BLS")

    # Switch to Jump Diffusion and run that pipeline
    m.setup_jd(possion_freq=1.5, jump_std=0.3, jump_mean=-0.1)
    for i in range(m.ta):
        m.step_jump_diff(i)
    call_jd, _ = m.price_option("JUD")

    assert math.isfinite(call_bls)
    assert math.isfinite(call_jd)
    # We expect a different price once jumps are enabled
    assert call_jd != call_bls


# =====================================================
# 3. Greeks Pipeline Integration
# =====================================================

def _make_bls_model_with_paths(S0, K, T_days, r, sigma, num_sims=5000):
    """
    Helper to construct a Model instance and run GBM simulation
    so PW_* Greeks can be computed.
    """
    m = Model(
        S0=S0,
        K=K,
        IR=r,
        ta=T_days,
        sig=sigma,
        sym="BTCUSDT",
        beg_date="2024-01-01",
        fin_date="2024-12-31",
        _seed=7,
        num_sims=num_sims,
        _AV=False,
    )
    for i in range(m.ta):
        m.step_lognormal(i)
    return m


def test_pathwise_delta_pipeline():
    """
    Full pipeline: simulation -> PW_delta().

    We only check that:
    - the pipeline runs end-to-end
    - outputs are finite
    - values lie within relaxed theoretical bounds.
    """
    m = _make_bls_model_with_paths(
        S0=20000,
        K=21000,
        T_days=30,
        r=0.05,
        sigma=0.6,
    )

    delta_call, delta_put = m.PW_delta()

    assert math.isfinite(delta_call)
    assert math.isfinite(delta_put)

    # Relaxed bounds to allow for MC noise
    assert -0.1 <= delta_call <= 1.1
    assert -1.1 <= delta_put <= 0.1


def test_pathwise_vega_pipeline():
    """
    Full pipeline: simulation -> PW_vega().
    Only check finiteness; sign/magnitude may depend on parameters.
    """
    m = _make_bls_model_with_paths(
        S0=20000,
        K=21000,
        T_days=30,
        r=0.05,
        sigma=0.6,
    )

    vega_call, vega_put = m.PW_vega()

    assert math.isfinite(vega_call)
    assert math.isfinite(vega_put)


# =====================================================
# 4. External-API / Historical Data Integration (via mocking)
# =====================================================

def test_historical_mode_uses_mocked_data():
    """
    Check that the model correctly consumes historical data
    (provided by the mocked get_historical_prices).
    """
    m = Model(
        S0=100,
        K=100,
        IR=0.0,
        ta=3,
        sig=0.2,
        sym="BTCUSDT",
        beg_date="2024-01-01",
        fin_date="2024-01-03",
        _seed=1,
        num_sims=1000,
        _AV=False,
    )

    # step_historical should now return the mocked series values:
    # 100, 101, 102 (from the fixture)
    assert m.step_historical(0) == 100.0
    assert m.step_historical(1) == 101.0
    assert m.step_historical(2) == 102.0
    # After the end, it returns None (by your implementation)
    assert m.step_historical(3) is None


# =====================================================
# 5. Threaded Simulation Integration
# =====================================================

def test_threaded_simulation_pipeline():
    """
    Run a full pricing pipeline inside a background thread and
    ensure that it completes and returns a sane result.
    """
    result = []

    def worker():
        m = Model(
            S0=20000,
            K=21000,
            IR=0.04,
            ta=20,
            sig=0.5,
            sym="BTCUSDT",
            beg_date="2024-01-01",
            fin_date="2024-12-31",
            _seed=99,
            num_sims=2000,
            _AV=False,
        )
        for i in range(m.ta):
            m.step_lognormal(i)
        result.append(m.price_option("BLS"))

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=10)

    # Integration check: thread finished and produced a sane result
    assert len(result) == 1
    call, put = result[0]
    assert math.isfinite(call)
    assert math.isfinite(put)
    assert call >= 0
    assert put >= 0


# =====================================================
# 6. Volatility Surface Integration (optional / adaptable)
# =====================================================

# If you have a dedicated vol surface builder, import it here.
# Adjust this block to match your actual module/function names.
try:
    # Example: from surface_module import build_vol_surface
    from surface import build_vol_surface  # <-- CHANGE IF NEEDED
except ImportError:  # pragma: no cover - gracefully skip if not available
    build_vol_surface = None


@pytest.mark.skipif(build_vol_surface is None, reason="Vol surface builder not available")
def test_vol_surface_pipeline_produces_finite_grid():
    """
    Example integration test for the volatility surface pipeline.
    Adapt build_vol_surface import + signature to your implementation.
    """
    # Synthetic strikes, maturities (in days), and vols
    strikes = np.array([15000, 20000, 25000])
    maturities = np.array([7, 30, 90])
    vols = np.array([
        [0.7, 0.6, 0.55],
        [0.75, 0.65, 0.6],
        [0.8, 0.7, 0.65],
    ])

    grid_S, grid_T, grid_vol = build_vol_surface(strikes, maturities, vols)

    assert grid_vol.ndim == 2
    assert np.isfinite(grid_vol).all()


# test_folder/test_integration.py

import math
import threading
import numpy as np
import pandas as pd
import pytest

from models import models as Model
import utils


# =====================================================
# GLOBAL FIXTURE — MOCK HISTORICAL PRICE API
# =====================================================
@pytest.fixture(autouse=True)
def mock_get_historical_prices(monkeypatch):
    """
    Automatically mock utils.get_historical_prices for ALL tests.
    This prevents ccxt/Binance live API calls and makes tests deterministic.
    """
    def fake_get_historical_prices(sym, beg_date, fin_date, ta):
        if ta <= 0:
            ta = 1
        prices = pd.Series(
            np.linspace(100.0, 100.0 + ta - 1, ta),
            name="close"
        )
        return prices, len(prices)

    monkeypatch.setattr(utils, "get_historical_prices", fake_get_historical_prices)


# =====================================================
# 1. PRICING PIPELINE INTEGRATION
# =====================================================

def test_bls_pricing_pipeline_basic():
    m = Model(
        S0=20000, K=21000, IR=0.05, ta=30, sig=0.6,
        sym="BTCUSDT", beg_date="2024-01-01", fin_date="2024-12-31",
        _seed=42, num_sims=5000, _AV=False,
    )

    for i in range(m.ta):
        m.step_lognormal(i)

    call, put = m.price_option("BLS")

    assert math.isfinite(call)
    assert math.isfinite(put)
    assert call >= 0
    assert put >= 0


# =====================================================
# 2. EDGE-CASE PIPELINE
# =====================================================

def test_pricing_pipeline_edge_parameters():
    m = Model(
        S0=20000, K=20000, IR=-0.01, ta=1, sig=3.0,
        sym="BTCUSDT", beg_date="2024-01-01", fin_date="2024-01-03",
        _seed=11, num_sims=3000, _AV=False,
    )

    for i in range(m.ta):
        m.step_lognormal(i)

    call, put = m.price_option("BLS")
    assert math.isfinite(call) and call >= 0
    assert math.isfinite(put) and put >= 0


# =====================================================
# 3. ANTITHETIC VARIATES PIPELINE
# =====================================================

def test_pricing_pipeline_with_antithetic_variates():
    m = Model(
        S0=25000, K=26000, IR=0.02, ta=20, sig=0.7,
        sym="BTCUSDT", beg_date="2024-01-01", fin_date="2024-12-31",
        _seed=21, num_sims=4000, _AV=True,
    )

    for i in range(m.ta):
        m.step_lognormal(i)

    call, put = m.price_option("BLS")
    assert math.isfinite(call)
    assert math.isfinite(put)


# =====================================================
# 4. RERUNNING SAME MODEL (STATE RESET)
# =====================================================

def test_pricing_pipeline_rerun_same_model():
    m = Model(
        S0=18000, K=19000, IR=0.03, ta=15, sig=0.5,
        sym="BTCUSDT", beg_date="2024-01-01", fin_date="2024-12-31",
        _seed=33, num_sims=3000, _AV=False,
    )

    # First run
    for i in range(m.ta):
        m.step_lognormal(i)
    call1, put1 = m.price_option("BLS")

    # Reset simulation
    m.draw_randoms()
    for i in range(m.ta):
        m.step_lognormal(i)
    call2, put2 = m.price_option("BLS")

    for v in (call1, put1, call2, put2):
        assert math.isfinite(v) and v >= 0


# =====================================================
# 5. DETERMINISM (SAME SEED → SAME RESULT)
# =====================================================

def test_pricing_pipeline_determinism_same_seed():
    params = dict(
        S0=22000, K=22500, IR=0.04, ta=25, sig=0.6,
        sym="BTCUSDT", beg_date="2024-01-01", fin_date="2024-12-31",
        num_sims=4000, _AV=False,
    )

    m1 = Model(_seed=99, **params)
    m2 = Model(_seed=99, **params)

    for i in range(m1.ta):
        m1.step_lognormal(i)
        m2.step_lognormal(i)

    call1, put1 = m1.price_option("BLS")
    call2, put2 = m2.price_option("BLS")

    assert math.isclose(call1, call2, rel_tol=1e-3)
    assert math.isclose(put1, put2, rel_tol=1e-3)


# =====================================================
# 6. MODEL-SWITCHING INTEGRATION
# =====================================================

def test_model_switching_pipeline():
    m = Model(
        S0=20000, K=21000, IR=0.03, ta=30, sig=0.5,
        sym="BTCUSDT", beg_date="2024-01-01", fin_date="2024-12-31",
        _seed=123, num_sims=3000, _AV=False,
    )

    # BLS
    for i in range(m.ta):
        m.step_lognormal(i)
    call_bls, _ = m.price_option("BLS")

    # JD
    m.setup_jd(possion_freq=0.5, jump_std=0.3, jump_mean=-0.05)
    for i in range(m.ta):
        m.step_jump_diff(i)
    call_jd, _ = m.price_option("JUD")

    assert math.isfinite(call_bls)
    assert math.isfinite(call_jd)
    assert call_jd != call_bls


# =====================================================
# 7. MODEL CONSISTENCY (JD → BS WHEN λ = 0)
# =====================================================

def test_jump_diffusion_reduces_to_bls_when_lambda_zero():
    params = dict(
        S0=20000, K=20500, IR=0.03, ta=30, sig=0.5,
        sym="BTCUSDT", beg_date="2024-01-01", fin_date="2024-12-31",
        num_sims=5000, _AV=False,
    )

    bls = Model(_seed=123, **params)
    for i in range(bls.ta):
        bls.step_lognormal(i)
    call_bls, _ = bls.price_option("BLS")

    jd = Model(_seed=123, **params)
    jd.setup_jd(possion_freq=0.0, jump_std=0.0, jump_mean=0.0)
    for i in range(jd.ta):
        jd.step_jump_diff(i)
    call_jd, _ = jd.price_option("JUD")

    assert math.isclose(call_bls, call_jd, rel_tol=5e-2)


# =====================================================
# 8. GREEKS PIPELINE
# =====================================================

def test_greeks_pipeline_pathwise_delta():
    m = Model(
        S0=20000, K=21000, IR=0.05, ta=30, sig=0.6,
        sym="BTCUSDT", beg_date="2024-01-01", fin_date="2024-12-31",
        _seed=7, num_sims=5000, _AV=False,
    )
    for i in range(m.ta):
        m.step_lognormal(i)

    delta_call, delta_put = m.PW_delta()
    assert math.isfinite(delta_call)
    assert math.isfinite(delta_put)
    assert -0.1 <= delta_call <= 1.1
    assert -1.1 <= delta_put <= 0.1


def test_greeks_pipeline_pathwise_vega():
    m = Model(
        S0=20000, K=21000, IR=0.05, ta=30, sig=0.6,
        sym="BTCUSDT", beg_date="2024-01-01", fin_date="2024-12-31",
        _seed=8, num_sims=5000, _AV=False,
    )
    for i in range(m.ta):
        m.step_lognormal(i)

    vega_call, vega_put = m.PW_vega()
    assert math.isfinite(vega_call)
    assert math.isfinite(vega_put)


# =====================================================
# 9. API INTEGRATION (MOCKED)
# =====================================================

def test_historical_data_pipeline_mocked():
    m = Model(
        S0=100, K=100, IR=0.0, ta=3, sig=0.2,
        sym="BTCUSDT", beg_date="2024-01-01", fin_date="2024-01-03",
        _seed=1, num_sims=1000, _AV=False,
    )

    assert m.step_historical(0) == 100.0
    assert m.step_historical(1) == 101.0
    assert m.step_historical(2) == 102.0
    assert m.step_historical(3) is None


# =====================================================
# 10. THREADING PIPELINE
# =====================================================

def test_threaded_simulation_pipeline():
    result = []

    def worker():
        m = Model(
            S0=20000, K=21000, IR=0.04, ta=20, sig=0.5,
            sym="BTCUSDT", beg_date="2024-01-01", fin_date="2024-12-31",
            _seed=99, num_sims=2000, _AV=False,
        )
        for i in range(m.ta):
            m.step_lognormal(i)
        result.append(m.price_option("BLS"))

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=10)

    assert len(result) == 1
    call, put = result[0]
    assert math.isfinite(call)
    assert math.isfinite(put)


# =====================================================
# 11. HISTORICAL + SIMULATION HYBRID
# =====================================================

def test_historical_then_simulation_hybrid_pipeline():
    m = Model(
        S0=100, K=105, IR=0.01, ta=10, sig=0.4,
        sym="BTCUSDT", beg_date="2024-01-01", fin_date="2024-01-10",
        _seed=55, num_sims=2000, _AV=False,
    )

    # Historical should be defined for indices [0, ..., historical_data_len-1]
    for i in range(m.historical_data_len):
        assert m.step_historical(i) is not None

    # Once we go past that, we should get None
    assert m.step_historical(m.historical_data_len) is None

    # Then run simulation and price as usual
    for i in range(m.ta):
        m.step_lognormal(i)

    call, put = m.price_option("BLS")
    assert math.isfinite(call)
    assert math.isfinite(put)

