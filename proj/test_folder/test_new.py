# test_gui_logic.py

import types
import numpy as np
import pandas as pd
import pytest

import new as ui


# ---------- Small helper dummies ----------

class DummyEntry:
    """Minimal stand-in for tk.Entry for logic tests."""
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class DummyVar:
    """Minimal stand-in for tk.BooleanVar/StringVar for logic tests."""
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


# ---------- Basic scalar helpers ----------

def test_validate_decimal_input_accepts_valid_numbers():
    assert ui.validate_decimal_input("")  # empty ok
    assert ui.validate_decimal_input("123")
    assert ui.validate_decimal_input("0.5")
    assert ui.validate_decimal_input("-3.14")


def test_validate_decimal_input_rejects_invalid_strings():
    assert not ui.validate_decimal_input("abc")
    assert not ui.validate_decimal_input("1.2.3")
    assert not ui.validate_decimal_input("10,5")


def test_get_ouput_blank_and_nonblank():
    assert ui.get_ouput("") == 0.0
    assert ui.get_ouput(None) == 0.0
    assert ui.get_ouput("1.5") == 1.5
    assert ui.get_ouput("0") == 0.0


def test_get_output_nonzero_blank_and_nonblank():
    assert ui.get_output_nonzero("") is None
    assert ui.get_output_nonzero(None) is None
    assert ui.get_output_nonzero("2.5") == 2.5
    assert ui.get_output_nonzero("0") == 0.0


# ---------- FD step size helpers (Delta / Theta / Rho / Vega) ----------

@pytest.mark.parametrize(
    "S0, expected_kind",
    [
        (0.005, "forward"),   # < 0.01
        (0.5, "central"),     # < 1
        (50.0, "central"),    # < 100
        (5000.0, "central"),  # < 10_000
        (20000.0, "central"), # >= 10_000
    ],
)
def test_compute_delta_h_branches(S0, expected_kind):
    ui.current_case = types.SimpleNamespace(S0=S0)
    h, kind = ui.compute_delta_h()
    assert h > 0.0
    assert kind == expected_kind


@pytest.mark.parametrize(
    "ta, expected_h, expected_kind",
    [
        (10, 1, "forward"),   # < 30
        (100, 2, "central"),  # < 182
        (365, 5, "central"),  # >= 182
    ],
)
def test_compute_theta_h_branches(ta, expected_h, expected_kind):
    ui.current_case = types.SimpleNamespace(ta=ta)
    h, kind = ui.compute_theta_h()
    assert h == expected_h
    assert kind == expected_kind


@pytest.mark.parametrize(
    "IR, expected_h, expected_kind",
    [
        (0.0001, 0.0005, "forward"),  # < 0.001
        (0.005, 0.001, "central"),    # < 0.01
        (0.05, 0.01, "central"),      # < 0.1
        (0.2, 0.02, "central"),       # >= 0.1
    ],
)
def test_compute_rho_h_branches(IR, expected_h, expected_kind):
    ui.current_case = types.SimpleNamespace(IR=IR)
    h, kind = ui.compute_rho_h()
    assert h == pytest.approx(expected_h)
    assert kind == expected_kind


@pytest.mark.parametrize(
    "sigma, expected_h, expected_kind",
    [
        (2.0, 0.05, "central"),   # > 1.5
        (1.0, 0.03, "central"),   # > 0.8
        (0.5, 0.02, "central"),   # > 0.3
        (0.2, 0.01, "forward"),   # <= 0.3
    ],
)
def test_compute_vega_h_branches(sigma, expected_h, expected_kind):
    ui.current_case = types.SimpleNamespace(sig=sigma)
    h, kind = ui.compute_vega_h()
    assert h == pytest.approx(expected_h)
    assert kind == expected_kind


# ---------- validate_vol_surface_inputs ----------

def test_validate_vol_surface_inputs_all_good_internet_mode(monkeypatch):
    """
    surf_vol_type = 'internet' means: ignore CSV stuff.
    Only interest rate 'input' branch can fail, but here it's valid.
    """
    monkeypatch.setattr(ui, "vol_surf_selected_IR", DummyVar("input"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_user_interest_rate", DummyEntry("5.0"), raising=False)
    monkeypatch.setattr(ui, "surf_vol_type", DummyVar("internet"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_intial_price_inp", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "vol_surf_uploaded_file_path", None, raising=False)

    ok, msg = ui.validate_vol_surface_inputs()
    assert ok
    assert msg == ""


def test_validate_vol_surface_inputs_bad_interest_rate(monkeypatch):
    """
    In input-IR mode, blank/None interest-rate must cause an error.
    """
    monkeypatch.setattr(ui, "vol_surf_selected_IR", DummyVar("input"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_user_interest_rate", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "surf_vol_type", DummyVar("internet"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_intial_price_inp", DummyEntry("123"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_uploaded_file_path", "dummy.csv", raising=False)

    ok, msg = ui.validate_vol_surface_inputs()
    assert not ok
    assert "interest rate" in msg.lower()


def test_validate_vol_surface_inputs_missing_initial_price_for_csv(monkeypatch):
    """
    In CSV mode, missing S0 should trigger an error.
    """
    monkeypatch.setattr(ui, "vol_surf_selected_IR", DummyVar("USDC"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_user_interest_rate", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "surf_vol_type", DummyVar("input"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_intial_price_inp", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "vol_surf_uploaded_file_path", "dummy.csv", raising=False)

    ok, msg = ui.validate_vol_surface_inputs()
    assert not ok
    assert "initial crypto price" in msg.lower()


def test_validate_vol_surface_inputs_missing_csv_file(monkeypatch):
    """
    In CSV mode, missing vol_surf_uploaded_file_path should trigger an error.
    """
    monkeypatch.setattr(ui, "vol_surf_selected_IR", DummyVar("USDC"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_user_interest_rate", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "surf_vol_type", DummyVar("input"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_intial_price_inp", DummyEntry("1000.0"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_uploaded_file_path", None, raising=False)

    ok, msg = ui.validate_vol_surface_inputs()
    assert not ok
    assert "surface" in msg.lower() or "file" in msg.lower()


# ---------- fetch_r behaviour ----------

def test_fetch_r_uses_usdc_rate(monkeypatch):
    """
    fetch_r('USDC', crypto) should call utils.get_usdc_lend_rate.
    """
    calls = {}

    class DummyUtils:
        @staticmethod
        def get_usdc_lend_rate(beg_date):
            calls['usdc'] = True
            return 0.07

        @staticmethod
        def get_risk_free_treasury_rate(beg, fin):
            calls['treasury'] = True
            return 0.01

        @staticmethod
        def get_historical_mean(sym, beg_date):
            calls['mean'] = True
            return 0.05

    # Patch the utils module used inside ui
    monkeypatch.setattr(ui, "utils", DummyUtils, raising=True)
    # We also need vol_surf_user_interest_rate to exist, even if not used
    monkeypatch.setattr(ui, "vol_surf_user_interest_rate", DummyEntry("3.0"), raising=False)

    r = ui.fetch_r("USDC", "BTC")
    assert r == 0.07
    assert calls.get('usdc', False)
    assert not calls.get('treasury', False)
    assert not calls.get('mean', False)


def test_fetch_r_uses_manual_input_for_input_mode(monkeypatch):
    """
    fetch_r('input', crypto) should read from vol_surf_user_interest_rate
    and divide by 100.
    """
    class DummyUtils:
        @staticmethod
        def get_usdc_lend_rate(beg_date):
            raise AssertionError("Should not be called in 'input' mode")

        @staticmethod
        def get_risk_free_treasury_rate(beg, fin):
            raise AssertionError("Should not be called in 'input' mode")

        @staticmethod
        def get_historical_mean(sym, beg_date):
            raise AssertionError("Should not be called in 'input' mode")

    monkeypatch.setattr(ui, "utils", DummyUtils, raising=True)
    monkeypatch.setattr(ui, "vol_surf_user_interest_rate", DummyEntry("4.5"), raising=False)

    r = ui.fetch_r("input", "BTC")
    assert r == pytest.approx(0.045)


# ---------- Small helper dummies ----------

class DummyEntry:
    """Minimal stand-in for tk.Entry for logic tests."""
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class DummyVar:
    """Minimal stand-in for tk.BooleanVar/StringVar for logic tests."""
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class DummyLabel:
    """Dummy label for testing display_error_message / display_vol_error."""
    def __init__(self):
        self.text = ""
        self.calls = []

    def config(self, **kwargs):
        if "text" in kwargs:
            self.text = kwargs["text"]
        self.calls.append(("config", kwargs))

    def grid(self, *args, **kwargs):
        self.calls.append(("grid", args, kwargs))

    def grid_remove(self, *args, **kwargs):
        self.calls.append(("grid_remove", args, kwargs))


# =====================================================================
# BASIC SCALAR HELPERS
# =====================================================================

def test_validate_decimal_input_accepts_valid_numbers():
    assert ui.validate_decimal_input("")  # empty ok
    assert ui.validate_decimal_input("123")
    assert ui.validate_decimal_input("0.5")
    assert ui.validate_decimal_input("-3.14")


def test_validate_decimal_input_rejects_invalid_strings():
    assert not ui.validate_decimal_input("abc")
    assert not ui.validate_decimal_input("1.2.3")
    assert not ui.validate_decimal_input("10,5")


def test_get_ouput_blank_and_nonblank():
    assert ui.get_ouput("") == 0.0
    assert ui.get_ouput(None) == 0.0
    assert ui.get_ouput("1.5") == 1.5
    assert ui.get_ouput("0") == 0.0


def test_get_output_nonzero_blank_and_nonblank():
    assert ui.get_output_nonzero("") is None
    assert ui.get_output_nonzero(None) is None
    assert ui.get_output_nonzero("2.5") == 2.5
    assert ui.get_output_nonzero("0") == 0.0


# =====================================================================
# FD STEP SIZE HELPERS (DELTA / THETA / RHO / VEGA)
# =====================================================================

@pytest.mark.parametrize(
    "S0, expected_kind",
    [
        (0.005, "forward"),   # < 0.01
        (0.5, "central"),     # < 1
        (50.0, "central"),    # < 100
        (5000.0, "central"),  # < 10_000
        (20000.0, "central"), # >= 10_000
    ],
)
def test_compute_delta_h_branches(S0, expected_kind):
    ui.current_case = types.SimpleNamespace(S0=S0)
    h, kind = ui.compute_delta_h()
    assert h > 0.0
    assert kind == expected_kind


@pytest.mark.parametrize(
    "ta, expected_h, expected_kind",
    [
        (10, 1, "forward"),   # < 30
        (100, 2, "central"),  # < 182
        (365, 5, "central"),  # >= 182
    ],
)
def test_compute_theta_h_branches(ta, expected_h, expected_kind):
    ui.current_case = types.SimpleNamespace(ta=ta)
    h, kind = ui.compute_theta_h()
    assert h == expected_h
    assert kind == expected_kind


@pytest.mark.parametrize(
    "IR, expected_h, expected_kind",
    [
        (0.0001, 0.0005, "forward"),  # < 0.001
        (0.005, 0.001, "central"),    # < 0.01
        (0.05, 0.01, "central"),      # < 0.1
        (0.2, 0.02, "central"),       # >= 0.1
    ],
)
def test_compute_rho_h_branches(IR, expected_h, expected_kind):
    ui.current_case = types.SimpleNamespace(IR=IR)
    h, kind = ui.compute_rho_h()
    assert h == pytest.approx(expected_h)
    assert kind == expected_kind


@pytest.mark.parametrize(
    "sigma, expected_h, expected_kind",
    [
        (2.0, 0.05, "central"),   # > 1.5
        (1.0, 0.03, "central"),   # > 0.8
        (0.5, 0.02, "central"),   # > 0.3
        (0.2, 0.01, "forward"),   # <= 0.3
    ],
)
def test_compute_vega_h_branches(sigma, expected_h, expected_kind):
    ui.current_case = types.SimpleNamespace(sig=sigma)
    h, kind = ui.compute_vega_h()
    assert h == pytest.approx(expected_h)
    assert kind == expected_kind


# =====================================================================
# VOL SURFACE VALIDATION
# =====================================================================

def test_validate_vol_surface_inputs_all_good_internet_mode(monkeypatch):
    """
    surf_vol_type = 'internet' means: ignore CSV stuff.
    Only interest rate 'input' branch can fail, but here it's valid.
    """
    monkeypatch.setattr(ui, "vol_surf_selected_IR", DummyVar("input"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_user_interest_rate", DummyEntry("5.0"), raising=False)
    monkeypatch.setattr(ui, "surf_vol_type", DummyVar("internet"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_intial_price_inp", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "vol_surf_uploaded_file_path", None, raising=False)

    ok, msg = ui.validate_vol_surface_inputs()
    assert ok
    assert msg == ""


def test_validate_vol_surface_inputs_bad_interest_rate(monkeypatch):
    """
    In input-IR mode, blank/None interest-rate must cause an error.
    """
    monkeypatch.setattr(ui, "vol_surf_selected_IR", DummyVar("input"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_user_interest_rate", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "surf_vol_type", DummyVar("internet"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_intial_price_inp", DummyEntry("123"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_uploaded_file_path", "dummy.csv", raising=False)

    ok, msg = ui.validate_vol_surface_inputs()
    assert not ok
    assert "interest rate" in msg.lower()


def test_validate_vol_surface_inputs_missing_initial_price_for_csv(monkeypatch):
    """
    In CSV mode, missing S0 should trigger an error.
    """
    monkeypatch.setattr(ui, "vol_surf_selected_IR", DummyVar("USDC"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_user_interest_rate", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "surf_vol_type", DummyVar("input"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_intial_price_inp", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "vol_surf_uploaded_file_path", "dummy.csv", raising=False)

    ok, msg = ui.validate_vol_surface_inputs()
    assert not ok
    assert "initial crypto price" in msg.lower()


def test_validate_vol_surface_inputs_missing_csv_file(monkeypatch):
    """
    In CSV mode, missing vol_surf_uploaded_file_path should trigger an error.
    """
    monkeypatch.setattr(ui, "vol_surf_selected_IR", DummyVar("USDC"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_user_interest_rate", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "surf_vol_type", DummyVar("input"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_intial_price_inp", DummyEntry("1000.0"), raising=False)
    monkeypatch.setattr(ui, "vol_surf_uploaded_file_path", None, raising=False)

    ok, msg = ui.validate_vol_surface_inputs()
    assert not ok
    assert "file" in msg.lower() or "surface" in msg.lower()


# =====================================================================
# fetch_r BEHAVIOUR
# =====================================================================

def test_fetch_r_uses_usdc_rate(monkeypatch):
    """
    fetch_r('USDC', crypto) should call utils.get_usdc_lend_rate.
    """
    calls = {}

    class DummyUtils:
        @staticmethod
        def get_usdc_lend_rate(beg_date):
            calls['usdc'] = True
            return 0.07

        @staticmethod
        def get_risk_free_treasury_rate(beg, fin):
            calls['treasury'] = True
            return 0.01

        @staticmethod
        def get_historical_mean(sym, beg_date):
            calls['mean'] = True
            return 0.05

    monkeypatch.setattr(ui, "utils", DummyUtils, raising=True)
    monkeypatch.setattr(ui, "vol_surf_user_interest_rate", DummyEntry("3.0"), raising=False)

    r = ui.fetch_r("USDC", "BTC")
    assert r == 0.07
    assert calls.get('usdc', False)
    assert not calls.get('treasury', False)
    assert not calls.get('mean', False)


def test_fetch_r_uses_manual_input_for_input_mode(monkeypatch):
    """
    fetch_r('input', crypto) should read from vol_surf_user_interest_rate
    and divide by 100.
    """
    class DummyUtils:
        @staticmethod
        def get_usdc_lend_rate(beg_date):
            raise AssertionError("Should not be called in 'input' mode")

        @staticmethod
        def get_risk_free_treasury_rate(beg, fin):
            raise AssertionError("Should not be called in 'input' mode")

        @staticmethod
        def get_historical_mean(sym, beg_date):
            raise AssertionError("Should not be called in 'input' mode")

    monkeypatch.setattr(ui, "utils", DummyUtils, raising=True)
    monkeypatch.setattr(ui, "vol_surf_user_interest_rate", DummyEntry("4.5"), raising=False)

    r = ui.fetch_r("input", "BTC")
    assert r == pytest.approx(0.045)


# =====================================================================
# verify_RH_params / verify_HS_params / verify_jump_params
# =====================================================================

def test_verify_RH_params_off_returns_true(monkeypatch):
    """If RH is turned off, verify_RH_params should always pass."""
    monkeypatch.setattr(ui, "RH_bool", DummyVar(False), raising=False)
    # other globals irrelevant in this branch, but we define them to be safe
    monkeypatch.setattr(ui, "RH_selected_type", DummyVar("input"), raising=False)
    monkeypatch.setattr(ui, "RH_theta_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "RH_rho_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "RH_kappa_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "RH_vov_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "RH_hurst_input", DummyEntry(""), raising=False)

    ok, msg = ui.verify_RH_params()
    assert ok
    assert msg == ""


def test_verify_RH_params_invalid_theta(monkeypatch):
    """If theta is invalid in input mode, should fail with RH_theta_missing."""
    monkeypatch.setattr(ui, "RH_bool", DummyVar(True), raising=False)
    monkeypatch.setattr(ui, "RH_selected_type", DummyVar("input"), raising=False)

    # theta invalid (blank), others valid
    monkeypatch.setattr(ui, "RH_theta_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "RH_rho_input", DummyEntry("0.0"), raising=False)
    monkeypatch.setattr(ui, "RH_kappa_input", DummyEntry("1.0"), raising=False)
    monkeypatch.setattr(ui, "RH_vov_input", DummyEntry("0.5"), raising=False)
    monkeypatch.setattr(ui, "RH_hurst_input", DummyEntry("0.2"), raising=False)

    ok, msg = ui.verify_RH_params()
    assert not ok
    assert msg == "RH_theta_missing"


def test_verify_RH_params_full_valid_input(monkeypatch):
    """Valid RH parameters in input mode should pass."""
    monkeypatch.setattr(ui, "RH_bool", DummyVar(True), raising=False)
    monkeypatch.setattr(ui, "RH_selected_type", DummyVar("input"), raising=False)

    monkeypatch.setattr(ui, "RH_theta_input", DummyEntry("0.1"), raising=False)
    monkeypatch.setattr(ui, "RH_rho_input", DummyEntry("-0.5"), raising=False)
    monkeypatch.setattr(ui, "RH_kappa_input", DummyEntry("2.0"), raising=False)
    monkeypatch.setattr(ui, "RH_vov_input", DummyEntry("0.8"), raising=False)
    monkeypatch.setattr(ui, "RH_hurst_input", DummyEntry("0.3"), raising=False)

    ok, msg = ui.verify_RH_params()
    assert ok
    assert msg == ""


def test_verify_HS_params_off_returns_true(monkeypatch):
    monkeypatch.setattr(ui, "HS_bool", DummyVar(False), raising=False)
    monkeypatch.setattr(ui, "HS_radio_button", DummyVar("input"), raising=False)
    monkeypatch.setattr(ui, "HS_kappa_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "HS_rho_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "HS_theta_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "HS_vov_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "uploaded_file_path_HS", None, raising=False)

    ok, msg = ui.verify_HS_params()
    assert ok
    assert msg == ""


def test_verify_HS_params_invalid_kappa(monkeypatch):
    monkeypatch.setattr(ui, "HS_bool", DummyVar(True), raising=False)
    monkeypatch.setattr(ui, "HS_radio_button", DummyVar("input"), raising=False)

    # invalid kappa: blank
    monkeypatch.setattr(ui, "HS_kappa_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "HS_rho_input", DummyEntry("0.0"), raising=False)
    monkeypatch.setattr(ui, "HS_theta_input", DummyEntry("0.2"), raising=False)
    monkeypatch.setattr(ui, "HS_vov_input", DummyEntry("0.5"), raising=False)
    monkeypatch.setattr(ui, "uploaded_file_path_HS", None, raising=False)

    ok, msg = ui.verify_HS_params()
    assert not ok
    assert msg == "kappa-missing"


def test_verify_HS_params_valid_input(monkeypatch):
    monkeypatch.setattr(ui, "HS_bool", DummyVar(True), raising=False)
    monkeypatch.setattr(ui, "HS_radio_button", DummyVar("input"), raising=False)

    monkeypatch.setattr(ui, "HS_kappa_input", DummyEntry("1.0"), raising=False)
    monkeypatch.setattr(ui, "HS_rho_input", DummyEntry("-0.3"), raising=False)
    monkeypatch.setattr(ui, "HS_theta_input", DummyEntry("0.1"), raising=False)
    monkeypatch.setattr(ui, "HS_vov_input", DummyEntry("0.4"), raising=False)
    monkeypatch.setattr(ui, "uploaded_file_path_HS", None, raising=False)

    ok, msg = ui.verify_HS_params()
    assert ok
    assert msg == ""


def test_verify_jump_params_off_returns_true(monkeypatch):
    monkeypatch.setattr(ui, "JD_bool", DummyVar(False), raising=False)
    monkeypatch.setattr(ui, "JD_radio_button", DummyVar("input"), raising=False)
    monkeypatch.setattr(ui, "jump_mean_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "jump_std_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "jump_rate_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "uploaded_file_path", None, raising=False)

    ok, msg = ui.verify_jump_params()
    assert ok
    assert msg == ""


def test_verify_jump_params_valid_input(monkeypatch):
    monkeypatch.setattr(ui, "JD_bool", DummyVar(True), raising=False)
    monkeypatch.setattr(ui, "JD_radio_button", DummyVar("input"), raising=False)

    monkeypatch.setattr(ui, "jump_mean_input", DummyEntry("0.2"), raising=False)
    monkeypatch.setattr(ui, "jump_std_input", DummyEntry("0.3"), raising=False)
    monkeypatch.setattr(ui, "jump_rate_input", DummyEntry("5.0"), raising=False)
    monkeypatch.setattr(ui, "uploaded_file_path", None, raising=False)

    ok, msg = ui.verify_jump_params()
    assert ok
    assert msg == ""


def test_verify_jump_params_missing_std(monkeypatch):
    monkeypatch.setattr(ui, "JD_bool", DummyVar(True), raising=False)
    monkeypatch.setattr(ui, "JD_radio_button", DummyVar("input"), raising=False)

    monkeypatch.setattr(ui, "jump_mean_input", DummyEntry("0.2"), raising=False)
    monkeypatch.setattr(ui, "jump_std_input", DummyEntry(""), raising=False)
    monkeypatch.setattr(ui, "jump_rate_input", DummyEntry("5.0"), raising=False)
    monkeypatch.setattr(ui, "uploaded_file_path", None, raising=False)

    ok, msg = ui.verify_jump_params()
    assert not ok
    assert msg == "std-jump-missing"


# =====================================================================
# display_error_message / display_vol_error
# =====================================================================

@pytest.mark.parametrize(
    "code, expected_substring",
    [
        ("date_error", "dates are wrong"),
        ("IR_error", "interest rate"),
        ("vol_error", "volatility"),
        ("inital_val_error", "inital value"),
        ("strike_error", "Invalid Stike"),
        ("HS-file-missing", "Heston Model option prices"),
        ("RH_theta_missing", "Rough Heston"),
    ],
)
def test_display_error_message_sets_text(monkeypatch, code, expected_substring):
    dummy = DummyLabel()
    monkeypatch.setattr(ui, "error", dummy, raising=False)

    ui.display_error_message(code)

    assert expected_substring.lower() in dummy.text.lower()
    # ensure grid() was called at least once
    assert any(call[0] == "grid" for call in dummy.calls)


def test_display_error_message_empty_for_unknown_code(monkeypatch):
    dummy = DummyLabel()
    monkeypatch.setattr(ui, "error", dummy, raising=False)

    ui.display_error_message("some_unknown_code")

    assert dummy.text == "" or dummy.text == ""
    assert any(call[0] == "grid" for call in dummy.calls)


def test_display_vol_error_sets_label_text(monkeypatch):
    dummy = DummyLabel()
    monkeypatch.setattr(ui, "vol_surf_error_message", dummy, raising=False)

    ui.display_vol_error("hello world")
    assert dummy.text == "hello world"
    assert any(call[0] == "config" for call in dummy.calls)
