# test_folder/conftest.py
import pytest
import pandas as pd
import numpy as np
import utils


@pytest.fixture(autouse=True)
def mock_get_historical_prices(monkeypatch):
    def fake_get_historical_prices(sym, beg_date, fin_date, ta):
        data = pd.Series(
            np.full(ta, 100.0),  # simple constant series
            name="close"
        )
        return data, ta

    monkeypatch.setattr(utils, "get_historical_prices", fake_get_historical_prices)
