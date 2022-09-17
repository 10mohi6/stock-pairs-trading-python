import pytest
from stock_pairs_trading import StockPairsTrading


@pytest.fixture(scope="module", autouse=True)
def scope_module():
    yield StockPairsTrading(
        start="2007-12-01",
        end="2017-12-01",
        window=1,
        transition_covariance=0.01,
    )


@pytest.fixture(scope="function", autouse=True)
def spt(scope_module):
    yield scope_module


# @pytest.mark.skip
def test_backtest(spt):
    spt.backtest(("ADBE", "MSFT"))


# @pytest.mark.skip
def test_latest_signal(spt):
    spt.latest_signal(("ADBE", "MSFT"))


# @pytest.mark.skip
def test_find_pairs(spt):
    spt.find_pairs(["AAPL", "ADBE", "MSFT", "IBM"])
