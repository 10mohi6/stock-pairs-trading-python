# stock-pairs-trading

[![PyPI](https://img.shields.io/pypi/v/stock-pairs-trading)](https://pypi.org/project/stock-pairs-trading/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/10mohi6/stock-pairs-trading-python/branch/main/graph/badge.svg?token=DukbkJ6Pnx)](https://codecov.io/gh/10mohi6/stock-pairs-trading-python)
[![Build Status](https://app.travis-ci.com/10mohi6/stock-pairs-trading-python.svg?branch=main)](https://app.travis-ci.com/10mohi6/stock-pairs-trading-python)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/stock-pairs-trading)](https://pypi.org/project/stock-pairs-trading/)
[![Downloads](https://pepy.tech/badge/stock-pairs-trading)](https://pepy.tech/project/stock-pairs-trading)

stock-pairs-trading is a python library for backtest with stock pairs trading using kalman filter on Python 3.8 and above.

## Installation

    $ pip install stock-pairs-trading

## Usage

### find pairs
```python
from stock_pairs_trading import StockPairsTrading

spt = StockPairsTrading(
    start="2007-12-01",
    end="2017-12-01",
)
r = spt.find_pairs(["AAPL", "ADBE", "MSFT", "IBM"])
print(r)
```
```python
[('ADBE', 'MSFT')]
```
![pairs.png](https://raw.githubusercontent.com/10mohi6/stock-pairs-trading-python/main/tests/pairs.png)

### backtest
```python
from pprint import pprint
from stock_pairs_trading import StockPairsTrading

spt = StockPairsTrading(
    start="2007-12-01",
    end="2017-12-01",
)
r = spt.backtest(('ADBE', 'MSFT'))
pprint(r)
```
```python
{'cointegration': 0.0018311528816901195,
 'correlation': 0.9858057442729853,
 'maximum_drawdown': 34.801876068115234,
 'profit_factor': 1.1214715644744209,
 'riskreward_ratio': 0.8095390763424627,
 'sharpe_ratio': 0.03606830691295276,
 'total_profit': 35.97085762023926,
 'total_trades': 520,
 'win_rate': 0.5807692307692308}
```
![performance.png](https://raw.githubusercontent.com/10mohi6/stock-pairs-trading-python/main/tests/performance.png)

### latest signal
```python
from pprint import pprint
from stock_pairs_trading import StockPairsTrading

spt = StockPairsTrading(
    start="2007-12-01",
    end="2017-12-01",
)
r = spt.latest_signal(("ADBE", "MSFT"))
pprint(r)
```
```python
{'ADBE Adj Close': 299.5,
 'ADBE Buy': True, # entry buy
 'ADBE Cover': False, # exit buy
 'ADBE Sell': False, # entry sell
 'ADBE Short': False, # exit sell
 'MSFT Adj Close': 244.74000549316406,
 'MSFT Buy': False, # entry buy
 'MSFT Cover': False, # exit buy
 'MSFT Sell': True, # entry sell
 'MSFT Short': False, # exit sell
 'date': '2022-09-16',
 'zscore': -36.830427514962274}
```
## Advanced Usage
```python
from pprint import pprint
from stock_pairs_trading import StockPairsTrading

spt = StockPairsTrading(
    start="2007-12-01",
    end="2017-12-01",
    outputs_dir_path = "outputs",
    data_dir_path = "data",
    column = "Adj Close",
    window = 1,
    transition_covariance = 0.01,
)
r = spt.backtest(('ADBE', 'MSFT'))
pprint(r)
```
