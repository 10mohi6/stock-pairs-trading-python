from setuptools import find_packages, setup

setup(
    name="stock-pairs-trading",
    version="0.1.1",
    description="stock-pairs-trading is a python library \
        for backtest with stock pairs trading using kalman filter on Python 3.8 and above.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    author="10mohi6",
    author_email="10.mohi.6.y@gmail.com",
    url="https://github.com/10mohi6/stock-pairs-trading-python",
    keywords="pairs trading python backtest stock kalman filter",
    packages=find_packages(),
    install_requires=[
        "yfinance",
        "matplotlib",
        "statsmodels",
        "pykalman",
        "seaborn",
    ],
    python_requires=">=3.8.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
    ],
)
