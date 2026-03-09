# Systematic Trading

An implementation of Rob Carver's [Systematic Trading](https://www.systematicmoney.org/systematic-trading) book
using the [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader) framework.

This project demonstrates how to build a systematic trading framework to target a specific annual volatility,
backtest it against historical data, and analyze trading performance.

### Strategy Details

1. Compute forecasts on the underlying instruments using the latest prices. Forecasts imply probability and direction of price movement by their sign and magnitude.
2. Translate forecasts to positions based on the current account value and the chosen annualized target volatility, essentially acting as a risk engine.

## Setup

### 1. Install uv (if not already installed)

Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your operating system.

### 2. Clone and navigate to the project

```bash
cd systematic-trading
```

### 3. Install dependencies with uv

```bash
uv sync
```

### 4. Activate the virtual environment

```bash
source .venv/bin/activate
```

## Running the CLI

```bash
python cli.py
```

### Available Commands

#### 1. `download_data`

Downloads historical end-of-day price data for AAPL from the [EODHD API](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes) using the demo account:

```
(Cmd) download_data
```

#### 2. `ingest_data`

Ingests CSV data files into a Nautilus Parquet catalog for efficient backtesting:

```
(Cmd) ingest_data
```

#### 3. `backtest`

Runs a backtest of the vol-targeting strategy against historical data:

```
(Cmd) backtest
```
