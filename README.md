# Futures Portfolio Risk Analytics

A Python toolkit for managing and stress-testing a multi-asset futures portfolio. The project covers portfolio P&L calculation, Value at Risk (VaR) and Expected Shortfall (ES) estimation via both historical and Monte Carlo methods, dynamic roll optimization, and historical scenario analysis.

---

## Portfolio

The default portfolio consists of 10 commodity futures contracts across energy, metals, and agriculture:

| Ticker | Description         | Exchange | Multiplier | Contracts |
|--------|---------------------|----------|------------|-----------|
| CL     | WTI Crude Oil       | NYMEX    | 1,000      | 145       |
| RB     | RBOB Gasoline       | NYMEX    | 42,000     | 206       |
| NG     | Natural Gas         | NYMEX    | 10,000     | 756       |
| GC     | Gold                | COMEX    | 100        | 57        |
| SI     | Silver              | COMEX    | 5,000      | 69        |
| PL     | Platinum            | NYMEX    | 50         | 363       |
| ALI    | Aluminum            | COMEX    | 25         | 911       |
| HG     | Copper              | COMEX    | 25,000     | 477       |
| LE     | Live Cattle         | CME      | 40,000     | 194       |
| ZS     | Soybeans            | CBOT     | 5,000      | 335       |

---

## Features

### `tv.py` — Portfolio Risk & Scenario Analysis

- **P&L calculation** — computes daily mark-to-market P&L from price data, contract sizes, and multipliers
- **Historical VaR & ES** — rolling-window historical simulation at configurable confidence intervals and holding periods
- **Monte Carlo VaR & ES** — multivariate Student-t simulation using Cholesky decomposition to preserve inter-asset correlations and capture fat tails
- **Equity curve charting** — visualizes cumulative portfolio P&L over time
- **Return distribution plots** — histograms with VaR and ES overlays
- **Data ingestion** — supports both live download via TvDatafeed and local CSV import

#### Stress Test Scenarios

| Scenario | Description | Period / Parameters |
|----------|-------------|---------------------|
| 1 | Oil Price Crash | Feb – Apr 2020 |
| 2 | Energy Crisis | Feb – Oct 2022 |
| 3 | Increased Volatility | 2× volatility scaler, ν = 5 |
| 4 | Increased Tail Risk | ν = 3 |
| 5 | Extreme Volatility + Tail Risk | 4× volatility scaler, ν = 3 |

---

### `dynamic_roll.py` — Dynamic Roll Optimizer

Implements the **S&P GSCI Dynamic Roll** methodology. Given a futures curve loaded from CSV, it:

1. Takes the front contract as the reference price
2. Computes the **annualized implied roll yield** for each deferred contract relative to the front
3. Identifies the **optimal roll-in contract** — the one maximizing implied roll yield

#### CSV Format

Input files should be comma-separated with no header, in the format:

```
TICKER, DATE (MM/DD/YYYY), PRICE
```

#### Example Output

```
--- S&P GSCI Dynamic Roll Recommendation ---
Current Front Contract: RBM25
Optimal Roll-In Contract: RBU25
Maximum Implied Roll Yield (Annualized): 12.34%
```

---

### Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `tvDatafeed` *(optional — only needed for live data download)*

---

## Usage

### Portfolio Risk Analysis

Place your price data CSV at `./data/tv_data.csv`, then run:

```bash
python src/tv.py
```

### Dynamic Roll Optimizer

Place your futures curve CSV at `./data/RB.csv`, then run:

```bash
python src/dynamic_roll.py
```

---

## Data

- **`./data/tv_data.csv`** — historical daily close prices for all 10 tickers (columns = tickers, index = dates)
- **`./data/RB.csv`** — futures curve data for roll optimization (TICKER, DATE, PRICE)

> **Note:** The TvDatafeed download method (`tv_download`) is available but may be unstable due to API limitations. Using local CSV import (`tv_import`) is recommended.

---

## Project Structure

```
futures_portfolio/
├── src/
│   ├── tv.py               # Portfolio risk analytics & stress testing
│   └── dynamic_roll.py     # Dynamic roll yield optimizer
├── data/
│   ├── tv_data.csv         # Historical price data
│   └── RB.csv              # Futures curve for roll optimization
└── README.md
```
