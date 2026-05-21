<p align="center">
  <h1 align="center">📈 Hybrid Markov + Neural Network Backtester</h1>
  <p align="center">
    <strong>Institutional-Grade Quantitative Trading Framework</strong><br>
    Combining Hidden Markov Models for Regime Detection with PyTorch Neural Networks for Signal Generation<br>
    <em>Hyperparameter Search Space Informed by CS 230 (Stanford) LSTM Research</em>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-streamlit-ui-complete-guide">UI Guide</a> •
    <a href="#-how-the-math-works-and-why-it-matters">Math</a> •
    <a href="#-installation">Installation</a> •
    <a href="#-who-benefits-and-how">Benefits</a>
  </p>
</p>

---

## 📑 Table of Contents

<details>
<summary><strong>Click to expand full table of contents</strong></summary>

1. [Overview](#-overview)
2. [Who Benefits and How](#-who-benefits-and-how)
3. [Installation](#-installation)
4. [Quick Start](#-quick-start)
5. [Streamlit UI — Complete Guide](#-streamlit-ui--complete-guide)
   - [Backtest Mode](#1-backtest-mode)
   - [Data Settings](#2-data-settings)
   - [Model Architectures](#3-model-architectures)
   - [Trading Constraints](#4-trading-constraints)
   - [Risk Management](#5-risk-management)
   - [Hyperparameter Tuning](#6-hyperparameter-tuning)
   - [Validation & Walk-Forward](#7-validation--walk-forward)
   - [Run Backtest Button](#8-run-backtest)
6. [What's New — CS 230 Paper Integration](#-whats-new--cs-230-paper-integration)
7. [Architecture](#-architecture)
8. [How the Math Works and Why It Matters](#-how-the-math-works-and-why-it-matters)
9. [Understanding the Outputs](#-understanding-the-outputs)
10. [Pipeline Deep Dive](#-pipeline-deep-dive)
11. [Code Architecture](#-code-architecture)
12. [How It Works — Decision Flow](#-how-it-works--decision-flow)
13. [Example Recipes](#-example-recipes)
14. [Extending the Framework](#-extending-the-framework)
15. [Known Limitations](#-known-limitations)
16. [Academic References](#-academic-references)
17. [Project Structure](#-project-structure)
18. [Performance Notes](#-performance-notes)
19. [Troubleshooting](#-troubleshooting)
20. [Dependencies](#-dependencies)
21. [Contributing](#-contributing)
22. [License](#-license)

</details>

---

## 🔭 Overview

### What This Project Does

This system predicts whether stock prices will go **up or down tomorrow**, then automatically decides whether to **buy, sell, or stay flat** — all while managing risk like a professional trading desk.

It combines two powerful ideas:

1. **Hidden Markov Models (HMM)** — A statistical method that figures out what "mood" the market is in (bull market, bear market, or uncertain). Think of it as a weather forecast for the stock market.

2. **Deep Learning Neural Networks (PyTorch)** — Four different AI architectures (MLP, ResNet, LSTM, Transformer) that learn patterns from 21 technical indicators to predict price direction.

The key innovation is that **the HMM acts as a gatekeeper** — the neural network's predictions are only acted upon when the market is in a clear, stable regime. This prevents the system from trading during chaotic, unpredictable periods.

### System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Yahoo      │───▶│  Feature     │───▶│  HMM Regime   │───▶│  PyTorch     │
│  Finance    │    │  Engineering │    │  Detection    │    │  Neural Net  │
│  (OHLCV)    │    │  (21 feats)  │    │  (Cholesky)   │    │  (4 archs)   │
└─────────────┘    └──────────────┘    └──────┬────────┘    └──────┬───────┘
                                              │                    │
                   ┌──────────────┐    ┌──────▼────────┐           │
                   │  Streamlit UI│◀───│  Walk-Forward │◀──────────┘
                   │  + Excel     │    │  Backtesting  │
                   │  + HTML/CSV  │    │  + Optuna     │
                   └──────────────┘    └───────────────┘
```

---

## 💰 Who Benefits and How

### For Hedge Funds & Asset Managers

- **Regime-aware trading** — HMM detects market shifts before your portfolio takes a hit
- **Walk-forward validation** — No overfitting; each fold trains only on past data
- **Statistical rigor** — Deflated Sharpe Ratio separates real alpha from luck
- **Transaction cost modeling** — Backtest Sharpe won't collapse when you go live

### For Retail & Algorithmic Traders

- **One-click operation** — Pick a ticker, click Run, get results
- **Built-in benchmarks** — Compared against Buy & Hold, Logistic Regression, and 500 random traders
- **Risk management** — Stop-losses, trailing stops, circuit breakers built in
- **No coding required** — Every parameter adjustable via sidebar sliders

### For Portfolio Managers

- **Multi-asset mode** — Run backtests across multiple tickers simultaneously
- **Correlation-adjusted weights** — Highly correlated assets get reduced allocation
- **Diversification metrics** — Full correlation matrix in output

### For Researchers & Students

- **Complete codebase** — Single file (`app.py`, ~1,800 lines), fully readable
- **CS 230 paper integration** — Stanford research informs the search space
- **Extensible** — Add new architectures in 20 lines, new features in 1 line

---

## 🛠️ Installation

```bash
git clone https://github.com/ranjithvijik/markov.git
cd markov
python -m venv markov-env
source markov-env/bin/activate  # Windows: markov-env\Scripts\activate
pip install -r requirements.txt
```

### Verify

```bash
python -c "import torch; import hmmlearn; import optuna; import streamlit; print('✅ Ready!')"
```

---

## ⚡ Quick Start

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Configure via the sidebar (detailed below), then click **🚀 Run Backtest**.

---

## 🖥️ Streamlit UI — Complete Guide

The entire system is controlled through the **left sidebar** in the Streamlit interface. This section explains every option, what values you can select, and when to change them.

---

### 1. Backtest Mode

```
┌─────────────────────────────┐
│  Backtest Mode              │
│  ● Single Asset             │
│  ○ Portfolio                │
└─────────────────────────────┘
```

| Option | What It Does |
|---|---|
| **Single Asset** | Runs the full backtest pipeline on ONE ticker symbol. Produces equity curves, drawdowns, signals, and statistical tests for that single asset. |
| **Portfolio** | Runs independent backtests on MULTIPLE tickers, then combines them into a portfolio with correlation-adjusted weights. Produces portfolio-level equity curves, correlation matrix, and asset allocation pie chart. |

**When to use each:**

| Scenario | Choose |
|---|---|
| Testing a strategy on SPY | Single Asset |
| Testing a strategy on BTC-USD | Single Asset |
| Building a diversified portfolio across SPY, QQQ, GLD, TLT | Portfolio |
| Comparing how the model performs on different assets | Run Single Asset multiple times |

**Portfolio Mode Details:**
- Enter tickers as comma-separated values: `SPY, QQQ, GLD, TLT`
- Each ticker gets its own independent backtest (no cross-asset information leakage)
- Results are combined using equal weights, then adjusted for correlation (pairs with correlation > 0.7 get 0.8× weight penalty)
- Minimum 2 successful backtests required for portfolio aggregation

---

### 2. Data Settings

```
┌─────────────────────────────┐
│  Data Settings              │
│                             │
│  Ticker Symbol              │
│  [SPY                    ]  │
│                             │
│  History Period             │
│  [10y              ▼]       │
│                             │
│  Interval                   │
│  [1d               ▼]       │
└─────────────────────────────┘
```

#### Ticker Symbol

| Field | Details |
|---|---|
| **Type** | Text input |
| **Default** | `SPY` |
| **Format** | Standard Yahoo Finance ticker format |

**Valid examples:**

| Asset Class | Examples |
|---|---|
| US Equities | `SPY`, `QQQ`, `AAPL`, `TSLA`, `MSFT` |
| Crypto | `BTC-USD`, `ETH-USD`, `SOL-USD` |
| Commodities | `GLD` (gold), `SLV` (silver), `USO` (oil) |
| Bonds | `TLT` (20yr treasury), `IEF` (7-10yr) |
| International | `EWJ` (Japan), `FXI` (China), `EFA` (developed) |
| Volatility | `VIXY` (VIX futures) |

**In Portfolio Mode:** Enter comma-separated tickers in a text area: `SPY, QQQ, GLD, TLT`

**Tips:**
- Use `BTC-USD` not `BTCUSD` for crypto
- Delisted tickers will cause errors — use currently active symbols
- ETFs generally work better than individual stocks (more data, less noise)

#### History Period

| Field | Details |
|---|---|
| **Type** | Dropdown selector |
| **Default** | `10y` |
| **Options** | `1y`, `2y`, `5y`, `10y`, `max` |

**What each option means:**

| Period | Approximate Data Points | Best For |
|---|---|---|
| `1y` | ~252 trading days | Quick tests, recent regime analysis |
| `2y` | ~504 trading days | Short-term strategy validation |
| `5y` | ~1,260 trading days | Medium-term strategies, includes 1-2 market cycles |
| **`10y`** | **~2,520 trading days** | **Recommended default — includes multiple bull/bear cycles** |
| `max` | All available history | Maximum statistical power, but older data may be less relevant |

**Guidance:**
- **Minimum recommended:** `5y` (need enough data for 5 walk-forward folds)
- **Optimal:** `10y` (captures 2008 crisis aftermath, 2020 COVID crash, 2022 bear market)
- **Crypto:** Use `5y` or less (most crypto data starts 2017-2018)
- **More data = more reliable statistics** but older patterns may not repeat

**⚠️ Minimum requirement:** The system needs at least 60 data points. With 5 CV splits, you realistically need 2+ years of daily data.

#### Interval

| Field | Details |
|---|---|
| **Type** | Dropdown selector |
| **Default** | `1d` |
| **Options** | `1d`, `1wk` |

| Interval | What It Means | When to Use |
|---|---|---|
| **`1d`** | Daily bars (Open, High, Low, Close, Volume per day) | **Default and recommended.** Most technical indicators are designed for daily data. |
| `1wk` | Weekly bars | Longer-term strategies, less noise, fewer signals. Reduces data points by 5×. |

**Why no intraday?** The feature engineering (RSI, MACD, Bollinger Bands, etc.) is calibrated for daily timeframes. Using hourly data would require recalibrating all indicator lookback periods.

---

### 3. Model Architectures

```
┌─────────────────────────────┐
│  Model Architectures        │
│                             │
│  NN Architecture            │
│  [mlp              ▼]       │
│                             │
│  Sequence Length (LSTM/Tr)  │
│  [20                    ]   │
│                             │
│  HMM Hidden States          │
│  [═══════●═══════] 3        │
│  min: 2        max: 5       │
│                             │
│  Hidden Dimension           │
│  [64               ▼]       │
│                             │
│  ☐ Use Student-t HMM       │
│  ☐ PyTorch AMP (GPU)       │
└─────────────────────────────┘
```

#### NN Architecture

| Field | Details |
|---|---|
| **Type** | Dropdown selector |
| **Default** | `mlp` |
| **Options** | `mlp`, `resnet`, `lstm`, `transformer` |

**Detailed comparison:**

| Architecture | How It Works | Strengths | Weaknesses | Speed |
|---|---|---|---|---|
| **`mlp`** | Feed-forward network with 3 layers, BatchNorm, ReLU, Dropout | Fast, reliable baseline, works well with tabular features | Cannot model sequential patterns | ⚡ Fastest |
| **`resnet`** | MLP with skip connections (residual blocks) | Can learn deeper representations without gradient vanishing | Slightly slower than MLP, may overfit with small data | ⚡ Fast |
| **`lstm`** | Recurrent network with memory cells + attention mechanism | Captures sequential patterns (e.g., "3 down days followed by reversal") | Slower to train, needs sequence length tuning | 🐢 Slow |
| **`transformer`** | Self-attention mechanism over sequence | Can learn long-range dependencies, parallelizable | Slowest, needs more data to avoid overfitting | 🐢 Slowest |

**When to use each:**

| Scenario | Recommended Architecture |
|---|---|
| First time running / quick test | `mlp` |
| Want best speed-to-performance ratio | `mlp` or `resnet` |
| Believe sequential patterns matter (momentum, mean reversion) | `lstm` |
| Have GPU and lots of data (10y+) | `transformer` |
| CS 230 paper replication | `lstm` with seq_len=30 |

**Note:** When Optuna tuning is enabled, it searches over `mlp` and `resnet` automatically. LSTM and Transformer are selected via this dropdown for the final model but are NOT in the Optuna search space (they're too slow for inner-fold tuning).

#### Sequence Length (LSTM/Transformer only)

| Field | Details |
|---|---|
| **Type** | Number input |
| **Default** | `20` |
| **Appears** | Only when architecture is `lstm` or `transformer` |
| **Tuned by Optuna** | Yes — searches `[20, 30, 40]` |

**What it means:** How many past days the LSTM/Transformer looks at when making a prediction.

| Value | Interpretation | Best For |
|---|---|---|
| `20` | ~1 month of trading days | Short-term patterns, mean reversion |
| `30` | ~1.5 months (CS 230 paper's value) | Balanced — captures weekly and monthly cycles |
| `40` | ~2 months | Longer-term trends, momentum strategies |

**CS 230 paper finding:** *"All models use a timestep of 30 during data processing"* [8]. This is why 30 is included in the Optuna search space.

#### HMM Hidden States

| Field | Details |
|---|---|
| **Type** | Slider |
| **Default** | `3` |
| **Range** | `2` to `5` |
| **Tuned by Optuna** | Yes — searches `[2, 3, 4]` |

**What it means:** How many distinct "market moods" the HMM tries to identify.

| States | Interpretation | When to Use |
|---|---|---|
| `2` | Bull vs Bear (simplest) | Crypto, assets with clear binary regimes |
| **`3`** | **Bull vs Bear vs Uncertain (recommended)** | **Most assets — captures the "I don't know" state** |
| `4` | Bull / Mild Bull / Mild Bear / Bear | Assets with gradual regime transitions |
| `5` | Very granular regime detection | Only with very long histories (10y+), risk of overfitting |

**Guidance:**
- **Start with 3** — it's the most interpretable (good/bad/uncertain)
- **Use 2 for crypto** — crypto tends to have sharper regime transitions
- **Avoid 5** unless you have 10+ years of data — more states = more parameters to estimate = more overfitting risk

#### Hidden Dimension

| Field | Details |
|---|---|
| **Type** | Dropdown selector |
| **Default** | `64` |
| **Options** | `32`, `64`, `128` |
| **Tuned by Optuna** | Yes — searches `[32, 64]` |

**What it means:** The number of neurons in each hidden layer of the neural network. Controls the model's "capacity" — how complex a pattern it can learn.

| Value | Capacity | When to Use |
|---|---|---|
| `32` | Low — simple patterns only | Small datasets (<3 years), fast iteration, less overfitting |
| **`64`** | **Medium (recommended)** | **Default for most use cases** |
| `128` | High — complex patterns | Large datasets (10y+), when 64 seems to underfit |

**CS 230 paper used 50 neurons** — our search space of [32, 64] brackets this value.

#### Use Student-t HMM

| Field | Details |
|---|---|
| **Type** | Checkbox (toggle) |
| **Default** | Unchecked (off) |

**What it means:**

| Setting | HMM Type | Distribution | Best For |
|---|---|---|---|
| **Unchecked** | Gaussian HMM (`hmmlearn`) | Normal distribution — assumes returns are bell-shaped | Most assets, faster, well-tested |
| **Checked** | Student-t HMM (custom implementation) | Heavy-tailed distribution — accounts for extreme moves | Crypto, volatile stocks, assets with frequent "black swan" events |

**When to enable:**
- ✅ Trading crypto (BTC, ETH) — extreme moves are common
- ✅ Trading volatile small-caps — fat tails are the norm
- ✅ You see the Gaussian HMM producing unrealistic regime classifications
- ❌ Trading SPY/QQQ — Gaussian is usually sufficient
- ❌ You want maximum speed — Student-t is ~2× slower

**Technical detail:** The Student-t HMM uses degrees of freedom ν=5, which gives heavier tails than Gaussian. It's more robust to outliers during the EM parameter estimation.

#### PyTorch AMP (GPU)

| Field | Details |
|---|---|
| **Type** | Checkbox (toggle) |
| **Default** | Unchecked (off) |

**What it means:** Enables Automatic Mixed Precision training, which uses 16-bit floating point for some operations to speed up GPU training.

| Setting | Effect |
|---|---|
| **Unchecked** | Standard 32-bit training on CPU or GPU |
| **Checked** | Mixed 16/32-bit training — ~2× faster on NVIDIA GPUs with CUDA |

**When to enable:**
- ✅ You have an NVIDIA GPU with CUDA installed
- ✅ Running many Optuna trials (15+) and want to save time
- ❌ You're on CPU only (will be ignored, no harm)
- ❌ You're getting NaN losses (AMP can cause numerical instability in rare cases)

**Requirements:** NVIDIA GPU + CUDA toolkit + `torch` compiled with CUDA support.

---

### 4. Trading Constraints

```
┌─────────────────────────────┐
│  Trading Constraints        │
│                             │
│  Probability Threshold (Long)│
│  [═══════════●══] 0.52      │
│  min: 0.50      max: 0.99   │
│                             │
│  Probability Threshold (Short)│
│  [══●═══════════] 0.48      │
│  min: 0.01      max: 0.50   │
│                             │
│  Regime Confidence Gate     │
│  [═══════●══════] 0.45      │
│  min: 0.00      max: 1.00   │
│                             │
│  Target Volatility          │
│  [0.15                  ]   │
│                             │
│  Fixed Cost (bps)           │
│  [2.00                  ]   │
│                             │
│  Spread (bps)               │
│  [1.00                  ]   │
└─────────────────────────────┘
```

#### Probability Threshold (Long)

| Field | Details |
|---|---|
| **Type** | Slider |
| **Default** | `0.52` |
| **Range** | `0.50` to `0.99` |

**What it means:** The neural network outputs a probability between 0 and 1 representing "how likely is the price to go UP tomorrow?" This threshold determines when to go LONG (buy).

```
If NN probability ≥ 0.52 → GO LONG (buy)
If NN probability < 0.52 → Don't go long
```

| Value | Behavior | Trade Frequency | Signal Quality |
|---|---|---|---|
| `0.50` | Go long whenever model says >50% up | Very frequent trading | Many false signals |
| **`0.52`** | **Slight edge required (recommended)** | **Moderate trading** | **Good balance** |
| `0.55` | Need 55% confidence to trade | Less frequent | Higher quality signals |
| `0.60` | Need 60% confidence | Rare trades | Very selective |
| `0.70+` | Extremely selective | Very few trades | May miss opportunities |

**Guidance:**
- **0.52** is the sweet spot — requires the model to be slightly more confident than a coin flip
- Increase to **0.55-0.60** if you're getting too many trades or too many false signals
- Decrease to **0.50-0.51** if the model is too conservative and rarely trades

#### Probability Threshold (Short)

| Field | Details |
|---|---|
| **Type** | Slider |
| **Default** | `0.48` |
| **Range** | `0.01` to `0.50` |

**What it means:** When the NN probability drops BELOW this threshold, the system goes SHORT (sells/bets on price decline).

```
If NN probability ≤ 0.48 → GO SHORT (sell)
If NN probability > 0.48 → Don't go short
```

| Value | Behavior | Short Frequency |
|---|---|---|
| `0.50` | Short whenever model says <50% up | Very frequent shorting |
| **`0.48`** | **Need slight bearish conviction (recommended)** | **Moderate** |
| `0.45` | Need 55% bearish conviction | Less frequent |
| `0.40` | Need 60% bearish conviction | Rare shorts |
| `0.01` | Effectively disables shorting | Never shorts |

**The "dead zone":** Between `prob_short` and `prob_long` (e.g., 0.48 to 0.52), the system stays FLAT — no position. This dead zone prevents trading on weak signals.

```
[SHORT]  ←── 0.48 ──── FLAT (no trade) ────  0.52 ──→  [LONG]
```

**Guidance:**
- Keep `prob_short < prob_long` always (the system validates this)
- Set `prob_short = 0.01` to create a **long-only** strategy (never shorts)
- Widen the dead zone (e.g., 0.45/0.55) for fewer but higher-quality trades

#### Regime Confidence Gate

| Field | Details |
|---|---|
| **Type** | Slider |
| **Default** | `0.45` |
| **Range** | `0.00` to `1.00` |

**What it means:** The HMM outputs a probability for each regime (e.g., 70% bull, 20% bear, 10% uncertain). This gate requires the HMM to be at least X% confident in the favorable regime before allowing trades.

```
For LONG trades:  P(bull_state) must be ≥ 0.45
For SHORT trades: P(bear_state) must be ≥ 0.45
Otherwise:        Signal = 0 (FLAT, no trade)
```

| Value | Behavior | Effect |
|---|---|---|
| `0.00` | Disabled — trade regardless of regime | Maximum trades, no regime filtering |
| `0.30` | Low bar — trade if regime is somewhat clear | More trades, some noise |
| **`0.45`** | **Moderate confidence required (recommended)** | **Good balance of activity and quality** |
| `0.60` | High confidence required | Fewer trades, only in clear regimes |
| `0.80` | Very high confidence | Very few trades, only in extreme regimes |
| `1.00` | Impossible to satisfy | No trades ever (effectively disables system) |

**Why this matters:** During regime transitions (e.g., bull → bear), the HMM is uncertain — probabilities might be 40%/35%/25%. Trading during these transitions is dangerous because the model doesn't know what's happening. The gate prevents this.

**Guidance:**
- **0.45** works well for most assets
- Lower to **0.30** if the system is too conservative (rarely trades)
- Raise to **0.60** if you want only the highest-conviction trades

#### Target Volatility

| Field | Details |
|---|---|
| **Type** | Number input |
| **Default** | `0.15` |
| **Range** | `0` to any positive number |

**What it means:** The annualized volatility you want your portfolio to have. The system automatically sizes positions to achieve this target.

```
position_size = min(target_vol / yesterday's_realized_vol, 1.5)
```

| Value | Meaning | Effect |
|---|---|---|
| `0` | **Disabled** — no volatility targeting, always trade at 1× | Raw signals, no position sizing |
| `0.10` | Target 10% annual vol | Conservative — smaller positions |
| **`0.15`** | **Target 15% annual vol (recommended)** | **Moderate risk — similar to S&P 500 long-term vol** |
| `0.20` | Target 20% annual vol | Aggressive — larger positions |
| `0.30` | Target 30% annual vol | Very aggressive — may use leverage |

**How it works in practice:**
- If SPY's recent volatility is 20% and target is 15%: position = 15%/20% = 0.75× (reduce exposure)
- If SPY's recent volatility is 10% and target is 15%: position = 15%/10% = 1.5× (increase exposure, capped at 1.5×)
- If SPY's recent volatility is 40% (crash): position = 15%/40% = 0.375× (dramatically reduce exposure)

**Guidance:**
- **0.15** is appropriate for most equity strategies
- Set to **0** for crypto (crypto vol is already 60-80%, targeting 15% would mean tiny positions)
- Increase to **0.20-0.25** if you're comfortable with higher drawdowns

#### Fixed Cost (bps)

| Field | Details |
|---|---|
| **Type** | Number input |
| **Default** | `2.00` |
| **Unit** | Basis points (1 bps = 0.01%) |

**What it means:** The fixed commission cost per trade, expressed in basis points of the trade value.

| Value | Meaning | Typical For |
|---|---|---|
| `0` | Zero commission | Commission-free brokers (Robinhood, etc.) |
| `1.0` | 0.01% per trade | Institutional rates |
| **`2.0`** | **0.02% per trade (recommended)** | **Conservative estimate for most brokers** |
| `5.0` | 0.05% per trade | High-cost brokers or exotic instruments |
| `10.0` | 0.10% per trade | Very expensive execution |

**Example:** Trading $100,000 with cost_bps=2.0 → each trade costs $100,000 × 0.0002 = $20.

#### Spread (bps)

| Field | Details |
|---|---|
| **Type** | Number input |
| **Default** | `1.00` |
| **Unit** | Basis points |

**What it means:** The bid-ask spread cost — the difference between the price you can buy at and the price you can sell at.

| Value | Meaning | Typical For |
|---|---|---|
| `0.5` | Very tight spread | SPY, QQQ (most liquid ETFs) |
| **`1.0`** | **Tight spread (recommended)** | **Large-cap stocks, major ETFs** |
| `3.0` | Moderate spread | Mid-cap stocks, less liquid ETFs |
| `5.0` | Wide spread | Small-caps, emerging market ETFs |
| `10.0+` | Very wide spread | Illiquid assets, crypto on some exchanges |

**Guidance:**
- SPY/QQQ: `0.5-1.0` bps
- Individual large-cap stocks: `1.0-2.0` bps
- Crypto: `3.0-10.0` bps (varies by exchange)
- Small-caps: `5.0-15.0` bps

---

### 5. Risk Management

```
┌─────────────────────────────┐
│  Risk Management            │
│                             │
│  ☐ Enable Stop Loss /       │
│    Take Profit              │
│                             │
│  Stop Loss %                │
│  [0.020                 ]   │
│                             │
│  Take Profit %              │
│  [0.050                 ]   │
│                             │
│  Max DD Halt %              │
│  [0.10                  ]   │
└─────────────────────────────┘
```

#### Enable Stop Loss / Take Profit

| Field | Details |
|---|---|
| **Type** | Checkbox (toggle) |
| **Default** | Unchecked (off) |

**What it means:** When enabled, activates the full risk management suite including stop-loss, take-profit, and trailing stop logic.

| Setting | Effect |
|---|---|
| **Unchecked** | No intraday risk management — positions held until signal changes |
| **Checked** | Active risk management — positions can be closed by stops |

**When to enable:**
- ✅ Trading volatile assets (crypto, small-caps)
- ✅ You want to cap maximum loss per trade
- ✅ You want to lock in profits automatically
- ❌ First time testing (adds complexity, harder to interpret results)
- ❌ You want to see the "pure" model performance without risk overlays

#### Stop Loss % (appears when risk enabled)

| Field | Details |
|---|---|
| **Type** | Number input |
| **Default** | `0.02` (2%) |

**What it means:** If a trade loses more than this percentage from entry, close it immediately.

| Value | Meaning | Appropriate For |
|---|---|---|
| `0.01` | 1% stop | Very tight — frequent stops, may get "stopped out" of good trades |
| **`0.02`** | **2% stop (recommended)** | **Standard for equities** |
| `0.03` | 3% stop | Slightly wider — fewer false stops |
| `0.05` | 5% stop | Wide — for volatile assets or longer holding periods |
| `0.10` | 10% stop | Very wide — only for crypto or extreme volatility |

#### Take Profit % (appears when risk enabled)

| Field | Details |
|---|---|
| **Type** | Number input |
| **Default** | `0.05` (5%) |

**What it means:** If a trade gains more than this percentage from entry, close it and lock in profits.

| Value | Meaning |
|---|---|
| `0.03` | Take profit at 3% — conservative, frequent profit-taking |
| **`0.05`** | **Take profit at 5% (recommended)** |
| `0.10` | Take profit at 10% — lets winners run longer |
| `0.20` | Take profit at 20% — very patient |

**Rule of thumb:** Take-profit should be 2-3× your stop-loss (reward:risk ratio ≥ 2:1).

#### Max DD Halt %

| Field | Details |
|---|---|
| **Type** | Number input |
| **Default** | `0.10` (10%) |

**What it means:** If the portfolio's total drawdown from its peak exceeds this percentage, **ALL trading is halted permanently** for the remainder of that fold. This is a "circuit breaker."

| Value | Meaning |
|---|---|
| `0.05` | Halt at 5% drawdown — very conservative |
| **`0.10`** | **Halt at 10% drawdown (recommended)** |
| `0.15` | Halt at 15% drawdown |
| `0.20` | Halt at 20% drawdown — aggressive |
| `1.00` | Effectively disabled (100% drawdown would mean total loss) |

**Why this exists:** Even with stop-losses on individual trades, a series of losing trades can compound. The circuit breaker prevents catastrophic loss by shutting everything down when things go wrong.

**⚠️ Important:** Once triggered, trading does NOT resume in that fold. This is intentional — it simulates a real fund manager pulling the plug.

---

### 6. Hyperparameter Tuning

```
┌─────────────────────────────┐
│  Hyperparameter Tuning      │
│                             │
│  ☑ Enable Optuna Tuning    │
│                             │
│  Optuna Trials per Fold     │
│  [═══●═════════════] 5      │
│  min: 1         max: 50     │
└─────────────────────────────┘
```

#### Enable Optuna Tuning

| Field | Details |
|---|---|
| **Type** | Checkbox (toggle) |
| **Default** | Checked (on) |

**What it means:** When enabled, the system uses Optuna's Bayesian optimization (Tree-structured Parzen Estimator) to automatically find the best hyperparameters within each walk-forward fold.

| Setting | Effect | Runtime |
|---|---|---|
| **Checked** | Optuna searches 432 possible configurations per fold | 3-15 minutes |
| **Unchecked** | Uses default config values (no search) | < 15 seconds |

**What Optuna searches (CS 230 enhanced):**

| Parameter | Search Space | Paper's Best [8] |
|---|---|---|
| `n_states` | [2, 3, 4] | — |
| `architecture` | [mlp, resnet] | — |
| `hidden_dim` | [32, 64] | 50 |
| `dropout` | [0.10, 0.20, 0.35] | **0.10** |
| `lr` | [5e-4, 1e-3] | — |
| `batch_size` | [32, 64] | **32** |
| `seq_len` | [20, 30, 40] | **30** |

**When to disable:**
- Quick environment test (verify everything works)
- Debugging feature engineering changes
- You want deterministic results with specific parameters

#### Optuna Trials per Fold

| Field | Details |
|---|---|
| **Type** | Slider |
| **Default** | `5` |
| **Range** | `1` to `50` |

**What it means:** How many different hyperparameter combinations Optuna tries within each walk-forward fold before selecting the best one.

| Value | Search Quality | Runtime (per fold) | When to Use |
|---|---|---|---|
| `1` | Minimal — essentially random | ~30 sec | Debugging only |
| **`5`** | **Good — finds reasonable configs (recommended)** | **~2-3 min** | **Default for most use cases** |
| `10` | Better — more thorough search | ~5-7 min | When you have time and want better results |
| `15` | Thorough — good coverage of search space | ~8-12 min | Serious strategy development |
| `25` | Very thorough | ~15-20 min | Final validation before deployment |
| `50` | Exhaustive — diminishing returns | ~30-40 min | Research purposes only |

**Guidance:**
- **5 trials** is the sweet spot for development (fast enough to iterate, good enough to find decent params)
- **15 trials** for final strategy validation
- **50 trials** only if you're running overnight and want maximum confidence
- More trials = better hyperparameters but with diminishing returns after ~15-20

**How Optuna works internally:**
1. Trial 1-2: Random exploration
2. Trial 3+: Bayesian optimization — uses results from previous trials to intelligently choose next configuration
3. Each trial trains a model on inner folds and evaluates on inner validation set
4. Best trial's hyperparameters are used for the final model on the outer test fold

---

### 7. Validation & Walk-Forward

```
┌─────────────────────────────┐
│  Validation & Walk-Forward  │
│                             │
│  CV Splits                  │
│  [═══════●═════════] 5      │
│  min: 2         max: 10     │
│                             │
│  ☐ Anchored Walk-Forward   │
└─────────────────────────────┘
```

#### CV Splits

| Field | Details |
|---|---|
| **Type** | Slider |
| **Default** | `5` |
| **Range** | `2` to `10` |

**What it means:** How many walk-forward folds to use. Each fold trains on past data and tests on a future chunk.

| Value | Test Periods | Data per Fold | Statistical Reliability |
|---|---|---|---|
| `2` | 2 test periods | Large train, large test | Low — only 2 out-of-sample evaluations |
| `3` | 3 test periods | Large train, medium test | Moderate |
| **`5`** | **5 test periods** | **Medium train, medium test** | **Good balance (recommended)** |
| `7` | 7 test periods | Smaller train, smaller test | High reliability, but each fold has less data |
| `10` | 10 test periods | Small train, small test | Maximum reliability, but may underfit |

**Trade-offs:**

| More Splits (7-10) | Fewer Splits (2-3) |
|---|---|
| ✅ More out-of-sample evaluations | ✅ More training data per fold |
| ✅ More statistically reliable | ✅ Model can learn more complex patterns |
| ❌ Less training data per fold | ❌ Fewer test evaluations |
| ❌ Slower (more folds to process) | ❌ Less reliable statistics |

**Guidance:**
- **5 splits** is optimal for 10 years of daily data (~500 test days per fold)
- Use **3 splits** if you only have 2-3 years of data
- Use **7 splits** if you have 10+ years and want maximum statistical confidence
- **Never use 2** unless you're just debugging

#### Anchored Walk-Forward

| Field | Details |
|---|---|
| **Type** | Checkbox (toggle) |
| **Default** | Unchecked (off) |

**What it means:** Controls whether the training window grows over time or stays fixed.

| Setting | Training Window | Effect |
|---|---|---|
| **Unchecked (Rolling)** | Fixed size — drops old data as new data is added | Each fold has the same amount of training data. Assumes recent data is more relevant than old data. |
| **Checked (Anchored)** | Expanding — always starts from the beginning | Later folds have MORE training data. Assumes all historical data is valuable. |

**Visual comparison:**

```
ROLLING (default):
Fold 1: [====TRAIN====][TEST]
Fold 2:    [====TRAIN====][TEST]
Fold 3:       [====TRAIN====][TEST]    ← Same train size each fold

ANCHORED:
Fold 1: [====TRAIN====][TEST]
Fold 2: [=======TRAIN=======][TEST]
Fold 3: [==========TRAIN==========][TEST]    ← Growing train size
```

| Use Rolling When | Use Anchored When |
|---|---|
| Market structure changes over time | You believe all history is informative |
| Recent data is more relevant | You want maximum training data in later folds |
| Asset has undergone structural changes | Asset has been stable for decades |
| **Most cases (recommended)** | Long-term equity indices (SPY 20y+) |

---

### 8. Run Backtest

```
┌─────────────────────────────┐
│                             │
│  [🚀 Run Backtest        ] │
│                             │
└─────────────────────────────┘
```

| Field | Details |
|---|---|
| **Type** | Button (primary, full-width) |
| **Color** | Red/Primary |

**What happens when you click:**

1. **Configuration is assembled** from all sidebar values
2. **Data is downloaded** from Yahoo Finance (cached for 1 hour)
3. **Features are computed** (21 technical indicators)
4. **Walk-forward loop begins:**
   - For each fold: Optuna tunes → HMM fits → NN trains → Signals generated → Performance computed
5. **Results are compiled** into equity curves, statistics, and exports
6. **Dashboard appears** in three tabs: Dashboard View, Data & Metrics, Downloads

**Progress indicators:**
- Progress bar shows fold completion (e.g., "Fold 3/5")
- Status text shows current operation (e.g., "Tuning Optuna Trial 4/5")
- Final message: "Execution Completed in X.Xs"

---

## 🚀 What's New — CS 230 Paper Integration

This framework incorporates findings from Stanford CS 230 [8] which tested 6 LSTM configurations:

| Model | Layers | Dropout | Batch Size | Best RMSE (FB) |
|---|---|---|---|---|
| 1 | 4 | 0.2 | 32 | 5.61 |
| 2 | 4 | 0.1 | 32 | 6.98 |
| 3 | 3 | 0.2 | 32 | 5.24 |
| **4** | **3** | **0.1** | **32** | **4.89** ✓ |
| 5 | 3 | 0.2 | 64 | 6.68 |
| 6 | 3 | 0.1 | 64 | 6.36 |

**Key findings integrated into our Optuna search space:**
- **Dropout 0.10** outperforms 0.20 → Added to search: `[0.10, 0.20, 0.35]`
- **Batch size 32** outperforms 64 → Added to search: `[32, 64]`
- **Timestep 30** used as default → Added to search: `[20, 30, 40]`

---

## 🏗️ Architecture

### The 6-Stage Pipeline

| Stage | What Happens | Why It Matters |
|---|---|---|
| **1. Data** | Downloads OHLCV with retry logic | Handles API failures gracefully |
| **2. Features** | 21 technical indicators | Transforms noise into learnable signals |
| **3. HMM** | Classifies market regime | Prevents trading in chaos |
| **4. Neural Net** | Predicts up/down probability | Core prediction engine |
| **5. Validation** | Walk-forward + Optuna | Ensures real out-of-sample performance |
| **6. Reporting** | Excel + HTML + CSV | Professional output for analysis |

---

## 🧠 How the Math Works and Why It Matters

### Hidden Markov Model — Reading the Market's Mood

The HMM identifies latent market states using:

**Transition probabilities** — How likely is the market to switch moods?
```
Aᵢⱼ = P(Sₜ₊₁ = j | Sₜ = i)
```

**Emission probabilities** — Given the mood, what patterns do we expect?
```
bᵢ(Xₜ) = 𝒩(Xₜ; μᵢ, Σᵢ)
```

**Posterior probability** — What mood is the market in NOW?
```
γₜ(i) = P(Sₜ = i | all observations)
```

**Cholesky decomposition** for numerical stability:
```
Σ = LLᵀ → D²(x,μ) = ‖L⁻¹(x-μ)‖²
```

### Neural Networks — Predicting Direction

**LSTM gates** (from CS 230 paper):
```
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)     Forget gate
iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)     Input gate
cₜ = fₜ⊙cₜ₋₁ + iₜ⊙tanh(Wc·[hₜ₋₁,xₜ]+bc)   Cell update
hₜ = σ(Wo·[hₜ₋₁,xₜ]+bo) ⊙ tanh(cₜ)          Output
```

**Attention mechanism** (our enhancement):
```
αₜ = softmax(vᵀ·tanh(Wₐhₜ))    Which past days matter most?
context = Σₜ αₜ·hₜ              Weighted combination
```

### Volatility Targeting — Sizing the Bet

```
position = min(σ_target / σ̂_{t-1}, 1.5)

Calm market (σ=10%): position = 15%/10% = 1.5× (max leverage)
Normal (σ=16%):      position = 15%/16% = 0.94×
Crash (σ=40%):       position = 15%/40% = 0.375× (cut exposure)
```

### Statistical Testing — Is It Real?

**Probabilistic Sharpe Ratio:**
```
PSR = Φ((SR̂ - 0) / √Var(SR̂))
Var(SR̂) = (1/(n-1))·[1 + ½SR² - γ₃·SR + (γ₄/4)·SR²]
```

**Deflated Sharpe Ratio** (multiple testing correction):
```
DSR = PSR(SR̂, E[max random SR], n, skew, kurtosis)
```

---

## 📈 Understanding the Outputs

### Dashboard Tab (10 Panels)

1. Cumulative equity curves (log scale)
2. Performance summary table
3. Drawdown analysis
4. Monthly return distributions
5. NN probabilities & trade signals
6. Regime stability mask
7. Hyperparameter tuning log
8. Statistical significance tests
9. Rolling 63-day Sharpe ratio
10. Annual returns comparison

### Downloads Tab

| Export | Format | Contents |
|---|---|---|
| Master Excel | `.xlsx` | 6 sheets: Summary, EquityCurve, MonthlyReturns, HyperparamLog, StatisticalTests, Configuration |
| Interactive HTML | `.html` | Full Plotly dashboard (shareable, no Python needed) |
| Raw Signals CSV | `.csv` | Daily signals, probabilities, returns for further analysis |

---

## 🚥 How It Works — Decision Flow

```
Day T arrives
    │
    ▼
┌─────────────────────────────────┐
│  GATE 1: Regime Stability       │
│  ≤6 flips in 20 days?          │──── NO ──▶ FLAT
└───────────────┬─────────────────┘
                │ YES
                ▼
┌─────────────────────────────────┐
│  GATE 2: HMM Confidence        │
│  P(regime) ≥ regime_gate?       │──── NO ──▶ FLAT
└───────────────┬─────────────────┘
                │ YES
                ▼
┌─────────────────────────────────┐
│  GATE 3: NN Probability         │
│  ≥ prob_long? → LONG            │
│  ≤ prob_short? → SHORT          │──── NEITHER ──▶ FLAT
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  GATE 4: Vol Sizing             │
│  size = target_vol / σ̂_{t-1}    │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  GATE 5: Risk Management        │
│  Stop-loss / Take-profit /      │
│  Trailing stop / DD halt        │
└───────────────┬─────────────────┘
                │
                ▼
         Execute Trade (minus costs)
```

---

## 🧪 Example Recipes

| Use Case | Settings |
|---|---|
| **Quick test** | SPY, 10y, mlp, Optuna OFF → <15 sec |
| **Standard backtest** | SPY, 10y, mlp, 5 trials, 5 folds → 3-5 min |
| **CS 230 replication** | SPY, 10y, lstm, seq_len=30, 15 trials → 15-20 min |
| **Crypto** | BTC-USD, 5y, mlp, Student-t HMM, vol_target=0 |
| **Portfolio** | Portfolio mode, `SPY, QQQ, GLD, TLT`, 5 trials |
| **Conservative** | prob_long=0.55, prob_short=0.45, regime_gate=0.60 |
| **Aggressive** | prob_long=0.51, prob_short=0.49, regime_gate=0.30 |

---

## 🧩 Extending the Framework

### Add a Feature (1 line)
```python
features["my_indicator"] = close.rolling(10).mean() / close.rolling(50).mean()
```

### Add an Architecture (20 lines)
```python
class MyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, **kwargs):
        super().__init__()
        self.net = nn.Sequential(...)
    def forward(self, x):
        if x.dim() == 3: x = x[:, -1, :]
        return self.net(x)
```

---

## 🛑 Known Limitations

| Limitation | Mitigation |
|---|---|
| Gaussian HMM may miss fat tails | Enable Student-t HMM |
| Assumes regime stability within folds | Use shorter periods or anchored |
| Daily frequency only | Supports `1wk`; intraday not implemented |
| No market microstructure | Increase `impact_factor` for illiquid assets |
| Survivorship bias possible | Use currently active tickers |

---

## 🎓 Academic References

1. **Hamilton, J.D. (1989).** *"A New Approach to the Economic Analysis of Nonstationary Time Series."* Econometrica.
2. **Gu, S., Kelly, B., & Xiu, D. (2020).** *"Empirical Asset Pricing via Machine Learning."* Review of Financial Studies.
3. **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley.
4. **Bailey, D.H., & López de Prado, M. (2012).** *"The Sharpe Ratio Efficient Frontier."* Journal of Risk.
5. **Rabiner, L.R. (1989).** *"A Tutorial on Hidden Markov Models."* Proceedings of the IEEE.
6. **Hochreiter, S., & Schmidhuber, J. (1997).** *"Long Short-Term Memory."* Neural Computation.
7. **Vaswani, A., et al. (2017).** *"Attention Is All You Need."* NeurIPS.
8. **Miao, Y. (2020).** *"CS 230: A Deep Learning Approach for Stock Market Prediction."* Stanford University.
9. **Li, A.W. & Bastos, G.S. (2020).** *"Stock Market Forecasting Using Deep Learning."* IEEE Access.

---

## 📁 Project Structure

```
markov/
├── app.py              # Complete engine + Streamlit UI (~1,800 lines)
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
└── .gitignore
```

---

## ⏱️ Performance Notes

| Scenario | Runtime | Memory |
|---|---|---|
| No tuning | < 15 sec | ~500 MB |
| MLP, 5 trials, 5 folds | 3–7 min | ~1.5 GB |
| LSTM, 15 trials, 5 folds | 15–25 min | ~1.5 GB |
| Portfolio (4 assets), 5 trials | 15–30 min | ~1.5 GB |

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| "No data returned" | Check ticker format (`BTC-USD` not `BTCUSD`) |
| "HMM did not converge" | Harmless — Cholesky fallback handles it |
| "Not enough samples" | Increase period or decrease CV splits |
| "CUDA out of memory" | Reduce hidden dim or disable AMP |
| Very slow | Reduce Optuna trials or disable tuning |
| No trades generated | Lower regime_gate or narrow prob thresholds |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `torch` | Neural networks (MLP, ResNet, LSTM, Transformer) |
| `hmmlearn` | Gaussian HMM regime detection |
| `scipy` | Cholesky decomposition, statistical tests |
| `optuna` | Bayesian hyperparameter optimization |
| `plotly` | Interactive dashboards |
| `openpyxl` | Excel workbook generation |
| `streamlit` | Web UI |
| `yfinance` | Market data |
| `scikit-learn` | Scaling, splitting, benchmarks |
| `pandas`, `numpy` | Data manipulation |

---