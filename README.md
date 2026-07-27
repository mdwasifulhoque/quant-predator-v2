# 🤖 Quant Predator v2 — Autonomous Multi-Timeframe PPO Trading System

> A fully autonomous, Multi-Timeframe PPO Reinforcement Learning trading system
> built on MetaTrader 5, Stable-Baselines3, and local Ollama Llama3.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![MT5](https://img.shields.io/badge/MetaTrader_5-API-black?style=for-the-badge)](https://www.metatrader5.com/)
[![SB3](https://img.shields.io/badge/RL-Stable--Baselines3-purple?style=for-the-badge)](https://stable-baselines3.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Postgres](https://img.shields.io/badge/DB-Postgres_(Supabase)-336791?style=for-the-badge&logo=postgresql)](https://supabase.com/)

---

## 📊 Results (Run 4 — currently deployed)

The model trains via PPO with an LSTM feature extractor on a 40-feature
multi-timeframe (H1→H4→D1) EURUSD observation space. **Run 4**, which
introduced continuous ATR-normalized Fair Value Gap distance features in
place of earlier binary flags, is the current best checkpoint and is
deployed to a demo account for live validation.

| Metric | Value |
|---|---|
| Sharpe Ratio (H1-annualised) | **1.090** |
| Sortino Ratio | 1.304 |
| Max Drawdown | -20.38% |
| Total Return (backtest) | +63.59% |
| Total Trades | 175 |
| Win Rate | 29.1% |
| Realised R:R | 3.19 : 1 |
| Profit Factor | 1.312 |
| Expectancy | +$36.34 / trade |

**Statistical honesty note:** at 175 trades, the 95% confidence interval on
win rate is approximately 22.4%–35.8%. The lower bound sits just below the
23.9% break-even threshold at this R:R — promising, but not yet statistically
conclusive on its own. This is the reason for an active 60-day live demo
validation period (target: 250+ live trades) before any live-capital
decision is made. 


---

## ⚡ What's New in v2 (vs original)

| Bug / Weakness                               | v1 Status               | v2 Fix                               |
| --------------------------------------------- | ------------------------ | ------------------------------------- |
| Fixed episode start (model memorised 1 path) | ❌ Always bar 124        | ✅ Random start per episode           |
| Lookahead bias in swing detection            | ❌ `center=True`         | ✅ `center=False` confirmed           |
| Non-stationary absolute price features       | ❌ Raw EMA/BB levels     | ✅ All stationary (% distance)        |
| Open bar used for inference                  | ❌ `iloc[-1]` open bar   | ✅ `iloc[-2]` closed bar              |
| MT5 not thread-safe — raw calls everywhere   | ❌ No serialisation      | ✅ MT5Gateway queue pattern           |
| Race condition on shared state writes        | ❌ Plain `json.dump()`   | ✅ Postgres transactions (see below)  |
| Position lost on bot restart                 | ❌ Always `position=0`   | ✅ `_recover_position_state()`        |
| SL/TP offline hits never logged              | ❌ Lost forever          | ✅ Detected on next startup           |
| AI reasoning columns all zero on close       | ❌ Hardcoded zeros       | ✅ Entry indicators cached            |
| MLP 64×64 on 890-dim input                   | ❌ 14:1 bottleneck       | ✅ LSTM(128) + MLP[256,128]           |
| Binary FVG presence flags (low signal)       | ❌ Active/inactive only  | ✅ Continuous ATR-normalized distance |
| JSON/CSV files as shared application state   | ❌ Hand-rolled, file-locking issues | ✅ Migrated to Postgres (Supabase) |

---

## 🏗️ System Architecture

```
quant-predator-v2/
├── live_trader.py          ← Main process — run this to start trading
│
├── core/
│   ├── config.py           ← All constants (NO magic numbers anywhere else)
│   ├── db.py                ← Postgres engine/session (SQLAlchemy)
│   ├── db_models.py          ← ORM models: live state, journal, eval log, run history
│   └── logger.py            ← Coloured console + rotating file logging
│
├── data/
│   ├── feature_engineer.py ← 40-feature stationary MTF matrix (H1→H4→D1)
│   └── fetcher.py          ← MT5 OHLCV data retrieval
│
├── engine/
│   ├── policy.py           ← LSTM feature extractor (custom SB3 policy)
│   ├── rl_env.py           ← Gymnasium trading environment v2.1
│   ├── trainer.py          ← Walk-forward PPO training pipeline
│   ├── backtester.py       ← Out-of-sample simulation engine
│   └── run_logger.py        ← Records each training run's config + results to Postgres
│
├── broker/
│   └── mt5_gateway.py      ← Thread-safe MT5 API serialiser
│
├── ledger/
│   ├── state_manager.py    ← Postgres-backed state persistence (live state, journal, eval log)
│   └── reconciler.py       ← Broker history reconciliation
│
├── news/
│   ├── forex_radar.py      ← ForexFactory high-impact event detector
│   └── llm_analyst.py      ← Ollama Llama3 market commentary + chat
│
├── app/
│   └── dashboard.py        ← Streamlit live trading dashboard
│
└── tools/
    └── tear_sheet.py       ← Performance metrics (Sharpe, Sortino, Calmar, expectancy)
```

**Why Postgres instead of JSON/CSV files:** the original design used flat
files as shared state between the trading engine and the dashboard, which
required hand-rolled atomic-write and retry logic to work around Windows
file-lock contention. This has been migrated to a hosted Postgres database
(Supabase), which handles concurrent reads/writes natively and — more
importantly — decouples the trading engine (which must run somewhere with
MetaTrader 5, i.e. Windows) from the dashboard (which can be deployed
independently on any host, including ephemeral/serverless platforms).

---
##  Technology Stack

- [MetaTrader 5 Python API](https://pypi.org/project/MetaTrader5/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Pandas-TA](https://github.com/twopirllc/pandas-ta)
- [SQLAlchemy](https://www.sqlalchemy.org/) + [Supabase](https://supabase.com/) (Postgres)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)
- [Ollama](https://ollama.com/) + [Llama3](https://ollama.com/library/llama3)
---
## 🚀 Installation

### Step 1 — Clone the repository

```
git clone https://github.com/mdwasifulhoque/quant-predator-v2.git
cd quant-predator-v2
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/macOS
```

### Step 2 — Install PyTorch with CUDA

```
# Check your CUDA version first:
nvidia-smi

# Install matching PyTorch (CUDA 12.1 example):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is detected:
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

### Step 3 — Install project dependencies

```
pip install -r requirements.txt
```

### Step 4 — Configure credentials

```
cp .env.template .env
# Edit .env with:
#   - your MT5 login, password, and server name
#   - DATABASE_URL (Postgres connection string )
```

### Step 5 — Install and configure Ollama (for chat interface)

```
# Download from: https://ollama.com/download
ollama pull llama3
ollama run llama3
```

### Step 6 — Prepare MetaTrader 5

- Open MT5 and log into your demo account
- Enable **Algo Trading**: Tools → Options → Expert Advisors → ✅ Allow Algorithmic Trading
- Ensure EURUSD is in your Market Watch

---

## 🧠 Training the Model

```
python engine/trainer.py
```

**What this does:**

1. Fetches 50,000 H1 EURUSD bars from MT5 (~6.25 years of data)
2. Builds the full 40-feature MTF matrix (H1 → H4 → D1)
3. Splits data: **70% train / 15% validation / 15% out-of-sample test**
4. Trains PPO with an LSTM feature extractor for **2,500,000 timesteps**
5. Evaluates on the held-out test set
6. Saves the best checkpoint (`best_model.zip`) and logs the run's config +
   results to the `training_runs` table in Postgres.

**Validation gate:** model proceeds to demo deployment only if out-of-sample
Sharpe ≥ 1.0 and max drawdown ≤ 20%. **Live-capital deployment additionally
requires 50+ live trades with Sharpe > 1.2 over a 60-day demo period** — Run
4 clears the first gate (1.090 backtest Sharpe) but has not yet cleared the
second; that validation period is the current critical path for this
project, not a formality.

---

## 🔁 Running the Backtester

```
python engine/backtester.py
```

Runs the trained model against a held-out dataset not used during training.
Outputs an equity curve CSV and prints the full tear sheet.

---

## 📊 Running the Performance Report

```
python tools/tear_sheet.py
```

Prints Sharpe (H1-annualised), Sortino, Calmar ratio, profit factor, and
expectancy.

---

## ⚡ Live Trading

```
# Terminal 1 — Start the trading engine
python live_trader.py

# Terminal 2 — Start the dashboard (separate process)
streamlit run app/dashboard.py
```

Navigate to `http://localhost:8501` to see the live dashboard.

> **⚠️ IMPORTANT:** Always run on a **demo account** for a minimum of 60 days
> and verify the model achieves 50+ trades with Sharpe > 1.2 before deploying
> real capital.

---

## 🔄 Journal Reconciliation

If the trade journal gets out of sync (connectivity dropout, manual restart):

```
python ledger/reconciler.py
```

Rebuilds the trade journal from MT5 broker history (last 90 days) against
the Postgres-backed ledger.

---

## 📡 Dashboard Features

| Panel        | Refresh  | Description                              |
| ------------ | -------- | ----------------------------------------- |
| Account HUD  | 2s       | Balance, equity, floating P&L, heartbeat |
| AI Signal    | 2s       | Current model signal + news radar status |
| Analytics    | 30s      | Win-rate and capital flow donut charts   |
| Chart        | 30s      | H1 candlestick with live SL/TP lines     |
| Trade Ledger | 30s      | Full journal with AI reasoning columns   |
| Chat         | On input | Interactive Q&A via local Llama3         |

---

## 🛡️ Risk Management

| Mechanism                  | Value                  | Configurable In                       |
| --------------------------- | ------------------------ | ---------------------------------------- |
| Risk per trade             | 1% of balance          | `TradingConfig.RISK_PER_TRADE`        |
| Volatility parity range    | 0.5%–1.5%              | `MIN/MAX_DYNAMIC_RISK`                |
| Max lot size               | 5.0 lots               | `TradingConfig.MAX_LOT_SIZE`          |
| Minimum SL distance        | max(3 pips, 0.5 × ATR) | `engine/rl_env.py` + `live_trader.py` |
| Breakeven trigger          | 1R in profit           | `TradingConfig.BREAKEVEN_R_MULT`      |
| News halt window           | 30 min before event    | `NewsConfig.HALT_HOURS_AHEAD`         |
| Daily loss circuit breaker | 3% equity drop         | `TradingConfig.DAILY_LOSS_LIMIT`      |
| Monthly drawdown limit     | 10%                    | `TradingConfig.MONTHLY_DD_LIMIT`      |

---

## 📋 Feature Engineering — 40 Stationary Features

| Category | Features | Count |
|---|---|---|
| H1 Base | ATR, RSI, MACD×3 (normalised), BB%B, BB Width, H1 Trend, EMA50/200 dist | 10 |
| Lagged Memory | RSI lag1-3, ATR lag1-3, MACD lag1-3 | 9 |
| Market Structure | Swing high/low flags, Bull/Bear FVG distance (continuous, ATR-normalized) | 4 |
| Session Context | Asian, London, NY, Overlap sessions, day of week | 5 |
| Volume & Momentum | Relative volume, ROC(3), ROC(12) | 3 |
| HTF Projections | H4 Trend, H4 EMA dist, D1 Trend, D1 EMA dist | 4 |
| Price Action Conviction | Candle body ratio, upper/lower wick ratio, session range %, volume spike | 5 |
| **Total** | | **40** |

All features are **stationary** (no absolute price levels) — the model
produces identical feature distributions regardless of the current EURUSD
price level. The switch from binary FVG presence flags to continuous
ATR-normalized zone distance (Run 4) was the single highest-impact feature
change across all training runs; see `PROJECT_BLUEPRINT.md` §5–6 for the
full before/after comparison.

---

## 🔧 Configuration Reference

All system parameters are defined in `core/config.py`. Key sections:

- `TradingConfig` — live trading risk and execution parameters
- `ModelConfig` — PPO architecture and training hyperparameters
- `RewardConfig` — reward shaping (entry cost, patience bonus, MTF gate)
- `BrokerConfig` — MT5 order settings (magic number, deviation, comment)
- `NewsConfig` — ForexFactory event detection thresholds
- `LLMConfig` — Ollama model name and timeout
- `Paths` — file paths still in use (chart data cache only — all other
  shared state now lives in Postgres, see `core/db_models.py`)

---


## 📈 Upgrade Roadmap

- **Now** — 60-day live demo validation of Run 4 (current critical path)
- **Next** — Strategy #2: a second, deliberately uncorrelated pair/approach
- **Then** — Portfolio/capital allocator (only once 2+ strategies are validated)
- **Exploration bucket (not scheduled)** — news-as-signal, strategy builder,
  multi-asset expansion, ensemble voting across seeds

---

## ⚠️ Disclaimer

*This repository is provided for educational and portfolio demonstration purposes only.
The quantitative models and execution logic do not constitute financial advice.
Trading foreign exchange on margin carries a high level of risk. Past backtest
performance does not guarantee future results. Never risk capital you cannot
afford to lose. Always validate on a demo account before live deployment.*

---


