# 🤖 Quant Predator v2 — Institutional Algorithmic Trading Terminal

> A fully autonomous, Multi-Timeframe PPO Reinforcement Learning trading system
> built on MetaTrader 5, Stable-Baselines3, and local Ollama Llama3.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![MT5](https://img.shields.io/badge/MetaTrader_5-API-black?style=for-the-badge)
![SB3](https://img.shields.io/badge/RL-Stable--Baselines3-purple?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Ollama](https://img.shields.io/badge/LLM-Ollama_Llama3-orange?style=for-the-badge)

---

## ⚡ What's New in v2 (vs original)

| Bug / Weakness | v1 Status | v2 Fix |
|---|---|---|
| Fixed episode start (model memorised 1 path) | ❌ Always bar 124 | ✅ Random start per episode |
| Lookahead bias in swing detection | ❌ `center=True` | ✅ `center=False` confirmed |
| 10 non-stationary absolute price features | ❌ Raw EMA/BB levels | ✅ All stationary (% distance) |
| Open bar used for inference | ❌ `iloc[-1]` open bar | ✅ `iloc[-2]` closed bar |
| Sharpe ratio overstated 5× | ❌ `sqrt(252)` daily | ✅ `sqrt(6240)` H1-correct |
| Floating P&L hardcoded to 0.0 in backtest | ❌ Silent obs corruption | ✅ Computed each step |
| 3-pip fixed SL floor caused giant lots | ❌ `MIN_SL = 0.00030` | ✅ `max(0.00030, ATR × 0.5)` |
| MT5 not thread-safe — raw calls everywhere | ❌ No serialisation | ✅ MT5Gateway queue pattern |
| Race condition: state.json partial writes | ❌ Plain `json.dump()` | ✅ Atomic `os.replace()` |
| Position lost on bot restart | ❌ Always `position=0` | ✅ `_recover_position_state()` |
| SL/TP offline hits never logged | ❌ Lost forever | ✅ Detected on next startup |
| AI reasoning columns all zero on close | ❌ Hardcoded zeros | ✅ Entry indicators cached |
| Magic number split-brain (100100 vs 777777) | ❌ Two different values | ✅ Single `BrokerConfig.MAGIC_NUMBER` |
| MLP 64×64 on 890-dim input | ❌ 14:1 bottleneck | ✅ LSTM(128) + MLP[256,128] |
| No train/test split | ❌ Same data for all | ✅ 70/15/15 temporal split |

---

## 🏗️ System Architecture

```
quant-predator-v2/
├── live_trader.py          ← Main process — run this to start trading
│
├── core/
│   ├── config.py           ← All constants (NO magic numbers anywhere else)
│   └── logger.py           ← Coloured console + rotating file logging
│
├── data/
│   ├── feature_engineer.py ← 35-feature stationary MTF matrix (H1→H4→D1)
│   └── fetcher.py          ← MT5 OHLCV data retrieval
│
├── engine/
│   ├── policy.py           ← LSTM feature extractor (custom SB3 policy)
│   ├── rl_env.py           ← Gymnasium trading environment v2
│   ├── trainer.py          ← Walk-forward PPO training pipeline
│   └── backtester.py       ← Out-of-sample simulation engine
│
├── broker/
│   └── mt5_gateway.py      ← Thread-safe MT5 API serialiser
│
├── ledger/
│   ├── state_manager.py    ← Atomic JSON/CSV state persistence
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
    └── tear_sheet.py       ← Institutional performance metrics
```

---

## 🚀 Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourusername/quant-predator-v2.git
cd quant-predator-v2
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
```

### Step 2 — Install PyTorch with CUDA (RTX 3050)
```bash
# Check your CUDA version first:
nvidia-smi

# Install matching PyTorch (CUDA 12.1 example):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is detected:
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
# Expected: CUDA: True | NVIDIA GeForce RTX 3050 Laptop GPU
```

### Step 3 — Install project dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Configure credentials
```bash
cp .env.template .env
# Edit .env with your MT5 login, password, and server name
```

### Step 5 — Install and configure Ollama (for chat interface)
```bash
# Download from: https://ollama.com/download
# After installation, pull the model:
ollama pull llama3

# Verify it works:
ollama run llama3 "Say hello"
```

### Step 6 — Prepare MetaTrader 5
- Open MT5 and log into your demo account
- Enable **Algo Trading**: Tools → Options → Expert Advisors → ✅ Allow Algorithmic Trading
- Ensure EURUSD is in your Market Watch

---

## 🧠 Training the Model

```bash
# Ensure MT5 is open and logged in, then:
python engine/trainer.py
```

**What this does:**
1. Fetches 50,000 H1 EURUSD bars from MT5 (~6.25 years of data)
2. Builds the full 35-feature MTF matrix (H1 → H4 → D1)
3. Splits data: **70% train / 15% validation / 15% out-of-sample test**
4. Trains PPO with LSTM feature extractor for **1,500,000 timesteps**
5. Evaluates on the held-out test set
6. Saves model to `models/eurusd/ppo_model.zip` only if validation passes

**Estimated time on RTX 3050:** ~45–70 minutes

**Validation gate:** Model proceeds to deployment only if:
- Out-of-sample Sharpe ratio ≥ 1.0
- Maximum drawdown ≤ 20%

---

## 🔁 Running the Backtester

```bash
python engine/backtester.py
```

Runs the trained model against a **fresh 15,000-bar dataset not used during training**.
Outputs an equity curve CSV and prints the full tear sheet.

---

## 📊 Running the Performance Report

```bash
python tools/tear_sheet.py
```

Prints institutional-grade metrics including Sharpe (H1-annualised), Sortino,
Calmar ratio, profit factor, and expectancy.

---

## ⚡ Live Trading

```bash
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

```bash
python ledger/reconciler.py
```

Rebuilds `app/state/trade_journal.csv` from MT5 broker history (last 90 days).
Creates a timestamped backup before overwriting.

---

## 📡 Dashboard Features

| Panel | Refresh | Description |
|---|---|---|
| Account HUD | 2s | Balance, equity, floating P&L, heartbeat |
| AI Signal | 2s | Current model signal + news radar status |
| Analytics | 10s | Win-rate and capital flow donut charts |
| Chart | 30s | H1 candlestick with live SL/TP lines |
| Trade Ledger | 30s | Full journal with AI reasoning columns |
| Chat | On input | Interactive Q&A via local Llama3 |

---

## 🛡️ Risk Management

| Mechanism | Value | Configurable In |
|---|---|---|
| Risk per trade | 1% of balance | `TradingConfig.RISK_PER_TRADE` |
| Volatility parity range | 0.5%–1.5% | `MIN/MAX_DYNAMIC_RISK` |
| Max lot size | 5.0 lots | `TradingConfig.MAX_LOT_SIZE` |
| Minimum SL distance | max(3 pips, 0.5 × ATR) | `engine/rl_env.py` + `live_trader.py` |
| Breakeven trigger | 1R in profit | `TradingConfig.BREAKEVEN_R_MULT` |
| News halt window | 30 min before event | `NewsConfig.HALT_HOURS_AHEAD` |
| Daily loss circuit breaker | 3% equity drop | `TradingConfig.DAILY_LOSS_LIMIT` |
| Monthly drawdown limit | 10% | `TradingConfig.MONTHLY_DD_LIMIT` |

---

## 📋 Feature Engineering — 35 Stationary Features

| Category | Features | Count |
|---|---|---|
| H1 Base | ATR, RSI, MACD×3 (normalised), BB%B, BB Width, H1 Trend, EMA50/200 dist | 10 |
| Lagged Memory | RSI lag1-3, ATR lag1-3, MACD lag1-3 | 9 |
| Market Structure | Swing high/low flags, Bull/Bear FVG active zones | 4 |
| Session Context | Asian, London, NY, Overlap sessions, day of week | 5 |
| Volume & Momentum | Relative volume, ROC(3), ROC(12) | 3 |
| HTF Projections | H4 Trend, H4 EMA dist, D1 Trend, D1 EMA dist | 4 |
| **Total** | | **35** |

All features are **stationary** (no absolute price levels). The model produces
identical feature distributions regardless of the current EURUSD price level.

---

## 🔧 Configuration Reference

All system parameters are defined in `core/config.py`. Key sections:

- `TradingConfig` — live trading risk and execution parameters
- `ModelConfig`   — PPO architecture and training hyperparameters
- `BrokerConfig`  — MT5 order settings (magic number, deviation, comment)
- `NewsConfig`    — ForexFactory event detection thresholds
- `LLMConfig`     — Ollama model name and timeout
- `Paths`         — All file paths (centralised, no hardcoded strings)

---

## 📈 Upgrade Roadmap

- **Phase 4** — XAUUSD second model (separate architecture per pair)
- **Phase 5** — Multi-pair market scanner with capital allocation manager
- **Phase 6** — FastAPI WebSocket backend + React frontend (via Lovable.dev)
- **Phase 7** — Ensemble voting (5 seeds per pair, majority decision)

---

## ⚠️ Disclaimer

*This repository is provided for educational and portfolio demonstration purposes only.
The quantitative models and execution logic do not constitute financial advice.
Trading foreign exchange on margin carries a high level of risk. Past backtest
performance does not guarantee future results. Never risk capital you cannot
afford to lose. Always validate on a demo account before live deployment.*

---

## 🙏 Technology Stack

- [MetaTrader 5 Python API](https://pypi.org/project/MetaTrader5/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Pandas-TA](https://github.com/twopirllc/pandas-ta)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)
- [Ollama](https://ollama.com/) + [Llama3](https://ollama.com/library/llama3)
