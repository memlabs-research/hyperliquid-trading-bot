# Hyperliquid Trading Bot

An automated cryptocurrency trading bot for Hyperliquid that uses machine learning predictions to execute trades at fixed intervals. The bot connects to live market data via WebSocket and makes trading decisions based on a linear regression model trained on historical price data.

## Features

- **Real-time Trading**: WebSocket connection for live price feeds
- **Predictive Model**: Linear regression model using autoregressive features
- **Scheduled Execution**: Trades at fixed intervals (e.g., hourly) synchronized to interval boundaries
- **Automatic Reconnection**: Handles connection failures with exponential backoff
- **Backtesting Framework**: Research tools for strategy development and evaluation
- **Modular Design**: Clean separation between data streaming, strategy logic, and execution

## Architecture

```
┌─────────────────┐
│   WebSocket     │ ──► Real-time price updates
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stream Module  │ ──► Log returns & lag calculation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Module   │ ──► Price prediction (LinReg)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Strategy Module │ ──► Generate trading signals
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Exchange     │ ──► Execute trades on Hyperliquid
└─────────────────┘
```

## Project Structure

```
.
├── main.py           # Application entry point
├── strategy.py       # Trading strategy implementation
├── models.py         # Machine learning models
├── stream.py         # Streaming data processing
├── research.py       # Backtesting and research tools
├── hl.py            # Hyperliquid exchange interface
├── hyperliquid-trading.yml  # Conda environment
└── README.md        # This file
```

## Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Hyperliquid account with API credentials

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd hyperliquid-trading-bot
   ```

2. **Create the conda environment**
   ```bash
   conda env create -f hyperliquid-trading.yml
   ```

3. **Activate the environment**
   ```bash
   conda activate hyperliquid-trading
   ```

4. **Set up environment variables**
   ```bash
   export HL_SECRET="your_hyperliquid_secret_key"
   export HL_WALLET="your_wallet_address"
   ```

   Or create a `.env` file:
   ```bash
   echo "HL_SECRET=your_hyperliquid_secret_key" > .env
   echo "HL_WALLET=your_wallet_address" >> .env
   ```

## Configuration

Edit the `params` dictionary in `main.py` to configure your strategy:

```python
params = {
    'sym': 'BTCUSD',        # Trading symbol
    'interval': '1h',       # Trading interval (1h, 15m, 1d, etc.)
    'model': {
        'weight': -0.0001,  # Model coefficient
        'bias': -0.0000002  # Model intercept
    }
}
```

### Strategy Parameters

- **sym**: The cryptocurrency pair to trade (e.g., 'BTCUSD', 'ETH', 'SOL')
- **interval**: Trading frequency
  - `'1m'` - Every minute
  - `'15m'` - Every 15 minutes
  - `'1h'` - Every hour
  - `'4h'` - Every 4 hours
  - `'1d'` - Daily
- **model.weight**: Linear regression coefficient (negative = mean reversion, positive = momentum)
- **model.bias**: Model intercept term

### Position Sizing

Edit `trade_sz` in the `create_strategy()` function in `main.py`:

```python
trade_sz = 0.0002  # Position size in BTC
```

## Usage

### Running the Bot

```bash
conda activate hyperliquid-trading
python main.py
```

The bot will:
1. Download historical price data to warm up the lag calculator
2. Connect to the Hyperliquid WebSocket feed
3. Execute trades at the specified interval (e.g., every hour at :00)
4. Automatically reconnect if the connection drops

### Expected Output

```
Connected to BTC stream
--- [Sync Every 1h] 14:00:00 | Price: 50123.45 | Lag: 0.0024 | Forecast: 0.0001 ---
Position closed: {'status': 'ok', ...}
Order opened: {'status': 'ok', ...}
--- [Sync Every 1h] 15:00:00 | Price: 50234.56 | Lag: 0.0022 | Forecast: -0.0003 ---
Position closed: {'status': 'ok', ...}
Order opened: {'status': 'ok', ...}
```

### Stopping the Bot

Press `Ctrl+C` to stop the bot. It will gracefully disconnect from the WebSocket.

## Research & Backtesting

Before running the bot live, use the research module to develop and test your strategy.

### Example Research Workflow

```python
import research

# 1. Create dataset with lagged features
df = research.create_ar_df(
    sym='BTC',
    interval='1h',
    start='2024-01-01',
    end='2024-12-31',
    no_lags=5
)

# 2. Test strategy with different features
features = ['close_log_return_lag_1']
results = research.eval_linreg(
    df,
    features=features,
    target='close_log_return',
    train_size=0.3  # 30% train, 70% test
)

# Output:
# Model coefficients: [-0.0001] + -0.0000002
# Win rate: 0.512
# Total return: 0.087
# (Plot of cumulative returns)

# 3. Copy model coefficients to main.py params
```

### Performance Metrics

- **Win Rate**: Percentage of profitable trades
- **Total Return**: Overall return from the strategy (as percentage)
- **Cumulative Returns Plot**: Visual representation of equity curve

## Module Documentation

### `stream.py` - Data Processing
- `Window`: Fixed-size sliding window for streaming data
- `LogReturn`: Calculate logarithmic returns from price ticks
- `Lags`: Store and retrieve lagged values

### `models.py` - Prediction Models
- `LinReg`: Linear regression model for price prediction

### `strategy.py` - Trading Logic
- `BasicTakerStrat`: Market-taking strategy using model predictions
- Automatically closes previous positions before opening new ones

### `research.py` - Backtesting Tools
- `create_ar_df()`: Create autoregressive dataset with lagged features
- `eval_linreg()`: Train and evaluate linear regression strategy

### `main.py` - Application
- WebSocket connection management
- Periodic trade execution
- Automatic reconnection with exponential backoff

## Safety & Risk Management

⚠️ **Important Considerations**:

1. **Start Small**: Begin with minimal position sizes to test the bot
2. **Paper Trading**: Test on testnet before using real funds
3. **Monitor Closely**: Keep an eye on the bot, especially initially
4. **Set Limits**: Consider implementing stop-loss and position limits
5. **API Keys**: Keep your API keys secure and never commit them to git
6. **Network**: Ensure stable internet connection for reliable operation

## Disclaimer

This software is for educational purposes only. Use at your own risk. Cryptocurrency trading carries significant risk of financial loss. The authors are not responsible for any financial losses incurred while using this software.

Always:
- Understand the code before running it
- Start with small amounts
- Never invest more than you can afford to lose
- Do your own research (DYOR)

---

**Happy Trading! 📈**