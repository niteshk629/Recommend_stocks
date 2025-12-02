# Stock Recommendation System for Indian Markets

An AI/ML-driven stock recommendation system specifically tailored for the Indian stock market (NSE and BSE). This system analyzes historical stock data, fundamental financial data, and news sentiment to provide actionable buy/sell/hold recommendations for mid-to-long term investment horizons.

## Features

- **Multi-Source Data Analysis**
  - Historical stock price data from NSE/BSE
  - Fundamental financial metrics (PE ratio, ROE, debt-to-equity, etc.)
  - News sentiment analysis

- **Technical Analysis**
  - Moving Averages (SMA, EMA)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - ATR (Average True Range)
  - Volume indicators

- **Fundamental Analysis**
  - Valuation metrics (PE, PB, PS ratios)
  - Profitability metrics (ROE, ROA, profit margins)
  - Financial health indicators (debt-to-equity, current ratio)
  - Growth metrics

- **Machine Learning Models**
  - Random Forest Classifier
  - XGBoost Classifier
  - Feature engineering pipeline
  - Model evaluation and feature importance

- **Recommendation Engine**
  - Composite scoring system
  - Five-level recommendations: Strong Buy, Buy, Hold, Sell, Strong Sell
  - Confidence levels for each recommendation
  - Portfolio-level recommendations

## Project Structure

```
Recommend_stocks/
├── config/
│   └── config.yaml           # Configuration file
├── src/
│   ├── data_collection/      # Data fetching modules
│   ├── fundamental_analysis/ # Fundamental metrics calculation
│   ├── sentiment_analysis/   # News sentiment analysis
│   ├── feature_engineering/  # Technical indicators
│   ├── models/              # Machine learning models
│   ├── recommendation_engine/# Recommendation logic
│   └── utils/               # Utility functions
├── data/                    # Data storage
├── tests/                   # Test files
├── notebooks/              # Jupyter notebooks
├── main.py                 # Main entry point
└── requirements.txt        # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/niteshk629/Recommend_stocks.git
cd Recommend_stocks
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The system uses a YAML configuration file located at `config/config.yaml`. You can customize:

- Stock universe (NSE/BSE stocks to analyze)
- Data collection settings (historical period, data sources)
- Technical indicators to calculate
- Model parameters (Random Forest, XGBoost, LSTM)
- Recommendation thresholds
- Investment horizon and risk tolerance

## Usage

### Basic Usage

Run the recommendation system with default settings:

```bash
python main.py
```

### Custom Configuration

Specify a custom configuration file:

```bash
python main.py --config path/to/config.yaml
```

### Analyze Specific Stocks

Analyze specific stock symbols:

```bash
python main.py --symbols RELIANCE.NS TCS.NS INFY.NS
```

### Save Recommendations

Save recommendations to a specific file:

```bash
python main.py --output results/my_recommendations.csv
```

## Output

The system generates:

1. **Console Output**: Summary of recommendations with key metrics
2. **CSV File**: Detailed recommendations with scores and confidence levels
3. **Log File**: Detailed execution logs in `logs/stock_recommender.log`

### Sample Output

```
Top 10 Stock Recommendations:
Symbol         Recommendation  Confidence  Composite Score  Technical  Fundamental  Sentiment
RELIANCE.NS    STRONG BUY     HIGH        0.812           0.75       0.85         0.85
TCS.NS         BUY            MEDIUM      0.687           0.65       0.70         0.72
INFY.NS        BUY            HIGH        0.702           0.68       0.72         0.70
...
```

## Stock Universe

The default configuration includes top NSE stocks:
- RELIANCE, TCS, HDFCBANK, INFY, HINDUNILVR
- ICICIBANK, SBIN, BHARTIARTL, ITC, KOTAKBANK
- LT, AXISBANK, WIPRO, MARUTI, TATAMOTORS
- HCLTECH, BAJFINANCE, SUNPHARMA, ONGC, TITAN

You can modify the stock universe in `config/config.yaml`.

## Recommendation Levels

- **STRONG BUY**: Composite score ≥ 0.75 - High confidence opportunity
- **BUY**: Composite score ≥ 0.60 - Good investment opportunity
- **HOLD**: Composite score ≥ 0.40 - Maintain current position
- **SELL**: Composite score ≥ 0.25 - Consider reducing position
- **STRONG SELL**: Composite score < 0.25 - High confidence to exit

## Methodology

The system uses a three-pillar approach:

1. **Technical Analysis (40%)**: Price patterns, momentum, volatility
2. **Fundamental Analysis (40%)**: Financial health, valuation, profitability
3. **Sentiment Analysis (20%)**: News sentiment, market mood

Each pillar generates a score (0-1), which are combined using weighted averaging to produce a composite score. The recommendation is then derived based on configurable thresholds.

## Machine Learning

The system includes ML models that can be trained on historical data to predict future price movements:

- Feature extraction from technical indicators
- Training on historical price movements
- Cross-validation and performance evaluation
- Model persistence for reuse

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- yfinance (for stock data)
- xgboost, tensorflow (for ML models)
- textblob, nltk (for sentiment analysis)
- See `requirements.txt` for complete list

## Limitations

- News sentiment analysis uses placeholder implementation (integrate with real news APIs in production)
- Historical data limited to what's available through yfinance
- Models require training before use in production
- Not suitable for high-frequency trading
- Past performance doesn't guarantee future results

## Disclaimer

This system is for educational and research purposes only. It should not be considered as financial advice. Always conduct your own research and consult with financial advisors before making investment decisions. The authors are not responsible for any financial losses incurred from using this system.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Contact

For questions or feedback, please open an issue on GitHub.
