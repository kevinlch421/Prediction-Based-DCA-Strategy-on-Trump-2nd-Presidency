# Prediction-Based DCA on Trump’s Second Presidency

## Overview

**Prediction-Based Dollar-Cost Averaging (DCA) on Trump’s Second Presidency** is an end-to-end machine learning project focused on forecasting QQQ ETF returns during a hypothetical second Trump presidency. The project integrates price, macroeconomic, and news sentiment data to inform an adaptive DCA strategy, aiming to outperform traditional approaches in politically volatile periods.

---

## Academic & Professional Contribution

- **Academic Value**:  
  Presents a reproducible framework for integrating structured market and macro data with unstructured news sentiment for financial time series prediction.

- **Professional Value**:  
  Demonstrates modern data science and engineering best practices, including data integration, feature engineering, deep learning model development, and robust validation.  
  Designed for extensibility and real-world application.

---

---

## Installation

1. **Clone this repository**
    ```bash
    git clone https://github.com/yourusername/prediction-based-dca-trump.git
    cd prediction-based-dca-trump
    ```

2. **(Recommended) Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate         # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    Or install manually if no requirements.txt:
    ```bash
    pip install pandas numpy scikit-learn torch sqlite3 pandas_ta vaderSentiment
    ```

---

## Usage

1. **Prepare your environment** (see above).

2. **Run the notebooks in order:**
    - `notebooks/01_Create_Database.ipynb`: Integrate all raw data into SQLite.
    - `notebooks/02_EDA_And_Market_Insight.ipynb`: Perform exploratory analysis.
    - `notebooks/03_Feature_Construction.ipynb`: Generate technical, macro, and sentiment features.
    - `notebooks/04_ML_Models.ipynb`: Train and evaluate return prediction models.

3. **Review the report**  
   See `/report/[FYP-21229570]-Prediction-Based-DCA-on-Trump-Second-Presidency.doc` for methodology, results, and discussion.

---

## Data Sources

- **Price Indicators**:  
  QQQ and related assets (2017–2025) from Yahoo Finance.

- **Macroeconomic Indicators**:  
  Historical macro data (2017–2024) from [Investing.com], with 2025 data scraped.

- **News Headlines**:  
  15.7M headlines (1999–2023) from FNSPID, plus 2025 headlines scraped from [FINVIZ.com] for sentiment analysis.

All data is integrated in `/DataBase/stock_price.sqlite`.

---

## Methodology

- **Database Design**:  
  Integrates all sources for unified querying.

- **EDA & Preprocessing**:  
  Statistical exploration, missing value treatment, text tokenization.

- **Feature Engineering**:  
  - Technical: RSI, VIX, moving averages, lags
  - Macro: Unemployment, inflation, etc.
  - Sentiment: VADER scores from headlines
  - Selection: RFECV, log/Z-score normalization

- **Modeling**:  
  Linear Regression, Vanilla LSTM, Bidirectional LSTM, CNN-LSTM

- **Evaluation**:  
  - MSE
  - 5-fold cross-validation
  - Backtesting of investment strategies

---

## Results

- **Best Model**:  
  Fine-tuned Bidirectional LSTM, MSE: **0.0568**

- **Investment Performance**:
    - Prediction-Based DCA: **15.18%** return
    - Adaptive DCA: 14.67%
    - Enhanced DCA: 15.12%

See the report for full results and backtesting from Trump’s first presidency.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Contact

Leung Cheuk Him  
[your.email@example.com] *(update before sharing)*

---

## Notes for Employers

- End-to-end, modular data science pipeline
- Modern ML (deep learning, NLP, time series)
- Clean, reproducible, and extensible codebase
- Suitable for research, finance, and production environments
