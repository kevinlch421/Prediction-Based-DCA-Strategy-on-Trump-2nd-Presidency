import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import spacy
import re
import matplotlib.pyplot as plt
from collections import Counter
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

class StockFeaturePipeline:
    def __init__(self, db_path):
        self.db_path = db_path
        self.nlp = spacy.load('en_core_web_sm')
        self.ticker_fallbacks = {
            '^VIX': 'VIX',
            'DX-Y.NYB': 'UUP',
            '000001.SS': '600036.SS',
        }
        self.financial_tickers = [
            '^VIX', 'BZ=F', 'XLK', 'BTC-USD', 'IXN', 'VXUS', '^TNX',
            'DX-Y.NYB', '^GSPC', '^VXN', '000001.SS', '^STOXX50E', '^N225',
            '^GDAXI', '^GSPTSE', '^MXX', '^FTSE', '^FCHI', '^HSI', '^BSESN',
            '^AXJO', '^KS11', '^BVSP'
        ]
        
        self.macro_features = [
            'Unemployment_Actual', 'Unemployment_Predicted',
            'CPI_Actual', 'CPI_Predicted',
            'Nonfarm_Payrolls_Actual', 'Nonfarm_Payrolls_Predicted',
            'Retail_Sales_Actual', 'Retail_Sales_Predicted',
            'Industrial_Production_Actual', 'Industrial_Production_Predicted',
            'Consumer_Confidence_Index_Actual', 'Consumer_Confidence_Index_Predicted',
            'Personal_Income_Actual', 'Personal_Income_Predicted'
        ]
        
        self.columns_to_lag = [
            'open', 'high', 'low', 'close', 'volume',
            '^VIX', 'BZ=F', 'XLK', 'BTC-USD', 'IXN', 'VXUS',
            '^TNX', 'DX-Y.NYB', '^GSPC', '^VXN', '000001.SS',
            '^STOXX50E', '^N225', '^GDAXI', '^GSPTSE', '^MXX',
            '^FTSE', '^FCHI', '^HSI', '^BSESN', '^AXJO', '^KS11', '^BVSP'
        ] 
        
    def add_target(self, df):
        df["Future_Log_Returns"] = np.log(df['close']).diff(1).shift(-1)
        return df

    class DataLoader:
        def __init__(self, db_path):
            self.db_path = db_path
            
        def load_data(self):
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT 
                s.date, s.open, s.high, s.low, s.close, s.volume, s.ticker, s.month_year,
                m.Release_Date, m.Unemployment_Actual, m.Unemployment_Predicted,
                m.CPI_Actual, m.CPI_Predicted, m.Nonfarm_Payrolls_Actual, m.Nonfarm_Payrolls_Predicted,
                m.Retail_Sales_Actual, m.Retail_Sales_Predicted, m.Industrial_Production_Actual,
                m.Industrial_Production_Predicted, m.Consumer_Confidence_Index_Actual,
                m.Consumer_Confidence_Index_Predicted, m.Personal_Income_Actual, m.Personal_Income_Predicted,
                n.titles
            FROM stock_data s
            LEFT JOIN macro_data m ON s.month_year = m.month_year
            LEFT JOIN (
                SELECT date, GROUP_CONCAT(title, ', ') AS titles, month_year, ticker
                FROM news_data
                GROUP BY date, month_year, ticker
            ) n ON s.date = n.date AND s.ticker = n.ticker
            """
            df = pd.read_sql(query, conn)
            conn.close()
            return df

    class FinancialFeatureEngineer:
        def __init__(self, tickers, fallbacks):
            self.tickers = tickers
            self.fallbacks = fallbacks
            
        def add_features(self, df, start_date, end_date):
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            
            for ticker in self.tickers:
                try:
                    data = yf.download(ticker, start=start_date, end=end_date)['Close']
                    if data.empty:
                        data = yf.download(self.fallbacks.get(ticker, ticker), 
                                         start=start_date, end=end_date)['Close']
                        
                    feature_df = data.reset_index().rename(columns={'Date': 'date', 'Close': ticker})
                    feature_df['date'] = pd.to_datetime(feature_df['date'])
                    df = pd.merge(df, feature_df, on='date', how='left')
                    
                    if df[ticker].isnull().all():
                        print(f"Warning: Failed to load data for {ticker}")
                        
                except Exception as e:
                    print(f"Error downloading {ticker}: {str(e)}")
                    df[ticker] = np.nan
                    
            return df

    class TechnicalFeatureEngineer:
        @staticmethod
        def add_features(df):
            df = df.copy()
            
            # Trend Indicators
            df['SMA_10'] = SMAIndicator(df['close'], 10).sma_indicator()
            df['SMA_20'] = SMAIndicator(df['close'], 20).sma_indicator()
            df['SMA_50'] = SMAIndicator(df['close'], 50).sma_indicator()
            df['EMA_10'] = EMAIndicator(df['close'], 10).ema_indicator()
            df['EMA_20'] = EMAIndicator(df['close'], 20).ema_indicator()
            df['EMA_50'] = EMAIndicator(df['close'], 50).ema_indicator()
            
            macd = MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_histogram'] = macd.macd_diff()
            
            # Momentum Indicators
            df['RSI_14'] = RSIIndicator(df['close'], 14).rsi()
            df['RSI_7'] = RSIIndicator(df['close'], 7).rsi()
            
            stoch = StochasticOscillator(df['high'], df['low'], df['close'], 14)
            df['Stoch_%K'] = stoch.stoch()
            df['Stoch_%D'] = stoch.stoch_signal()
            
            df['ROC_10'] = ROCIndicator(df['close'], 10).roc()
            df['ROC_21'] = ROCIndicator(df['close'], 21).roc()
                
            # Volatility Indicators
            bb = BollingerBands(df['close'], 20)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            
            df['ATR_14'] = AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
            df['Std_20'] = df['close'].rolling(20).std()
            
            # Volume Indicators
            df['OBV'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['CMF_20'] = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], 20).chaikin_money_flow()
            
            df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3
            df['VWAP_20'] = (df['Typical_Price'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            
            return df

    class MacroFeatureEngineer:
        @staticmethod
        def add_errors(df, macro_features):
            df = df.copy()
            for i in range(0, len(macro_features), 2):
                actual, pred = macro_features[i], macro_features[i+1]
                indicator = actual.split('_')[0]
                error_col = f"{indicator}_error"
                if actual in df.columns and pred in df.columns:
                    df[error_col] = df[actual] - df[pred]
                else:
                    print(f"Warning: Missing columns for {error_col} calculation")
                    df[error_col] = np.nan
            return df

    class LagFeatureEngineer:
        @staticmethod
        def add_lags(df, columns, lags):
            sorted_df = df.sort_values('date').copy()
            valid_columns = [col for col in columns if col in sorted_df.columns]
            missing = set(columns) - set(valid_columns)
            
            if missing:
                print(f"Missing columns for lagging: {', '.join(missing)}")
                print("Available columns:", list(sorted_df.columns))
    
            lagged_features = {}
            for col in valid_columns:
                if sorted_df[col].isnull().all():
                    print(f"Skipping lagging for empty column: {col}")
                    continue
                for lag in lags:
                    new_col = f'{col}_lag{lag}'
                    lagged_features[new_col] = sorted_df[col].shift(lag)
    
            if lagged_features:
                lagged_df = pd.DataFrame(lagged_features)
                return pd.concat([sorted_df, lagged_df], axis=1)
            return sorted_df

    class NLPProcessor:
        def __init__(self, nlp):
            self.nlp = nlp
            self.top_n = 10
            
        def process_text(self, df):
            df = df.copy()
            df['titles'] = df['titles'].fillna('')
            df['preprocessed'] = df['titles'].str.lower()
            texts = df['preprocessed'].tolist()
            processed_docs = list(self.nlp.pipe(texts))
            df['processed_title'] = [
                ' '.join([token.lemma_ for token in doc if not token.is_stop])
                for doc in processed_docs
            ]
            df['processed_title'] = df['processed_title'].str.rstrip()
            df['processed_title'] = df['processed_title'].apply(
                lambda x: x + '.' if x and not x.endswith('.') else x
            )
            all_tokens = []
            for title in df['processed_title'].dropna():
                all_tokens.extend(re.findall(r'\w+|\.', title))
            word_tokens = [token for token in all_tokens if token.isalnum()]
            word_freq = Counter(word_tokens)
            unique_words = len(word_freq)
            N = max(1, int(0.01 * unique_words))
            high_freq_words = set(word for word, freq in word_freq.most_common(N))
            
            def filter_text(text):
                tokens = re.findall(r'\w+|\.', text)
                return ' '.join([
                    token for token in tokens
                    if not (token.isalnum() and token in high_freq_words)
                ])
            
            df['filtered_title'] = df['processed_title'].apply(filter_text)
            self._plot_word_frequencies(df)
            self._print_sample_comparison(df)
            df = df.drop(columns=['preprocessed', 'processed_title', 'titles'])
            return df.rename(columns={"filtered_title": "titles"})
        
        def _plot_word_frequencies(self, df):
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            all_text_before = ' '.join(df['processed_title'].dropna())
            words_before = re.findall(r'\w+', all_text_before)
            top_words_before = Counter(words_before).most_common(self.top_n)
            axes[0].bar(
                [w[0] for w in top_words_before],
                [w[1] for w in top_words_before],
                color='steelblue',
                edgecolor='black'
            )
            axes[0].set_title("Top Words Before Filtering")
            all_text_after = ' '.join(df['filtered_title'].dropna())
            words_after = re.findall(r'\w+', all_text_after)
            top_words_after = Counter(words_after).most_common(self.top_n)
            axes[1].bar(
                [w[0] for w in top_words_after],
                [w[1] for w in top_words_after],
                color='steelblue',
                hatch='/',
                edgecolor='black'
            )
            axes[1].set_title("Top Words After Filtering")
            plt.tight_layout()
            plt.savefig('word_frequency_comparison.png')
            plt.close()
        
        def _print_sample_comparison(self, df):
            pass

    class SentimentAnalyzer:
        def __init__(self):
            nltk.download('vader_lexicon', quiet=True)
            self.analyzer = SentimentIntensityAnalyzer()
            
        def add_sentiment(self, df):
            df = df.copy()
            df['sentiment_score'] = df['titles'].apply(
                lambda x: self.analyzer.polarity_scores(str(x))['compound']
            )
            return df

    class DataProcessor:
        def process_final(self, df, start_date, end_date):
            df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')
            sentiment_df = df.groupby('month_year').agg({
                'titles': lambda x: ' '.join(x.dropna()),
                'sentiment_score': 'mean'
            }).reset_index().rename(columns={'sentiment_score': 'avg_sentiment_score'})
            final_df = pd.merge(
                df,
                sentiment_df,
                on='month_year',
                how='left',
                suffixes=('', '_agg')
            )
            final_df['date'] = pd.to_datetime(final_df['date'])
            final_df.set_index('date', inplace=True)
            final_df = final_df.loc[start_date:end_date]
            cols_to_drop = ['ticker', 'Release_Date', 'titles', 'sentiment_score']
            final_df = final_df.drop(
                columns=[c for c in cols_to_drop if c in final_df.columns]
            )
            return final_df.resample('M').last().dropna(how='all')

    def run_pipeline(self, start_date, end_date):
        loader = self.DataLoader(self.db_path)
        financial_engineer = self.FinancialFeatureEngineer(
            self.financial_tickers,
            self.ticker_fallbacks
        )
        technical_engineer = self.TechnicalFeatureEngineer()
        macro_engineer = self.MacroFeatureEngineer()
        lag_engineer = self.LagFeatureEngineer()
        nlp_processor = self.NLPProcessor(self.nlp)
        sentiment_analyzer = self.SentimentAnalyzer()
        data_processor = self.DataProcessor()

        df = loader.load_data()
        df = financial_engineer.add_features(df, start_date, end_date)
        df = technical_engineer.add_features(df)
        df = macro_engineer.add_errors(df, self.macro_features)
        df = lag_engineer.add_lags(df, self.columns_to_lag, [1, 10, 42, 63])
        df = nlp_processor.process_text(df)
        df = sentiment_analyzer.add_sentiment(df)
        final_df = data_processor.process_final(df, start_date, end_date)
        
        # Impute missing values with mean for numerical columns
        numerical_cols = final_df.select_dtypes(include=['float64', 'int64']).columns
        final_df[numerical_cols] = final_df[numerical_cols].fillna(final_df[numerical_cols].mean())
        
        final_df = self.add_target(final_df)  # Integrate the target feature here
        
        print("\nFinal DataFrame Info:")
        print(final_df.info())
        
        return final_df