import matplotlib.pyplot as plt
import pandas as pd

# Define the lists of columns for each group
competitors_indicators = [
    '^VIX', 'BZ=F', 'XLK', 'BTC-USD', 'IXN', 'VXUS', '^TNX', 'DX-Y.NYB', 
    '^GSPC', '^VXN', '000001.SS', '^STOXX50E', '^N225', '^GDAXI', '^GSPTSE', 
    '^MXX', '^FTSE', '^FCHI', '^HSI', '^BSESN', '^AXJO', '^KS11', '^BVSP'
]

technical_indicators = [
    'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20', 'EMA_50', 'MACD', 
    'MACD_signal', 'MACD_histogram', 'RSI_14', 'RSI_7', 'Stoch_%K', 'Stoch_%D', 
    'ROC_10', 'ROC_21', 'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR_14', 'Std_20', 
    'OBV', 'CMF_20', 'Typical_Price', 'VWAP_20', 'open', 'high', 'low', 'close', 
    'volume'
]

macro_indicators = [
    'Unemployment_Actual', 'Unemployment_Predicted', 'CPI_Actual', 'CPI_Predicted', 
    'Nonfarm_Payrolls_Actual', 'Nonfarm_Payrolls_Predicted', 'Retail_Sales_Actual', 
    'Retail_Sales_Predicted', 'Industrial_Production_Actual', 'Industrial_Production_Predicted', 
    'Consumer_Confidence_Index_Actual', 'Consumer_Confidence_Index_Predicted', 
    'Personal_Income_Actual', 'Personal_Income_Predicted', 'Unemployment_error', 
    'CPI_error', 'Nonfarm_error', 'Retail_error', 'Industrial_error', 'Consumer_error', 
    'Personal_error'
]

news_indicators = ['avg_sentiment_score']

def plot_group_indicators(data, columns, group_name, cols=3):
    """
    Plot a group of indicators in a grid of subplots with 3 columns.
    
    Parameters:
    - data: pandas DataFrame with datetime index and columns to plot
    - columns: list of column names to plot
    - group_name: name of the indicator group (e.g., 'Competitors')
    - cols: number of columns in the subplot grid (default is 3)
    """
    n = len(columns)
    rows = (n + cols - 1) // cols  # Calculate rows needed (ceiling division)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows), sharex=True)
    axes = axes.flatten()  # Flatten 2D array of axes to 1D for easier indexing
    
    # Plot each indicator in its own subplot
    for i, col in enumerate(columns):
        ax = axes[i]
        ax.plot(data.index, data[col], label=col)
        ax.set_title(col, fontsize=10)
        ax.grid(True)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Rotate x-axis labels for the last row for better readability
    for ax in axes[(rows - 1) * cols:]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add a title to the entire figure
    fig.suptitle(f"{group_name} Indicators", fontsize=16)
    plt.tight_layout