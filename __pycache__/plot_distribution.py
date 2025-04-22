import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_competitors_distribution(df, competitors_indicators, bins=30, cols_per_figure=3, rows_per_figure=4):
    """
    Plot histogram distribution bar plots for the specified competitors_indicators columns.
    
    Parameters:
    - df: pandas DataFrame containing the competitors_indicators columns
    - competitors_indicators: list of column names to plot
    - bins: number of bins for the histogram (default=30)
    - cols_per_figure: number of columns in each subplot grid (default=3)
    - rows_per_figure: number of rows in each subplot grid (default=4)
    
    Returns:
    - None (saves plots to files)
    """
    # Filter columns that exist in the DataFrame
    valid_cols = [col for col in competitors_indicators if col in df.columns]
    missing_cols = set(competitors_indicators) - set(valid_cols)
    if missing_cols:
        print(f"Warning: The following columns are missing: {missing_cols}")
    
    if not valid_cols:
        print("No valid competitors_indicators columns found in the DataFrame.")
        return
    
    # Calculate number of plots and figures needed
    plots_per_figure = cols_per_figure * rows_per_figure
    num_figures = (len(valid_cols) + plots_per_figure - 1) // plots_per_figure
    
    for fig_num in range(num_figures):
        start_idx = fig_num * plots_per_figure
        end_idx = min((fig_num + 1) * plots_per_figure, len(valid_cols))
        current_cols = valid_cols[start_idx:end_idx]
        
        if not current_cols:
            break
        
        # Create a new figure
        fig, axes = plt.subplots(rows_per_figure, cols_per_figure, figsize=(15, 4 * rows_per_figure))
        axes = axes.flatten()  # Flatten for easier indexing
        
        for i, col in enumerate(current_cols):
            ax = axes[i]
            # Plot histogram for the column
            ax.hist(df[col].dropna(), bins=bins, color='skyblue', edgecolor='black')
            ax.set_title(col, fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.show()
    
    print(f"Distribution plots saved as 'competitors_distribution_figX.png' for {len(valid_cols)} columns.")

# Example usage
# Assuming `outlier_df` is your DataFrame
# For testing, let's create a sample DataFrame
"""
outlier_df = pd.DataFrame({
    '^VIX': np.random.normal(20, 5, 100),
    'BZ=F': np.random.normal(80, 10, 100),
    'XLK': np.random.normal(150, 20, 100),
    # Add more columns as needed
})
"""

# List of competitors indicators
competitors_indicators = [
    '^VIX', 'BZ=F', 'XLK', 'BTC-USD', 'IXN', 'VXUS', '^TNX', 'DX-Y.NYB', 
    '^GSPC', '^VXN', '000001.SS', '^STOXX50E', '^N225', '^GDAXI', '^GSPTSE', 
    '^MXX', '^FTSE', '^FCHI', '^HSI', '^BSESN', '^AXJO', '^KS11', '^BVSP'
]

# Apply the plotting function
# plot_competitors_distribution(outlier_df, competitors_indicators)