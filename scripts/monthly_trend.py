from typing import Dict, Optional
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def plot_monthly_trends(df, date_column, aggregation_column: Dict[str, str], title_map=None):
    """
    Plots monthly trends of a specified column in a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - date_column: string, name of the column with date information.
    - aggregation_column: string, name of the column to aggregate.
    - title: string, title of the plot.
    """
    try:
        # Aggregate monthly data
        monthly_data = aggregate_monthly_trends(
            df, date_column, aggregation_column)
        length = len(aggregation_column)
        fig, axes = plt.subplots(
            length, 1, figsize=(10, 5 * length), sharex=True)
        axes = axes if isinstance(axes, np.ndarray) else [axes]
        for i, (col, _) in enumerate(aggregation_column.items()):
            sns.lineplot(monthly_data, x="Month", y=col, ax=axes[i])
            title = title_map.get(col, f"Monthly Trend of {col}")
            axes[i].set_title(title)
            axes[i].set_ylabel(col)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True)
        plt.xlabel("Month")
        plt.tight_layout()
        plt.show()
        logging.info("Monthly trends plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting monthly trends: {e}")
        return None


def aggregate_monthly_trends(df: pd.DataFrame, date_column: str, aggregation_column: Dict[str, str], title: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Aggregates monthly trends of a specified column in a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - date_column: string, name of the column with date information.
    - aggregation_column: string, name of the column to aggregate.

    Returns:
    - pandas Series with monthly aggregated data.
    """
    try:
        # Ensure date_column is in datetime format
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df['Month'] = df[date_column].dt.to_period('M')
        grouped_by = df.groupby('Month').agg(aggregation_column).reset_index()
        grouped_by['Month'] = grouped_by['Month'].dt.to_timestamp()
        grouped_by.set_index('Month', inplace=True)
        logging.info(
            f"Monthly aggregation successful for {aggregation_column}.")
    except Exception as e:
        logging.error(f"Error during monthly aggregation: {e}")
        return None

    return grouped_by
