import io

import matplotlib

# displaying the image in the user's browser on client side with matplotlib.use
# Use the "Agg" backend for non-GUI environments
# AGG used in the example above is a backend that renders graphs as PNGs.
# matplotlib.use() must be used before importing pyplot so placing it here
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_plot_image(mkt_df: pd.DataFrame) -> io.BytesIO:
    """
    Method to generate a plot of telecom media channel sales data

    Args:
        mkt_df (pd.DataFrame): Pandas dataframe of telecom media channel sales data

        model_path (str): File path to a derived XGBoost model.

    Returns:
        (io.BytesIO): Matplotlib image out of the price sales data inputs.
    """
    mkt_df = mkt_df.sort_values(by="start_date")
    df = mkt_df.set_index("start_date")
    df.index = pd.to_datetime(df.index)

    color_palette = sns.color_palette("tab10") #seaborn color palette
    df["sales"].plot(figsize=(10, 5),
                                    color=color_palette[0],
                                    ms=1, lw=1,
                                    xlabel="Dates",
                                    ylabel="Sales in USD Dollars",
                                    title="Plot of Telecom Media Channel Sales Data")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close()
    return img_buf