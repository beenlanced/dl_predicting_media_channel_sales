import io
from pathlib import Path

import pandas as pd
import pytest
from src.create_plot_image import create_plot_image


# Get Path to the /src/eda folder - contains theXGBoost model and future data set to show forecasting
data_directory_path = Path(__file__).resolve().parent.parent.parent /"data"
clean_data_path = str(data_directory_path.joinpath("clean_marketing_telecom.csv"))

#Get data frame
mkt_df = pd.read_csv(clean_data_path)

# Define the test data with pytext fixtures
test_data = [
    (mkt_df),
]

@pytest.mark.parametrize("mkt_df", test_data)
def test_create_plot_image(mkt_df: pd.DataFrame) -> None :
    """
    GIVEN a Pandas dataframe of feature values and variables used as input to an Keras model
    WHEN the dataframe's sales values are available
    THEN the sale's data can be rendered as a Matplotlib plot
    """
    result = create_plot_image(mkt_df)
    assert isinstance(result, io.BytesIO)