from pathlib import Path

import keras
from keras import utils
import numpy as np


# This script was used to test loading a prediction model and testing single instance

# Test budget values from the validated request data
digital_budget = 345.15
tv_budget = 156.0
radio_budget = 37.8
newspaper_budget = 69.2

PARENT_DIR = Path(__file__).resolve().parent
model_file=str(Path(PARENT_DIR, "prediction_model.keras"))

# Retrieve the prediction model
model = keras.models.load_model(model_file)
prediction_data = [digital_budget, tv_budget, radio_budget, newspaper_budget]

# Normalize data 
normalized_feature =  utils.normalize(prediction_data)
print(f"normalized_features = {normalized_feature}")

# --- Prepare the single array of data to be used for prediction ---
# Keras models typically expect input data in batches, even when predicting on a single instance.
# Therefore, the single array representing the data point must be reshaped to include a 
# batch dimension. This can be achieved using np.expand_dims() or by adding a new axis with np.newaxis.

input_array = np.array(normalized_feature)
input_data = np.expand_dims(input_array, axis=0)

# Get the output from the model for the prepared input data
prediction =  model.predict(input_data)
print(f"prediction result = {prediction[0][0]}")
