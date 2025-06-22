from pathlib import Path
from typing import Dict, Any

from create_plot_image import create_plot_image
from fastapi import BackgroundTasks, HTTPException, FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import keras
from keras import utils
import matplotlib
import numpy as np

# displaying the image in the user's browser on client side with matplotlib.use
# Use the "Agg" backend for non-GUI environments
# AGG used in the example above is a backend that renders graphs as PNGs.
# matplotlib.use() must be used before importing pyplot so placing it here
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import uvicorn


# Get Base path
BASE_DIR = Path(__file__).resolve().parent.parent
template_path=str(Path(BASE_DIR, "templates"))

# Set up the application
app = FastAPI(
    title="Media Channel Sales Prediction App",
    description="A simple FastAPI backend to serve an HTML frontend that provides sales predictions."
)
templates = Jinja2Templates(directory=template_path) # Create a "templates" directory

# The pre-existing keras derived prediction model
model_file = "prediction_model.keras"

@app.get("/info")
def info() -> dict[str, str]:
    """
    FASTAPI route that returns simple dictionary/JSON message to show site

    Returns:
        (dict[str, str]): Dict with text describing he purpose of the Fast API web application
    """
    return {"name": "Predicting Media Channel Sales", "description": "The goal of the project is take raw marketing telecommunication data from a fictitious company and build a predictive model for media sales. Essentially, this is a typical machine learning project which gets an extra zhuzh by building the model using a deep learning/neural network model and making a web accessible application to provide access to the predicative model."}

# async def allows you to write efficient, responsive programs, 
# especially for I/O-bound tasks, by enabling the program to perform 
# other work while waiting for slow operations to finish.
@app.get("/plot-sales")
async def get_plot(background_tasks: BackgroundTasks) -> Response:
    """
    Method to Plot Matplotlib image of Existing sales data

    Args:
        background_tasks (BackgroundTasks): Runs task to close image buffer after sending image

    Returns:
        (Response): Returns an image that is rendered in the this FAST API route
    """
    try:
         BASE_DATA_PATH = Path(__file__).resolve().parent.parent /"data"
         clean_data_path = str(Path(BASE_DATA_PATH, "clean_marketing_telecom.csv"))
         mkt_df = pd.read_csv(clean_data_path)

    except FileNotFoundError as e:
        # If parsing fails, raise an HTTP exception
        raise HTTPException(status_code=500, detail=f"Data File not found")
    
    except Exception as e:
        # Any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
    img_buf = create_plot_image(mkt_df)
    background_tasks.add_task(img_buf.close) # Ensure the buffer is closed after sending

    # Set the Content-Disposition header, so that the image can be viewed in the browser
    headers = {"Content-Disposition": 'inline; filename="plot.png"'}
    return Response(img_buf.getvalue(), media_type="image/png", headers=headers)

# Serve the prediction_template HTML file at the root directory
@app.get("/", response_class=HTMLResponse, summary="Serve the Sales Prediction HTML page")
async def read_root(request: Request) -> HTMLResponse:
    """
    Serves the `index.html` file using Jinja2Templates when the root URL is accessed.
    This requires `index.html` to be placed inside a 'templates' subdirectory.
    The `request` object is required by Jinja2Templates.

    Args:
        request (Request): HTML request

    Returns:
        (HTMLResponse):  the index.html file page
    """
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint that handles POST requests from the HTML form
@app.post("/predict_sales", summary="Predict sales based on marketing budgets")
async def predict_sales_endpoint(
    request: Request, # Expects a JSON body matching PredictionRequest model
) -> Dict[str, float]:
    """
    Receives marketing budget data and prediction model and then returns the prediction.

    Args:
        request_data (PredictionRequest): JSON Body of data matching Class: PredictionRequest model

    Returns:
        Dict[str, float]: A dictionary containing the predicted sales. Example: {"predicted_sales": 123456.78}
    """
    try:
        # Get the JSON data from the request body
        request_data: Dict[str, Any] = await request.json()

        # Extract budget values, defaulting to 0 if not provided or invalid
        digital_budget: float = float(request_data.get('digital', 0.0))
        tv_budget: float = float(request_data.get('tv', 0.0))
        radio_budget: float = float(request_data.get('radio', 0.0))
        newspaper_budget: float = float(request_data.get('newspaper', 0.0))

        # Retrieve the prediction model
        PARENT_DIR = Path(__file__).resolve().parent
        model_file=str(Path(PARENT_DIR, "prediction_model.keras"))

        model = keras.models.load_model(model_file)
        prediction_data = [digital_budget, tv_budget, radio_budget, newspaper_budget]

        # Normalize data 
        normalized_feature =  utils.normalize(prediction_data)
    
        input_array = np.array(normalized_feature)
        input_data = np.expand_dims(input_array, axis=0)

        # Get the output from the model for the prepared input data
        prediction =  model.predict(input_data)

        # Return the prediction as a JSON response
        return {"predicted_sales": prediction}
    
    except ValueError:
        # Handle cases where input data cannot be converted to float
        raise HTTPException(
            status_code=400,
            detail="Invalid budget values provided. Please ensure they are numbers."
        )
    except Exception as e:
        # Catch any other unexpected errors and return a 500 status
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {str(e)}"
        )