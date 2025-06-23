from fastapi.testclient import TestClient

from src.app import app


client = TestClient(app)

def test_info_endpoint() -> None:
    """ 
    GIVEN a FASTAPI application configured for testing
    WHEN the '/info' endpoint is requested (GET)
    THEN check for 200 status return code and that
        {"name": "Predicting Media Channel Sales", "description": "The goal of the project is take raw marketing telecommunication data from a fictitious company and build a predictive model for media sales. Essentially, this is a typical machine learning project which gets an extra zhuzh by building the model using a deep learning/neural network model and making a web accessible application to provide access to the predicative model."}
        was received
    """
    response = client.get("/info")
    assert response.status_code == 200
    assert response.json() == {"name": "Predicting Media Channel Sales", "description": "The goal of the project is take raw marketing telecommunication data from a fictitious company and build a predictive model for media sales. Essentially, this is a typical machine learning project which gets an extra zhuzh by building the model using a deep learning/neural network model and making a web accessible application to provide access to the predicative model."}

def test_default_endpoint() -> None:
    """ 
    GIVEN a FASTAPI application configured for testing
    WHEN the '/' endpoint is requested (GET)
    THEN check for 200 status return code and that title of the HTML is 
         'Sales Prediction Input'
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Sales Prediction Input</title>" in response.text

def test_plot_sales_endpoint() -> None:
    """ 
    GIVEN a FASTAPI application configured for testing
    WHEN the '/plot-sales' endpoint is requested (GET)
    THEN check for 200 status return code and that 
        an image was produced by checking the response type
    """
    response = client.get("/plot-sales")
    assert response.status_code == 200
    assert  response.headers["Content-Type"] == "image/png"

def test_predict_sales_endpoint() -> None:
    """ 
    GIVEN a FASTAPI application configured for testing and
    WHEN the 'predict_sales' endpoint is requested (POST) and the following values are present:
        * digital_budget
        * tv_budget
        * radio_budget
        * newspaper_budget
    THEN check for 200 status return code and that a JSON response of
        "{predicted_sales": prediction} was returned with prediction being the
        Deep Learning prediction model's prediction value.
    """
    # See requestData in index.html for structure the post expects to see
    post_data = {
        "digital": 1000.0,
        "tv": 1000.0,
        "radio": 1000.0,
        "newspaper": 1000
    }
    response = client.post("/predict_sales", json=post_data) #use JSON here
    response_data = response.json()
    assert response.status_code == 200
    assert isinstance(response_data, dict) # Check that response type structure
    assert response.headers["Content-Type"] == "application/json"
    assert "predicted_sales" in response_data # Check that key is present