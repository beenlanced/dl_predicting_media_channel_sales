# Predicting Media Channel Sales Using Deep Learning with Tensorflow and Keras

<p>
  <img alt="Telecom Media Channels" src="imgs/media_channels.jpeg"/>
</p>

[img source: Pinterest](https://www.pinterest.com/pin/mess-media--731131320746574258/)

![Code Coverage](https://img.shields.io/badge/coverage-100%25-green)
[![DL Prediction Tests](https://github.com/beenlanced/dl_predicting_media_channel_sales/actions/workflows/dl_prediction_test.yml/badge.svg)](https://github.com/beenlanced/dl_predicting_media_channel_sales/actions/workflows/dl_prediction_test.yml)

## Project Description

This project came out of the `LinkedIn Learning course`: [Deep Learning and Generative AI with Python: Data Processing and Analytics](https://www.linkedin.com/learning/deep-learning-and-generative-ai-data-prep-analysis-and-visualization-with-python)

The goal of the project is to take raw marketing telecommunication data from a fictitious company and build a predictive model for media sales. Essentially, this is a typical machine learning project which gets an extra zhuzh by building the model using a deep learning/neural network model and making a web accessible application to provide access to the predictive model.

### What this Project Does Specifically

The project:

- Loads and inspects the marketing telecommunication data
- Preprocesses/cleans the data
- Performs exploratory data analysis
- Creates a deep learning network to build the prediction model
- Conducts analysis of the predictive model results
- Makes any improvements
- Makes the predicative model accessible via a web application
- Hosts the web application using the Amazon Web Services - Elastic Cloud Service (AWS-ECS).

---

## Objective

The project contains the key elements:

- `AWS-ECS`, Amazon Web Services - Elastic Cloud Service, runs virtual machine to render the semantic search web application supplied by building a Docker container from a deployed Docker image from Docker hub,
- `CI/CD` automated pipeline created using `GitHub Actions`,
- `Deep Learning` for neural networks building,
- `FastAPI` to render the app,
- `Functional tests` and `unit tests` using `pytest`,
- `Git` (version control),
- `Jupyter` Python coded notebooks,
- `Keras` to build nodes and layers,
- `Matplotlib` visualization of spaces,
- `Numpy` for arrays and numerical operations,
- `Pandas` for dataframe usage,
- `Python` the standard modules,
- `Pydantic` to define structure of incoming request body and validate input data,
- `Scikit-Learn` to get training and test datasets,
- `TensorFlow` to build nodes and layers,
- `Type` hinting using `Pylance`,
- `Web application` using `FastAPI`,
- `uv` package management including use of `ruff` for linting and formatting, and
- ***

## Tech Stack

![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?logo=amazon-web-services&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)
![HTML](https://img.shields.io/badge/HTML-%23E34F26.svg?logo=html5&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=white)
![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![YAML](https://img.shields.io/badge/YAML-CB171E?logo=yaml&logoColor=fff)

---

## Getting Started

Here are some instructions to help you set up this project locally.

---

## Installation Steps

The Python version used for this project is `Python 3.11` to be compatible with TensorFlow.

Follow the requirements for using TensorFlow [here](https://www.tensorflow.org/install/pip#macos)

use `uv pip install tensorflow`

- Make sure to use python versions `Python 3.9–3.12
- pip version 19.0 or higher for Linux (requires manylinux2014 support) and Windows. pip version 20.3 or higher for macOS.
- Windows Native Requires Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019

### Clone the Repo

1. Clone the repo (or download it as a zip file):

   ```bash
   git clone https://github.com/beenlanced/dl_predicting_media_channel_sales.git
   ```

2. Create a virtual environment named `.venv` using `uv` Python version 3.11:

   ```bash
   uv venv --python=3.11
   ```

3. Activate the virtual environment: `.venv`

   On macOs and Linux:

   ```bash
   source .venv/bin/activate #mac
   ```

   On Windows:

   ```bash
    # In cmd.exe
    venv\Scripts\activate.bat
   ```

4. Install packages using `pyproject.toml` or (see special notes section)

   ```bash
   uv pip install -r pyproject.toml
   ```

### Running the App

1. **Run the App**

   - Start the project by running the appropriate command at the `src` directory
     ```
     uv run fastapi dev app.py
     ```
     or if at root level
     ```
     uv run fastapi dev src/app.py
     ```
     or
     ```
     uv uvicorn app:app --reload
     ```

2. **Access the Project's Web Interface**

   - Open a web browser with the following url.
     - http://127.0.0.1:8000/

3. **Running the application** by entering the following URL's in a browser

   - The entry API: `http://127.0.0.1:8000/`
       <p>
           <img src="./imgs/entry_endpoint.png"/>
       </p>

     Entry point show that the app is working. A set of default budgets are present, but the user can change these values. To get a prediction of the sales simply press the predict button and the predicted sales value using the Deep Learning prediction model created in this project will be displayed.

   - The Info API: `http://127.0.0.1:8000/info`
       <p>
           <img src="./imgs/info_endpoint.png"/>
       </p>

     Provides overview of the application.

   - The plot-forecast API: `http://127.0.0.1:8000/plot-sales`

       <p>
           <img src="./imgs/plot_sales_endpoint.png"/>
       </p>

     Plots the ingested telecommunication sales data at various start dates. This sales data is the target or label data used in the training and testing datasets used to build the prediction model.

---

## Special Notes

- Note the Docker file here also has to specify Python 3.11 because of the `TensorFlow` restrictions with Python 3.13 as of the time of this project's creation.

---

### Final Words

Thanks for visiting.

Give the project a star (⭐) if you liked it or if it was helpful to you!

You've `beenlanced`! 😉

---

## Acknowledgements

I would like to extend my gratitude to all the individuals and organizations who helped in the development and success of this project. Your support, whether through contributions, inspiration, or encouragement, have been invaluable. Thank you.

Specifically, I would like to acknowledge:

As this work was a project for the [LinkedIn Learning course: Deep Learning and Generative AI with Python: Data Processing and Analytics](https://www.linkedin.com/learning/deep-learning-and-generative-ai-data-prep-analysis-and-visualization-with-python), a shout out to the folks at LinkedIn Learning.

- [Hema Kalyan Murapaka](https://www.linkedin.com/in/hemakalyan) and [Benito Martin](https://martindatasol.com/blog) for sharing their README.md templates upon which I have derived my README.md.

- The folks at Astral for their UV [documentation](https://docs.astral.sh/uv/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details
